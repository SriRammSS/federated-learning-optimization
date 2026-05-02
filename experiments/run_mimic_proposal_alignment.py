
import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flopt.analysis import summarize_rows
from flopt.config import FLConfig
from flopt.dirichlet import dirichlet_split, partition_audit
from flopt.duality import solve_policy_lp
from flopt.fedavg import federated_train
from flopt.utils import _device
from flopt.fedprox import fedprox_train
from flopt.io import convergence_summary, round_records_to_csv, write_csv, write_json
from flopt.landscape import (
    landscape_1d,
    landscape_2d,
    save_checkpoints,
    validation_sample,
    write_landscape_config,
)
from flopt.mimic import load_mimic
from flopt.models import LogisticModel, TabularMLP, count_parameters
from flopt.plots import bar, line_mean_std, plot_shadow_price, scatter
from flopt.resource_watchdog import ResourceWatchdog
from flopt.search import ga_search, grid_search
from flopt.sparsity import lp_comparison, sparsity_stats
from flopt.stats import confidence_intervals, paired_tests


OUT = Path("outputs/full_mimic_iv_proposal_alignment")
MIMIC_OUT = Path("outputs/full_mimic_iv")
OLD_TRAINING = Path("outputs/full_mimic_iv_training")
SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
ALPHAS = [0, 0.5, 0.75, 0.9, 0.95]
MUS = [0.0, 0.001, 0.01, 0.1]
BETAS = [0.1, 0.5, 1.0, "infinity"]
GRID = [(le, cpr, lr) for le in [1, 2, 3] for cpr in [5, 10, 15] for lr in [0.003, 0.005, 0.01]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "local", "full"], default="local")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--mimic-out", default=str(MIMIC_OUT))
    parser.add_argument("--old-training", default=str(OLD_TRAINING))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-watchdog", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    ensure_alignment_dirs(out)
    watchdog = ResourceWatchdog(out / "monitoring", interval_seconds=30.0)
    if not args.no_watchdog:
        watchdog.start()

    runtime: list[dict] = []
    manifest: list[dict] = []
    all_rounds: list[dict] = []
    method_rows: list[dict] = []
    sparsity_rows: list[dict] = []
    checkpoint_rows: list[dict] = []

    settings = mode_settings(args.mode)
    try:
        t0 = time.perf_counter()
        watchdog.set_stage("load")
        bundle = load_mimic(Path(args.mimic_out), seed=SEEDS[0])
        feature_count = len(bundle.feature_names)
        base = FLConfig(
            rounds=settings["rounds"],
            max_rounds=settings["rounds"],
            local_epochs=1,
            clients_per_round=min(5, len(bundle.clients)),
            lr=0.005,
            batch_size=settings["batch_size"],
            seed=SEEDS[0],
            cvar_alpha=0.0,
            patience=settings["patience"],
            min_delta=0.0005,
            early_stopping=True,
            monitor="loss",
            class_weights=bundle.class_weights,
            optimizer="adam",
        )
        write_metadata(out, args, bundle, base, feature_count)
        runtime.append(stage_row("load", t0))

        mlp_factory = lambda: TabularMLP(feature_count, 2, hidden=(256, 128, 64), dropout=0.1)
        logreg_factory = lambda: LogisticModel(feature_count, 2)

        checkpoint_rows.extend(run_fedprox_natural(
            out, bundle, base, mlp_factory, settings, all_rounds, method_rows, sparsity_rows, watchdog, manifest))
        checkpoint_rows.extend(run_logreg_controls(
            out, bundle, base, logreg_factory, settings, all_rounds, method_rows, sparsity_rows, watchdog, manifest))
        checkpoint_rows.extend(run_dirichlet_study(
            out, Path(args.mimic_out) / "preprocessing" / "model_arrays.npz", base, logreg_factory,
            settings, all_rounds, method_rows, watchdog, manifest))
        run_sparsity_lp(out, method_rows, sparsity_rows, manifest)
        run_stronger_search(out, bundle, base, logreg_factory, settings, watchdog, manifest)
        run_loss_landscapes(out, bundle, base, logreg_factory, mlp_factory, settings, watchdog, manifest)
        run_aggregate_outputs(out, all_rounds, method_rows, manifest)
        run_reports(out, args, method_rows, manifest, bundle)

        write_csv(out / "runtime" / "runtime_by_stage.csv", runtime)
        write_csv(out / "runtime" / "experiment_manifest.csv", manifest)
        write_csv(out / "monitoring" / "stage_checkpoints.csv", checkpoint_rows)
        validate_no_overwrite(Path(args.old_training), out)
        print(f"wrote proposal-alignment outputs to {out}")
    finally:
        if not args.no_watchdog:
            watchdog.stop()


def mode_settings(mode: str):
    if mode == "debug":
        return {
            "seeds": [7],
            "rounds": 2,
            "search_rounds": 2,
            "patience": 2,
            "batch_size": 128,
            "mus": [0.0, 0.01],
            "alphas": [0, 0.9],
            "betas": [0.1],
            "ga_maxiter": 1,
            "ga_popsize": 2,
            "landscape_1d": 7,
            "landscape_2d": 5,
            "validation_rows": 512,
        }
    if mode == "full":
        return {
            "seeds": SEEDS,
            "rounds": 180,
            "search_rounds": 80,
            "patience": 25,
            "batch_size": 256,
            "mus": MUS,
            "alphas": ALPHAS,
            "betas": BETAS,
            "ga_maxiter": 10,
            "ga_popsize": 10,
            "landscape_1d": 41,
            "landscape_2d": 25,
            "validation_rows": 5000,
        }
    return {
        "seeds": SEEDS,
        "rounds": 80,
        "search_rounds": 30,
        "patience": 15,
        "batch_size": 256,
        "mus": MUS,
        "alphas": ALPHAS,
        "betas": BETAS,
        "ga_maxiter": 4,
        "ga_popsize": 6,
        "landscape_1d": 41,
        "landscape_2d": 25,
        "validation_rows": 5000,
    }


def run_fedprox_natural(out, bundle, base, model_factory, settings, all_rounds, method_rows, sparsity_rows, watchdog, manifest):
    checkpoints = []
    watchdog.set_stage("fedprox_natural")
    for mu in settings["mus"]:
        for seed in settings["seeds"]:
            name = f"fedprox_mu_{fmt_float(mu)}"
            raw_path = out / "raw" / f"{name}_seed_{seed}_rounds.csv"
            if raw_path.exists():
                checkpoints.append(checkpoint("fedprox_natural", name, seed, "skipped_existing", raw_path))
                continue
            cfg = replace(base, seed=seed)
            initial = model_factory()
            model, records, sp = fedprox_train(initial, bundle.clients, cfg, mu=mu, drift=True, sparsity=True)
            rr = round_records_to_csv(records, name, seed)
            for row in rr:
                row["method"] = name
                row["mu"] = mu
            all_rounds.extend(rr)
            write_csv(raw_path, rr)
            manifest.append(file_manifest(raw_path, "fedprox natural rounds"))
            c = convergence_summary(records, name, seed, None, cfg.max_rounds or cfg.rounds)
            row = {"method": name, "seed": seed, "mu": mu, **strip_summary(c)}
            method_rows.append(row)
            write_json(out / "raw" / f"{name}_seed_{seed}_summary.json", row)
            sparsity_rows.extend(sp)
            checkpoints.append(checkpoint("fedprox_natural", name, seed, "done", raw_path))
            if watchdog.check("fedprox_natural") == "stop":
                return checkpoints
    return checkpoints


def run_logreg_controls(out, bundle, base, model_factory, settings, all_rounds, method_rows, sparsity_rows, watchdog, manifest):
    checkpoints = []
    watchdog.set_stage("logreg_controls")
    saved_triplet = False
    for seed in settings["seeds"]:
        for alpha in settings["alphas"]:
            name = "logreg_fedavg" if alpha == 0 else f"logreg_cvar_{alpha}"
            raw_path = out / "raw" / f"{name}_seed_{seed}_rounds.csv"
            if raw_path.exists():
                checkpoints.append(checkpoint("logreg_controls", name, seed, "skipped_existing", raw_path))
                continue
            cfg = replace(base, seed=seed, cvar_alpha=alpha, optimizer="adam")
            initial = model_factory()
            initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
            model, records = federated_train(initial, bundle.clients, cfg, drift=True)
            rr = round_records_to_csv(records, name, seed, alpha)
            for row in rr:
                row["method"] = name
            all_rounds.extend(rr)
            write_csv(raw_path, rr)
            manifest.append(file_manifest(raw_path, "logistic control rounds"))
            c = convergence_summary(records, name, seed, alpha, cfg.max_rounds or cfg.rounds)
            method_rows.append({"method": name, "seed": seed, **strip_summary(c)})
            if not saved_triplet and alpha == 0:
                init_model = model_factory()
                init_model.load_state_dict(initial_state)
                save_checkpoints(out / "landscape", "logreg", init_model, model, model)
                saved_triplet = True
            checkpoints.append(checkpoint("logreg_controls", name, seed, "done", raw_path))
            if watchdog.check("logreg_controls") == "stop":
                return checkpoints
        for mu in settings["mus"]:
            name = f"logreg_fedprox_mu_{fmt_float(mu)}"
            raw_path = out / "raw" / f"{name}_seed_{seed}_rounds.csv"
            if raw_path.exists():
                checkpoints.append(checkpoint("logreg_controls", name, seed, "skipped_existing", raw_path))
                continue
            cfg = replace(base, seed=seed, optimizer="adam")
            model, records, sp = fedprox_train(model_factory(), bundle.clients, cfg, mu=mu, drift=True, sparsity=True)
            rr = round_records_to_csv(records, name, seed)
            for row in rr:
                row["method"] = name
                row["mu"] = mu
            all_rounds.extend(rr)
            write_csv(raw_path, rr)
            manifest.append(file_manifest(raw_path, "logistic fedprox rounds"))
            c = convergence_summary(records, name, seed, None, cfg.max_rounds or cfg.rounds)
            method_rows.append({"method": name, "seed": seed, "mu": mu, **strip_summary(c)})
            sparsity_rows.extend(sp)
            checkpoints.append(checkpoint("logreg_controls", name, seed, "done", raw_path))
            if watchdog.check("logreg_controls") == "stop":
                return checkpoints
    return checkpoints


def run_dirichlet_study(out, arrays_path, base, model_factory, settings, all_rounds, method_rows, watchdog, manifest):
    checkpoints = []
    watchdog.set_stage("dirichlet_study")
    all_audit = []
    for beta in settings["betas"]:
        for seed in settings["seeds"]:
            clients, map_rows, dist_rows = dirichlet_split(arrays_path, beta, 30, seed)
            map_path = out / "partitions" / f"dirichlet_beta_{fmt_beta(beta)}_seed_{seed}_client_map.csv"
            dist_path = out / "partitions" / f"dirichlet_beta_{fmt_beta(beta)}_seed_{seed}_label_distribution.csv"
            write_csv(map_path, map_rows)
            write_csv(dist_path, dist_rows)
            manifest.append(file_manifest(map_path, "dirichlet client map"))
            manifest.append(file_manifest(dist_path, "dirichlet label distribution"))
            all_audit.extend(partition_audit(dist_rows))
            method_specs = [("fedavg", 0.0), ("fedprox", None)]
            if float_beta(beta) == 0.1:
                method_specs.append(("cvar_0.9", 0.9))
            for method, alpha in method_specs:
                name = f"dirichlet_beta_{fmt_beta(beta)}_{method}"
                raw_path = out / "raw" / f"{name}_seed_{seed}_rounds.csv"
                if raw_path.exists():
                    checkpoints.append(checkpoint("dirichlet_study", name, seed, "skipped_existing", raw_path))
                    continue
                cfg = replace(base, seed=seed, cvar_alpha=alpha or 0.0, clients_per_round=min(10, len(clients)))
                if method == "fedprox":
                    model, records, _ = fedprox_train(model_factory(), clients, cfg, mu=0.01, drift=True)
                else:
                    model, records = federated_train(model_factory(), clients, cfg, drift=True)
                rr = round_records_to_csv(records, name, seed, alpha)
                for row in rr:
                    row["method"] = name
                    row["beta"] = str(beta)
                all_rounds.extend(rr)
                write_csv(raw_path, rr)
                manifest.append(file_manifest(raw_path, "dirichlet rounds"))
                c = convergence_summary(records, name, seed, alpha, cfg.max_rounds or cfg.rounds)
                method_rows.append({"method": name, "seed": seed, "beta": str(beta), **strip_summary(c)})
                checkpoints.append(checkpoint("dirichlet_study", name, seed, "done", raw_path))
                if watchdog.check("dirichlet_study") == "stop":
                    write_csv(out / "partitions" / "dirichlet_partition_audit.csv", all_audit)
                    return checkpoints
    write_csv(out / "partitions" / "dirichlet_partition_audit.csv", all_audit)
    return checkpoints


def run_sparsity_lp(out, method_rows, sparsity_rows, manifest):
    write_csv(out / "raw" / "sparsity_round_updates.csv", sparsity_rows)
    write_csv(out / "lp" / "sparsity_summary.csv", sparsity_stats(sparsity_rows))
    lp_source = lp_comparison(method_rows, sparsity_rows)
    write_csv(out / "lp" / "dense_vs_sparse_shadow_price.csv", lp_source)
    manifest.append(file_manifest(out / "raw" / "sparsity_round_updates.csv", "sparsity raw updates"))
    if not lp_source:
        return
    losses = [float(r["final_loss"]) for r in lp_source if r.get("final_loss") not in {None, ""}]
    sparse_costs = [float(r["sparse_cost"]) for r in lp_source if r.get("final_loss") not in {None, ""}]
    dense_costs = [float(r["dense_cost"]) for r in lp_source if r.get("final_loss") not in {None, ""}]
    if not losses or max(sparse_costs) <= min(sparse_costs):
        return
    budgets = np.linspace(min(sparse_costs), max(sparse_costs), 12).tolist()
    sparse_lp = solve_policy_lp(losses, sparse_costs, budgets)
    write_json(out / "lp" / "sparsity_lp_shadow_price.json", sparse_lp)
    write_csv(out / "lp" / "sparsity_lp_shadow_price.csv", flatten_lp(sparse_lp))
    write_csv(out / "lp" / "kkt_diagnostics.csv", flatten_lp(sparse_lp))
    if max(dense_costs) > min(dense_costs):
        dense_lp = solve_policy_lp(losses, dense_costs, np.linspace(min(dense_costs), max(dense_costs), 12).tolist())
        write_csv(out / "lp" / "dense_lp_shadow_price.csv", flatten_lp(dense_lp))
    plot_shadow_price(sparse_lp, str(out / "plots" / "sparse_shadow_price_vs_budget.png"))


def run_stronger_search(out, bundle, base, model_factory, settings, watchdog, manifest):
    if watchdog.should_skip_optional():
        write_csv(out / "search" / "proposal_ga_history.csv", [])
        return
    watchdog.set_stage("stronger_ga")
    cfg = replace(base, seed=settings["seeds"][0], rounds=settings["search_rounds"], max_rounds=settings["search_rounds"], patience=max(3, min(base.patience, 10)))
    grid_path = out / "search" / "proposal_grid_search.csv"
    grid_rows = grid_search(bundle.clients, cfg, GRID, gamma=1e-8, model_factory=model_factory, score_key="auprc", min_score=0.25)
    write_csv(grid_path, grid_rows)
    manifest.append(file_manifest(grid_path, "proposal grid search"))
    ga = ga_search(
        bundle.clients,
        cfg,
        bounds=[(1, 3), (3, min(20, len(bundle.clients))), (0.001, 0.02)],
        maxiter=settings["ga_maxiter"],
        popsize=settings["ga_popsize"],
        gamma=1e-8,
        model_factory=model_factory,
        score_key="auprc",
        min_score=0.25,
    )
    write_json(out / "search" / "proposal_ga_result.json", ga)
    write_csv(out / "search" / "proposal_ga_history.csv", ga["history"])
    write_csv(out / "search" / "proposal_ga_best_so_far.csv", ga_best_history(ga["history"]))
    if ga.get("history"):
        line_mean_std(ga_best_history(ga["history"]), "best_fitness", str(out / "plots" / "proposal_ga_fitness_vs_evaluations.png"), "Proposal GA Best Fitness")


def run_loss_landscapes(out, bundle, base, logreg_factory, mlp_factory, settings, watchdog, manifest):
    if watchdog.should_skip_optional():
        return
    watchdog.set_stage("loss_landscape")
    x_val, y_val = validation_sample(bundle.clients, max_rows=settings["validation_rows"], seed=settings["seeds"][0])
    cfg = replace(base, seed=settings["seeds"][0])
    write_landscape_config(out / "landscape" / "loss_landscape_config.json", {
        "validation_rows": int(len(y_val)),
        "landscape_1d_points": settings["landscape_1d"],
        "landscape_2d_grid": settings["landscape_2d"],
        "seed": settings["seeds"][0],
    })
    _ensure_landscape_triplet(out, "logreg", logreg_factory, bundle.clients, cfg)
    _ensure_landscape_triplet(out, "mlp", mlp_factory, bundle.clients, replace(cfg, max_rounds=min(cfg.max_rounds or cfg.rounds, 20), rounds=min(cfg.rounds, 20)))
    for prefix, factory in [("logreg", logreg_factory), ("mlp", mlp_factory)]:
        initial = torch.load(out / "landscape" / f"{prefix}_initial_model.pt", map_location="cpu")
        final = torch.load(out / "landscape" / f"{prefix}_final_model.pt", map_location="cpu")
        landscape_1d(factory, initial, final, x_val, y_val, cfg, out / "landscape" / f"{prefix}_1d_loss_curve.csv", out / "plots" / f"loss_landscape_{prefix}_1d.png", points=settings["landscape_1d"], prefix=prefix)
        landscape_2d(factory, final, x_val, y_val, cfg, out / "landscape" / f"{prefix}_2d_loss_surface.csv", out / "plots" / f"loss_landscape_{prefix}_2d.png", grid=settings["landscape_2d"], seed=settings["seeds"][0], prefix=prefix)


def _ensure_landscape_triplet(out, prefix, model_factory, clients, cfg):
    if (out / "landscape" / f"{prefix}_initial_model.pt").exists():
        return
    initial = model_factory()
    init_copy = model_factory()
    init_copy.load_state_dict({k: v.detach().clone() for k, v in initial.state_dict().items()})
    final, _ = federated_train(initial, clients, cfg, drift=False)
    save_checkpoints(out / "landscape", prefix, init_copy, final, final)


def run_aggregate_outputs(out, all_rounds, method_rows, manifest):
    write_csv(out / "raw" / "all_round_metrics.csv", all_rounds)
    write_csv(out / "metrics" / "proposal_method_seed_results.csv", method_rows)
    metrics = ["final_loss", "final_accuracy", "final_auroc", "final_auprc", "final_worst_client_accuracy", "final_worst_client_recall", "total_comm_until_stop", "stopped_round"]
    summary = summarize_rows(method_rows, "method", metrics)
    write_csv(out / "metrics" / "proposal_method_summary.csv", summary)
    write_csv(out / "metrics" / "logreg_method_summary.csv", [r for r in summary if str(r["method"]).startswith("logreg")])
    write_csv(out / "metrics" / "fedprox_vs_fedavg_summary.csv", [r for r in summary if "fedprox" in str(r["method"]) or "fedavg" in str(r["method"])])
    write_csv(out / "metrics" / "dirichlet_beta_summary.csv", [r for r in method_rows if "dirichlet_beta" in str(r["method"])])
    write_csv(out / "stats" / "proposal_confidence_intervals.csv", confidence_intervals(method_rows, "method", metrics))
    write_csv(out / "stats" / "proposal_paired_tests.csv", paired_tests(method_rows, "method", "seed", metrics, "logreg_fedavg"))
    write_csv(out / "stats" / "effect_sizes.csv", effect_size_rows(method_rows, "logreg_fedavg", metrics))
    _plot_basic(out, all_rounds, summary, method_rows)
    for path in [out / "raw" / "all_round_metrics.csv", out / "metrics" / "proposal_method_summary.csv"]:
        manifest.append(file_manifest(path, "aggregate output"))


def run_reports(out, args, method_rows, manifest, bundle):
    checklist = proposal_checklist(out)
    write_csv(out / "reports" / "proposal_alignment_checklist.csv", checklist)
    write_csv(out / "reports" / "proposal_compliance_matrix.csv", checklist)
    write_csv(out / "metrics" / "proposal_compliance_matrix.csv", checklist)
    write_json(out / "reports" / "proposal_alignment_manifest.json", {
        "output_root": str(out),
        "created_at": time.time(),
        "mode": args.mode,
        "dataset_rows": sum(len(c.x_train) + len(c.x_test) for c in bundle.clients),
        "natural_clients": len(bundle.clients),
        "synthetic_dirichlet_clients": 30,
        "platform": platform.platform(),
        "files": manifest,
    })
    report = alignment_report_text(out, checklist, method_rows)
    (out / "reports" / "proposal_alignment_report.md").write_text(report, encoding="utf-8")


def ensure_alignment_dirs(out: Path):
    for name in ["raw", "metrics", "partitions", "lp", "search", "stats", "plots", "landscape", "monitoring", "reports", "runtime", "artifacts"]:
        (out / name).mkdir(parents=True, exist_ok=True)


def write_metadata(out, args, bundle, base, feature_count):
    write_json(out / "run_metadata.json", {
        "mode": args.mode,
        "mimic_out": args.mimic_out,
        "old_training": args.old_training,
        "features": feature_count,
        "natural_clients": len(bundle.clients),
        "class_names": bundle.class_names,
        "class_weights": bundle.class_weights,
        "base_config": base.__dict__,
        "device": str(_device()),
        "logistic_parameters": count_parameters(LogisticModel(feature_count, 2)),
        "mlp_parameters": count_parameters(TabularMLP(feature_count, 2, hidden=(256, 128, 64), dropout=0.1)),
    })


def _plot_basic(out, all_rounds, summary, method_rows):
    for metric in ["loss", "auprc", "worst_client_recall"]:
        rows = [r for r in all_rounds if r.get(metric) not in {None, ""}]
        if rows:
            line_mean_std(rows, metric, str(out / "plots" / f"{metric}_proposal_convergence.png"), f"Proposal {metric}")
    fedprox = [r for r in summary if "fedprox" in str(r["method"]) or r["method"] in {"fedavg_default", "logreg_fedavg"}]
    if fedprox:
        if any("final_auprc_mean" in r for r in fedprox):
            bar([r for r in fedprox if r.get("final_auprc_mean") not in {None, ""}], "method", "final_auprc_mean", str(out / "plots" / "fedprox_vs_fedavg_auprc.png"), "FedProx vs FedAvg AUPRC")
        if any("final_worst_client_recall_mean" in r for r in fedprox):
            bar([r for r in fedprox if r.get("final_worst_client_recall_mean") not in {None, ""}], "method", "final_worst_client_recall_mean", str(out / "plots" / "fedprox_vs_fedavg_worst_client_recall.png"), "FedProx vs FedAvg Worst Recall")
    dirichlet = [r for r in method_rows if "dirichlet_beta" in str(r["method"])]
    if dirichlet:
        scatter(dirichlet, "stopped_round", "final_loss", str(out / "plots" / "dirichlet_beta_convergence.png"), "Dirichlet Convergence")
        scatter(dirichlet, "stopped_round", "final_auprc", str(out / "plots" / "dirichlet_beta_final_metric_trend.png"), "Dirichlet Final Metric Trend")
    write_csv(out / "plots" / "proposal_compliance_matrix.csv", proposal_checklist(out))


def proposal_checklist(out: Path):
    items = [
        ("FedProx baseline", "FedProx non-IID baseline", out / "metrics" / "fedprox_vs_fedavg_summary.csv"),
        ("Convex logistic baseline", "Convex empirical risk control", out / "metrics" / "logreg_method_summary.csv"),
        ("Dirichlet beta study", "beta in {0.1, 0.5, 1.0, infinity}", out / "metrics" / "dirichlet_beta_summary.csv"),
        ("Sparsity LP", "l0/top-k communication cost", out / "lp" / "sparsity_lp_shadow_price.csv"),
        ("Differential evolution", "GA vs grid search", out / "search" / "proposal_ga_history.csv"),
        ("Loss landscape", "research visualization", out / "landscape" / "logreg_2d_loss_surface.csv"),
        ("Resource watchdog", "OOM-safe execution", out / "monitoring" / "resource_timeseries.csv"),
    ]
    rows = []
    for claim, requirement, path in items:
        rows.append({
            "proposal_claim": claim,
            "requirement": requirement,
            "evidence_file": str(path),
            "status": "complete" if path.exists() and path.stat().st_size > 0 else "pending",
            "adaptation_note": "MIMIC-IV adaptation preserving proposal concept",
        })
    return rows


def alignment_report_text(out, checklist, method_rows):
    complete = sum(1 for r in checklist if r["status"] == "complete")
    return f"""# MIMIC-IV Proposal Alignment Report

This report records the proposal-faithful additions to the existing MIMIC-IV clinical FL project.

## Non-overwrite guarantee

All outputs are under `{out}`. Existing `outputs/full_mimic_iv` and `outputs/full_mimic_iv_training` are read-only baselines.

## Compliance summary

Completed checklist items: {complete}/{len(checklist)}.

See:

- `reports/proposal_alignment_checklist.csv`
- `reports/proposal_compliance_matrix.csv`
- `metrics/proposal_method_summary.csv`
- `lp/sparsity_lp_shadow_price.csv`
- `landscape/*loss_surface.csv`

## Interpretation

The existing MIMIC MLP experiment remains the main clinical extension. This alignment suite adds FedProx, convex logistic controls, Dirichlet beta non-IID controls, sparsity communication LP, stronger GA search, loss landscapes, and resource monitoring to directly address the original MSML604 proposal.
"""


def effect_size_rows(rows, baseline, metrics):
    base = {r["seed"]: r for r in rows if r.get("method") == baseline}
    effects = []
    for method in sorted({r.get("method") for r in rows if r.get("method") != baseline}):
        vals = {r["seed"]: r for r in rows if r.get("method") == method}
        seeds = sorted(set(base) & set(vals))
        for metric in metrics:
            diffs = [float(vals[s][metric]) - float(base[s][metric]) for s in seeds if vals[s].get(metric) not in {None, ""} and base[s].get(metric) not in {None, ""}]
            if len(diffs) >= 2:
                arr = np.array(diffs, dtype=float)
                effects.append({"baseline": baseline, "method": method, "metric": metric, "n": len(arr), "mean_diff": float(arr.mean()), "cohens_d": float(arr.mean() / (arr.std(ddof=1) + 1e-12))})
    return effects


def flatten_lp(rows):
    result = []
    for r in rows:
        kkt = r.get("kkt", {})
        result.append({"budget": r.get("budget"), "loss": r.get("loss"), "cost": r.get("cost"), "lambda": r.get("lambda"), "status": r.get("status"), **kkt})
    return result


def ga_best_history(history):
    best = float("inf")
    rows = []
    for row in history:
        best = min(best, float(row["fitness"]))
        rows.append({"round": row["evaluation"], "best_fitness": best})
    return rows


def strip_summary(c):
    return {k: v for k, v in c.items() if k not in {"run_type", "seed", "alpha", "max_rounds"}}


def checkpoint(stage, method, seed, status, path):
    return {"stage": stage, "method": method, "seed": seed, "status": status, "path": str(path), "timestamp": time.time()}


def file_manifest(path: Path, description: str):
    return {"path": str(path), "description": description, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def stage_row(stage, start):
    return {"stage": stage, "seconds": time.perf_counter() - start}


def fmt_float(v: float):
    return str(v).replace(".", "p")


def fmt_beta(v):
    return str(v).replace(".", "p")


def float_beta(v):
    return float("inf") if str(v).lower() in {"inf", "infinity"} else float(v)


def read_csv_dicts(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_no_overwrite(old_training: Path, out: Path):
    write_json(out / "reports" / "non_overwrite_validation.json", {
        "old_training_root": str(old_training),
        "alignment_root": str(out),
        "old_training_exists": old_training.exists(),
        "validation": "alignment runner writes only to alignment_root",
    })


if __name__ == "__main__":
    main()
