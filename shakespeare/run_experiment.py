#!/usr/bin/env python3
"""
Full Shakespeare FL Experiment — GPU-optimized, 100% self-contained.

10 seeds x (FedAvg + 3 CVaR + Centralized + Local-only) + Grid Search + GA Search + LP Duality
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from shakespeare.config import ShakespeareConfig
from shakespeare.models import CharTransformer, count_parameters
from shakespeare.data_loader import load_shakespeare
from shakespeare.training import (federated_train, centralized_train,
                                   local_only_summary, predict_clients,
                                   evaluate_all, _device)
from shakespeare.search import ga_search, grid_search
from shakespeare.eda import run_eda
from shakespeare.utils import (
    solve_policy_lp, write_json, write_csv, flatten_round_records,
    convergence_summary, confidence_rows, paired_tests,
    plot_convergence, plot_cvar, plot_shadow_price, plot_search_comparison,
    line_mean_std, bar,
)

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "outputs"

SEEDS = list(range(10))
CVaR_ALPHAS = [0.0, 0.1, 0.3, 0.5]

BASE = ShakespeareConfig(
    rounds=100, max_rounds=100, local_epochs=1, clients_per_round=10,
    lr=0.8, batch_size=256, cvar_alpha=0.0, patience=15, min_delta=0.001,
    early_stopping=True, monitor="loss", optimizer="sgd",
    vocab_size=80, d_model=128, nhead=4, num_layers=2, dim_ff=256,
    grad_clip=5.0, use_amp=True, num_workers=4,
)


def main():
    t0 = time.time()
    device = _device()
    print("=" * 70)
    print("SHAKESPEARE FL EXPERIMENT  (CharTransformer, GPU-optimized)")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("=" * 70)

    # ── 1. Load data ──
    print("\n[1/7] Loading & preprocessing data ...")
    bundle = load_shakespeare()
    clients = bundle.clients
    cfg = replace(BASE, vocab_size=bundle.vocab_size)
    print(f"  {len(clients)} clients, vocab_size={bundle.vocab_size}, "
          f"total_samples={bundle.total_samples:,}")
    mf = lambda: CharTransformer(cfg.vocab_size, cfg.d_model, cfg.nhead,
                                  cfg.num_layers, cfg.dim_ff)
    print(f"  Model params: {count_parameters(mf()):,}")
    print(f"  AMP: {cfg.use_amp}, batch_size: {cfg.batch_size}, "
          f"num_workers: {cfg.num_workers}")

    # ── 2. EDA ──
    print("\n[2/7] Running EDA ...")
    run_eda(bundle)

    # ── 3. Multi-seed FL runs ──
    n_methods = len(CVaR_ALPHAS) + 2
    print(f"\n[3/7] Multi-seed training "
          f"({len(SEEDS)} seeds x {n_methods} methods) ...")
    all_round_rows: list[dict] = []
    all_conv: list[dict] = []
    cvar_summary_rows: list[dict] = []
    best_fedavg_model = None
    best_fedavg_loss = float("inf")

    for seed in SEEDS:
        seed_t0 = time.time()
        print(f"\n  ── Seed {seed} ──")
        seed_dir = _OUT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for alpha in CVaR_ALPHAS:
            tag = "fedavg" if alpha == 0.0 else f"cvar_{alpha}"
            print(f"    {tag} ...", flush=True)
            c = replace(cfg, seed=seed, cvar_alpha=alpha)
            model, recs = federated_train(
                mf(), clients, c, track_drift=(alpha == 0.0))
            rr = flatten_round_records(recs, tag, seed, alpha)
            all_round_rows.extend(rr)
            cs = _conv_summary(recs, tag, seed, alpha,
                               c.max_rounds or c.rounds)
            all_conv.append(cs)
            write_csv(seed_dir / f"{tag}_rounds.csv", rr)
            write_json(seed_dir / f"{tag}_summary.json", cs)
            last = recs[-1]
            print(f"    -> loss={last['loss']:.4f} acc={last['accuracy']:.4f} "
                  f"ppl={last.get('perplexity', 0):.1f} rnd={last['round']}")
            if alpha == 0.0 and last["loss"] < best_fedavg_loss:
                best_fedavg_loss = last["loss"]
                best_fedavg_model = model
            if alpha > 0:
                cvar_summary_rows.append({
                    "seed": seed, "alpha": alpha,
                    "accuracy": last["accuracy"],
                    "worst_client_accuracy": last["worst_client_accuracy"],
                    "perplexity": last.get("perplexity", 0),
                    "loss": last["loss"],
                })

        # Centralized
        print("    centralized ...", flush=True)
        c = replace(cfg, seed=seed, batch_size=512)
        _, recs = centralized_train(mf(), clients, c)
        rr = flatten_round_records(recs, "centralized", seed)
        all_round_rows.extend(rr)
        cs = _conv_summary(recs, "centralized", seed, None,
                           c.max_rounds or c.rounds)
        all_conv.append(cs)
        write_csv(seed_dir / "centralized_rounds.csv", rr)
        last = recs[-1]
        print(f"    -> loss={last['loss']:.4f} acc={last['accuracy']:.4f} "
              f"rnd={last['round']}")

        # Local-only
        print(f"    local_only ({len(clients)} clients) ...", flush=True)
        c = replace(cfg, seed=seed)
        lo_rows, lo_rr = local_only_summary(mf, clients, c)
        write_csv(seed_dir / "local_only.csv", lo_rows)
        mean_acc = float(np.mean([r["accuracy"] for r in lo_rows]))
        mean_loss = float(np.mean([r["loss"] for r in lo_rows]))
        all_conv.append({
            "run_type": "local_only", "seed": seed, "alpha": None,
            "max_rounds": c.max_rounds or c.rounds,
            "stopped_round": 0, "stopped_early": False,
            "best_round": 0, "best_loss": mean_loss,
            "final_loss": mean_loss, "final_accuracy": mean_acc,
            "final_worst_client_accuracy": float(
                np.min([r["accuracy"] for r in lo_rows])),
            "total_comm_until_stop": 0,
            "final_perplexity": float(
                np.mean([r.get("perplexity", 0) for r in lo_rows])),
        })
        print(f"    -> mean_acc={mean_acc:.4f} mean_loss={mean_loss:.4f}")

        elapsed_seed = time.time() - seed_t0
        print(f"  Seed {seed} done in {elapsed_seed / 60:.1f} min")

    # ── 4. Search ──
    print("\n[4/7] Hyperparameter search ...")
    search_cfg = replace(cfg, seed=42, rounds=30, max_rounds=30, patience=8)
    full_grid = [(le, cpr, lr)
                 for le in [1, 2, 3]
                 for cpr in [5, 10, 15]
                 for lr in [0.5, 0.8, 1.0]]
    print(f"  Grid search ({len(full_grid)} combos) ...")
    grid_rows = grid_search(clients, search_cfg, grid=full_grid,
                            model_factory=mf)
    (_OUT / "summary").mkdir(parents=True, exist_ok=True)
    write_csv(_OUT / "summary" / "grid_search.csv", grid_rows)
    print(f"  Best grid: fitness={grid_rows[0]['fitness']:.4f} "
          f"acc={grid_rows[0]['accuracy']:.4f}")

    print("  GA search (maxiter=5, popsize=6) ...")
    ga_result = ga_search(clients, search_cfg, maxiter=5, popsize=6,
                          model_factory=mf)
    write_json(_OUT / "summary" / "ga_search.json", ga_result)
    if ga_result.get("history"):
        write_csv(_OUT / "summary" / "ga_search.csv", ga_result["history"])
    print(f"  GA: fitness={ga_result['fitness']:.4f} "
          f"evals={ga_result['evaluations']}")

    # ── 5. LP shadow prices ──
    print("\n[5/7] LP shadow price analysis ...")
    fedavg_conv = [c for c in all_conv if c["run_type"] == "fedavg"]
    losses = [c["final_loss"] for c in fedavg_conv]
    costs = [c["total_comm_until_stop"] for c in fedavg_conv]
    if losses and costs:
        max_cost = max(costs) if costs else 1
        budgets = np.linspace(min(costs) * 0.5, max_cost * 1.5, 20).tolist()
        lp_rows = solve_policy_lp(losses, costs, budgets)
        write_json(_OUT / "summary" / "lp_shadow.json", lp_rows)
        lp_csv = [{k: v for k, v in r.items() if k != "weights"}
                  for r in lp_rows]
        write_csv(_OUT / "summary" / "lp_shadow.csv", lp_csv)
        plot_shadow_price(lp_rows,
                          str(_OUT / "plots" / "shadow_price.png"))
        n_opt = len([r for r in lp_rows
                     if r.get("status") == "optimal"])
        print(f"  {n_opt} optimal solutions")
    else:
        lp_rows = []
        print("  Skipped (no data)")

    # ── 6. Aggregate & plot ──
    print("\n[6/7] Aggregating results & plotting ...")
    write_csv(_OUT / "summary" / "all_round_rows.csv", all_round_rows)
    write_csv(_OUT / "summary" / "convergence_summary.csv", all_conv)

    _metrics = ["final_loss", "final_accuracy", "final_perplexity",
                "final_worst_client_accuracy"]
    ci_rows = confidence_rows(all_conv, "run_type", _metrics)
    write_csv(_OUT / "summary" / "confidence_intervals.csv", ci_rows)

    pt_rows = paired_tests(all_conv, "run_type", "seed", _metrics, "fedavg")
    write_csv(_OUT / "summary" / "paired_tests.csv", pt_rows)

    _OUT_P = _OUT / "plots"
    _OUT_P.mkdir(parents=True, exist_ok=True)

    for rt in ["fedavg", "centralized"]:
        rr = [r for r in all_round_rows if r["run_type"] == rt]
        if rr:
            line_mean_std(rr, "loss",
                          str(_OUT_P / f"{rt}_loss_mean_std.png"),
                          title=f"{rt} Loss")
            line_mean_std(rr, "accuracy",
                          str(_OUT_P / f"{rt}_acc_mean_std.png"),
                          title=f"{rt} Accuracy")
            ppl = [{**r, "perplexity": r.get("perplexity", 0)}
                   for r in rr if r.get("perplexity") is not None]
            if ppl:
                line_mean_std(ppl, "perplexity",
                              str(_OUT_P / f"{rt}_ppl_mean_std.png"),
                              title=f"{rt} Perplexity")
            seed0 = [r for r in rr if r["seed"] == 0]
            if seed0:
                plot_convergence(seed0,
                                 str(_OUT_P / f"{rt}_convergence_seed0.png"))

    if cvar_summary_rows:
        write_csv(_OUT / "summary" / "cvar_summary.csv", cvar_summary_rows)
        avg_by_alpha: dict[float, list] = {}
        for r in cvar_summary_rows:
            avg_by_alpha.setdefault(r["alpha"], []).append(r)
        cvar_plot = [{
            "alpha": a,
            "accuracy": float(np.mean([r["accuracy"] for r in rs])),
            "worst_client_accuracy": float(
                np.mean([r["worst_client_accuracy"] for r in rs])),
        } for a, rs in sorted(avg_by_alpha.items())]
        plot_cvar(cvar_plot, str(_OUT_P / "cvar_tradeoff.png"))

    if grid_rows and ga_result.get("history"):
        plot_search_comparison(grid_rows, ga_result,
                               str(_OUT_P / "search_comparison.png"))

    bar(all_conv, "run_type", "final_accuracy",
        str(_OUT_P / "method_accuracy_bar.png"),
        title="Final Accuracy by Method")
    bar(all_conv, "run_type", "final_loss",
        str(_OUT_P / "method_loss_bar.png"),
        title="Final Loss by Method")

    if best_fedavg_model is not None:
        print("  Per-client accuracy (best FedAvg) ...")
        preds = predict_clients(best_fedavg_model, clients)
        acc_by_client: dict[int, list] = {}
        for r in preds:
            acc_by_client.setdefault(r["client_id"], []).append(
                int(r["y_true"] == r["y_pred"]))
        client_acc_rows = [{
            "client": (_short(bundle.client_names[cid])
                       if cid < len(bundle.client_names) else str(cid)),
            "accuracy": float(np.mean(hits)), "count": len(hits),
        } for cid, hits in sorted(acc_by_client.items())]
        write_csv(_OUT / "summary" / "per_client_accuracy.csv",
                  client_acc_rows)
        bar(client_acc_rows, "client", "accuracy",
            str(_OUT_P / "per_client_accuracy.png"),
            title="Per-Client Accuracy (best FedAvg seed)", rotation=90)

    # Save artifacts
    art = _OUT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    write_json(art / "vocab.json", bundle.vocab)
    write_json(art / "client_names.json", bundle.client_names)
    write_json(art / "config.json", asdict(cfg))
    write_json(art / "experiment_meta.json", {
        "n_clients": len(clients), "vocab_size": bundle.vocab_size,
        "total_samples": bundle.total_samples,
        "model_params": count_parameters(mf()),
        "seeds": SEEDS, "cvar_alphas": CVaR_ALPHAS,
        "device": str(device),
        "gpu": (torch.cuda.get_device_name(0)
                if device.type == "cuda" else "N/A"),
        "model_type": "CharTransformer",
        "d_model": cfg.d_model, "nhead": cfg.nhead,
        "num_layers": cfg.num_layers, "dim_ff": cfg.dim_ff,
        "use_amp": cfg.use_amp, "batch_size": cfg.batch_size,
    })

    # ── 7. Done ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed / 60:.1f} min ({elapsed / 3600:.2f} hr)")
    print(f"Outputs: {_OUT}")
    print(f"{'=' * 70}")


def _conv_summary(records, run_type, seed, alpha, max_rounds):
    out = convergence_summary(records, run_type, seed, alpha, max_rounds)
    last = records[-1]
    if "perplexity" in last:
        out["final_perplexity"] = last["perplexity"]
    return out


def _short(name: str, maxlen: int = 20) -> str:
    return name[:maxlen] + "..." if len(name) > maxlen else name


if __name__ == "__main__":
    main()
