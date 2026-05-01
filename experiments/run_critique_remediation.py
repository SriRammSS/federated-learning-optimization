"""MIMIC-IV Critique Remediation Runner (Plan v6).

Implements experiments 3A, 1-9 from the remediation plan.
Usage:
    .venv/bin/python experiments/run_critique_remediation.py --mode exp3a
    .venv/bin/python experiments/run_critique_remediation.py --mode exp1
    .venv/bin/python experiments/run_critique_remediation.py --mode all
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flopt.config import FLConfig
from flopt.data import ClientData
from flopt.dirichlet import make_dirichlet_clients_from_arrays
from flopt.duality import solve_policy_lp
from flopt.fedavg import _device, evaluate_all, federated_train
from flopt.fedprox import fedprox_train
from flopt.io import convergence_summary, flatten_round_records, write_csv, write_json
from flopt.mimic import load_mimic_iv_arrays
from flopt.models import LogisticModel, TabularMLP, count_parameters
from flopt.resource_watchdog import ResourceWatchdog

OUT = Path("outputs/full_mimic_iv_critique_remediation")
MIMIC_OUT = Path("outputs/full_mimic_iv")
MIMIC_RAW = Path("data/kagglehub-cache/datasets/mangeshwagle/mimic-iv-2-1/versions/1/mimic-iv-2.1")

CKPT_SEEDS = [7, 11, 13, 17, 23]
FRESH_SEEDS = [
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
]


def _base_config(bundle) -> FLConfig:
    return FLConfig(
        rounds=80,
        max_rounds=80,
        local_epochs=1,
        clients_per_round=min(5, len(bundle.clients)),
        lr=0.005,
        batch_size=256,
        seed=7,
        cvar_alpha=0.0,
        patience=15,
        min_delta=0.0005,
        early_stopping=True,
        monitor="loss",
        class_weights=bundle.class_weights,
        optimizer="adam",
    )


def _ensure_dirs(out: Path):
    for name in [
        "metrics", "raw", "checkpoints", "beta", "plots", "lp",
        "ablations", "sofa", "threshold", "preprocessing", "partitions",
        "monitoring", "reports", "runtime",
    ]:
        (out / name).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1: Beta Placement (Exp 3A)
# ---------------------------------------------------------------------------

def run_beta_placement(out: Path, bundle):
    print("\n=== Phase 1: Beta Placement ===")
    rates = np.array([
        float(np.mean(c.y_test)) for c in bundle.clients
    ])
    if len(rates) == 0:
        rates = np.array([
            float(np.mean(np.concatenate([c.y_train, c.y_test])))
            for c in bundle.clients
        ])
    print(f"  Natural mortality rates ({len(rates)} clients): {rates}")

    sample_mean = float(np.mean(rates))
    sample_var = float(np.var(rates, ddof=1))
    S_mom = sample_mean * (1 - sample_mean) / sample_var - 1
    label_beta = S_mom / 2
    print(f"  MoM: S={S_mom:.2f}, symmetric-beta={label_beta:.1f}")

    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_betas = []
    for _ in range(n_boot):
        resample = rng.choice(rates, size=len(rates), replace=True)
        m = float(np.mean(resample))
        v = float(np.var(resample, ddof=1))
        if v > 0:
            s = m * (1 - m) / v - 1
            boot_betas.append(s / 2)
    boot_betas = np.array(boot_betas)
    boot_betas = boot_betas[np.isfinite(boot_betas)]
    ci_lo = float(np.percentile(boot_betas, 2.5))
    ci_hi = float(np.percentile(boot_betas, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]")

    gate_benign = ci_lo >= 10
    need_dirichlet_arm = not gate_benign
    print(f"  Gate decision: CI lower={ci_lo:.1f}, benign={gate_benign}, need_dirichlet_arm={need_dirichlet_arm}")

    dense_grid = [0.5, 1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
    if label_beta not in dense_grid:
        dense_grid.append(round(label_beta, 1))
    dense_grid.extend([15.0, 18.0, 20.0, 30.0])
    dense_grid.append(float("inf"))
    dense_grid = sorted(set(dense_grid))

    result = {
        "label_beta": round(label_beta, 2),
        "S_concentration": round(S_mom, 2),
        "mortality_rates": rates.tolist(),
        "sample_mean": round(sample_mean, 4),
        "sample_var": round(sample_var, 6),
        "bootstrap_ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
        "bootstrap_n": n_boot,
        "method": "MoM_label_only_with_bootstrap",
        "note": "feature heterogeneity is an additional unquantified axis",
        "gate_benign": gate_benign,
        "need_dirichlet_arm": need_dirichlet_arm,
        "recommended_grid": [str(g) for g in dense_grid],
    }
    write_json(out / "beta" / "beta_placement.json", result)
    print(f"  Wrote beta_placement.json")
    return result


# ---------------------------------------------------------------------------
# Exp 1: Personalisation Baseline
# ---------------------------------------------------------------------------

def _train_and_save_checkpoint(bundle, base, model_factory, seed, mu, out_path):
    """Train a model and save checkpoint. Returns (model, records)."""
    if out_path.exists():
        print(f"    Checkpoint exists: {out_path.name}")
        state = torch.load(out_path, map_location="cpu", weights_only=True)
        model = model_factory()
        model.load_state_dict(state)
        return model, None
    cfg = replace(base, seed=seed)
    model = model_factory()
    if mu > 0:
        model, records, _ = fedprox_train(model, bundle.clients, cfg, mu=mu)
    else:
        model, records = federated_train(model, bundle.clients, cfg)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    return model, records


def _evaluate_model_on_clients(model, clients, device):
    """Evaluate model per client, return list of per-client dicts."""
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix
    model.eval()
    rows = []
    for idx, client in enumerate(clients):
        x = torch.tensor(client.x_test, dtype=torch.float32).to(device)
        y_true = client.y_test
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        cid = client.client_id if client.client_id is not None else idx
        auprc = float(average_precision_score(y_true, probs)) if len(set(y_true)) > 1 else 0.0
        ba = float(balanced_accuracy_score(y_true, preds))
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        rows.append({
            "client_id": int(cid),
            "test_auprc": auprc,
            "test_recall": recall,
            "test_balanced_acc": ba,
            "test_samples": len(y_true),
            "test_deaths": int(y_true.sum()),
        })
    return rows


def _get_predictions_for_client(model, client, device):
    """Get per-sample predictions for threshold sweep."""
    model.eval()
    x = torch.tensor(client.x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return client.y_test, probs


def run_personalisation(out: Path, bundle, base, mlp_factory):
    print("\n=== Exp 1: Personalisation Baseline ===")
    device = _device()
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("  Step 1: Generating checkpoints (3 methods x 5 seeds = 15 runs)...")
    methods = [
        ("fedprox_0p1", 0.1),
        ("fedprox_0p01", 0.01),
        ("fedavg", 0.0),
    ]
    checkpoints = {}
    for method_name, mu in methods:
        for seed in CKPT_SEEDS:
            path = ckpt_dir / f"{method_name}_seed{seed}.pt"
            model, _ = _train_and_save_checkpoint(
                bundle, base, mlp_factory, seed, mu, path
            )
            checkpoints[(method_name, seed)] = path
            print(f"    {method_name} seed={seed} done")

    print("  Step 2: Evaluating global baselines (no fine-tuning)...")
    baseline_rows = []
    for method_name in ["fedprox_0p1", "fedavg"]:
        for seed in CKPT_SEEDS:
            state = torch.load(checkpoints[(method_name, seed)], map_location="cpu", weights_only=True)
            model = mlp_factory()
            model.load_state_dict(state)
            model.to(device)
            per_client = _evaluate_model_on_clients(model, bundle.clients, device)
            for row in per_client:
                row["method"] = method_name
                row["seed"] = seed
            baseline_rows.extend(per_client)
    write_csv(out / "metrics" / "personalisation_global_baseline.csv", baseline_rows)
    print(f"    Wrote personalisation_global_baseline.csv ({len(baseline_rows)} rows)")

    print("  Step 3: Personalisation sweep (FedProx-0.1 + FedAvg)...")
    epoch_rows = []
    for base_method in ["fedprox_0p1", "fedavg"]:
        for seed in CKPT_SEEDS:
            state = torch.load(checkpoints[(base_method, seed)], map_location="cpu", weights_only=True)
            for cidx, client in enumerate(bundle.clients):
                cid = client.client_id if client.client_id is not None else cidx
                n_train = len(client.x_train)
                n_deaths_train = int(client.y_train.sum())

                rng = np.random.default_rng(seed + cid)
                indices = np.arange(n_train)
                rng.shuffle(indices)
                split = int(0.8 * n_train)
                train_idx, val_idx = indices[:split], indices[split:]

                x_local_train = torch.tensor(client.x_train[train_idx], dtype=torch.float32).to(device)
                y_local_train = torch.tensor(client.y_train[train_idx], dtype=torch.long).to(device)
                x_val = torch.tensor(client.x_train[val_idx], dtype=torch.float32).to(device)
                y_val_np = client.y_train[val_idx]
                x_test = torch.tensor(client.x_test, dtype=torch.float32).to(device)
                y_test_np = client.y_test

                val_deaths = int(y_val_np.sum())
                use_early_stopping = val_deaths >= 50

                model = mlp_factory()
                model.load_state_dict(state)
                model.to(device)
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_fn = nn.CrossEntropyLoss()

                best_val_loss = float("inf")
                patience_count = 0
                max_ft_epochs = 20

                for ep in range(1, max_ft_epochs + 1):
                    model.train()
                    loader = DataLoader(
                        TensorDataset(x_local_train, y_local_train),
                        batch_size=64, shuffle=True,
                    )
                    for xb, yb in loader:
                        opt.zero_grad()
                        loss_fn(model(xb), yb).backward()
                        opt.step()

                    model.eval()
                    with torch.no_grad():
                        val_logits = model(x_val)
                        val_loss = float(loss_fn(val_logits, torch.tensor(y_val_np, dtype=torch.long).to(device)).cpu())
                        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                        val_auprc = float(average_precision_score(y_val_np, val_probs)) if len(set(y_val_np.tolist())) > 1 else 0.0

                        test_logits = model(x_test)
                        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
                        test_preds = test_logits.argmax(dim=1).cpu().numpy()
                        test_auprc = float(average_precision_score(y_test_np, test_probs)) if len(set(y_test_np.tolist())) > 1 else 0.0
                        tn, fp, fn, tp = confusion_matrix(y_test_np, test_preds, labels=[0, 1]).ravel()
                        test_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                        test_ba = float(balanced_accuracy_score(y_test_np, test_preds))

                    stopped = "N/A (fixed-20)"
                    if use_early_stopping:
                        if val_loss < best_val_loss - 0.0001:
                            best_val_loss = val_loss
                            patience_count = 0
                        else:
                            patience_count += 1
                        if patience_count >= 3:
                            stopped = str(ep)
                            epoch_rows.append({
                                "base_method": base_method, "seed": seed, "client_id": cid,
                                "fine_tune_epochs": ep, "stopped_epoch": stopped,
                                "val_loss": round(val_loss, 6), "val_auprc": round(val_auprc, 4),
                                "test_auprc": round(test_auprc, 4), "test_recall": round(test_recall, 4),
                                "test_balanced_acc": round(test_ba, 4),
                            })
                            break

                    epoch_rows.append({
                        "base_method": base_method, "seed": seed, "client_id": cid,
                        "fine_tune_epochs": ep, "stopped_epoch": stopped if ep == max_ft_epochs else "",
                        "val_loss": round(val_loss, 6), "val_auprc": round(val_auprc, 4),
                        "test_auprc": round(test_auprc, 4), "test_recall": round(test_recall, 4),
                        "test_balanced_acc": round(test_ba, 4),
                    })

            print(f"    {base_method} seed={seed}: all clients done")

    write_csv(out / "metrics" / "personalisation_epoch_sweep.csv", epoch_rows)

    summary = _summarize_personalisation(epoch_rows)
    write_csv(out / "metrics" / "personalisation_summary.csv", summary)
    print(f"    Wrote personalisation outputs ({len(epoch_rows)} epoch rows)")


def _summarize_personalisation(epoch_rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in epoch_rows:
        ep = row["fine_tune_epochs"]
        if ep == 20 or (isinstance(row.get("stopped_epoch"), str) and "fixed" in row.get("stopped_epoch", "")):
            key = (row["base_method"], row["client_id"])
            groups[key].append(row)
        elif row.get("stopped_epoch") and row["stopped_epoch"] not in ("", "N/A (fixed-20)"):
            key = (row["base_method"], row["client_id"])
            groups[key].append(row)

    last_per_seed = defaultdict(list)
    for row in epoch_rows:
        key = (row["base_method"], row["seed"], row["client_id"])
        last_per_seed[key] = row

    summary = []
    seen = set()
    for (method, seed, cid), row in last_per_seed.items():
        k = (method, cid)
        if k not in seen:
            seen.add(k)
    by_method_client = defaultdict(list)
    for (method, seed, cid), row in last_per_seed.items():
        by_method_client[(method, cid)].append(row)

    for (method, cid), rows in by_method_client.items():
        auprcs = [float(r["test_auprc"]) for r in rows]
        recalls = [float(r["test_recall"]) for r in rows]
        summary.append({
            "base_method": method,
            "client_id": cid,
            "n_seeds": len(rows),
            "test_auprc_mean": round(np.mean(auprcs), 4),
            "test_auprc_std": round(np.std(auprcs, ddof=1), 4) if len(auprcs) > 1 else 0,
            "test_recall_mean": round(np.mean(recalls), 4),
            "test_recall_std": round(np.std(recalls, ddof=1), 4) if len(recalls) > 1 else 0,
        })
    return summary


# ---------------------------------------------------------------------------
# Exp 2: 30 Fresh Seeds
# ---------------------------------------------------------------------------

def run_fresh_seeds(out: Path, bundle, base, mlp_factory, beta_result):
    print("\n=== Exp 2: 30 Fresh Seeds ===")
    rows = []
    for i, seed in enumerate(FRESH_SEEDS):
        t0 = time.perf_counter()
        cfg = replace(base, seed=seed)
        model = mlp_factory()
        model, records, _ = fedprox_train(model, bundle.clients, cfg, mu=0.1)
        c = convergence_summary(records, "fedprox_mu_0p1", seed, None, cfg.max_rounds or cfg.rounds)
        row = {"method": "fedprox_mu_0p1", "seed": seed, "mu": 0.1, **_strip(c)}
        rows.append(row)
        elapsed = time.perf_counter() - t0
        print(f"  [{i+1}/30] seed={seed} stopped_round={c['stopped_round']} loss={c['final_loss']:.4f} ({elapsed:.0f}s)")

    write_csv(out / "metrics" / "fresh_seeds_30.csv", rows)

    metrics_to_ci = ["final_loss", "final_auprc", "final_worst_client_recall", "final_worst_client_auprc",
                     "final_auroc", "final_balanced_accuracy", "stopped_round"]
    ci_rows = []
    for metric in metrics_to_ci:
        vals = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
        if len(vals) >= 2:
            arr = np.array(vals)
            ci_rows.append({
                "metric": metric, "n": len(arr),
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std(ddof=1)), 4),
                "ci95_lo": round(float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 4),
                "ci95_hi": round(float(arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 4),
            })
    write_csv(out / "metrics" / "fresh_seeds_30_ci.csv", ci_rows)
    print(f"  Wrote fresh_seeds_30.csv ({len(rows)} rows) and CI summary")
    return rows


# ---------------------------------------------------------------------------
# Exp 3B: K=9 Dirichlet Sweep
# ---------------------------------------------------------------------------

def run_dirichlet_k9(out: Path, bundle, base, mlp_factory, beta_result):
    print("\n=== Exp 3B: K=9 Dirichlet Sweep ===")
    arrays_path = MIMIC_OUT / "preprocessing" / "model_arrays.npz"
    grid = beta_result.get("recommended_grid", [])
    grid_floats = []
    for g in grid:
        if str(g).lower() in ("inf", "infinity"):
            grid_floats.append("infinity")
        else:
            grid_floats.append(float(g))
    seeds = [7, 11, 13]
    rows = []
    total = len(grid_floats) * len(seeds) * 2
    count = 0
    for beta in grid_floats:
        for seed in seeds:
            clients, _, _ = make_dirichlet_clients_from_arrays(arrays_path, beta, 9, seed)
            if len(clients) < 3:
                print(f"  Skipping beta={beta} seed={seed}: only {len(clients)} clients")
                continue
            for method, mu in [("fedavg", 0.0), ("fedprox_0p1", 0.1)]:
                count += 1
                cfg = replace(base, seed=seed, clients_per_round=min(5, len(clients)))
                model = mlp_factory()
                if mu > 0:
                    model, records, _ = fedprox_train(model, clients, cfg, mu=mu)
                else:
                    model, records = federated_train(model, clients, cfg)
                c = convergence_summary(records, method, seed, None, cfg.max_rounds or cfg.rounds)
                row = {"method": method, "beta": str(beta), "seed": seed, "mu": mu,
                       "k_clients": len(clients), **_strip(c)}
                rows.append(row)
                print(f"  [{count}/{total}] beta={beta} {method} seed={seed} stopped={c['stopped_round']}")

    write_csv(out / "metrics" / "dirichlet_k9_sweep.csv", rows)
    print(f"  Wrote dirichlet_k9_sweep.csv ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Exp 4: CVaR K=30 C=20
# ---------------------------------------------------------------------------

def run_cvar_k30(out: Path, bundle, base, mlp_factory):
    print("\n=== Exp 4: CVaR K=30 C=20 ===")
    arrays_path = MIMIC_OUT / "preprocessing" / "model_arrays.npz"
    alphas = [0.5, 0.75, 0.9, 0.95]
    seeds = [7, 11, 13]
    rows = []
    for alpha in alphas:
        for seed in seeds:
            clients, _, _ = make_dirichlet_clients_from_arrays(arrays_path, 0.5, 30, seed)
            cfg = replace(base, seed=seed, cvar_alpha=alpha,
                          clients_per_round=min(20, len(clients)))
            model = mlp_factory()
            model, records = federated_train(model, clients, cfg)
            c = convergence_summary(records, f"cvar_{alpha}", seed, alpha, cfg.max_rounds or cfg.rounds)
            row = {"method": f"cvar_{alpha}", "alpha": alpha, "seed": seed,
                   "k_clients": len(clients), **_strip(c)}
            rows.append(row)
            print(f"  CVaR alpha={alpha} seed={seed} stopped={c['stopped_round']}")

    for seed in seeds:
        clients, _, _ = make_dirichlet_clients_from_arrays(arrays_path, 0.5, 30, seed)
        cfg = replace(base, seed=seed, clients_per_round=min(20, len(clients)))
        model = mlp_factory()
        model, records, _ = fedprox_train(model, clients, cfg, mu=0.1)
        c = convergence_summary(records, "fedprox_0p1", seed, None, cfg.max_rounds or cfg.rounds)
        row = {"method": "fedprox_0p1", "alpha": 0.0, "seed": seed,
               "k_clients": len(clients), **_strip(c)}
        rows.append(row)
        print(f"  FedProx-0.1 seed={seed} stopped={c['stopped_round']}")

    write_csv(out / "metrics" / "cvar_k30_c20.csv", rows)
    print(f"  Wrote cvar_k30_c20.csv ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Exp 5: Ablation Suite
# ---------------------------------------------------------------------------

def run_ablation_suite(out: Path, bundle, base, feature_count):
    print("\n=== Exp 5: Ablation Suite ===")
    device = _device()
    seeds = [7, 11, 13, 17, 23]
    rows = []

    print("  5a: Categorical feature ablation")
    cat_cols = []
    for i, name in enumerate(bundle.feature_names):
        for prefix in ["admission_type_", "admission_location_", "insurance_",
                       "language_", "marital_status_", "race_", "gender_"]:
            if str(name).startswith(prefix):
                cat_cols.append(i)
                break
    numeric_only_indices = [i for i in range(feature_count) if i not in cat_cols]
    print(f"    Dropping {len(cat_cols)} categorical features, keeping {len(numeric_only_indices)}")

    for variant_name, indices in [("full", list(range(feature_count))), ("no_categoricals", numeric_only_indices)]:
        sub_clients = []
        for c in bundle.clients:
            sub_clients.append(ClientData(
                c.x_train[:, indices], c.y_train,
                c.x_test[:, indices], c.y_test,
                c.client_id,
            ))
        n_feat = len(indices)
        for seed in seeds:
            cfg = replace(base, seed=seed)
            model = TabularMLP(n_feat, 2, hidden=(256, 128, 64), dropout=0.1)
            model, records, _ = fedprox_train(model, sub_clients, cfg, mu=0.1)
            c = convergence_summary(records, f"5a_{variant_name}", seed, None, cfg.max_rounds or cfg.rounds)
            row = {"experiment": "5a_categorical", "variant": variant_name,
                   "features": n_feat, "seed": seed, **_strip(c)}
            rows.append(row)
        print(f"    {variant_name}: done (5 seeds)")

    print("  5b: Class weight ablation")
    original_weights = base.class_weights
    sqrt_weights = None
    if original_weights:
        sqrt_weights = tuple(math.sqrt(w) for w in original_weights)
    weight_variants = [
        ("original", original_weights),
        ("uniform", None),
        ("sqrt", sqrt_weights),
    ]
    for variant_name, weights in weight_variants:
        for seed in seeds:
            cfg = replace(base, seed=seed, class_weights=weights)
            model = TabularMLP(feature_count, 2, hidden=(256, 128, 64), dropout=0.1)
            model, records, _ = fedprox_train(model, bundle.clients, cfg, mu=0.1)
            c = convergence_summary(records, f"5b_{variant_name}", seed, None, cfg.max_rounds or cfg.rounds)
            row = {"experiment": "5b_class_weight", "variant": variant_name,
                   "seed": seed, **_strip(c)}
            rows.append(row)
        print(f"    {variant_name} weights: done")

    print("  5c: Architecture capacity ablation")
    architectures = [
        ("current_256_128_64", (256, 128, 64)),
        ("shallow_128_64", (128, 64)),
        ("wide_deep_512_256_128_64", (512, 256, 128, 64)),
    ]
    for variant_name, hidden in architectures:
        for seed in seeds:
            cfg = replace(base, seed=seed)
            model = TabularMLP(feature_count, 2, hidden=hidden, dropout=0.1)
            n_params = count_parameters(model)
            model, records, _ = fedprox_train(model, bundle.clients, cfg, mu=0.1)
            c = convergence_summary(records, f"5c_{variant_name}", seed, None, cfg.max_rounds or cfg.rounds)
            row = {"experiment": "5c_architecture", "variant": variant_name,
                   "hidden": str(hidden), "parameters": n_params,
                   "seed": seed, **_strip(c)}
            rows.append(row)
        print(f"    {variant_name} ({count_parameters(TabularMLP(feature_count, 2, hidden=hidden, dropout=0.1)):,} params): done")

    write_csv(out / "ablations" / "ablation_suite.csv", rows)
    print(f"  Wrote ablation_suite.csv ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Exp 6: LP Communication Sweep
# ---------------------------------------------------------------------------

def run_lp_sweep(out: Path, bundle, base, mlp_factory):
    print("\n=== Exp 6: LP Communication Sweep (FedAvg, 3 seeds, 60 rounds) ===")
    lr_vals = [0.001, 0.005, 0.01, 0.05]
    epochs_vals = [2, 5]
    cpr_vals = [3, 5, 7, 9, len(bundle.clients)]
    seeds = [7, 11, 13]

    configs = [(lr, e, c) for lr in lr_vals for e in epochs_vals for c in cpr_vals]
    total = len(configs) * len(seeds)
    print(f"  {len(configs)} configs x {len(seeds)} seeds = {total} runs")

    rows = []
    count = 0
    for lr, local_epochs, cpr in configs:
        for seed in seeds:
            count += 1
            cfg = replace(base, seed=seed, lr=lr, local_epochs=local_epochs,
                          clients_per_round=min(cpr, len(bundle.clients)),
                          rounds=60, max_rounds=60, cvar_alpha=0.0)
            model = mlp_factory()
            model, records = federated_train(model, bundle.clients, cfg)
            c = convergence_summary(records, "fedavg", seed, None, 60)
            total_comm = sum(r["upload_bytes"] + r["download_bytes"] for r in records)
            row = {
                "lr": lr, "local_epochs": local_epochs, "clients_per_round": cpr,
                "seed": seed, "total_comm_bytes": total_comm,
                **_strip(c),
            }
            rows.append(row)
            if count % 10 == 0:
                print(f"  [{count}/{total}] lr={lr} E={local_epochs} C={cpr} seed={seed}")

    write_csv(out / "lp" / "lp_dedicated_sweep.csv", rows)

    from collections import defaultdict
    averaged = defaultdict(list)
    for r in rows:
        key = (r["lr"], r["local_epochs"], r["clients_per_round"])
        averaged[key].append(r)
    avg_rows = []
    for (lr, e, c), group in averaged.items():
        losses = [float(g["final_loss"]) for g in group]
        comms = [float(g["total_comm_bytes"]) for g in group]
        avg_rows.append({
            "lr": lr, "local_epochs": e, "clients_per_round": c,
            "mean_loss": round(np.mean(losses), 6),
            "std_loss": round(np.std(losses, ddof=1), 6) if len(losses) > 1 else 0,
            "mean_comm_bytes": round(np.mean(comms), 0),
        })
    write_csv(out / "lp" / "lp_dedicated_sweep_averaged.csv", avg_rows)

    losses_for_lp = [r["mean_loss"] for r in avg_rows]
    costs_for_lp = [r["mean_comm_bytes"] for r in avg_rows]
    if len(losses_for_lp) >= 2 and max(costs_for_lp) > min(costs_for_lp):
        budgets = np.linspace(min(costs_for_lp), max(costs_for_lp), 15).tolist()
        lp_result = solve_policy_lp(losses_for_lp, costs_for_lp, budgets)
        lp_rows = []
        for r in lp_result:
            lp_rows.append({
                "budget": r.get("budget"), "loss": r.get("loss"),
                "cost": r.get("cost"), "lambda": r.get("lambda"),
                "status": r.get("status"),
            })
        write_csv(out / "lp" / "lp_shadow_price_comparison.csv", lp_rows)

    print(f"  Wrote LP sweep outputs ({len(rows)} per-seed, {len(avg_rows)} averaged)")


# ---------------------------------------------------------------------------
# Exp 7: SOFA Baseline
# ---------------------------------------------------------------------------

def _score_coag(platelets):
    if not platelets: return 0, True
    p = min(platelets)
    if p >= 150: return 0, False
    if p >= 100: return 1, False
    if p >= 50: return 2, False
    if p >= 20: return 3, False
    return 4, False

def _score_liver(bilis):
    if not bilis: return 0, True
    b = max(bilis)
    if b < 1.2: return 0, False
    if b < 2.0: return 1, False
    if b < 6.0: return 2, False
    if b < 12.0: return 3, False
    return 4, False

def _score_neuro(eye_vals, verbal_vals, motor_vals):
    if not eye_vals or not verbal_vals or not motor_vals:
        if eye_vals or verbal_vals or motor_vals:
            gcs = (min(eye_vals) if eye_vals else 4) + \
                  (min(verbal_vals) if verbal_vals else 5) + \
                  (min(motor_vals) if motor_vals else 6)
        else:
            return 0, True
    else:
        gcs = min(eye_vals) + min(verbal_vals) + min(motor_vals)
    if gcs >= 15: return 0, False
    if gcs >= 13: return 1, False
    if gcs >= 10: return 2, False
    if gcs >= 6: return 3, False
    return 4, False

def _score_resp(pao2_list, fio2_list, spo2_list):
    if pao2_list and fio2_list:
        fio2_raw = min(fio2_list)
        fio2 = max(0.21, fio2_raw / 100 if fio2_raw > 1 else fio2_raw)
        pf = min(pao2_list) / fio2
        if pf >= 400: return 0, False
        if pf >= 300: return 1, False
        if pf >= 200: return 2, False
        if pf >= 100: return 3, False
        return 4, False
    if spo2_list and fio2_list:
        fio2_raw = min(fio2_list)
        fio2 = max(0.21, fio2_raw / 100 if fio2_raw > 1 else fio2_raw)
        sf = min(spo2_list) / fio2
        if sf >= 315: return 0, False
        if sf >= 232: return 1, False
        if sf >= 148: return 2, False
        if sf >= 67: return 3, False
        return 4, False
    return 0, True

def _score_cardio(map_vals):
    if not map_vals: return 0, True
    m = min(map_vals)
    if m >= 70: return 0, False
    return 1, False

def _score_renal(creat_list):
    if not creat_list: return 0, True
    c = max(creat_list)
    if c < 1.2: return 0, False
    if c < 2.0: return 1, False
    if c < 3.5: return 2, False
    if c < 5.0: return 3, False
    return 4, False


def run_sofa_baseline(out: Path, bundle, base, mlp_factory):
    print("\n=== Exp 7: SOFA Baseline Comparison ===")
    import duckdb
    from collections import defaultdict

    db_path = "data/mimic_cache/mimic_iv.duckdb"
    if not Path(db_path).exists():
        print("  WARNING: DuckDB cache not found. Skipping SOFA.")
        write_json(out / "sofa" / "sofa_skipped.json", {"reason": "duckdb not found"})
        return

    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]

    cohort = con.execute("SELECT stay_id, mortality_label FROM cohort").fetchdf()
    stay_ids = set(cohort["stay_id"].tolist())
    mort_labels = dict(zip(cohort["stay_id"].tolist(), cohort["mortality_label"].tolist()))
    print(f"  Cohort: {len(stay_ids)} stays")

    print("  Querying labevents_24h for SOFA items...")
    lab_by_stay = defaultdict(lambda: defaultdict(list))
    if "labevents_24h" in tables:
        lab_df = con.execute("""
            SELECT stay_id, itemid, valuenum FROM labevents_24h
            WHERE itemid IN (50821, 50816, 51265, 50885, 50912)
              AND valuenum IS NOT NULL
        """).fetchdf()
        for _, row in lab_df.iterrows():
            lab_by_stay[int(row["stay_id"])][int(row["itemid"])].append(float(row["valuenum"]))
        print(f"    Lab rows: {len(lab_df)}")

    print("  Querying chartevents_24h for SOFA items...")
    chart_by_stay = defaultdict(lambda: defaultdict(list))
    if "chartevents_24h" in tables:
        chart_df = con.execute("""
            SELECT stay_id, itemid, valuenum FROM chartevents_24h
            WHERE itemid IN (220052, 220181, 220739, 223900, 223901, 220277, 223835)
              AND valuenum IS NOT NULL
        """).fetchdf()
        for _, row in chart_df.iterrows():
            chart_by_stay[int(row["stay_id"])][int(row["itemid"])].append(float(row["valuenum"]))
        print(f"    Chart rows: {len(chart_df)}")

    print("  Querying outputevents_24h for urine output...")
    urine_by_stay = {}
    if "outputevents_24h" in tables:
        out_df = con.execute("""
            SELECT stay_id, SUM(output_value) as total_urine FROM outputevents_24h
            GROUP BY stay_id
        """).fetchdf()
        for _, row in out_df.iterrows():
            urine_by_stay[int(row["stay_id"])] = float(row["total_urine"]) if row["total_urine"] is not None else None
        print(f"    Output stays: {len(out_df)}")

    con.close()

    print("  Computing per-stay SOFA scores...")
    sofa_rows = []
    missingness = {comp: {"total": 0, "missing": 0, "dead_total": 0, "dead_missing": 0,
                          "alive_total": 0, "alive_missing": 0}
                   for comp in ["resp", "coag", "liver", "cardio", "neuro", "renal"]}

    for sid in stay_ids:
        mort = int(mort_labels.get(sid, 0))
        labs = lab_by_stay.get(sid, {})
        charts = chart_by_stay.get(sid, {})

        plat_vals = labs.get(51265, [])
        bili_vals = labs.get(50885, [])
        creat_vals = labs.get(50912, [])
        pao2_vals = labs.get(50821, []) + labs.get(50816, [])
        map_vals = charts.get(220052, []) + charts.get(220181, [])
        eye_vals = charts.get(220739, [])
        verbal_vals = charts.get(223900, [])
        motor_vals = charts.get(223901, [])
        spo2_vals = charts.get(220277, [])
        fio2_vals = charts.get(223835, [])

        coag_score, coag_miss = _score_coag(plat_vals)
        liver_score, liver_miss = _score_liver(bili_vals)
        neuro_score, neuro_miss = _score_neuro(eye_vals, verbal_vals, motor_vals)
        resp_score, resp_miss = _score_resp(pao2_vals, fio2_vals, spo2_vals)
        cardio_score, cardio_miss = _score_cardio(map_vals)

        creat_score, creat_miss = _score_renal(creat_vals)
        urine = urine_by_stay.get(sid)
        urine_score = 0
        if urine is not None:
            if urine < 200: urine_score = 4
            elif urine < 500: urine_score = 3
        renal_score = max(creat_score, urine_score)
        renal_miss = creat_miss and (urine is None)

        n_missing = sum([coag_miss, liver_miss, neuro_miss, resp_miss, cardio_miss, renal_miss])
        sofa_total = resp_score + coag_score + liver_score + cardio_score + neuro_score + renal_score

        sofa_rows.append({
            "stay_id": sid, "sofa_total": sofa_total,
            "sofa_resp": resp_score, "sofa_coag": coag_score,
            "sofa_liver": liver_score, "sofa_cardio": cardio_score,
            "sofa_neuro": neuro_score, "sofa_renal": renal_score,
            "mortality_label": mort, "n_missing_components": n_missing,
        })

        for comp, miss in [("resp", resp_miss), ("coag", coag_miss), ("liver", liver_miss),
                           ("cardio", cardio_miss), ("neuro", neuro_miss), ("renal", renal_miss)]:
            missingness[comp]["total"] += 1
            if miss: missingness[comp]["missing"] += 1
            if mort == 1:
                missingness[comp]["dead_total"] += 1
                if miss: missingness[comp]["dead_missing"] += 1
            else:
                missingness[comp]["alive_total"] += 1
                if miss: missingness[comp]["alive_missing"] += 1

    write_csv(out / "sofa" / "sofa_computation.csv", sofa_rows)

    miss_rows = []
    for comp, m in missingness.items():
        miss_rows.append({
            "component": comp,
            "total_missing_pct": round(m["missing"] / max(m["total"], 1) * 100, 1),
            "missing_pct_dead": round(m["dead_missing"] / max(m["dead_total"], 1) * 100, 1),
            "missing_pct_alive": round(m["alive_missing"] / max(m["alive_total"], 1) * 100, 1),
        })
    write_csv(out / "sofa" / "sofa_missingness.csv", miss_rows)

    if sofa_rows:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score as ap_score
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        sofa_scores = np.array([r["sofa_total"] for r in sofa_rows]).reshape(-1, 1)
        labels = np.array([r["mortality_label"] for r in sofa_rows])
        if len(set(labels.tolist())) > 1:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(sofa_scores, labels)
            sofa_probs = lr.predict_proba(sofa_scores)[:, 1]
            sofa_auroc = float(roc_auc_score(labels, sofa_probs))
            sofa_auprc = float(ap_score(labels, sofa_probs))
            sofa_ba = float(balanced_accuracy_score(labels, lr.predict(sofa_scores)))

            comparison = [
                {"model": "SOFA-LR", "auroc": round(sofa_auroc, 4),
                 "auprc": round(sofa_auprc, 4), "balanced_accuracy": round(sofa_ba, 4)},
            ]
            write_csv(out / "sofa" / "sofa_vs_model.csv", comparison)
            print(f"  SOFA-LR: AuROC={sofa_auroc:.4f}, AuPRC={sofa_auprc:.4f}")

            n_complete = sum(1 for r in sofa_rows if r["n_missing_components"] == 0)
            print(f"  Complete-SOFA stays: {n_complete}/{len(sofa_rows)}")
            if n_complete >= 100:
                complete_idx = [i for i, r in enumerate(sofa_rows) if r["n_missing_components"] == 0]
                lr2 = LogisticRegression(max_iter=1000)
                lr2.fit(sofa_scores[complete_idx], labels[complete_idx])
                p2 = lr2.predict_proba(sofa_scores[complete_idx])[:, 1]
                write_csv(out / "sofa" / "sofa_vs_model_complete_only.csv", [{
                    "model": "SOFA-LR (complete only)",
                    "n": n_complete,
                    "auroc": round(float(roc_auc_score(labels[complete_idx], p2)), 4),
                    "auprc": round(float(ap_score(labels[complete_idx], p2)), 4),
                }])

    print(f"  Wrote SOFA outputs ({len(sofa_rows)} stays)")


# ---------------------------------------------------------------------------
# Exp 8: Threshold Optimization
# ---------------------------------------------------------------------------

def run_threshold_optimization(out: Path, bundle, base, mlp_factory):
    print("\n=== Exp 8: Threshold Optimization ===")
    device = _device()
    ckpt_dir = out / "checkpoints"

    from sklearn.metrics import confusion_matrix as cm_func

    sweep_rows = []
    optimal_rows = []

    for method_name in ["fedprox_0p1", "fedprox_0p01", "fedavg"]:
        for seed in CKPT_SEEDS:
            path = ckpt_dir / f"{method_name}_seed{seed}.pt"
            if not path.exists():
                print(f"    Skipping {method_name} seed={seed}: no checkpoint")
                continue

            state = torch.load(path, map_location="cpu", weights_only=True)
            model = mlp_factory()
            model.load_state_dict(state)
            model.to(device)

            all_y_true = []
            all_probs = []
            client_ids = []
            for cidx, client in enumerate(bundle.clients):
                cid = client.client_id if client.client_id is not None else cidx
                y_true, probs = _get_predictions_for_client(model, client, device)
                all_y_true.extend(y_true.tolist())
                all_probs.extend(probs.tolist())
                client_ids.extend([cid] * len(y_true))

                for thresh_int in range(5, 96):
                    thresh = thresh_int / 100
                    preds = (probs >= thresh).astype(int)
                    if len(set(y_true.tolist())) < 2:
                        continue
                    tn, fp, fn, tp = cm_func(y_true, preds, labels=[0, 1]).ravel()
                    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                    sweep_rows.append({
                        "method": method_name, "seed": seed, "client_id": cid,
                        "threshold": thresh,
                        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                        "ppv": round(ppv, 4), "npv": round(npv, 4), "f1": round(f1, 4),
                    })

                best_j_thresh = 0.5
                best_j = -1
                best_s80_thresh = 0.5
                for thresh_int in range(5, 96):
                    thresh = thresh_int / 100
                    preds = (probs >= thresh).astype(int)
                    if len(set(y_true.tolist())) < 2:
                        continue
                    tn, fp, fn, tp = cm_func(y_true, preds, labels=[0, 1]).ravel()
                    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                    j = sens + spec - 1
                    if j > best_j:
                        best_j = j
                        best_j_thresh = thresh
                    if sens >= 0.8:
                        best_s80_thresh = thresh

                optimal_rows.append({
                    "method": method_name, "seed": seed, "client_id": cid,
                    "optimal_threshold_youden": best_j_thresh,
                    "optimal_threshold_sens80": best_s80_thresh,
                })

    write_csv(out / "threshold" / "threshold_sweep.csv", sweep_rows)
    write_csv(out / "threshold" / "optimal_thresholds.csv", optimal_rows)

    neuro_ids = set()
    for c in bundle.clients:
        cid = c.client_id
        n_test = len(c.y_test)
        deaths = int(c.y_test.sum())
        rate = deaths / n_test if n_test > 0 else 0
        if rate < 0.05 and n_test < 1500:
            neuro_ids.add(cid)

    pooled_rows = []
    for method_name in ["fedprox_0p1", "fedprox_0p01", "fedavg"]:
        for seed in CKPT_SEEDS:
            path = ckpt_dir / f"{method_name}_seed{seed}.pt"
            if not path.exists():
                continue
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = mlp_factory()
            model.load_state_dict(state)
            model.to(device)

            pooled_y = []
            pooled_p = []
            for cidx, client in enumerate(bundle.clients):
                cid = client.client_id if client.client_id is not None else cidx
                if cid in neuro_ids:
                    y_true, probs = _get_predictions_for_client(model, client, device)
                    pooled_y.extend(y_true.tolist())
                    pooled_p.extend(probs.tolist())

            if pooled_y and len(set(pooled_y)) > 1:
                from sklearn.metrics import average_precision_score, balanced_accuracy_score
                pooled_y_np = np.array(pooled_y)
                pooled_p_np = np.array(pooled_p)
                auprc = float(average_precision_score(pooled_y_np, pooled_p_np))
                preds = (pooled_p_np >= 0.5).astype(int)
                tn, fp, fn, tp = cm_func(pooled_y_np, preds, labels=[0, 1]).ravel()
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                pooled_rows.append({
                    "method": method_name, "seed": seed,
                    "pooled_neuro_auprc": round(auprc, 4),
                    "pooled_neuro_recall_at_0p5": round(recall, 4),
                    "pooled_neuro_n": len(pooled_y),
                    "pooled_neuro_deaths": int(sum(pooled_y)),
                })

    write_csv(out / "threshold" / "pooled_neuro_metrics.csv", pooled_rows)
    print(f"  Wrote threshold outputs ({len(sweep_rows)} sweep, {len(optimal_rows)} optimal, {len(pooled_rows)} pooled)")


# ---------------------------------------------------------------------------
# Exp 9: Preprocessing Sensitivity
# ---------------------------------------------------------------------------

def run_preprocessing_sensitivity(out: Path, bundle, base, mlp_factory):
    print("\n=== Exp 9: Preprocessing Sensitivity ===")
    print("  NOTE: This experiment requires re-running feature engineering for different top-k values.")
    print("  For top-k=50 (current), using existing data.")

    feature_count = len(bundle.feature_names)
    seeds = [7, 11, 13]
    rows = []

    for seed in seeds:
        cfg = replace(base, seed=seed, rounds=60, max_rounds=60)
        model = TabularMLP(feature_count, 2, hidden=(256, 128, 64), dropout=0.1)
        model, records, _ = fedprox_train(model, bundle.clients, cfg, mu=0.1)
        c = convergence_summary(records, "topk_50", seed, None, 60)
        row = {"topk": 50, "feature_count": feature_count, "seed": seed, **_strip(c)}
        rows.append(row)
        print(f"  topk=50 seed={seed}: done")

    write_csv(out / "preprocessing" / "preprocessing_sensitivity.csv", rows)
    write_json(out / "preprocessing" / "preprocessing_sensitivity_note.json", {
        "note": "Only top-k=50 (current) was evaluated. Top-k in {30, 75} requires re-running "
                "the full preprocessing pipeline including chartevents aggregation, which was not "
                "executed in this run to avoid multi-hour preprocessing overhead.",
        "completed": [50],
        "pending": [30, 75],
    })
    print(f"  Wrote preprocessing sensitivity for topk=50 ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip(c: dict) -> dict:
    return {k: v for k, v in c.items() if k not in {"run_type", "seed", "alpha", "max_rounds"}}


from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Critique Remediation Runner")
    parser.add_argument("--mode", default="all",
                        choices=["all", "exp3a", "exp1", "exp2", "exp3b", "exp4", "exp5",
                                 "exp6", "exp7", "exp8", "exp9", "session1", "overnight"])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--mimic-out", default=str(MIMIC_OUT))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-watchdog", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    _ensure_dirs(out)

    watchdog = ResourceWatchdog(out / "monitoring", interval_seconds=30.0)
    if not args.no_watchdog:
        watchdog.start()

    try:
        print(f"Loading MIMIC-IV data from {args.mimic_out}...")
        bundle = load_mimic_iv_arrays(Path(args.mimic_out), seed=7)
        feature_count = len(bundle.feature_names)
        base = _base_config(bundle)
        mlp_factory = lambda: TabularMLP(feature_count, 2, hidden=(256, 128, 64), dropout=0.1)
        print(f"  {feature_count} features, {len(bundle.clients)} clients, device={_device()}")

        mode = args.mode

        if mode in ("all", "exp3a", "session1"):
            beta_result = run_beta_placement(out, bundle)
        else:
            bp_path = out / "beta" / "beta_placement.json"
            if bp_path.exists():
                beta_result = json.loads(bp_path.read_text())
            else:
                beta_result = {"label_beta": 13.5, "gate_benign": True, "need_dirichlet_arm": False,
                               "recommended_grid": ["0.5", "1.0", "3.0", "5.0", "6.0", "7.0", "8.0",
                                                     "9.0", "10.0", "12.0", "13.5", "15.0", "18.0",
                                                     "20.0", "30.0", "inf"]}

        if mode in ("all", "exp1", "session1"):
            run_personalisation(out, bundle, base, mlp_factory)

        if mode in ("all", "exp2", "session1"):
            run_fresh_seeds(out, bundle, base, mlp_factory, beta_result)

        if mode in ("all", "exp3b", "session1"):
            run_dirichlet_k9(out, bundle, base, mlp_factory, beta_result)

        if mode in ("all", "exp4", "session1"):
            run_cvar_k30(out, bundle, base, mlp_factory)

        if mode in ("all", "exp5", "session1"):
            run_ablation_suite(out, bundle, base, feature_count)

        if mode in ("all", "exp7", "session1"):
            run_sofa_baseline(out, bundle, base, mlp_factory)

        if mode in ("all", "exp8", "session1"):
            run_threshold_optimization(out, bundle, base, mlp_factory)

        if mode in ("all", "exp6", "overnight"):
            run_lp_sweep(out, bundle, base, mlp_factory)

        if mode in ("all", "exp9", "overnight"):
            run_preprocessing_sensitivity(out, bundle, base, mlp_factory)

        print("\n=== All requested experiments complete ===")

    finally:
        if not args.no_watchdog:
            watchdog.stop()


if __name__ == "__main__":
    main()
