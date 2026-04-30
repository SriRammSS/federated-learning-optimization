"""Self-contained utilities — inlined from flopt so the package has zero external project deps."""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ClientData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    client_id: int | None = None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_round_records(records: list[dict], run_type: str, seed: int,
                          alpha: float | None = None) -> list[dict]:
    base_keys = {
        "round", "loss", "accuracy", "worst_client_accuracy", "upload_bytes",
        "download_bytes", "selected_clients", "best_loss_so_far", "best_round",
        "rounds_since_improvement", "stopped_early", "client_loss",
        "client_accuracy", "config", "drift_client_ids", "drift_update_norms",
        "drift_cosine_to_mean", "drift_distance_to_mean",
    }
    rows = []
    for r in records:
        row = {
            "run_type": run_type, "seed": seed, "alpha": alpha,
            "round": r["round"], "loss": r["loss"], "accuracy": r["accuracy"],
            "worst_client_accuracy": r["worst_client_accuracy"],
            "upload_bytes": r["upload_bytes"],
            "download_bytes": r["download_bytes"],
            "total_comm_bytes": r["upload_bytes"] + r["download_bytes"],
            "selected_clients": " ".join(map(str, r["selected_clients"])),
            "best_loss_so_far": r.get("best_loss_so_far"),
            "best_round": r.get("best_round"),
            "rounds_since_improvement": r.get("rounds_since_improvement"),
            "stopped_early": r.get("stopped_early", False),
        }
        for k, v in r.items():
            if k not in base_keys and isinstance(v, (int, float, bool, str)):
                row[k] = v
        rows.append(row)
    return rows


def convergence_summary(records: list[dict], run_type: str, seed: int,
                        alpha: float | None, max_rounds: int) -> dict:
    last = records[-1]
    best = min(records, key=lambda r: r["loss"])
    total_comm = sum(r["upload_bytes"] + r["download_bytes"] for r in records)
    return {
        "run_type": run_type, "seed": seed, "alpha": alpha,
        "max_rounds": max_rounds, "stopped_round": last["round"],
        "stopped_early": last.get("stopped_early", False),
        "best_round": best["round"], "best_loss": best["loss"],
        "final_loss": last["loss"], "final_accuracy": last["accuracy"],
        "final_worst_client_accuracy": last["worst_client_accuracy"],
        "total_comm_until_stop": total_comm,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def confidence_rows(rows: list[dict], group_key: str,
                    metrics: list[str]) -> list[dict]:
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out = []
    for group, items in groups.items():
        for metric in metrics:
            vals = np.array(
                [float(r[metric]) for r in items
                 if r.get(metric) not in {None, ""}], dtype=float)
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            se = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci = 1.96 * se
            out.append({
                "group": group, "metric": metric, "n": len(vals),
                "mean": mean, "std": std,
                "ci95_low": mean - ci, "ci95_high": mean + ci,
            })
    return out


def paired_tests(rows: list[dict], method_key: str, seed_key: str,
                 metrics: list[str], baseline: str) -> list[dict]:
    methods = sorted({r[method_key] for r in rows})
    out = []
    for method in methods:
        if method == baseline:
            continue
        for metric in metrics:
            b = {r[seed_key]: float(r[metric]) for r in rows
                 if r[method_key] == baseline and r.get(metric) not in {None, ""}}
            m = {r[seed_key]: float(r[metric]) for r in rows
                 if r[method_key] == method and r.get(metric) not in {None, ""}}
            seeds = sorted(set(b) & set(m))
            if len(seeds) < 2:
                continue
            diff = np.array([m[s] - b[s] for s in seeds], dtype=float)
            t_p = float(scipy_stats.ttest_rel(
                [m[s] for s in seeds], [b[s] for s in seeds]).pvalue)
            try:
                w_p = float(scipy_stats.wilcoxon(diff).pvalue)
            except ValueError:
                w_p = 1.0
            effect = float(diff.mean() / (diff.std(ddof=1) + 1e-12)) \
                if len(diff) > 1 else 0.0
            out.append({
                "baseline": baseline, "method": method, "metric": metric,
                "n": len(seeds), "mean_diff": float(diff.mean()),
                "paired_t_p": t_p, "wilcoxon_p": w_p, "effect_size": effect,
            })
    return out


# ---------------------------------------------------------------------------
# LP Duality
# ---------------------------------------------------------------------------

def solve_policy_lp(losses: list[float], costs: list[float],
                    budgets: list[float]) -> list[dict]:
    import cvxpy as cp

    losses_np = np.array(losses, dtype="float64")
    costs_np = np.array(costs, dtype="float64")
    cost_scale = max(float(np.max(np.abs(costs_np))), 1.0)
    scaled_costs = costs_np / cost_scale
    rows = []
    for budget in budgets:
        scaled_budget = budget / cost_scale
        x = cp.Variable(len(losses_np), nonneg=True)
        simplex = sum(x) == 1
        budget_con = scaled_costs @ x <= scaled_budget
        problem = cp.Problem(cp.Minimize(losses_np @ x), [simplex, budget_con])
        for solver in [cp.CLARABEL, cp.HIGHS, cp.OSQP, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in {"optimal", "optimal_inaccurate"}:
                    break
            except cp.SolverError:
                continue
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            rows.append({"budget": budget, "status": problem.status})
            continue
        weights = np.asarray(x.value).reshape(-1)
        lam = float(budget_con.dual_value) / cost_scale
        rows.append({
            "budget": float(budget), "loss": float(losses_np @ weights),
            "cost": float(costs_np @ weights), "lambda": lam,
            "weights": weights.tolist(), "status": problem.status,
        })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_mpl_cache = _HERE / "outputs" / "mpl-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prep(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_convergence(records: list[dict], path: str) -> None:
    rounds = [r["round"] for r in records]
    loss = [r["loss"] for r in records]
    ref = loss[0] / np.sqrt(np.maximum(rounds, 1))
    _prep(path)
    plt.figure(figsize=(7, 4))
    plt.plot(rounds, loss, label="FedAvg loss")
    plt.plot(rounds, ref, label="O(1/sqrt(T)) reference", linestyle="--")
    plt.xlabel("Round"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_cvar(rows: list[dict], path: str) -> None:
    _prep(path)
    plt.figure(figsize=(6, 4))
    for row in rows:
        plt.scatter(row["accuracy"], row["worst_client_accuracy"])
        plt.text(row["accuracy"], row["worst_client_accuracy"],
                 f"alpha={row['alpha']}")
    plt.xlabel("Average client accuracy")
    plt.ylabel("Worst-client accuracy")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_shadow_price(rows: list[dict], path: str) -> None:
    clean = [r for r in rows
             if r.get("status") in {"optimal", "optimal_inaccurate"}]
    _prep(path)
    plt.figure(figsize=(6, 4))
    plt.plot([r["budget"] for r in clean],
             [r["lambda"] for r in clean], marker="o")
    plt.xlabel("Communication budget"); plt.ylabel("lambda*")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def plot_search_comparison(grid_rows: list[dict], ga_result: dict,
                           path: str) -> None:
    best_grid = min(float(r["fitness"]) for r in grid_rows)
    _prep(path)
    plt.figure(figsize=(5, 4))
    plt.bar(["Best grid", "GA"], [best_grid, float(ga_result["fitness"])])
    plt.ylabel("Fitness")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def line_mean_std(round_rows: list[dict], metric: str, path: str,
                  title: str = "") -> None:
    _prep(path)
    rounds = sorted({int(r["round"]) for r in round_rows})
    mean, std = [], []
    for rnd in rounds:
        vals = [float(r[metric]) for r in round_rows
                if int(r["round"]) == rnd and r.get(metric) not in {None, ""}]
        mean.append(np.mean(vals)); std.append(np.std(vals))
    mean, std = np.array(mean), np.array(std)
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, mean, label=metric)
    plt.fill_between(rounds, mean - std, mean + std, alpha=0.2, label="std")
    plt.title(title); plt.xlabel("round"); plt.ylabel(metric); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def bar(rows: list[dict], x: str, y: str, path: str, title: str = "",
        rotation: int = 45) -> None:
    _prep(path)
    plt.figure(figsize=(8, 4))
    plt.bar([str(r[x]) for r in rows], [float(r[y]) for r in rows])
    plt.title(title); plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(y)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def scatter(rows: list[dict], x: str, y: str, path: str,
            title: str = "") -> None:
    _prep(path)
    plt.figure(figsize=(6, 4))
    plt.scatter([float(r[x]) for r in rows], [float(r[y]) for r in rows])
    plt.title(title); plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def scatter3(rows: list[dict], x: str, y: str, z: str, path: str,
             title: str = "") -> None:
    _prep(path)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([float(r[x]) for r in rows], [float(r[y]) for r in rows],
               [float(r[z]) for r in rows])
    ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y); ax.set_zlabel(z)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
