"""GA + Grid hyperparameter search for Shakespeare FL."""
from __future__ import annotations
from dataclasses import replace

import numpy as np
from scipy.optimize import differential_evolution

from .config import ShakespeareConfig
from .training import federated_train
from .models import CharTransformer


def _default_factory(cfg: ShakespeareConfig):
    return lambda: CharTransformer(cfg.vocab_size, cfg.d_model, cfg.nhead,
                                   cfg.num_layers, cfg.dim_ff)


def ga_search(clients, base_cfg: ShakespeareConfig, bounds=None,
              maxiter: int = 5, popsize: int = 6, gamma: float = 1e-8,
              model_factory=None, score_key: str = "accuracy",
              min_score: float = 0.40) -> dict:
    model_factory = model_factory or _default_factory(base_cfg)
    if bounds is None:
        bounds = [(1, 5), (3, min(20, len(clients))), (0.1, 2.0)]
    history: list[dict] = []

    def objective(raw):
        le = max(1, int(round(raw[0])))
        cpr = max(1, int(round(raw[1])))
        lr = float(raw[2])
        cfg = replace(base_cfg, local_epochs=le, clients_per_round=cpr, lr=lr)
        _, records = federated_train(model_factory(), clients, cfg)
        last = records[-1]
        comm = sum(r["upload_bytes"] + r["download_bytes"] for r in records)
        score = float(last.get(score_key, last["accuracy"]))
        penalty = max(0, min_score - score) * 10
        fitness = last["loss"] + gamma * comm + penalty
        history.append({
            "evaluation": len(history) + 1,
            "local_epochs": le, "clients_per_round": cpr, "lr": lr,
            "loss": last["loss"], "accuracy": last["accuracy"],
            "perplexity": last.get("perplexity", 0),
            score_key: score, "comm": comm, "fitness": fitness,
        })
        print(f"    GA eval {len(history)}: le={le} cpr={cpr} lr={lr:.2f} "
              f"fitness={fitness:.4f} acc={last['accuracy']:.4f}", flush=True)
        return fitness

    result = differential_evolution(objective, bounds, maxiter=maxiter,
                                    popsize=popsize, polish=False,
                                    seed=base_cfg.seed)
    return {"x": result.x.tolist(), "fitness": float(result.fun),
            "evaluations": int(result.nfev), "history": history}


def grid_search(clients, base_cfg: ShakespeareConfig,
                grid: list[tuple[int, int, float]] | None = None,
                gamma: float = 1e-8, model_factory=None,
                score_key: str = "accuracy",
                min_score: float = 0.40) -> list[dict]:
    model_factory = model_factory or _default_factory(base_cfg)
    if grid is None:
        grid = [(le, cpr, lr)
                for le in [1, 2, 3]
                for cpr in [5, 10, 15]
                for lr in [0.5, 0.8, 1.0]]
    rows = []
    total = len(grid)
    for i, (le, cpr, lr) in enumerate(grid, 1):
        cfg = replace(base_cfg, local_epochs=le, clients_per_round=cpr, lr=lr)
        _, records = federated_train(model_factory(), clients, cfg)
        last = records[-1]
        comm = sum(r["upload_bytes"] + r["download_bytes"] for r in records)
        score = float(last.get(score_key, last["accuracy"]))
        fitness = last["loss"] + gamma * comm + max(0, min_score - score) * 10
        rows.append({
            "local_epochs": le, "clients_per_round": cpr, "lr": lr,
            "loss": last["loss"], "accuracy": last["accuracy"],
            "perplexity": last.get("perplexity", 0),
            score_key: score, "comm": comm, "fitness": fitness,
        })
        print(f"    Grid {i}/{total}: le={le} cpr={cpr} lr={lr:.1f} "
              f"fitness={fitness:.4f} acc={last['accuracy']:.4f}", flush=True)
    return sorted(rows, key=lambda r: r["fitness"])
