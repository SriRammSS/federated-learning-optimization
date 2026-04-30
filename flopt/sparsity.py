from __future__ import annotations

import numpy as np
import torch


TOPK_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)


def flatten_update(base_state: dict, local_state: dict) -> torch.Tensor:
    parts = []
    for key in base_state:
        if torch.is_floating_point(base_state[key]):
            parts.append((local_state[key].float() - base_state[key].float()).reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def update_sparsity_rows(
    base_state: dict,
    local_state: dict,
    round_id: int,
    client_id: int,
    method: str,
    seed: int,
    mu: float | None = None,
    topk_fractions: tuple[float, ...] = TOPK_FRACTIONS,
) -> list[dict]:
    update = flatten_update(base_state, local_state)
    n_params = int(update.numel())
    if n_params == 0:
        return []
    nonzero = int((update != 0).sum().item())
    dense_bytes = n_params * 4
    rows = []
    for frac in topk_fractions:
        k = max(1, int(np.ceil(n_params * frac)))
        sparse_bytes = k * 8  # value float32 + index int32
        rows.append({
            "method": method,
            "seed": seed,
            "mu": mu,
            "round": round_id,
            "client_id": client_id,
            "parameters": n_params,
            "l0_update_nonzeros": nonzero,
            "l0_fraction_nonzero": float(nonzero / n_params),
            "topk_fraction": frac,
            "topk_k": k,
            "dense_comm_bytes": dense_bytes,
            "sparse_comm_bytes": sparse_bytes,
            "compression_ratio": float(sparse_bytes / dense_bytes),
            "savings_bytes": dense_bytes - sparse_bytes,
        })
    return rows


def summarize_sparsity(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["method"], row.get("mu"), row["topk_fraction"])
        groups.setdefault(key, []).append(row)
    out = []
    for (method, mu, frac), items in groups.items():
        out.append({
            "method": method,
            "mu": mu,
            "topk_fraction": frac,
            "n": len(items),
            "dense_comm_bytes_mean": float(np.mean([r["dense_comm_bytes"] for r in items])),
            "sparse_comm_bytes_mean": float(np.mean([r["sparse_comm_bytes"] for r in items])),
            "compression_ratio_mean": float(np.mean([r["compression_ratio"] for r in items])),
            "l0_fraction_nonzero_mean": float(np.mean([r["l0_fraction_nonzero"] for r in items])),
        })
    return sorted(out, key=lambda r: (str(r["method"]), str(r["mu"]), float(r["topk_fraction"])))


def dense_vs_sparse_lp_source(method_rows: list[dict], sparsity_rows: list[dict]) -> list[dict]:
    sparse_by_method = {}
    for row in summarize_sparsity(sparsity_rows):
        if float(row["topk_fraction"]) == 0.1:
            sparse_by_method[(row["method"], row.get("mu"))] = row["sparse_comm_bytes_mean"]
    out = []
    for row in method_rows:
        method = row.get("method", row.get("run_type", ""))
        mu = row.get("mu")
        key = (method, mu)
        dense = float(row.get("total_comm_until_stop", row.get("comm", 0)) or 0)
        sparse_unit = sparse_by_method.get(key)
        stopped = float(row.get("stopped_round", 1) or 1)
        out.append({
            **row,
            "dense_cost": dense,
            "sparse_cost": float(sparse_unit * stopped) if sparse_unit is not None else dense,
        })
    return out
