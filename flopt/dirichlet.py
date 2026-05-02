
from pathlib import Path

import numpy as np

from .data import ClientData


def dirichlet_split(
    arrays_path: Path,
    beta: float | str,
    k_clients: int,
    seed: int,
    min_train: int = 10,
    min_test: int = 2,
):
    arr = np.load(arrays_path, allow_pickle=True)
    x = arr["x"].astype('float32')
    y = arr["y"].astype('int64')
    splits = arr["split"].astype(str)
    train_idx = np.where(splits == 'train')[0]
    test_idx = np.where(splits == 'test')[0]
    train_parts = _partition_indices(train_idx, y[train_idx], beta, k_clients, seed)
    test_parts = _partition_indices(test_idx, y[test_idx], beta, k_clients, seed + 100_000)

    clients: list[ClientData] = []
    map_rows: list[dict] = []
    dist_rows: list[dict] = []
    for cid in range(k_clients):
        tr = train_parts[cid]
        te = test_parts[cid]
        if len(tr) < min_train or len(te) < min_test:
            continue
        new_cid = len(clients)
        clients.append(ClientData(x[tr], y[tr], x[te], y[te], client_id=new_cid))
        for split_name, idxs in [("train", tr), ("test", te)]:
            labels = y[idxs]
            deaths = int(labels.sum())
            rows = int(len(labels))
            dist_rows.append({
                "synthetic_client_id": new_cid,
                "original_slot": cid,
                "split": split_name,
                "rows": rows,
                "deaths": deaths,
                "mortality_rate": float(deaths / rows) if rows else 0.0,
                "beta": str(beta),
                "seed": seed,
            })
        map_rows.append({
            "synthetic_client_id": new_cid,
            "original_slot": cid,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "beta": str(beta),
            "seed": seed,
        })
    return clients, map_rows, dist_rows


def _partition_indices(indices, labels, beta, k_clients, seed):
    rng = np.random.default_rng(seed)
    by_label = [indices[labels == lab].copy() for lab in sorted(set(labels.tolist()))]
    for idxs in by_label:
        rng.shuffle(idxs)

    if str(beta).lower() in {'inf', 'infinity', 'none'}:
        parts = [[] for _ in range(k_clients)]
        merged = indices.copy()
        rng.shuffle(merged)
        for i, idx in enumerate(merged):
            parts[i % k_clients].append(int(idx))
        return [np.array(p, dtype=np.int64) for p in parts]

    beta_val = float(beta)
    parts = [[] for _ in range(k_clients)]
    for label_indices in by_label:
        proportions = rng.dirichlet(np.full(k_clients, beta_val))
        cuts = (np.cumsum(proportions)[:-1] * len(label_indices)).astype(int)
        splits = np.split(label_indices, cuts)
        for cid, split in enumerate(splits):
            parts[cid].extend(int(v) for v in split)
    for p in parts:
        rng.shuffle(p)
    return [np.array(p, dtype=np.int64) for p in parts]


def partition_audit(dist_rows):
    groups = {}
    for row in dist_rows:
        key = (row["beta"], row["seed"], row["split"])
        groups.setdefault(key, []).append(row)
    audit = []
    for (beta, seed, split), items in groups.items():
        rates = np.array([float(r["mortality_rate"]) for r in items], dtype=float)
        sizes = np.array([int(r["rows"]) for r in items], dtype=float)
        audit.append({
            "beta": beta,
            "seed": seed,
            "split": split,
            "clients": len(items),
            "rows_total": int(sizes.sum()),
            "rows_min": int(sizes.min()) if len(sizes) else 0,
            "rows_max": int(sizes.max()) if len(sizes) else 0,
            "mortality_rate_mean": float(rates.mean()) if len(rates) else 0.0,
            "mortality_rate_std": float(rates.std()) if len(rates) else 0.0,
            "mortality_rate_min": float(rates.min()) if len(rates) else 0.0,
            "mortality_rate_max": float(rates.max()) if len(rates) else 0.0,
        })
    return audit
