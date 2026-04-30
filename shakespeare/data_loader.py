"""Download flwrlabs/shakespeare from HuggingFace, tokenize, partition into FL clients."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from .utils import ClientData

_CSV_URL = "https://huggingface.co/datasets/flwrlabs/shakespeare/resolve/main/shakespeare.csv"
_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_MIN_SAMPLES = 2000
_TRAIN_FRAC = 0.80


@dataclass
class ShakespeareBundle:
    clients: list[ClientData]
    client_names: list[str]
    vocab: dict[str, int]
    inv_vocab: dict[int, str]
    vocab_size: int
    total_samples: int
    raw_csv: Path


def load_shakespeare(min_samples: int = _MIN_SAMPLES,
                     force: bool = False) -> ShakespeareBundle:
    npz_path = _DATA / "processed.npz"
    meta_path = _DATA / "meta.json"
    if npz_path.exists() and meta_path.exists() and not force:
        return _load_cached(npz_path, meta_path)
    csv_path = _ensure_csv()
    print("  Parsing CSV with pandas ...")
    df = pd.read_csv(csv_path, usecols=["x", "y", "character_id"],
                     dtype=str, engine="c")
    df.dropna(subset=["x", "y", "character_id"], inplace=True)
    total_samples = len(df)
    print(f"  {total_samples:,} samples loaded")
    all_chars: set[str] = set()
    for s in df["x"]:
        all_chars.update(s)
    all_chars.update(df["y"].unique())
    ordered = sorted(all_chars)
    vocab = {ch: i for i, ch in enumerate(ordered)}
    inv_vocab = {i: ch for ch, i in vocab.items()}
    vs = len(vocab)
    print(f"  vocab_size={vs}")
    print("  Tokenizing ...")
    x_all = np.array([[vocab.get(ch, 0) for ch in s[:80]]
                       for s in df["x"].values], dtype=np.int16)
    y_all = np.array([vocab.get(ch, 0) for ch in df["y"].values],
                     dtype=np.int16)
    cid_all = df["character_id"].values
    clients: list[ClientData] = []
    client_names: list[str] = []
    arrays_for_save: dict[str, np.ndarray] = {}
    unique_cids = sorted(set(cid_all))
    print(f"  {len(unique_cids)} unique character_ids, "
          f"filtering >= {min_samples} samples ...")
    for cid in unique_cids:
        mask = cid_all == cid
        n = int(mask.sum())
        if n < min_samples:
            continue
        xc = x_all[mask].astype(np.int64)
        yc = y_all[mask].astype(np.int64)
        split = int(n * _TRAIN_FRAC)
        if split < 1 or n - split < 1:
            continue
        idx = len(clients)
        clients.append(ClientData(xc[:split], yc[:split],
                                  xc[split:], yc[split:], client_id=idx))
        client_names.append(cid)
        arrays_for_save[f"x_train_{idx}"] = xc[:split]
        arrays_for_save[f"y_train_{idx}"] = yc[:split]
        arrays_for_save[f"x_test_{idx}"] = xc[split:]
        arrays_for_save[f"y_test_{idx}"] = yc[split:]
    del df, x_all, y_all, cid_all
    _DATA.mkdir(parents=True, exist_ok=True)
    print(f"  {len(clients)} clients after filtering, saving cache ...")
    np.savez_compressed(npz_path, **arrays_for_save)
    meta = {"vocab": vocab, "client_names": client_names, "vocab_size": vs,
            "total_samples": total_samples, "n_clients": len(clients)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Cached to {npz_path}")
    return ShakespeareBundle(clients, client_names, vocab, inv_vocab,
                             vs, total_samples, csv_path)


def _load_cached(npz_path: Path, meta_path: Path) -> ShakespeareBundle:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    inv_vocab = {int(v): k for k, v in vocab.items()}
    data = np.load(npz_path)
    clients = []
    for i in range(meta["n_clients"]):
        clients.append(ClientData(
            data[f"x_train_{i}"], data[f"y_train_{i}"],
            data[f"x_test_{i}"], data[f"y_test_{i}"], client_id=i))
    csv_path = _DATA / "shakespeare.csv"
    return ShakespeareBundle(clients, meta["client_names"], vocab, inv_vocab,
                             meta["vocab_size"], meta["total_samples"], csv_path)


def _ensure_csv() -> Path:
    _DATA.mkdir(parents=True, exist_ok=True)
    csv_path = _DATA / "shakespeare.csv"
    if csv_path.exists():
        return csv_path
    print(f"Downloading Shakespeare dataset to {csv_path} ...")
    urlretrieve(_CSV_URL, csv_path)
    print(f"Done ({csv_path.stat().st_size / 1e6:.1f} MB)")
    return csv_path
