"""FL training loops for Shakespeare — CUDA-optimized with AMP mixed precision."""
from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import ShakespeareConfig
from .models import count_parameters
from .utils import ClientData

_EVAL_SUBSET = 30


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optim(model: nn.Module, cfg: ShakespeareConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr)


def _use_amp(cfg: ShakespeareConfig) -> bool:
    return cfg.use_amp and torch.cuda.is_available()


def _clear_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _loader(x: np.ndarray, y: np.ndarray, cfg: ShakespeareConfig,
            shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.long),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=True,
                      persistent_workers=cfg.num_workers > 0)


# ---------------------------------------------------------------------------
# FedAvg with CVaR, early stopping, drift tracking
# ---------------------------------------------------------------------------

def federated_train(model: nn.Module, clients: list[ClientData],
                    cfg: ShakespeareConfig,
                    track_drift: bool = False) -> tuple[nn.Module, list[dict]]:
    _set_seed(cfg.seed)
    device = _device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    gm = deepcopy(model).to(device)
    records: list[dict] = []
    cids = list(range(len(clients)))
    mr = cfg.max_rounds or cfg.rounds
    best_val = float("inf"); best_rnd = 0; stale = 0; best_st = None
    eval_sub = [clients[i] for i in range(min(_EVAL_SUBSET, len(clients)))]

    for rnd in range(1, mr + 1):
        sel = random.sample(cids, min(cfg.clients_per_round, len(cids)))
        ls, lsz, ll = [], [], []
        base = {k: v.detach().cpu().clone() for k, v in gm.state_dict().items()}
        for c in sel:
            lm = deepcopy(gm)
            loss = train_one_client(lm, clients[c], cfg, device)
            ls.append({k: v.detach().cpu() for k, v in lm.state_dict().items()})
            lsz.append(len(clients[c].x_train))
            ll.append(loss)
        w = _agg_weights(np.array(lsz), np.array(ll), cfg)
        drift = _drift_stats(base, ls, sel, w) if track_drift else {}
        _load_weighted(gm, ls, w, device)

        met = evaluate_all(gm, eval_sub, device)
        cur = float(met[cfg.monitor])
        if cur < best_val - cfg.min_delta:
            best_val = cur; best_rnd = rnd; stale = 0
            best_st = {k: v.detach().cpu().clone()
                       for k, v in gm.state_dict().items()}
        else:
            stale += 1
        stop = bool(cfg.early_stopping and stale >= cfg.patience)
        met.update({
            "round": rnd, "selected_clients": sel,
            "upload_bytes": count_parameters(gm) * 4 * len(sel),
            "download_bytes": count_parameters(gm) * 4 * len(sel),
            "config": asdict(cfg), "best_loss_so_far": best_val,
            "best_round": best_rnd, "rounds_since_improvement": stale,
            "stopped_early": stop, **drift,
        })
        records.append(met)
        if rnd % 10 == 0 or stop:
            print(f"      [rnd {rnd:3d}] loss={met['loss']:.4f} "
                  f"acc={met['accuracy']:.4f} stale={stale}", flush=True)
        if stop:
            break

    if best_st is not None:
        gm.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    final = evaluate_all(gm, clients, device)
    records[-1].update({f"full_{k}": v for k, v in final.items()
                        if k in ("loss", "accuracy",
                                 "worst_client_accuracy", "perplexity")})
    for k in ("loss", "accuracy", "worst_client_accuracy", "perplexity"):
        records[-1][k] = final[k]
    return gm, records


def train_one_client(model: nn.Module, client: ClientData,
                     cfg: ShakespeareConfig,
                     device: torch.device) -> float:
    model.train()
    loader = _loader(client.x_train, client.y_train, cfg)
    opt = _optim(model, cfg)
    loss_fn = nn.CrossEntropyLoss()
    amp = _use_amp(cfg)
    scaler = GradScaler(enabled=amp)
    last = 0.0
    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            last = float(loss.detach())
    _clear_cache(device)
    return last


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_all(model: nn.Module, clients: list[ClientData],
                 device: torch.device) -> dict:
    """Batched evaluation with AMP: single forward pass over all test data."""
    model.eval()
    sizes = [len(c.x_test) for c in clients]
    all_x = np.concatenate([c.x_test for c in clients])
    all_y = np.concatenate([c.y_test for c in clients])
    x_t = torch.tensor(all_x, dtype=torch.long, device=device)
    y_t = torch.tensor(all_y, dtype=torch.long, device=device)

    chunks = []
    bs = 8192
    amp = torch.cuda.is_available()
    for i in range(0, len(x_t), bs):
        with autocast(enabled=amp):
            chunks.append(model(x_t[i:i + bs]).float())
    logits = torch.cat(chunks)
    pred = logits.argmax(1)

    cl, ca = [], []
    off = 0
    for s in sizes:
        sl = logits[off:off + s]; sy = y_t[off:off + s]
        cl.append(float(nn.CrossEntropyLoss()(sl, sy)))
        ca.append(float((pred[off:off + s] == sy).float().mean()))
        off += s
    del logits, x_t, y_t, chunks, pred
    _clear_cache(device)
    ml = float(np.mean(cl))
    return {
        "loss": ml, "accuracy": float(np.mean(ca)),
        "worst_client_accuracy": float(np.min(ca)),
        "perplexity": math.exp(min(ml, 20.0)),
        "client_loss": cl, "client_accuracy": ca,
    }


@torch.no_grad()
def predict_clients(model: nn.Module, clients: list[ClientData],
                    device: torch.device | None = None) -> list[dict]:
    device = device or _device()
    model = model.to(device); model.eval()
    sizes = [len(c.x_test) for c in clients]
    all_x = np.concatenate([c.x_test for c in clients])
    all_y = np.concatenate([c.y_test for c in clients])
    x_t = torch.tensor(all_x, dtype=torch.long, device=device)
    y_t = torch.tensor(all_y, dtype=torch.long, device=device)
    chunks = []
    bs = 8192
    amp = torch.cuda.is_available()
    for i in range(0, len(x_t), bs):
        with autocast(enabled=amp):
            chunks.append(model(x_t[i:i + bs]).float())
    logits = torch.cat(chunks)
    pred = logits.argmax(1)
    conf = torch.softmax(logits, dim=1).max(1).values
    rows = []
    off = 0
    for idx, s in enumerate(sizes):
        cid = clients[idx].client_id if clients[idx].client_id is not None else idx
        for i in range(s):
            rows.append({
                "client_id": int(cid), "row": int(i),
                "y_true": int(y_t[off + i].cpu()),
                "y_pred": int(pred[off + i].cpu()),
                "confidence": float(conf[off + i].cpu()),
            })
        off += s
    del logits, x_t, y_t, chunks, pred, conf
    _clear_cache(device)
    return rows


# ---------------------------------------------------------------------------
# Centralized baseline
# ---------------------------------------------------------------------------

def centralized_train(model: nn.Module, clients: list[ClientData],
                      cfg: ShakespeareConfig) -> tuple[nn.Module, list[dict]]:
    device = _device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = deepcopy(model).to(device)
    x = np.concatenate([c.x_train for c in clients])
    y = np.concatenate([c.y_train for c in clients])
    loader = _loader(x, y, cfg)
    opt = _optim(model, cfg)
    loss_fn = nn.CrossEntropyLoss()
    amp = _use_amp(cfg)
    scaler = GradScaler(enabled=amp)
    eval_sub = [clients[i] for i in range(min(_EVAL_SUBSET, len(clients)))]
    records = []; best = float("inf"); stale = 0
    mr = cfg.max_rounds or cfg.rounds
    for ep in range(1, mr + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        met = evaluate_all(model, eval_sub, device)
        if met["loss"] < best - cfg.min_delta:
            best = met["loss"]; stale = 0
        else:
            stale += 1
        met.update({
            "round": ep, "upload_bytes": 0, "download_bytes": 0,
            "selected_clients": [], "best_loss_so_far": best,
            "rounds_since_improvement": stale,
            "stopped_early": bool(cfg.early_stopping and stale >= cfg.patience),
        })
        records.append(met)
        if ep % 5 == 0 or met["stopped_early"]:
            print(f"      [ep {ep:3d}] loss={met['loss']:.4f} "
                  f"acc={met['accuracy']:.4f} stale={stale}", flush=True)
        if met["stopped_early"]:
            break
    final = evaluate_all(model, clients, device)
    for k in ("loss", "accuracy", "worst_client_accuracy", "perplexity"):
        records[-1][k] = final[k]
    return model, records


# ---------------------------------------------------------------------------
# Local-only baseline
# ---------------------------------------------------------------------------

def local_only_summary(model_factory, clients: list[ClientData],
                       cfg: ShakespeareConfig) -> tuple[list[dict], list[dict]]:
    device = _device()
    rows, rr = [], []
    n = len(clients)
    for idx, c in enumerate(clients):
        m = model_factory().to(device)
        _train_local(m, c, cfg, device)
        met = evaluate_all(m, [c], device)
        cid = c.client_id if c.client_id is not None else idx
        rows.append({
            "client_id": int(cid), "loss": met["loss"],
            "accuracy": met["accuracy"], "perplexity": met["perplexity"],
            "test_samples": len(c.x_test), "train_samples": len(c.x_train),
        })
        rr.append({
            "run_type": "local_only", "seed": cfg.seed,
            "client_id": int(cid), "loss": met["loss"],
            "accuracy": met["accuracy"], "perplexity": met["perplexity"],
        })
        if (idx + 1) % 50 == 0:
            print(f"      local {idx + 1}/{n} done", flush=True)
    return rows, rr


def _train_local(model: nn.Module, client: ClientData,
                 cfg: ShakespeareConfig, device: torch.device) -> None:
    loader = _loader(client.x_train, client.y_train, cfg)
    opt = _optim(model, cfg)
    loss_fn = nn.CrossEntropyLoss()
    amp = _use_amp(cfg)
    scaler = GradScaler(enabled=amp)
    epochs = cfg.local_epochs * max(1, min(cfg.max_rounds or cfg.rounds, 20))
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
    _clear_cache(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agg_weights(sizes: np.ndarray, losses: np.ndarray,
                 cfg: ShakespeareConfig) -> np.ndarray:
    w = sizes.astype("float64") / sizes.sum()
    if cfg.cvar_alpha <= 0:
        return w
    tau = np.quantile(losses, cfg.cvar_alpha)
    tail = np.maximum(losses - tau, 0)
    if tail.sum() == 0:
        return w
    w = w * (1 + cfg.fairness_strength * tail / tail.sum())
    return w / w.sum()


def _load_weighted(model: nn.Module, states: list[dict],
                   w: np.ndarray, device: torch.device) -> None:
    avg = {}
    for key in states[0]:
        avg[key] = sum(w[i] * states[i][key]
                       for i in range(len(states))).to(device)
    model.load_state_dict(avg)


def _drift_stats(base: dict, states: list[dict],
                 sel: list[int], w: np.ndarray) -> dict:
    updates, norms = [], []
    for st in states:
        parts = [(st[k] - base[k]).reshape(-1).float() for k in st]
        u = torch.cat(parts)
        updates.append(u)
        norms.append(float(torch.linalg.vector_norm(u)))
    avg = sum(float(w[i]) * updates[i] for i in range(len(updates)))
    an = float(torch.linalg.vector_norm(avg))
    cos, dist = [], []
    for u in updates:
        d = float(torch.linalg.vector_norm(u)) * max(an, 1e-12)
        cos.append(float(torch.dot(u, avg) / d) if d > 0 else 0.0)
        dist.append(float(torch.linalg.vector_norm(u - avg)))
    return {
        "drift_client_ids": sel, "drift_update_norms": norms,
        "drift_cosine_to_mean": cos, "drift_distance_to_mean": dist,
        "drift_mean_update_norm": float(np.mean(norms)) if norms else 0.0,
        "drift_avg_update_norm": an,
    }
