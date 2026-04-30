"""Comprehensive EDA for Shakespeare FL dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "outputs" / "eda"
_cache = _HERE / "outputs" / "mpl-cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.manifold import TSNE


def run_eda(bundle) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    clients = bundle.clients
    names = bundle.client_names
    inv_vocab = bundle.inv_vocab
    vs = bundle.vocab_size
    print(f"[EDA] {len(clients)} clients, vocab_size={vs}, "
          f"total_samples={bundle.total_samples}")

    sizes_tr = [len(c.x_train) for c in clients]
    sizes_te = [len(c.x_test) for c in clients]

    _client_sample_bar(names, sizes_tr, sizes_te)
    _client_pie(names, sizes_tr)
    _vocab_frequency(clients, inv_vocab, vs)
    _per_client_char_heatmap(clients, names, inv_vocab, vs)
    _label_distribution(clients, inv_vocab, vs)

    global_dist = _global_label_dist(clients, vs)
    kl, js, ent = _noniid_metrics(clients, vs, global_dist)
    _noniid_bar(names, kl, "KL Divergence", "kl_divergence")
    _noniid_bar(names, js, "Jensen-Shannon Divergence", "js_divergence")
    _noniid_bar(names, ent, "Label Entropy", "label_entropy")

    dists = _client_label_dists(clients, vs)
    _similarity_heatmap(names, dists)
    _tsne_2d(names, dists, sizes_tr)
    _tsne_3d(names, dists, sizes_tr)
    _top_bottom_clients(names, sizes_tr)
    _sample_vs_entropy(names, sizes_tr, ent)

    summary = {
        "n_clients": len(clients), "vocab_size": vs,
        "total_samples": bundle.total_samples,
        "total_train": sum(sizes_tr), "total_test": sum(sizes_te),
        "mean_train": float(np.mean(sizes_tr)),
        "std_train": float(np.std(sizes_tr)),
        "min_train": int(np.min(sizes_tr)),
        "max_train": int(np.max(sizes_tr)),
        "mean_kl": float(np.mean(kl)), "mean_js": float(np.mean(js)),
        "mean_entropy": float(np.mean(ent)),
    }
    (_OUT / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[EDA] done — {len(list(_OUT.glob('*.png')))} plots saved to {_OUT}")


def _short(name: str, maxlen: int = 20) -> str:
    return name[:maxlen] + "..." if len(name) > maxlen else name


def _client_sample_bar(names, sizes_tr, sizes_te):
    order = np.argsort(sizes_tr)[::-1]
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.25), 5))
    x = np.arange(len(names)); w = 0.4
    ax.bar(x, [sizes_tr[i] for i in order], w, label="train")
    ax.bar(x + w, [sizes_te[i] for i in order], w, label="test")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([_short(names[i]) for i in order],
                       rotation=90, fontsize=6)
    ax.set_ylabel("samples")
    ax.set_title("Samples per Client (train / test)")
    ax.legend(); fig.tight_layout()
    fig.savefig(_OUT / "client_samples_bar.png", dpi=160); plt.close(fig)


def _client_pie(names, sizes):
    top_k = 15
    order = np.argsort(sizes)[::-1]
    labels = [_short(names[i]) for i in order[:top_k]] + ["Other"]
    vals = ([sizes[i] for i in order[:top_k]]
            + [sum(sizes[i] for i in order[top_k:])])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", textprops={"fontsize": 7})
    ax.set_title("Client Size Distribution"); fig.tight_layout()
    fig.savefig(_OUT / "client_pie.png", dpi=160); plt.close(fig)


def _vocab_frequency(clients, inv_vocab, vs):
    counts = np.zeros(vs, dtype=np.int64)
    for c in clients:
        for row in c.x_train:
            for tok in row:
                counts[tok] += 1
        for tok in c.y_train:
            counts[tok] += 1
    order = np.argsort(counts)[::-1]
    labels = [repr(inv_vocab.get(int(i), "?")) for i in order]
    fig, ax = plt.subplots(figsize=(max(14, vs * 0.2), 5))
    ax.bar(range(vs), counts[order])
    ax.set_xticks(range(vs))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("count"); ax.set_title("Global Character Frequency")
    fig.tight_layout()
    fig.savefig(_OUT / "vocab_frequency.png", dpi=160); plt.close(fig)


def _per_client_char_heatmap(clients, names, inv_vocab, vs):
    mat = np.zeros((len(clients), vs), dtype=np.float64)
    for i, c in enumerate(clients):
        for tok in c.y_train:
            mat[i, tok] += 1
        s = mat[i].sum()
        if s > 0:
            mat[i] /= s
    order = np.argsort([len(c.x_train) for c in clients])[::-1]
    mat = mat[order]
    xlabels = [repr(inv_vocab.get(i, "?")) for i in range(vs)]
    ylabels = [_short(names[i]) for i in order]
    fig, ax = plt.subplots(
        figsize=(max(14, vs * 0.18), max(8, len(clients) * 0.15)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(vs))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=4)
    step = max(1, len(ylabels) // 30)
    ax.set_yticks(range(0, len(ylabels), step))
    ax.set_yticklabels([ylabels[i] for i in range(0, len(ylabels), step)],
                       fontsize=5)
    ax.set_title("Per-Client Target Character Distribution")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(_OUT / "client_char_heatmap.png", dpi=160); plt.close(fig)


def _label_distribution(clients, inv_vocab, vs):
    counts = np.zeros(vs, dtype=np.int64)
    for c in clients:
        for tok in c.y_train:
            counts[tok] += 1
    order = np.argsort(counts)[::-1][:30]
    labels = [repr(inv_vocab.get(int(i), "?")) for i in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(order)), counts[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Top-30 Target Character Distribution (y labels)")
    fig.tight_layout()
    fig.savefig(_OUT / "label_distribution.png", dpi=160); plt.close(fig)


def _global_label_dist(clients, vs):
    counts = np.zeros(vs, dtype=np.float64)
    for c in clients:
        for tok in c.y_train:
            counts[tok] += 1
    s = counts.sum()
    return counts / s if s > 0 else counts


def _client_label_dists(clients, vs):
    dists = np.zeros((len(clients), vs), dtype=np.float64)
    for i, c in enumerate(clients):
        for tok in c.y_train:
            dists[i, tok] += 1
        s = dists[i].sum()
        if s > 0:
            dists[i] /= s
    return dists


def _noniid_metrics(clients, vs, global_dist):
    eps = 1e-12
    kl, js, ent = [], [], []
    for c in clients:
        cd = np.zeros(vs, dtype=np.float64)
        for tok in c.y_train:
            cd[tok] += 1
        s = cd.sum()
        if s > 0:
            cd /= s
        kl.append(float(entropy(cd + eps, global_dist + eps)))
        m = 0.5 * (cd + global_dist)
        js.append(float(
            0.5 * entropy(cd + eps, m + eps)
            + 0.5 * entropy(global_dist + eps, m + eps)))
        ent.append(float(entropy(cd + eps)))
    return kl, js, ent


def _noniid_bar(names, vals, title, fname):
    order = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.22), 5))
    ax.bar(range(len(names)), [vals[i] for i in order])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_short(names[i]) for i in order],
                       rotation=90, fontsize=5)
    ax.set_ylabel(title); ax.set_title(f"Per-Client {title}")
    fig.tight_layout()
    fig.savefig(_OUT / f"{fname}.png", dpi=160); plt.close(fig)


def _similarity_heatmap(names, dists):
    n = len(names)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = (cosine(dists[i], dists[j])
                 if (dists[i].sum() > 0 and dists[j].sum() > 0) else 1.0)
            sim[i, j] = 1.0 - d
    fig, ax = plt.subplots(
        figsize=(max(10, n * 0.12), max(8, n * 0.12)))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    step = max(1, n // 25)
    short = [_short(nm) for nm in names]
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([short[i] for i in range(0, n, step)],
                       rotation=90, fontsize=5)
    ax.set_yticks(range(0, n, step))
    ax.set_yticklabels([short[i] for i in range(0, n, step)], fontsize=5)
    ax.set_title("Client Label-Distribution Cosine Similarity")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(_OUT / "client_similarity_heatmap.png", dpi=160)
    plt.close(fig)


def _tsne_2d(names, dists, sizes):
    perp = min(30, len(names) - 1)
    if perp < 2:
        return
    pts = TSNE(n_components=2, perplexity=perp,
               random_state=0).fit_transform(dists)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=sizes, cmap="viridis",
                    s=30, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="train samples")
    for i, nm in enumerate(names):
        ax.annotate(_short(nm), (pts[i, 0], pts[i, 1]),
                    fontsize=4, alpha=0.6)
    ax.set_title("t-SNE of Client Label Distributions (2D)")
    fig.tight_layout()
    fig.savefig(_OUT / "tsne_clients_2d.png", dpi=160); plt.close(fig)


def _tsne_3d(names, dists, sizes):
    perp = min(30, len(names) - 1)
    if perp < 2:
        return
    pts = TSNE(n_components=3, perplexity=perp,
               random_state=0).fit_transform(dists)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=sizes,
                    cmap="viridis", s=25, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="train samples", shrink=0.6)
    ax.set_title("t-SNE of Client Label Distributions (3D)")
    fig.tight_layout()
    fig.savefig(_OUT / "tsne_clients_3d.png", dpi=160); plt.close(fig)


def _top_bottom_clients(names, sizes, k=10):
    order = np.argsort(sizes)[::-1]
    top = order[:k]; bot = order[-k:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(range(k), [sizes[i] for i in top])
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels([_short(names[i]) for i in top], fontsize=7)
    axes[0].set_title(f"Top-{k} Clients by Size")
    axes[0].set_xlabel("samples")
    axes[1].barh(range(k), [sizes[i] for i in bot])
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([_short(names[i]) for i in bot], fontsize=7)
    axes[1].set_title(f"Bottom-{k} Clients by Size")
    axes[1].set_xlabel("samples")
    fig.tight_layout()
    fig.savefig(_OUT / "top_bottom_clients.png", dpi=160); plt.close(fig)


def _sample_vs_entropy(names, sizes, ent):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes, ent, s=20, alpha=0.7)
    ax.set_xlabel("train samples"); ax.set_ylabel("label entropy")
    ax.set_title("Client Size vs Label Entropy")
    fig.tight_layout()
    fig.savefig(_OUT / "size_vs_entropy.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    from .data_loader import load_shakespeare
    bundle = load_shakespeare()
    run_eda(bundle)
