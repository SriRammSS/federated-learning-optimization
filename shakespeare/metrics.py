from __future__ import annotations
import math
import numpy as np


def perplexity_from_loss(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def per_client_summary(client_accs: list[float],
                       client_losses: list[float]) -> dict:
    accs = np.array(client_accs)
    losses = np.array(client_losses)
    return {
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "worst_accuracy": float(accs.min()),
        "best_accuracy": float(accs.max()),
        "mean_loss": float(losses.mean()),
        "mean_perplexity": perplexity_from_loss(float(losses.mean())),
    }


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray,
                   k: int = 5) -> float:
    top = np.argsort(logits, axis=1)[:, -k:]
    hits = np.any(top == targets[:, None], axis=1)
    return float(hits.mean())
