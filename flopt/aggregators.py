from __future__ import annotations

import numpy as np

from .config import FLConfig


def sample_size_weights(sizes:np.ndarray)->np.ndarray:
    weights=sizes.astype("float64")
    total=weights.sum()
    if total <= 0:
        raise ValueError("client sizes must sum to a positive value")
    return weights / total


def cvar_reweighted_sizes(sizes:np.ndarray,losses:np.ndarray,alpha:float,fairness_strength:float)->np.ndarray:
    base_weights=sample_size_weights(sizes)
    if alpha <= 0:
        return base_weights

    tau=np.quantile(losses,alpha)
    tail=np.maximum(losses - tau,0.0)
    if tail.sum() == 0:
        return base_weights

    # No hard cutoff here, just giving high-loss clients a bit extra weight, that's the scene.
    adjusted=base_weights * (1 + fairness_strength * tail / tail.sum())
    return adjusted / adjusted.sum()


def aggregation_weights(sizes:np.ndarray,losses:np.ndarray,cfg:FLConfig)->np.ndarray:
    return cvar_reweighted_sizes(sizes=sizes,losses=losses,alpha=cfg.cvar_alpha,fairness_strength=cfg.fairness_strength)
