from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ShakespeareConfig:
    rounds: int = 100
    max_rounds: int | None = None
    local_epochs: int = 1
    clients_per_round: int = 10
    lr: float = 0.8
    batch_size: int = 256
    seed: int = 7
    cvar_alpha: float = 0.0
    fairness_strength: float = 1.0
    patience: int = 15
    min_delta: float = 0.001
    early_stopping: bool = True
    monitor: str = "loss"
    optimizer: str = "sgd"
    vocab_size: int = 80
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 256
    grad_clip: float = 5.0
    use_amp: bool = True
    num_workers: int = 4
