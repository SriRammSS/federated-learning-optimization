from dataclasses import dataclass


@dataclass(frozen=True)
class FLConfig:
    rounds:int=20
    max_rounds:int|None=None
    local_epochs:int=2
    clients_per_round:int=10
    lr:float=0.01
    batch_size:int=32
    seed:int=7
    cvar_alpha:float=0.0
    fairness_strength:float=1.0
    patience:int=30
    min_delta:float=0.001
    early_stopping:bool=False
    monitor:str="loss"

