from __future__ import annotations

from dataclasses import replace

from scipy.optimize import differential_evolution

from .config import FLConfig
from .fedavg import federated_train
from .models import HARMLP


def ga_search(clients,base_cfg:FLConfig,bounds=None,maxiter:int=4,popsize:int=5,gamma:float=1e-8,
              model_factory=None,score_key:str="accuracy",min_score:float=0.80)->dict:
    model_factory=model_factory or HARMLP
    if bounds is None:
        bounds=[(1,8),(3,min(20,len(clients))),(0.001,0.08)]

    history=[]

    def objective(raw):
        local_epochs=max(1,int(round(raw[0])))
        clients_per_round=max(1,int(round(raw[1])))
        learning_rate=float(raw[2])
        cfg=replace(base_cfg,local_epochs=local_epochs,clients_per_round=clients_per_round,lr=learning_rate)
        _,records=federated_train(model_factory(),clients,cfg)
        last=records[-1]
        communication=sum(row["upload_bytes"] + row["download_bytes"] for row in records)
        score=float(last.get(score_key,last["accuracy"]))
        penalty=max(0.0,min_score - score) * 10
        fitness=last["loss"] + gamma * communication + penalty
        history.append({
            "evaluation":len(history) + 1,
            "local_epochs":local_epochs,
            "clients_per_round":clients_per_round,
            "lr":learning_rate,
            "loss":last["loss"],
            "accuracy":last["accuracy"],
            score_key:score,
            "comm":communication,
            "fitness":fitness,
        })
        return fitness

    result=differential_evolution(objective,bounds,maxiter=maxiter,popsize=popsize,polish=False,seed=base_cfg.seed)
    return {
        "x":result.x.tolist(),
        "fitness":float(result.fun),
        "evaluations":int(result.nfev),
        "history":history,
    }


def grid_search(clients,base_cfg:FLConfig,grid:list[tuple[int,int,float]],gamma:float=1e-8,
                model_factory=None,score_key:str="accuracy",min_score:float=0.80)->list[dict]:
    model_factory=model_factory or HARMLP
    rows=[]

    for local_epochs,clients_per_round,learning_rate in grid:
        cfg=replace(base_cfg,local_epochs=local_epochs,clients_per_round=clients_per_round,lr=learning_rate)
        _,records=federated_train(model_factory(),clients,cfg)
        last=records[-1]
        communication=sum(row["upload_bytes"] + row["download_bytes"] for row in records)
        score=float(last.get(score_key,last["accuracy"]))
        rows.append({
            "local_epochs":local_epochs,
            "clients_per_round":clients_per_round,
            "lr":learning_rate,
            "loss":last["loss"],
            "accuracy":last["accuracy"],
            score_key:score,
            "comm":communication,
            "fitness":last["loss"] + gamma * communication + max(0.0,min_score - score) * 10,
        })

    return sorted(rows,key=lambda row:row["fitness"])
