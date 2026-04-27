from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import differential_evolution

from .config import FLConfig
from .fedavg import federated_train
from .models import HARMLP


def ga_search(clients,base_cfg:FLConfig,bounds=None,maxiter:int=4,popsize:int=5,gamma:float=1e-8)->dict:
    if bounds is None:
        bounds=[(1,8),(3,min(20,len(clients))),(0.001,0.08)]
    history=[]

    def objective(raw):
        local_epochs=max(1,int(round(raw[0])))
        clients_per_round=max(1,int(round(raw[1])))
        lr=float(raw[2])
        cfg=replace(base_cfg,local_epochs=local_epochs,clients_per_round=clients_per_round,lr=lr)
        _,records=federated_train(HARMLP(),clients,cfg)
        last=records[-1]
        comm=sum(r["upload_bytes"]+r["download_bytes"] for r in records)
        penalty=max(0,0.80-last["accuracy"])*10
        fitness=last["loss"]+gamma*comm+penalty
        history.append({"evaluation":len(history)+1,"local_epochs":local_epochs,"clients_per_round":clients_per_round,"lr":lr,"loss":last["loss"],"accuracy":last["accuracy"],"comm":comm,"fitness":fitness})
        return fitness

    result=differential_evolution(objective,bounds,maxiter=maxiter,popsize=popsize,polish=False,seed=base_cfg.seed)
    return {"x":result.x.tolist(),"fitness":float(result.fun),"evaluations":int(result.nfev),"history":history}


def grid_search(clients,base_cfg:FLConfig,grid:list[tuple[int,int,float]],gamma:float=1e-8)->list[dict]:
    rows=[]
    for local_epochs,clients_per_round,lr in grid:
        cfg=replace(base_cfg,local_epochs=local_epochs,clients_per_round=clients_per_round,lr=lr)
        _,records=federated_train(HARMLP(),clients,cfg)
        last=records[-1]
        comm=sum(r["upload_bytes"]+r["download_bytes"] for r in records)
        rows.append({
            "local_epochs":local_epochs,
            "clients_per_round":clients_per_round,
            "lr":lr,
            "loss":last["loss"],
            "accuracy":last["accuracy"],
            "comm":comm,
            "fitness":last["loss"]+gamma*comm+max(0,0.80-last["accuracy"])*10,
        })
    return sorted(rows,key=lambda r:r["fitness"])

