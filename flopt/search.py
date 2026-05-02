
from dataclasses import replace

import numpy as np
from scipy.optimize import differential_evolution

from .config import FLConfig
from .fedavg import federated_train
from .models import HARMLP


def ga_search(clients,base_cfg:FLConfig,bounds=None,maxiter:int=4,popsize:int=5,gamma:float=1e-8,model_factory=None,score_key:str="accuracy",min_score:float=0.80):
    model_factory=model_factory or HARMLP
    if bounds is None:
        bounds=[(1,8),(3,min(20,len(clients))),(0.001,0.08)]
    history=[]

    def objective(raw):
        local_epochs=max(1,int(round(raw[0])))
        clients_per_round=max(1,int(round(raw[1])))
        lr=float(raw[2])
        cfg=replace(base_cfg,local_epochs=local_epochs,clients_per_round=clients_per_round,lr=lr)
        _,records=federated_train(model_factory(),clients,cfg)
        last=records[-1]
        comm=sum(r["upload_bytes"]+r["download_bytes"] for r in records)
        score=float(last.get(score_key,last["accuracy"]))
        penalty=max(0,min_score-score)*10
        fitness=last["loss"]+gamma*comm+penalty
        history.append({"evaluation":len(history)+1,"local_epochs":local_epochs,"clients_per_round":clients_per_round,"lr":lr,"loss":last["loss"],"accuracy":last["accuracy"],score_key:score,"comm":comm,"fitness":fitness})
        return fitness

    result=differential_evolution(objective,bounds,maxiter=maxiter,popsize=popsize,polish=False,seed=base_cfg.seed)
    return {"x":result.x.tolist(),"fitness":float(result.fun),"evaluations":int(result.nfev),"history":history}


def grid_search(clients,base_cfg:FLConfig,grid:list[tuple[int,int,float]],gamma:float=1e-8,model_factory=None,score_key:str="accuracy",min_score:float=0.80):
    model_factory=model_factory or HARMLP
    rows=[]
    for local_epochs,clients_per_round,lr in grid:
        cfg=replace(base_cfg,local_epochs=local_epochs,clients_per_round=clients_per_round,lr=lr)
        _,records=federated_train(model_factory(),clients,cfg)
        last=records[-1]
        comm=sum(r["upload_bytes"]+r["download_bytes"] for r in records)
        score=float(last.get(score_key,last["accuracy"]))
        rows.append({
            "local_epochs":local_epochs,
            "clients_per_round":clients_per_round,
            "lr":lr,
            "loss":last["loss"],
            "accuracy":last["accuracy"],
            score_key:score,
            "comm":comm,
            "fitness":last["loss"]+gamma*comm+max(0,min_score-score)*10,
        })
    return sorted(rows,key=lambda r:r["fitness"])

