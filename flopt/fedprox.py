from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import FLConfig
from .data import ClientData
from .fedavg import _aggregation_weights,_drift_stats,evaluate_all
from .models import count_parameters
from .sparsity import update_sparsity_rows
from .utils import _device,_load_weighted_state,_loss_fn,_optimizer,_set_seed


def fedprox_train(
    model:nn.Module,
    clients:list[ClientData],
    cfg:FLConfig,
    mu:float=0.01,
    track_drift:bool=False,
    track_sparsity:bool=False,
)->tuple[nn.Module,list[dict],list[dict]]:
    """FedProx training with the same record schema as FedAvg.

    The local objective is CE(w; D_k) + mu/2 * ||w - w_global||^2. Setting
    mu=0 is a sanity check that this reduces to FedAvg local training.
    """
    _set_seed(cfg.seed)
    device=_device()
    global_model=deepcopy(model).to(device)
    records:list[dict]=[]
    sparsity_rows:list[dict]=[]
    client_ids=list(range(len(clients)))
    max_rounds=cfg.max_rounds or cfg.rounds
    best_value=float("inf")
    best_round=0
    stale_rounds=0
    best_state=None

    for round_id in range(1,max_rounds+1):
        selected=random.sample(client_ids,min(cfg.clients_per_round,len(client_ids)))
        local_states=[]
        local_sizes=[]
        local_losses=[]
        base_state={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
        global_ref={k:v.detach().clone() for k,v in global_model.state_dict().items()}

        for cid in selected:
            local_model=deepcopy(global_model)
            loss=train_one_client_fedprox(local_model,clients[cid],cfg,device,global_ref,mu)
            state_cpu={k:v.detach().cpu() for k,v in local_model.state_dict().items()}
            if track_sparsity:
                sparsity_rows.extend(update_sparsity_rows(base_state,state_cpu,round_id,cid,"fedprox",cfg.seed,mu))
            local_states.append(state_cpu)
            local_sizes.append(len(clients[cid].x_train))
            local_losses.append(loss)

        weights=_aggregation_weights(np.array(local_sizes),np.array(local_losses),cfg)
        drift=_drift_stats(base_state,local_states,selected,weights) if track_drift else {}
        _load_weighted_state(global_model,local_states,weights,device)
        metrics=evaluate_all(global_model,clients,device)
        current=float(metrics[cfg.monitor])
        if current<best_value-cfg.min_delta:
            best_value=current
            best_round=round_id
            stale_rounds=0
            best_state={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
        else:
            stale_rounds+=1
        stopped=bool(cfg.early_stopping and stale_rounds>=cfg.patience)
        metrics.update({
            "round":round_id,
            "selected_clients":selected,
            "upload_bytes":count_parameters(global_model)*4*len(selected),
            "download_bytes":count_parameters(global_model)*4*len(selected),
            "mu":mu,
            "best_loss_so_far":best_value,
            "best_round":best_round,
            "rounds_since_improvement":stale_rounds,
            "stopped_early":stopped,
            **drift,
        })
        records.append(metrics)
        if stopped:
            break

    if best_state is not None:
        global_model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    return global_model,records,sparsity_rows


def train_one_client_fedprox(
    model:nn.Module,
    client:ClientData,
    cfg:FLConfig,
    device:torch.device,
    global_ref:dict[str,torch.Tensor],
    mu:float,
)->float:
    model.train()
    x=torch.tensor(client.x_train,dtype=torch.float32)
    y=torch.tensor(client.y_train,dtype=torch.long)
    loader=DataLoader(TensorDataset(x,y),batch_size=cfg.batch_size,shuffle=True)
    opt=_optimizer(model,cfg)
    loss_fn=_loss_fn(cfg,device)
    last_loss=0.0
    ref={k:v.to(device) for k,v in global_ref.items()}
    for _ in range(cfg.local_epochs):
        for xb,yb in loader:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            if mu>0:
                prox=torch.zeros((),device=device)
                for name,param in model.named_parameters():
                    prox=prox+torch.sum((param-ref[name])**2)
                loss=loss+0.5*mu*prox
            loss.backward()
            opt.step()
            last_loss=float(loss.detach().cpu())
    return last_loss
