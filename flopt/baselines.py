
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from .config import FLConfig
from .data import ClientData
from .fedavg import evaluate_all
from .utils import _device,_loss_fn,_optimizer


def centralized_train(model:nn.Module,clients:list[ClientData],cfg:FLConfig):
    device=_device()
    model=deepcopy(model).to(device)
    x=np.concatenate([c.x_train for c in clients]).astype("float32")
    y=np.concatenate([c.y_train for c in clients]).astype("int64")
    loader=DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=cfg.batch_size,shuffle=True)
    opt=_optimizer(model,cfg)
    loss_fn=_loss_fn(cfg,device)
    records=[]
    best=float("inf")
    stale=0
    max_rounds=cfg.max_rounds or cfg.rounds
    for epoch in range(1,max_rounds+1):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            opt.step()
        metrics=evaluate_all(model,clients,device)
        improved=metrics["loss"]<best-cfg.min_delta
        if improved:
            best=metrics["loss"]
            stale=0
        else:
            stale+=1
        metrics.update({"round":epoch,"upload_bytes":0,"download_bytes":0,"selected_clients":[],"best_loss_so_far":best,"rounds_since_improvement":stale,"stopped_early":bool(cfg.early_stopping and stale>=cfg.patience)})
        records.append(metrics)
        if metrics["stopped_early"]:
            break
    return model,records


def local_only_summary(model_factory,clients:list[ClientData],cfg:FLConfig):
    device=_device()
    rows=[]
    round_rows=[]
    for idx,client in enumerate(clients):
        model=model_factory().to(device)
        train_client_model(model,client,cfg,device)
        metrics=evaluate_all(model,[client],device)
        cid=client.client_id if client.client_id is not None else idx
        rows.append({"client_id":int(cid),"loss":metrics["loss"],"accuracy":metrics["accuracy"],"test_samples":len(client.x_test),"train_samples":len(client.x_train)})
        round_rows.append({"run_type":"local_only","seed":cfg.seed,"client_id":int(cid),"loss":metrics["loss"],"accuracy":metrics["accuracy"]})
    return rows,round_rows


def train_client_model(model:nn.Module,client:ClientData,cfg:FLConfig,device:torch.device):
    x=torch.tensor(client.x_train,dtype=torch.float32)
    y=torch.tensor(client.y_train,dtype=torch.long)
    loader=DataLoader(TensorDataset(x,y),batch_size=cfg.batch_size,shuffle=True)
    opt=_optimizer(model,cfg)
    loss_fn=_loss_fn(cfg,device)
    epochs=cfg.local_epochs*max(1,min(cfg.max_rounds or cfg.rounds,50))
    for _ in range(epochs):
        for xb,yb in loader:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            opt.step()
