
import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score,balanced_accuracy_score,confusion_matrix,roc_auc_score

from .config import FLConfig
from .data import ClientData
from .models import count_parameters
from .utils import _device,_load_weighted_state,_loss_fn,_optimizer,_set_seed


def federated_train(model:nn.Module,clients:list[ClientData],cfg:FLConfig,drift:bool=False):
    _set_seed(cfg.seed)
    device=_device()
    global_model=deepcopy(model).to(device)
    records=[]
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
        for cid in selected:
            local_model=deepcopy(global_model)
            loss=train_one_client(local_model,clients[cid],cfg,device)
            local_states.append({k:v.detach().cpu() for k,v in local_model.state_dict().items()})
            local_sizes.append(len(clients[cid].x_train))
            local_losses.append(loss)
        weights=_aggregation_weights(np.array(local_sizes),np.array(local_losses),cfg)
        drift_stats=_drift_stats(base_state,local_states,selected,weights) if drift else {}
        _load_weighted_state(global_model,local_states,weights,device)
        metrics=evaluate_all(global_model,clients,device)
        current=float(metrics[cfg.monitor])
        improved=current<best_value-cfg.min_delta
        if improved:
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
            "best_loss_so_far":best_value,
            "best_round":best_round,
            "rounds_since_improvement":stale_rounds,
            "stopped_early":stopped,
            **drift_stats,
        })
        records.append(metrics)
        if stopped:
            break
    if best_state is not None:
        global_model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    return global_model,records


def train_one_client(model:nn.Module,client:ClientData,cfg:FLConfig,device:torch.device):
    model.train()
    x=torch.tensor(client.x_train,dtype=torch.float32)
    y=torch.tensor(client.y_train,dtype=torch.long)
    loader=DataLoader(TensorDataset(x,y),batch_size=cfg.batch_size,shuffle=True)
    opt=_optimizer(model,cfg)
    loss_fn=_loss_fn(cfg,device)
    last_loss=0.0
    for _ in range(cfg.local_epochs):
        for xb,yb in loader:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            opt.step()
            last_loss=float(loss.detach().cpu())
    return last_loss


@torch.no_grad()
def evaluate_all(model:nn.Module,clients:list[ClientData],device:torch.device):
    client_loss=[]
    client_acc=[]
    client_recall=[]
    client_auprc=[]
    all_true=[]
    all_pred=[]
    all_prob=[]
    for client in clients:
        loss,acc,y_true,y_pred,prob=evaluate_details(model,client,device)
        client_loss.append(loss)
        client_acc.append(acc)
        if len(set(y_true.tolist()))>1:
            client_auprc.append(float(average_precision_score(y_true,prob)))
        else:
            client_auprc.append(0.0)
        tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
        client_recall.append(float(tp/(tp+fn)) if tp+fn else 0.0)
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_prob.append(prob)
    y_true=np.concatenate(all_true) if all_true else np.array([])
    y_pred=np.concatenate(all_pred) if all_pred else np.array([])
    prob=np.concatenate(all_prob) if all_prob else np.array([])
    clinical={}
    if len(y_true) and set(y_true.tolist())<= {0,1}:
        tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
        try:
            auroc=float(roc_auc_score(y_true,prob))
        except ValueError:
            auroc=0.0
        try:
            auprc=float(average_precision_score(y_true,prob))
        except ValueError:
            auprc=0.0
        clinical={
            "auroc":auroc,
            "auprc":auprc,
            "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
            "sensitivity":float(tp/(tp+fn)) if tp+fn else 0.0,
            "specificity":float(tn/(tn+fp)) if tn+fp else 0.0,
            "worst_client_recall":float(np.min(client_recall)),
            "worst_client_auprc":float(np.min(client_auprc)),
        }
    return {
        "loss":float(np.mean(client_loss)),
        "accuracy":float(np.mean(client_acc)),
        "worst_client_accuracy":float(np.min(client_acc)),
        "client_loss":client_loss,
        "client_accuracy":client_acc,
        **clinical,
    }


@torch.no_grad()
def evaluate(model:nn.Module,client:ClientData,device:torch.device):
    loss,acc,_,_,_=evaluate_details(model,client,device)
    return loss,acc


@torch.no_grad()
def evaluate_details(model:nn.Module,client:ClientData,device:torch.device):
    model.eval()
    x=torch.tensor(client.x_test,dtype=torch.float32,device=device)
    y=torch.tensor(client.y_test,dtype=torch.long,device=device)
    logits=model(x)
    loss=float(nn.CrossEntropyLoss()(logits,y).detach().cpu())
    probs=torch.softmax(logits,dim=1)
    pred=probs.argmax(1)
    acc=float((pred==y).float().mean().detach().cpu())
    prob=probs[:,1].detach().cpu().numpy() if probs.shape[1]>1 else probs[:,0].detach().cpu().numpy()
    return loss,acc,y.detach().cpu().numpy(),pred.detach().cpu().numpy(),prob


@torch.no_grad()
def predict_clients(model:nn.Module,clients:list[ClientData],device:torch.device|None=None):
    device=device or _device()
    model=model.to(device)
    model.eval()
    rows=[]
    for idx,client in enumerate(clients):
        x=torch.tensor(client.x_test,dtype=torch.float32,device=device)
        y=torch.tensor(client.y_test,dtype=torch.long,device=device)
        logits=model(x)
        probs=torch.softmax(logits,dim=1)
        pred=probs.argmax(1)
        conf=probs.max(1).values
        cid=client.client_id if client.client_id is not None else idx
        for i in range(len(y)):
            rows.append({
                "client_id":int(cid),
                "row":int(i),
                "y_true":int(y[i].cpu()),
                "y_pred":int(pred[i].cpu()),
                "confidence":float(conf[i].cpu()),
                **{f"prob_{j}":float(probs[i,j].cpu()) for j in range(probs.shape[1])},
            })
    return rows


def _aggregation_weights(sizes,losses,cfg):
    weights=sizes.astype("float64")/sizes.sum()
    if cfg.cvar_alpha<=0:
        return weights
    tau=np.quantile(losses,cfg.cvar_alpha)
    tail=np.maximum(losses-tau,0)
    if tail.sum()==0:
        return weights
    # CVaR weighting keeps FedAvg's sample-size signal while lifting high-loss clients.
    weights=weights*(1+cfg.fairness_strength*tail/tail.sum())
    return weights/weights.sum()


def _drift_stats(base_state,states,selected,weights):
    updates=[]
    norms=[]
    for state in states:
        parts=[(state[k]-base_state[k]).reshape(-1).float() for k in state]
        update=torch.cat(parts)
        updates.append(update)
        norms.append(float(torch.linalg.vector_norm(update)))
    avg=sum(float(weights[i])*updates[i] for i in range(len(updates)))
    avg_norm=float(torch.linalg.vector_norm(avg))
    cos=[]
    dist=[]
    for update in updates:
        denom=float(torch.linalg.vector_norm(update))*max(avg_norm,1e-12)
        cos.append(float(torch.dot(update,avg)/denom) if denom>0 else 0.0)
        dist.append(float(torch.linalg.vector_norm(update-avg)))
    return {
        "drift_client_ids":selected,
        "drift_update_norms":norms,
        "drift_cosine_to_mean":cos,
        "drift_distance_to_mean":dist,
        "drift_mean_update_norm":float(np.mean(norms)) if norms else 0.0,
        "drift_avg_update_norm":avg_norm,
    }

