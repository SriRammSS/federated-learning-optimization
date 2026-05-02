from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from .aggregators import aggregation_weights
from .config import FLConfig
from .data import ClientData
from .models import count_parameters


def federated_train(model:nn.Module,clients:list[ClientData],cfg:FLConfig,track_drift:bool=False)->tuple[nn.Module,list[dict]]:
    _set_seed(cfg.seed)
    device=_device()
    global_model=deepcopy(model).to(device)
    records:list[dict]=[]
    client_ids=list(range(len(clients)))
    max_rounds=cfg.max_rounds or cfg.rounds

    best_value=float("inf")
    best_round=0
    stale_rounds=0
    best_state=None

    for round_id in range(1,max_rounds + 1):
        selected=random.sample(client_ids,min(cfg.clients_per_round,len(client_ids)))
        local_states=[]
        local_sizes=[]
        local_losses=[]
        base_state={key:value.detach().cpu().clone() for key,value in global_model.state_dict().items()}

        for client_id in selected:
            local_model=deepcopy(global_model)
            loss=train_one_client(local_model,clients[client_id],cfg,device)
            local_states.append({key:value.detach().cpu() for key,value in local_model.state_dict().items()})
            local_sizes.append(len(clients[client_id].x_train))
            local_losses.append(loss)

        weights=aggregation_weights(np.array(local_sizes),np.array(local_losses),cfg)
        drift=_drift_stats(base_state,local_states,selected,weights) if track_drift else {}
        _load_weighted_state(global_model,local_states,weights,device)

        metrics=evaluate_all(global_model,clients,device)
        current=float(metrics[cfg.monitor])
        improved=current < best_value - cfg.min_delta
        if improved:
            best_value=current
            best_round=round_id
            stale_rounds=0
            best_state={key:value.detach().cpu().clone() for key,value in global_model.state_dict().items()}
        else:
            stale_rounds+=1

        stopped=bool(cfg.early_stopping and stale_rounds >= cfg.patience)
        metrics.update(
            {
                "round":round_id,
                "selected_clients":selected,
                "upload_bytes":_round_transfer_bytes(global_model,len(selected)),
                "download_bytes":_round_transfer_bytes(global_model,len(selected)),
                "config":asdict(cfg),
                "best_loss_so_far":best_value,
                "best_round":best_round,
                "rounds_since_improvement":stale_rounds,
                "stopped_early":stopped,
                **drift,
            }
        )
        records.append(metrics)
        if stopped:
            break

    if best_state is not None:
        global_model.load_state_dict({key:value.to(device) for key,value in best_state.items()})
    return global_model,records


def train_one_client(model:nn.Module,client:ClientData,cfg:FLConfig,device:torch.device)->float:
    model.train()
    loader=_client_loader(client,cfg)
    optimizer=_optimizer(model,cfg)
    loss_fn=_loss_fn(cfg,device)
    last_loss=0.0

    for _ in range(cfg.local_epochs):
        for xb,yb in loader:
            xb=xb.to(device)
            yb=yb.to(device)
            optimizer.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            optimizer.step()
            last_loss=float(loss.detach().cpu())

    return last_loss


@torch.no_grad()
def evaluate_all(model:nn.Module,clients:list[ClientData],device:torch.device)->dict:
    client_loss=[]
    client_accuracy=[]
    client_recall=[]
    client_auprc=[]
    all_true=[]
    all_pred=[]
    all_prob=[]

    for client in clients:
        loss,accuracy,y_true,y_pred,probability=evaluate_details(model,client,device)
        client_loss.append(loss)
        client_accuracy.append(accuracy)
        client_auprc.append(_binary_auprc(y_true,probability))
        _,_,fn,tp=_binary_confusion_terms(y_true,y_pred)
        client_recall.append(float(tp / (tp + fn)) if tp + fn else 0.0)
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_prob.append(probability)

    y_true=np.concatenate(all_true) if all_true else np.array([])
    y_pred=np.concatenate(all_pred) if all_pred else np.array([])
    probability=np.concatenate(all_prob) if all_prob else np.array([])

    clinical={}
    if len(y_true) and set(y_true.tolist()) <= {0,1}:
        tn,fp,fn,tp=_binary_confusion_terms(y_true,y_pred)
        clinical={
            "auroc":_safe_metric(lambda:roc_auc_score(y_true,probability)),
            "auprc":_safe_metric(lambda:average_precision_score(y_true,probability)),
            "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
            "sensitivity":float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity":float(tn / (tn + fp)) if tn + fp else 0.0,
            "worst_client_recall":float(np.min(client_recall)),
            "worst_client_auprc":float(np.min(client_auprc)),
        }

    return {
        "loss":float(np.mean(client_loss)),
        "accuracy":float(np.mean(client_accuracy)),
        "worst_client_accuracy":float(np.min(client_accuracy)),
        "client_loss":client_loss,
        "client_accuracy":client_accuracy,
        **clinical,
    }


@torch.no_grad()
def evaluate_details(model:nn.Module,client:ClientData,device:torch.device)->tuple[float,float,np.ndarray,np.ndarray,np.ndarray]:
    model.eval()
    x=torch.tensor(client.x_test,dtype=torch.float32,device=device)
    y=torch.tensor(client.y_test,dtype=torch.long,device=device)
    logits=model(x)
    loss=float(nn.CrossEntropyLoss()(logits,y).detach().cpu())
    probabilities=torch.softmax(logits,dim=1)
    predictions=probabilities.argmax(1)
    accuracy=float((predictions == y).float().mean().detach().cpu())
    positive_probability=probabilities[:,1].detach().cpu().numpy() if probabilities.shape[1] > 1 else probabilities[:,0].detach().cpu().numpy()
    return (
        loss,
        accuracy,
        y.detach().cpu().numpy(),
        predictions.detach().cpu().numpy(),
        positive_probability,
    )


@torch.no_grad()
def predict_clients(
    model:nn.Module,
    clients:list[ClientData],
    device:torch.device | None=None,
) -> list[dict]:
    device=device or _device()
    model=model.to(device)
    model.eval()
    rows=[]

    for index,client in enumerate(clients):
        x=torch.tensor(client.x_test,dtype=torch.float32,device=device)
        y=torch.tensor(client.y_test,dtype=torch.long,device=device)
        logits=model(x)
        probabilities=torch.softmax(logits,dim=1)
        predictions=probabilities.argmax(1)
        confidence=probabilities.max(1).values
        client_id=client.client_id if client.client_id is not None else index

        for row_index in range(len(y)):
            rows.append(
                {
                    "client_id":int(client_id),
                    "row":int(row_index),
                    "y_true":int(y[row_index].cpu()),
                    "y_pred":int(predictions[row_index].cpu()),
                    "confidence":float(confidence[row_index].cpu()),
                    **{
                        f"prob_{class_index}":float(
                            probabilities[row_index,class_index].cpu()
                        )
                        for class_index in range(probabilities.shape[1])
                    },
                }
            )

    return rows


def _client_loader(client:ClientData,cfg:FLConfig) -> DataLoader:
    x=torch.tensor(client.x_train,dtype=torch.float32)
    y=torch.tensor(client.y_train,dtype=torch.long)
    dataset=TensorDataset(x,y)
    return DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True)


def _loss_fn(cfg:FLConfig,device:torch.device) -> nn.Module:
    weights=None
    if cfg.class_weights:
        weights=torch.tensor(cfg.class_weights,dtype=torch.float32,device=device)
    return nn.CrossEntropyLoss(weight=weights)


def _optimizer(model:nn.Module,cfg:FLConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(),lr=cfg.lr)
    return torch.optim.SGD(model.parameters(),lr=cfg.lr)


def _safe_metric(fn) -> float:
    try:
        return float(fn())
    except ValueError:
        return 0.0


def _load_weighted_state(
    model:nn.Module,
    states:list[dict],
    weights:np.ndarray,
    device:torch.device,
) -> None:
    averaged_state={}
    for key in states[0]:
        averaged_state[key]=sum(
            weights[index] * states[index][key]
            for index in range(len(states))
        ).to(device)
    model.load_state_dict(averaged_state)


def _round_transfer_bytes(model:nn.Module,selected_clients:int) -> int:
    return count_parameters(model) * 4 * selected_clients


def _binary_auprc(y_true:np.ndarray,probability:np.ndarray) -> float:
    if len(set(y_true.tolist())) <= 1:
        return 0.0
    return float(average_precision_score(y_true,probability))


def _binary_confusion_terms(
    y_true:np.ndarray,
    y_pred:np.ndarray,
) -> tuple[int,int,int,int]:
    return tuple(
        int(value)
        for value in confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    )


def _set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _drift_stats(
    base_state:dict,
    states:list[dict],
    selected:list[int],
    weights:np.ndarray,
) -> dict:
    updates=[]
    norms=[]

    for state in states:
        parts=[(state[key] - base_state[key]).reshape(-1).float() for key in state]
        update=torch.cat(parts)
        updates.append(update)
        norms.append(float(torch.linalg.vector_norm(update)))

    average_update=sum(
        float(weights[index]) * updates[index]
        for index in range(len(updates))
    )
    average_norm=float(torch.linalg.vector_norm(average_update))
    cosine_to_mean=[]
    distance_to_mean=[]

    for update in updates:
        denominator=float(torch.linalg.vector_norm(update)) * max(average_norm,1e-12)
        cosine_to_mean.append(
            float(torch.dot(update,average_update) / denominator)
            if denominator > 0
            else 0.0
        )
        distance_to_mean.append(float(torch.linalg.vector_norm(update - average_update)))

    return {
        "drift_client_ids":selected,
        "drift_update_norms":norms,
        "drift_cosine_to_mean":cosine_to_mean,
        "drift_distance_to_mean":distance_to_mean,
        "drift_mean_update_norm":float(np.mean(norms)) if norms else 0.0,
        "drift_avg_update_norm":average_norm,
    }
