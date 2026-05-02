from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .config import FLConfig


def _set_seed(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device()->torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loss_fn(cfg:FLConfig,device:torch.device)->nn.Module:
    weights=None
    if cfg.class_weights:
        weights=torch.tensor(cfg.class_weights,dtype=torch.float32,device=device)
    return nn.CrossEntropyLoss(weight=weights)


def _optimizer(model:nn.Module,cfg:FLConfig):
    if cfg.optimizer=="adam":
        return torch.optim.Adam(model.parameters(),lr=cfg.lr)
    return torch.optim.SGD(model.parameters(),lr=cfg.lr)


def _load_weighted_state(model:nn.Module,states:list[dict],weights:np.ndarray,device:torch.device)->None:
    avg={}
    for key in states[0]:
        avg[key]=sum(weights[i]*states[i][key] for i in range(len(states))).to(device)
    model.load_state_dict(avg)
