
import random

import numpy as np
import torch
from torch import nn

from .config import FLConfig


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _loss_fn(cfg,device):
    weights=None
    if cfg.class_weights:
        weights=torch.tensor(cfg.class_weights,dtype=torch.float32,device=device)
    return nn.CrossEntropyLoss(weight=weights)


def _optimizer(model,cfg):
    if cfg.optimizer=='adam':
        return torch.optim.Adam(model.parameters(),lr=cfg.lr)
    return torch.optim.SGD(model.parameters(),lr=cfg.lr)


def _load_weighted_state(model,states,weights,device):
    avg={}
    for key in states[0]:
        avg[key]=sum(weights[i]*states[i][key] for i in range(len(states))).to(device)
    model.load_state_dict(avg)
