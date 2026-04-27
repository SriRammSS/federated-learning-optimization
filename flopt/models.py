import torch
from torch import nn


class HARMLP(nn.Module):
    def __init__(self,features:int=561,classes:int=6):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(features,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,classes),
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x)


class LogisticModel(nn.Module):
    def __init__(self,features:int=561,classes:int=6):
        super().__init__()
        self.linear=nn.Linear(features,classes)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.linear(x)


def count_parameters(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters())


def clone_model(model:nn.Module)->nn.Module:
    clone=type(model)()
    clone.load_state_dict(model.state_dict())
    return clone

