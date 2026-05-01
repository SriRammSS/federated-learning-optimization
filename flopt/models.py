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


class TabularMLP(nn.Module):
    def __init__(self,features:int,classes:int=2,hidden:tuple[int,...]=(128,64),dropout:float=0.0):
        super().__init__()
        layers=[]
        in_features=features
        for width in hidden:
            layers.append(nn.Linear(in_features,width))
            layers.append(nn.ReLU())
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            in_features=width
        layers.append(nn.Linear(in_features,classes))
        self.net=nn.Sequential(*layers)

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


