
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .fedavg import _device,_loss_fn
from .io import write_csv,write_json


def save_model_triplet(out_dir:Path,prefix:str,initial:nn.Module,final:nn.Module,best:nn.Module) -> None:
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.save(initial.state_dict(),out_dir / f"{prefix}_initial_model.pt")
    torch.save(final.state_dict(),out_dir / f"{prefix}_final_model.pt")
    torch.save(best.state_dict(),out_dir / f"{prefix}_best_model.pt")


def landscape_1d(
    model_factory,
    initial_state:dict,
    final_state:dict,
    x_val:np.ndarray,
    y_val:np.ndarray,
    cfg,
    out_csv:Path,
    out_png:Path,
    points:int=41,
    prefix:str="model",
) -> list[dict]:
    alphas=np.linspace(-0.5,1.5,points)
    rows=[]
    for alpha in alphas:
        model=model_factory()
        state=_interpolate_state(initial_state,final_state,float(alpha))
        model.load_state_dict(state)
        loss,acc=_eval_model(model,x_val,y_val,cfg)
        rows.append({"model":prefix,"alpha":float(alpha),"loss":loss,"accuracy":acc})
    write_csv(out_csv,rows)
    _plot_1d(rows,out_png,f"{prefix} 1D Loss Landscape")
    return rows


def landscape_2d(
    model_factory,
    final_state:dict,
    x_val:np.ndarray,
    y_val:np.ndarray,
    cfg,
    out_csv:Path,
    out_png:Path,
    grid:int=25,
    radius:float=1.0,
    seed:int=7,
    prefix:str="model",
) -> list[dict]:
    rng=torch.Generator().manual_seed(seed)
    d1=_random_direction(final_state,rng)
    d2=_random_direction(final_state,rng)
    coords=np.linspace(-radius,radius,grid)
    rows=[]
    for a in coords:
        for b in coords:
            model=model_factory()
            state=_plane_state(final_state,d1,d2,float(a),float(b))
            model.load_state_dict(state)
            loss,acc=_eval_model(model,x_val,y_val,cfg)
            rows.append({"model":prefix,"x":float(a),"y":float(b),"loss":loss,"accuracy":acc})
    write_csv(out_csv,rows)
    _plot_2d(rows,out_png,f"{prefix} 2D Loss Landscape")
    return rows


def stratified_validation_subset(clients,max_rows:int=5000,seed:int=7) -> tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    x=np.concatenate([c.x_test for c in clients]).astype("float32")
    y=np.concatenate([c.y_test for c in clients]).astype("int64")
    idxs=[]
    per_class=max(1,max_rows // max(1,len(set(y.tolist()))))
    for label in sorted(set(y.tolist())):
        label_idx=np.where(y == label)[0]
        take=min(per_class,len(label_idx))
        idxs.extend(rng.choice(label_idx,size=take,replace=False).tolist())
    if len(idxs) < min(max_rows,len(y)):
        remaining=np.setdiff1d(np.arange(len(y)),np.array(idxs),assume_unique=False)
        take=min(max_rows - len(idxs),len(remaining))
        if take > 0:
            idxs.extend(rng.choice(remaining,size=take,replace=False).tolist())
    rng.shuffle(idxs)
    idx=np.array(idxs,dtype=np.int64)
    return x[idx],y[idx]


def _eval_model(model:nn.Module,x:np.ndarray,y:np.ndarray,cfg) -> tuple[float,float]:
    device=_device()
    model=deepcopy(model).to(device)
    model.eval()
    loss_fn=_loss_fn(cfg,device)
    with torch.no_grad():
        xb=torch.tensor(x,dtype=torch.float32,device=device)
        yb=torch.tensor(y,dtype=torch.long,device=device)
        logits=model(xb)
        loss=float(loss_fn(logits,yb).detach().cpu())
        pred=logits.argmax(1)
        acc=float((pred == yb).float().mean().detach().cpu())
    return loss,acc


def _interpolate_state(initial:dict,final:dict,alpha:float) -> dict:
    out={}
    for key,val in final.items():
        if torch.is_floating_point(val):
            out[key]=initial[key] + alpha * (final[key] - initial[key])
        else:
            out[key]=val
    return out


def _random_direction(state:dict,rng:torch.Generator) -> dict:
    direction={}
    norm_sq=0.0
    for key,val in state.items():
        if torch.is_floating_point(val):
            d=torch.randn(val.shape,generator=rng,dtype=val.dtype)
            direction[key]=d
            norm_sq+=float(torch.sum(d.float() ** 2))
        else:
            direction[key]=torch.zeros_like(val)
    norm=max(norm_sq ** 0.5,1e-12)
    return {k:(v / norm if torch.is_floating_point(v) else v) for k,v in direction.items()}


def _plane_state(center:dict,d1:dict,d2:dict,a:float,b:float) -> dict:
    out={}
    scale=_state_norm(center)
    for key,val in center.items():
        if torch.is_floating_point(val):
            out[key]=val + scale * (a * d1[key] + b * d2[key])
        else:
            out[key]=val
    return out


def _state_norm(state:dict) -> float:
    total=0.0
    for val in state.values():
        if torch.is_floating_point(val):
            total+=float(torch.sum(val.float() ** 2))
    return max(total ** 0.5,1e-12)


def _plot_1d(rows:list[dict],path:Path,title:str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.plot([r["alpha"] for r in rows],[r["loss"] for r in rows],marker="o")
    plt.xlabel("interpolation alpha")
    plt.ylabel("loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _plot_2d(rows:list[dict],path:Path,title:str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    xs=sorted({r["x"] for r in rows})
    ys=sorted({r["y"] for r in rows})
    mat=np.zeros((len(ys),len(xs)))
    lookup={(r["x"],r["y"]):r["loss"] for r in rows}
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            mat[i,j]=lookup[(x,y)]
    plt.figure(figsize=(6,5))
    plt.imshow(mat,origin="lower",extent=[min(xs),max(xs),min(ys),max(ys)],aspect="auto",cmap="viridis")
    plt.colorbar(label="loss")
    plt.xlabel("direction 1")
    plt.ylabel("direction 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def write_landscape_config(path:Path,config:dict) -> None:
    write_json(path,config)
