
import os
from pathlib import Path

_mpl_cache=Path("outputs")/"mpl-cache"
_mpl_cache.mkdir(parents=True,exist_ok=True)
_xdg_cache=Path("outputs")/"cache"
_xdg_cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME",str(_xdg_cache))

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import numpy as np

def plot_shadow_price(rows:list[dict],path:str):
    clean=[r for r in rows if r.get("status") in {"optimal","optimal_inaccurate"}]
    _prep(path)
    plt.figure(figsize=(6,4))
    plt.plot([r["budget"] for r in clean],[r["lambda"] for r in clean],marker="o")
    plt.xlabel("Communication budget")
    plt.ylabel("lambda*")
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _prep(path:str)->None:
    Path(path).parent.mkdir(parents=True,exist_ok=True)


def bar(rows:list[dict],x:str,y:str,path:str,title:str="",rotation:int=45):
    _prep(path)
    plt.figure(figsize=(8,4))
    plt.bar([str(r[x]) for r in rows],[float(r[y]) for r in rows])
    plt.title(title)
    plt.xticks(rotation=rotation,ha="right")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def line_mean_std(round_rows:list[dict],metric:str,path:str,title:str="")->None:
    _prep(path)
    rounds=sorted({int(r["round"]) for r in round_rows})
    mean=[]
    std=[]
    for rnd in rounds:
        vals=[float(r[metric]) for r in round_rows if int(r["round"])==rnd and r.get(metric) not in {None,""}]
        mean.append(np.mean(vals))
        std.append(np.std(vals))
    mean=np.array(mean)
    std=np.array(std)
    plt.figure(figsize=(8,4))
    plt.plot(rounds,mean,label=metric)
    plt.fill_between(rounds,mean-std,mean+std,alpha=0.2,label="std")
    plt.title(title)
    plt.xlabel("round")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def scatter(rows:list[dict],x:str,y:str,path:str,title:str="",color:str|None=None)->None:
    _prep(path)
    plt.figure(figsize=(6,4))
    c=[float(r[color]) for r in rows] if color else None
    sc=plt.scatter([float(r[x]) for r in rows],[float(r[y]) for r in rows],c=c,cmap="viridis" if color else None)
    if color:
        plt.colorbar(sc,label=color)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def scatter3(rows:list[dict],x:str,y:str,z:str,path:str,title:str="",color:str|None=None):
    _prep(path)
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111,projection="3d")
    c=[float(r[color]) for r in rows] if color else None
    sc=ax.scatter([float(r[x]) for r in rows],[float(r[y]) for r in rows],[float(r[z]) for r in rows],c=c,cmap="viridis" if color else None)
    if color:
        fig.colorbar(sc,ax=ax,label=color,shrink=0.7)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


