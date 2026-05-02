from __future__ import annotations

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
from sklearn.decomposition import PCA


def plot_convergence(records:list[dict],path:str)->None:
    rounds=[r["round"] for r in records]
    loss=[r["loss"] for r in records]
    ref=loss[0]/np.sqrt(np.maximum(rounds,1))
    _prep(path)
    plt.figure(figsize=(7,4))
    plt.plot(rounds,loss,label="FedAvg loss")
    plt.plot(rounds,ref,label="O(1/sqrt(T)) reference",linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()



def plot_shadow_price(rows:list[dict],path:str)->None:
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


def bar(rows:list[dict],x:str,y:str,path:str,title:str="",rotation:int=45)->None:
    _prep(path)
    plt.figure(figsize=(8,4))
    plt.bar([str(r[x]) for r in rows],[float(r[y]) for r in rows])
    plt.title(title)
    plt.xticks(rotation=rotation,ha="right")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def grouped_bar(rows:list[dict],x:str,ys:list[str],path:str,title:str="")->None:
    _prep(path)
    labels=[str(r[x]) for r in rows]
    pos=np.arange(len(labels))
    width=0.8/max(1,len(ys))
    plt.figure(figsize=(9,4))
    for i,y in enumerate(ys):
        plt.bar(pos+i*width,[float(r.get(y,0) or 0) for r in rows],width,label=y)
    plt.xticks(pos+width*(len(ys)-1)/2,labels,rotation=45,ha="right")
    plt.title(title)
    plt.legend()
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


def heatmap(matrix:np.ndarray,xlabels:list[str],ylabels:list[str],path:str,title:str="",xlabel:str="",ylabel:str="")->None:
    _prep(path)
    plt.figure(figsize=(9,6))
    plt.imshow(matrix,aspect="auto",cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(xlabels)),xlabels,rotation=45,ha="right")
    step=max(1,len(ylabels)//20)
    plt.yticks(range(0,len(ylabels),step),[ylabels[i] for i in range(0,len(ylabels),step)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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


def scatter3(rows:list[dict],x:str,y:str,z:str,path:str,title:str="",color:str|None=None)->None:
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


def pca_plots(x:np.ndarray,labels:list,activity_names:list[str],path2d:str,path3d:str,title_prefix:str)->None:
    pca=PCA(n_components=3,random_state=0)
    pts=pca.fit_transform(x)
    _prep(path2d)
    plt.figure(figsize=(7,5))
    uniq=sorted(set(labels))
    for lab in uniq:
        mask=np.array(labels)==lab
        name=activity_names[int(lab)] if isinstance(lab,(int,np.integer)) and int(lab)<len(activity_names) else str(lab)
        plt.scatter(pts[mask,0],pts[mask,1],s=8,label=name,alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{title_prefix} 2D PCA")
    if len(uniq)<=12:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path2d,dpi=160)
    plt.close()
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    for lab in uniq:
        mask=np.array(labels)==lab
        name=activity_names[int(lab)] if isinstance(lab,(int,np.integer)) and int(lab)<len(activity_names) else str(lab)
        ax.scatter(pts[mask,0],pts[mask,1],pts[mask,2],s=8,label=name,alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"{title_prefix} 3D PCA")
    if len(uniq)<=12:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path3d,dpi=160)
    plt.close()

