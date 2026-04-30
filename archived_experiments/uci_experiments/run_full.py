from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from flopt.config import FLConfig
from flopt.data import load_clients
from flopt.duality import solve_policy_lp
from flopt.fedavg import federated_train
from flopt.models import HARMLP
from flopt.plots import plot_convergence,plot_cvar,plot_search_comparison,plot_shadow_price
from flopt.search import ga_search,grid_search


ALPHAS=[0,0.5,0.75,0.9,0.95]
GRID=[(1,5,0.01),(1,10,0.03),(2,10,0.03),(4,10,0.02),(2,15,0.01),(4,15,0.02)]


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="uci",choices=["uci"])
    parser.add_argument("--rounds",type=int,default=30)
    parser.add_argument("--seeds",default="7,11,19")
    parser.add_argument("--out",default="outputs/full_uci")
    parser.add_argument("--skip-ga",action="store_true")
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    clients=load_clients(source=args.source,seed=seeds[0])
    base=FLConfig(rounds=args.rounds,local_epochs=2,clients_per_round=min(10,len(clients)),lr=0.03)

    fedavg_runs=[]
    for seed in seeds:
        cfg=replace(base,seed=seed)
        _,records=federated_train(HARMLP(),clients,cfg)
        fedavg_runs.append({"seed":seed,"records":records})
        _write_json(out/f"fedavg_seed_{seed}.json",records)
    _write_csv(out/"fedavg_summary.csv",_summary_rows(fedavg_runs))
    plot_convergence(fedavg_runs[0]["records"],str(out/"convergence_seed_first.png"))
    _plot_mean_convergence(fedavg_runs,out/"convergence_mean.png")

    cvar_rows=[]
    cvar_raw={}
    for alpha in ALPHAS:
        seed_rows=[]
        for seed in seeds:
            cfg=replace(base,seed=seed,cvar_alpha=alpha)
            _,records=federated_train(HARMLP(),clients,cfg)
            last=records[-1]
            row={"alpha":alpha,"seed":seed,"loss":last["loss"],"accuracy":last["accuracy"],"worst_client_accuracy":last["worst_client_accuracy"]}
            seed_rows.append(row)
            cvar_raw[f"alpha_{alpha}_seed_{seed}"]=records
        cvar_rows.append(_mean_row(seed_rows,"alpha",alpha))
    _write_json(out/"cvar_raw.json",cvar_raw)
    _write_csv(out/"cvar_summary.csv",cvar_rows)
    plot_cvar(cvar_rows,str(out/"cvar_pareto.png"))

    grid_cfg=replace(base,rounds=max(8,args.rounds//3),seed=seeds[0])
    grid_rows=grid_search(clients,grid_cfg,GRID)
    _write_csv(out/"grid_search.csv",grid_rows)
    losses=[r["loss"] for r in grid_rows]
    costs=[r["comm"] for r in grid_rows]
    budgets=np.linspace(min(costs),max(costs),10).tolist()
    lp_rows=solve_policy_lp(losses,costs,budgets)
    _write_json(out/"lp_shadow_price.json",lp_rows)
    _write_csv(out/"lp_shadow_price.csv",_flatten_lp(lp_rows))
    plot_shadow_price(lp_rows,str(out/"shadow_price.png"))

    ga_result=None
    if not args.skip_ga:
        ga_result=ga_search(clients,replace(base,rounds=max(5,args.rounds//5),seed=seeds[0]),maxiter=2,popsize=4)
        _write_json(out/"ga_result.json",ga_result)
        plot_search_comparison(grid_rows,ga_result,str(out/"ga_vs_grid.png"))

    _write_report(out,args.source,args.rounds,seeds,len(clients),fedavg_runs,cvar_rows,grid_rows,lp_rows,ga_result)
    print(f"wrote full experiment artifacts to {out}")


def _summary_rows(runs:list[dict])->list[dict]:
    rows=[]
    for run in runs:
        last=run["records"][-1]
        rows.append({"seed":run["seed"],"loss":last["loss"],"accuracy":last["accuracy"],"worst_client_accuracy":last["worst_client_accuracy"],"total_comm":sum(r["upload_bytes"]+r["download_bytes"] for r in run["records"])})
    rows.append(_mean_row(rows,"seed","mean"))
    return rows


def _mean_row(rows:list[dict],key:str,value)->dict:
    metrics=[k for k in rows[0] if k not in {key,"seed"}]
    out={key:value}
    for metric in metrics:
        vals=np.array([float(r[metric]) for r in rows],dtype="float64")
        out[metric]=float(vals.mean())
        out[f"{metric}_std"]=float(vals.std())
    return out


def _flatten_lp(rows:list[dict])->list[dict]:
    flat=[]
    for row in rows:
        kkt=row.get("kkt",{})
        flat.append({"budget":row.get("budget"),"loss":row.get("loss"),"cost":row.get("cost"),"lambda":row.get("lambda"),"status":row.get("status"),**kkt})
    return flat


def _plot_mean_convergence(runs:list[dict],path:Path)->None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True,exist_ok=True)
    losses=np.array([[r["loss"] for r in run["records"]] for run in runs],dtype="float64")
    rounds=np.array([r["round"] for r in runs[0]["records"]])
    mean=losses.mean(axis=0)
    std=losses.std(axis=0)
    ref=mean[0]/np.sqrt(np.maximum(rounds,1))
    plt.figure(figsize=(7,4))
    plt.plot(rounds,mean,label="Mean FedAvg loss")
    plt.fill_between(rounds,mean-std,mean+std,alpha=0.2,label="Seed std")
    plt.plot(rounds,ref,linestyle="--",label="O(1/sqrt(T)) reference")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _write_report(out:Path,source:str,rounds:int,seeds:list[int],client_count:int,fedavg_runs:list[dict],cvar_rows:list[dict],grid_rows:list[dict],lp_rows:list[dict],ga_result:dict|None)->None:
    fedavg_mean=_summary_rows(fedavg_runs)[-1]
    best_grid=grid_rows[0]
    active_lp=[r for r in lp_rows if r.get("lambda",0)>1e-8]
    best_cvar=max(cvar_rows,key=lambda r:r["worst_client_accuracy"])
    text=f"""# Federated Learning Optimization Report Draft

## Setup

Dataset source: `{source}`  
Clients: `{client_count}`  
Rounds: `{rounds}`  
Seeds: `{seeds}`  
Model: `HARMLP`, a 561-feature activity classifier with two hidden layers.

## What Was Trained

We trained one global federated model with FedAvg. Each round selected clients, trained local copies on client data, and aggregated local weights into the next global model.

## Main Results

- Mean FedAvg accuracy: `{fedavg_mean['accuracy']:.4f}` with worst-client accuracy `{fedavg_mean['worst_client_accuracy']:.4f}`.
- Best CVaR setting by worst-client accuracy: `alpha={best_cvar['alpha']}`, worst-client accuracy `{best_cvar['worst_client_accuracy']:.4f}`.
- Best grid policy: `E={best_grid['local_epochs']}`, `K={best_grid['clients_per_round']}`, `lr={best_grid['lr']}`, loss `{best_grid['loss']:.4f}`.
- LP budgets with positive communication shadow price: `{len(active_lp)}` out of `{len(lp_rows)}`.
"""
    if ga_result is not None:
        text+=f"- Differential evolution best fitness: `{ga_result['fitness']:.4f}` after `{ga_result['evaluations']}` evaluations.\n"
    text+="""
## Figures

- `convergence_mean.png`: mean convergence across seeds with an `O(1/sqrt(T))` reference.
- `cvar_pareto.png`: average-client accuracy vs worst-client accuracy.
- `shadow_price.png`: LP dual variable `lambda*` vs communication budget.

## Interpretation

The convergence plot supports the FedAvg training story. The CVaR plot shows whether emphasizing high-loss clients improves the worst-client outcome. The shadow-price plot translates communication limits into an optimization quantity: when `lambda*` is positive, bandwidth is binding and extra communication has measurable value.
"""
    (out/"report_draft.md").write_text(text,encoding="utf-8")


def _write_json(path:Path,obj)->None:
    path.write_text(json.dumps(obj,indent=2),encoding="utf-8")


def _write_csv(path:Path,rows:list[dict])->None:
    keys=sorted({k for row in rows for k in row})
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()

