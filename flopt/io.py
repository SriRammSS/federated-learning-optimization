from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_dirs(root:Path)->None:
    for name in [
        "eda","raw","metrics","baselines","cvar","stats","ablations","drift","noniid",
        "calibration","case_studies","errors","efficiency","sensitivity","runtime",
        "failure_modes","search","lp","artifacts","reports",
    ]:
        (root/name).mkdir(parents=True,exist_ok=True)
    for name in [
        "eda","training","fairness","classification","optimization","search","baselines",
        "statistics","ablations","drift","noniid","calibration","case_studies","errors",
        "efficiency","sensitivity","runtime","failure_modes",
    ]:
        (root/"plots"/name).mkdir(parents=True,exist_ok=True)


def write_json(path:Path,obj)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")


def write_csv(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_round_records(records:list[dict],run_type:str,seed:int,alpha:float|None=None)->list[dict]:
    rows=[]
    for r in records:
        rows.append({
            "run_type":run_type,
            "seed":seed,
            "alpha":alpha,
            "round":r["round"],
            "loss":r["loss"],
            "accuracy":r["accuracy"],
            "worst_client_accuracy":r["worst_client_accuracy"],
            "upload_bytes":r["upload_bytes"],
            "download_bytes":r["download_bytes"],
            "total_comm_bytes":r["upload_bytes"]+r["download_bytes"],
            "selected_clients":" ".join(map(str,r["selected_clients"])),
            "best_loss_so_far":r.get("best_loss_so_far"),
            "best_round":r.get("best_round"),
            "rounds_since_improvement":r.get("rounds_since_improvement"),
            "stopped_early":r.get("stopped_early",False),
        })
    return rows


def convergence_summary(records:list[dict],run_type:str,seed:int,alpha:float|None,max_rounds:int)->dict:
    last=records[-1]
    best=min(records,key=lambda r:r["loss"])
    total_comm=sum(r["upload_bytes"]+r["download_bytes"] for r in records)
    return {
        "run_type":run_type,
        "seed":seed,
        "alpha":alpha,
        "max_rounds":max_rounds,
        "stopped_round":last["round"],
        "stopped_early":last.get("stopped_early",False),
        "best_round":best["round"],
        "best_loss":best["loss"],
        "final_loss":last["loss"],
        "final_accuracy":last["accuracy"],
        "final_worst_client_accuracy":last["worst_client_accuracy"],
        "total_comm_until_stop":total_comm,
    }
