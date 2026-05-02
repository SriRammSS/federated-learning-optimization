
import numpy as np


def communication_efficiency(summary_rows,threshold:float=0.90):
    result=[]
    for row in summary_rows:
        comm=float(row.get("total_comm_until_stop",row.get("total_comm",0)) or 0)
        acc=float(row.get("final_accuracy",row.get("accuracy",0)) or 0)
        worst=float(row.get("final_worst_client_accuracy",row.get("worst_client_accuracy",0)) or 0)
        loss=float(row.get("final_loss",row.get("loss",0)) or 0)
        mb=comm/1_000_000 if comm else 0.0
        result.append({
            **row,
            "comm_mb":mb,
            "accuracy_per_mb":acc/mb if mb else 0.0,
            "worst_accuracy_per_mb":worst/mb if mb else 0.0,
            "loss_per_mb":loss/mb if mb else 0.0,
            "reached_threshold":bool(acc>=threshold),
        })
    return out
def ablation_rows(rows:list[dict],baseline_name:str="fedavg_default")->list[dict]:
    baseline=[r for r in rows if r.get("method")==baseline_name]
    base_by_seed={r["seed"]:r for r in baseline}
    deltas=[]
    for row in rows:
        base=base_by_seed.get(row.get("seed"))
        if not base:
            continue
        deltas.append({
            **row,
            "delta_accuracy":float(row.get("final_accuracy",row.get("accuracy",0)))-float(base.get("final_accuracy",base.get("accuracy",0))),
            "delta_worst_client_accuracy":float(row.get("final_worst_client_accuracy",row.get("worst_client_accuracy",0)))-float(base.get("final_worst_client_accuracy",base.get("worst_client_accuracy",0))),
            "delta_comm":float(row.get("total_comm_until_stop",row.get("total_comm",0)))-float(base.get("total_comm_until_stop",base.get("total_comm",0))),
        })
    return deltas


def failure_modes(rows):
    flags=[]
    for row in rows:
        acc=float(row.get("final_accuracy",row.get("accuracy",0)) or 0)
        worst=float(row.get("final_worst_client_accuracy",row.get("worst_client_accuracy",0)) or 0)
        if worst<0.70:
            flags.append({**row,"failure_type":"low_worst_client_accuracy","description":"Worst-client accuracy remains below 0.70."})
        if acc<0.90:
            flags.append({**row,"failure_type":"low_average_accuracy","description":"Average accuracy remains below 0.90."})
        if row.get("stopped_early") is False and row.get("max_rounds")==row.get("stopped_round"):
            flags.append({**row,"failure_type":"no_early_convergence","description":"Run reached max_rounds without early stopping."})
    return flags


def selected_case_clients(per_client,n:int=5):
    rows=sorted(per_client,key=lambda r:float(r["accuracy"]))
    ids=[int(rows[0]["client_id"]),int(rows[len(rows)//2]["client_id"]),int(rows[-1]["client_id"])]
    return list(dict.fromkeys(ids[:n]))


def summarize_rows(rows,group_key:str,metrics):
    groups={}
    for row in rows:
        groups.setdefault(row[group_key],[]).append(row)
    summary=[]
    for group,items in groups.items():
        r={group_key:group,"n":len(items)}
        for metric in metrics:
            vals=np.array([float(item[metric]) for item in items if item.get(metric) not in {None,""}],dtype=float)
            if len(vals):
                r[f"{metric}_mean"]=float(vals.mean())
                r[f"{metric}_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        summary.append(r)
    return summary
