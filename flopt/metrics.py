from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support

from .data import ClientData


def prediction_arrays(pred_rows:list[dict])->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    y_true=np.array([r["y_true"] for r in pred_rows],dtype=int)
    y_pred=np.array([r["y_pred"] for r in pred_rows],dtype=int)
    conf=np.array([r["confidence"] for r in pred_rows],dtype=float)
    client=np.array([r["client_id"] for r in pred_rows],dtype=int)
    return y_true,y_pred,conf,client


def classification_rows(pred_rows:list[dict],activity_names:list[str])->list[dict]:
    y_true,y_pred,_,_=prediction_arrays(pred_rows)
    report=classification_report(y_true,y_pred,target_names=activity_names,output_dict=True,zero_division=0)
    rows=[]
    for label,vals in report.items():
        if isinstance(vals,dict):
            rows.append({"label":label,**{k:float(v) for k,v in vals.items()}})
        else:
            rows.append({"label":label,"score":float(vals)})
    return rows


def confusion_rows(pred_rows:list[dict],activity_names:list[str],normalize:bool=False)->list[dict]:
    y_true,y_pred,_,_=prediction_arrays(pred_rows)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(activity_names))),normalize="true" if normalize else None)
    rows=[]
    for i,true_name in enumerate(activity_names):
        row={"true_label":true_name}
        for j,pred_name in enumerate(activity_names):
            row[pred_name]=float(cm[i,j])
        rows.append(row)
    return rows


def per_client_rows(pred_rows:list[dict],clients:list[ClientData])->list[dict]:
    y_true,y_pred,_,client_ids=prediction_arrays(pred_rows)
    rows=[]
    for idx,client in enumerate(clients):
        cid=client.client_id if client.client_id is not None else idx
        mask=client_ids==cid
        if not mask.any():
            continue
        rows.append({
            "client_id":int(cid),
            "test_samples":int(mask.sum()),
            "train_samples":int(len(client.x_train)),
            "accuracy":float((y_true[mask]==y_pred[mask]).mean()),
            "error_rate":float((y_true[mask]!=y_pred[mask]).mean()),
        })
    return rows


def aggregate_scores(pred_rows:list[dict])->dict:
    y_true,y_pred,_,_=prediction_arrays(pred_rows)
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    w_precision,w_recall,w_f1,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    return {
        "accuracy":float((y_true==y_pred).mean()),
        "macro_precision":float(precision),
        "macro_recall":float(recall),
        "macro_f1":float(f1),
        "weighted_precision":float(w_precision),
        "weighted_recall":float(w_recall),
        "weighted_f1":float(w_f1),
    }


def top_confusions(pred_rows:list[dict],activity_names:list[str],limit:int=10)->list[dict]:
    y_true,y_pred,_,_=prediction_arrays(pred_rows)
    rows=[]
    for i,name_i in enumerate(activity_names):
        for j,name_j in enumerate(activity_names):
            if i==j:
                continue
            count=int(((y_true==i)&(y_pred==j)).sum())
            if count:
                rows.append({"true_label":name_i,"pred_label":name_j,"count":count})
    return sorted(rows,key=lambda r:r["count"],reverse=True)[:limit]


def per_class_error_rows(pred_rows:list[dict],activity_names:list[str])->list[dict]:
    y_true,y_pred,_,_=prediction_arrays(pred_rows)
    rows=[]
    for i,name in enumerate(activity_names):
        mask=y_true==i
        rows.append({
            "label":name,
            "support":int(mask.sum()),
            "error_rate":float((y_pred[mask]!=i).mean()) if mask.any() else 0.0,
            "recall":float((y_pred[mask]==i).mean()) if mask.any() else 0.0,
        })
    return rows
