
import numpy as np
from sklearn.metrics import average_precision_score,balanced_accuracy_score,brier_score_loss,classification_report as _sklearn_classification_report,confusion_matrix,precision_recall_curve,precision_recall_fscore_support,roc_auc_score,roc_curve as _sklearn_roc_curve

from .data import ClientData


def prediction_arrays(preds:list[dict]):
    y_true=np.array([r["y_true"] for r in preds],dtype=int)
    y_pred=np.array([r["y_pred"] for r in preds],dtype=int)
    conf=np.array([r["confidence"] for r in preds],dtype=float)
    client=np.array([r["client_id"] for r in preds],dtype=int)
    return y_true,y_pred,conf,client


def classification_report(preds:list[dict],activity_names:list[str]):
    y_true,y_pred,_,_=prediction_arrays(preds)
    report=_sklearn_classification_report(y_true,y_pred,target_names=activity_names,output_dict=True,zero_division=0)
    rows=[]
    for label,vals in report.items():
        if isinstance(vals,dict):
            rows.append({"label":label,**{k:float(v) for k,v in vals.items()}})
        else:
            rows.append({"label":label,"score":float(vals)})
    return rows


def confusion_table(preds:list[dict],activity_names:list[str],normalize:bool=False):
    y_true,y_pred,_,_=prediction_arrays(preds)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(activity_names))),normalize="true" if normalize else None)
    rows=[]
    for i,true_name in enumerate(activity_names):
        row={"true_label":true_name}
        for j,pred_name in enumerate(activity_names):
            row[pred_name]=float(cm[i,j])
        rows.append(row)
    return rows


def client_breakdown(preds:list[dict],clients:list[ClientData]):
    y_true,y_pred,_,client_ids=prediction_arrays(preds)
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


def aggregate_scores(preds:list[dict]):
    y_true,y_pred,_,_=prediction_arrays(preds)
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


def binary_clinical_scores(preds:list[dict]):
    y_true,y_pred,conf,_=prediction_arrays(preds)
    prob=_positive_prob(preds)
    tn,fp,fn,tp=_binary_counts(y_true,y_pred)
    try:
        auroc=float(roc_auc_score(y_true,prob))
    except ValueError:
        auroc=0.0
    try:
        auprc=float(average_precision_score(y_true,prob))
    except ValueError:
        auprc=0.0
    return {
        "accuracy":float((y_true==y_pred).mean()),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)) if len(set(y_true.tolist()))>1 else 0.0,
        "auroc":auroc,
        "auprc":auprc,
        "sensitivity":float(tp/(tp+fn)) if tp+fn else 0.0,
        "specificity":float(tn/(tn+fp)) if tn+fp else 0.0,
        "precision_ppv":float(tp/(tp+fp)) if tp+fp else 0.0,
        "npv":float(tn/(tn+fn)) if tn+fn else 0.0,
        "f1_death":float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
        "brier":float(brier_score_loss(y_true,prob)),
        "mean_confidence":float(conf.mean()),
        "positive_rate":float(y_true.mean()),
        "predicted_positive_rate":float(y_pred.mean()),
        "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn),
    }


def client_scores(preds:list[dict],client_names:dict[int,str]|None=None):
    y_true,y_pred,_,client_ids=prediction_arrays(preds)
    prob=_positive_prob(preds)
    rows=[]
    client_names=client_names or {}
    for cid in sorted(set(client_ids.tolist())):
        mask=client_ids==cid
        sub=[preds[i] for i in np.where(mask)[0]]
        scores=binary_clinical_scores(sub)
        rows.append({"client_id":int(cid),"client_name":client_names.get(int(cid),str(cid)),"test_samples":int(mask.sum()),"deaths":int(y_true[mask].sum()),**scores})
    return rows


def roc_curve(preds:list[dict]):
    y_true,_,_,_=prediction_arrays(preds)
    prob=_positive_prob(preds)
    if len(set(y_true.tolist()))<2:
        return []
    fpr,tpr,thr=_sklearn_roc_curve(y_true,prob)
    return [{"fpr":float(fpr[i]),"tpr":float(tpr[i]),"threshold":float(thr[i])} for i in range(len(fpr))]


def pr_curve(preds:list[dict]):
    y_true,_,_,_=prediction_arrays(preds)
    prob=_positive_prob(preds)
    precision,recall,thr=precision_recall_curve(y_true,prob)
    rows=[]
    for i in range(len(precision)):
        rows.append({"precision":float(precision[i]),"recall":float(recall[i]),"threshold":float(thr[i]) if i<len(thr) else ""})
    return rows


def top_confusions(preds:list[dict],activity_names:list[str],limit:int=10):
    y_true,y_pred,_,_=prediction_arrays(preds)
    rows=[]
    for i,name_i in enumerate(activity_names):
        for j,name_j in enumerate(activity_names):
            if i==j:
                continue
            count=int(((y_true==i)&(y_pred==j)).sum())
            if count:
                rows.append({"true_label":name_i,"pred_label":name_j,"count":count})
    return sorted(rows,key=lambda r:r["count"],reverse=True)[:limit]


def per_class_errors(preds:list[dict],activity_names:list[str]):
    y_true,y_pred,_,_=prediction_arrays(preds)
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


def _positive_prob(preds):
    return np.array([r.get("prob_1",r.get("confidence",0.0)) for r in preds],dtype=float)


def _binary_counts(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    tn,fp,fn,tp=cm.ravel()
    return int(tn),int(fp),int(fn),int(tp)
