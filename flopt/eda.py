
from collections import Counter

import numpy as np

from .data import UCIBundle


def eda_tables(bundle:UCIBundle):
    xtr=bundle.x_train_raw
    xte=bundle.x_test_raw
    train_counts=Counter(bundle.y_train.tolist())
    test_counts=Counter(bundle.y_test.tolist())
    class_rows=[]
    for i,name in enumerate(bundle.activity_names):
        class_rows.append({"class_id":i,"activity":name,"train_count":train_counts[i],"test_count":test_counts[i],"total_count":train_counts[i]+test_counts[i]})
    client_rows=[]
    label_rows=[]
    all_subjects=sorted(set(bundle.subject_train)|set(bundle.subject_test))
    for sid in all_subjects:
        tr=bundle.subject_train==sid
        te=bundle.subject_test==sid
        labels=np.concatenate([bundle.y_train[tr],bundle.y_test[te]])
        counts=np.bincount(labels,minlength=len(bundle.activity_names))
        total=int(counts.sum())
        probs=counts/total if total else counts
        entropy=float(-(probs[probs>0]*np.log2(probs[probs>0])).sum()) if total else 0.0
        client_rows.append({"client_id":int(sid),"train_samples":int(tr.sum()),"test_samples":int(te.sum()),"total_samples":total,"label_entropy":entropy,"dominant_class_ratio":float(probs.max()) if total else 0.0})
        for i,name in enumerate(bundle.activity_names):
            label_rows.append({"client_id":int(sid),"activity":name,"count":int(counts[i]),"proportion":float(probs[i]) if total else 0.0})
    summary=[{
        "train_rows":int(xtr.shape[0]),
        "test_rows":int(xte.shape[0]),
        "features":int(xtr.shape[1]),
        "clients":len(all_subjects),
        "classes":len(bundle.activity_names),
        "train_missing":int(np.isnan(xtr).sum()),
        "test_missing":int(np.isnan(xte).sum()),
        "train_duplicates":int(xtr.shape[0]-np.unique(xtr,axis=0).shape[0]),
        "feature_min":float(xtr.min()),
        "feature_max":float(xtr.max()),
        "feature_mean":float(xtr.mean()),
        "feature_std":float(xtr.std()),
    }]
    return {
        "dataset_summary":summary,
        "class_distribution":class_rows,
        "client_sample_counts":client_rows,
        "client_label_distribution":label_rows,
    }


def noniid_stats(bundle:UCIBundle):
    rows=[]
    global_counts=np.bincount(np.concatenate([bundle.y_train,bundle.y_test]),minlength=len(bundle.activity_names)).astype(float)
    global_p=(global_counts+1e-9)/(global_counts.sum()+1e-9*len(global_counts))
    labels_by_subject={}
    for sid in sorted(set(bundle.subject_train)|set(bundle.subject_test)):
        labels=np.concatenate([bundle.y_train[bundle.subject_train==sid],bundle.y_test[bundle.subject_test==sid]])
        labels_by_subject[int(sid)]=labels
    for sid,labels in labels_by_subject.items():
        counts=np.bincount(labels,minlength=len(bundle.activity_names)).astype(float)
        p=(counts+1e-9)/(counts.sum()+1e-9*len(counts))
        m=0.5*(p+global_p)
        kl=float((p*np.log(p/global_p)).sum())
        js=float(0.5*(p*np.log(p/m)).sum()+0.5*(global_p*np.log(global_p/m)).sum())
        entropy=float(-(p*np.log2(p)).sum())
        rows.append({"client_id":sid,"kl_divergence":kl,"js_divergence":js,"label_entropy":entropy,"dominant_class_ratio":float(p.max()),"samples":int(counts.sum())})
    return rows
