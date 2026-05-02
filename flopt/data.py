from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


UCI_URL="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


@dataclass
class ClientData:
    x_train:np.ndarray
    y_train:np.ndarray
    x_test:np.ndarray
    y_test:np.ndarray
    client_id:int|None=None


def load_clients(source:str="uci",data_dir:str="data",seed:int=7)->list[ClientData]:
    if source=="uci":
        return load_uci_har(Path(data_dir),seed).clients
    if source=="mimic":
        from .mimic import load_mimic
        return load_mimic(Path(data_dir),seed).clients
    raise ValueError(f"unknown source: {source}")


@dataclass
class UCIBundle:
    clients:list[ClientData]
    scaler:StandardScaler
    activity_names:list[str]
    root:Path
    x_train_raw:np.ndarray
    y_train:np.ndarray
    subject_train:np.ndarray
    x_test_raw:np.ndarray
    y_test:np.ndarray
    subject_test:np.ndarray


def load_uci_har(data_dir:Path,seed:int=7)->UCIBundle:
    root=_ensure_uci_har(data_dir)
    x_train_raw=np.loadtxt(root/"train"/"X_train.txt",dtype="float32")
    y_train=np.loadtxt(root/"train"/"y_train.txt",dtype="int64")-1
    s_train=np.loadtxt(root/"train"/"subject_train.txt",dtype="int64")
    x_test_raw=np.loadtxt(root/"test"/"X_test.txt",dtype="float32")
    y_test=np.loadtxt(root/"test"/"y_test.txt",dtype="int64")-1
    s_test=np.loadtxt(root/"test"/"subject_test.txt",dtype="int64")
    labels=np.loadtxt(root/"activity_labels.txt",dtype=str)
    activity_names=[name for _,name in sorted((int(idx),name) for idx,name in labels)]
    scaler=StandardScaler().fit(x_train_raw)
    x_train=scaler.transform(x_train_raw).astype("float32")
    x_test=scaler.transform(x_test_raw).astype("float32")
    clients=[]
    for subject in sorted(set(s_train)|set(s_test)):
        tr=s_train==subject
        te=s_test==subject
        if tr.sum()==0 or te.sum()==0:
            x=np.concatenate([x_train[tr],x_test[te]])
            y=np.concatenate([y_train[tr],y_test[te]])
            xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.25,random_state=seed+int(subject))
        else:
            xtr,xte,ytr,yte=x_train[tr],x_test[te],y_train[tr],y_test[te]
        clients.append(ClientData(xtr,ytr,xte,yte,int(subject)))
    return UCIBundle(clients,scaler,activity_names,root,x_train_raw,y_train,s_train,x_test_raw,y_test,s_test)


def _ensure_uci_har(data_dir:Path)->Path:
    root=data_dir/"UCI HAR Dataset"
    if root.exists():
        return root
    data_dir.mkdir(parents=True,exist_ok=True)
    zip_path=data_dir/"uci_har.zip"
    if not zip_path.exists():
        urlretrieve(UCI_URL,zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    nested=data_dir/"UCI HAR Dataset.zip"
    if nested.exists() and not root.exists():
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(data_dir)
    return root

