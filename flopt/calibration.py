
import numpy as np

from .metrics import prediction_arrays


def calibration_bins(preds:list[dict],bins:int=10):
    y_true,y_pred,conf,_=prediction_arrays(preds)
    correct=(y_true==y_pred).astype(float)
    edges=np.linspace(0,1,bins+1)
    rows=[]
    ece=0.0
    mce=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf>=lo)&(conf<=hi if i==bins-1 else conf<hi)
        if mask.any():
            acc=float(correct[mask].mean())
            avg_conf=float(conf[mask].mean())
            gap=abs(acc-avg_conf)
            ece+=float(mask.mean())*gap
            mce=max(mce,gap)
            count=int(mask.sum())
        else:
            acc=avg_conf=gap=0.0
            count=0
        rows.append({"bin":i,"lower":float(lo),"upper":float(hi),"count":count,"accuracy":acc,"confidence":avg_conf,"gap":gap})
    summary={"ece":float(ece),"mce":float(mce),"mean_confidence":float(conf.mean()),"accuracy":float(correct.mean())}
    return rows,summary
