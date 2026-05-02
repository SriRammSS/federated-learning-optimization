
import math

import numpy as np
from scipy import stats as scipy_stats


def confidence_intervals(rows:list[dict],group_key:str,metrics:list[str]):
    groups={}
    for row in rows:
        groups.setdefault(row[group_key],[]).append(row)
    result=[]
    for group,items in groups.items():
        for metric in metrics:
            vals=np.array([float(r[metric]) for r in items if r.get(metric) not in {None,""}],dtype=float)
            if len(vals)==0:
                continue
            mean=float(vals.mean())
            std=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
            se=std/math.sqrt(len(vals)) if len(vals)>1 else 0.0
            ci=1.96*se
            result.append({"group":group,"metric":metric,"n":len(vals),"mean":mean,"std":std,"ci95_low":mean-ci,"ci95_high":mean+ci})
    return result


def paired_tests(rows:list[dict],method_key:str,seed_key:str,metrics:list[str],baseline:str):
    methods=sorted({r[method_key] for r in rows})
    tests=[]
    for method in methods:
        if method==baseline:
            continue
        for metric in metrics:
            b={r[seed_key]:float(r[metric]) for r in rows if r[method_key]==baseline and r.get(metric) not in {None,""}}
            m={r[seed_key]:float(r[metric]) for r in rows if r[method_key]==method and r.get(metric) not in {None,""}}
            seeds=sorted(set(b)&set(m))
            if len(seeds)<2:
                continue
            diff=np.array([m[s]-b[s] for s in seeds],dtype=float)
            t_p=float(scipy_stats.ttest_rel([m[s] for s in seeds],[b[s] for s in seeds]).pvalue)
            try:
                w_p=float(scipy_stats.wilcoxon(diff).pvalue)
            except ValueError:
                w_p=1.0
            effect=float(diff.mean()/(diff.std(ddof=1)+1e-12)) if len(diff)>1 else 0.0
            tests.append({"baseline":baseline,"method":method,"metric":metric,"n":len(seeds),"mean_diff":float(diff.mean()),"paired_t_p":t_p,"wilcoxon_p":w_p,"effect_size":effect})
    return tests


def correlations(left:list[dict],right:list[dict],key:str,left_metrics:list[str],right_metrics:list[str]):
    rmap={r[key]:r for r in right}
    pairs=[]
    for lm in left_metrics:
        for rm in right_metrics:
            xs=[]
            ys=[]
            for row in left:
                if row[key] in rmap and row.get(lm) not in {None,""} and rmap[row[key]].get(rm) not in {None,""}:
                    xs.append(float(row[lm]))
                    ys.append(float(rmap[row[key]][rm]))
            if len(xs)>2:
                corr,p=scipy_stats.pearsonr(xs,ys)
                pairs.append({"x_metric":lm,"y_metric":rm,"n":len(xs),"pearson_r":float(corr),"p_value":float(p)})
    return pairs
