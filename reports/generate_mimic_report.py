from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"full_mimic_iv_training"
PRE=ROOT/"outputs"/"full_mimic_iv"
TEX=ROOT/"reports"/"mimic_iv_report.tex"


def read_csv(path:Path)->list[dict]:
    if not path.exists() or path.stat().st_size==0:
        return []
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def esc(s)->str:
    s=str(s)
    for a,b in [("\\","\\textbackslash{}"),("&","\\&"),("%","\\%"),("$","\\$"),("#","\\#"),("_","\\_"),("{","\\{"),("}","\\}"),("~","\\textasciitilde{}"),("^","\\textasciicircum{}")]:
        s=s.replace(a,b)
    return s


def fnum(v,digits:int=4)->str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return esc(v)


def table(rows:list[dict],cols:list[str],caption:str,label:str,limit:int|None=None)->str:
    rows=rows[:limit] if limit else rows
    if not rows:
        return ""
    spec="l"+"r"*(len(cols)-1)
    body=["\\begin{table}[H]","\\centering",f"\\caption{{{esc(caption)}}}",f"\\label{{{label}}}",f"\\begin{{tabular}}{{{spec}}}","\\toprule"]
    body.append(" & ".join(esc(c) for c in cols)+" \\\\")
    body.append("\\midrule")
    for row in rows:
        vals=[]
        for col in cols:
            vals.append(fnum(row.get(col,"")) if _numeric(row.get(col,"")) else esc(row.get(col,"")))
        body.append(" & ".join(vals)+" \\\\")
    body.extend(["\\bottomrule","\\end{tabular}","\\end{table}"])
    return "\n".join(body)


def _numeric(v)->bool:
    try:
        float(v)
        return v not in {None,""}
    except Exception:
        return False


def figure(path:str,caption:str,label:str,width:str="0.86\\textwidth")->str:
    return f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width={width}]{{{path}}}
\\caption{{{esc(caption)}}}
\\label{{{label}}}
\\end{{figure}}"""


def main()->None:
    meta=read_json(OUT/"run_metadata.json")
    ds=read_csv(PRE/"eda"/"dataset_summary.csv")
    clients=read_csv(PRE/"eda"/"client_summary.csv")
    methods=read_csv(OUT/"metrics"/"method_summary.csv")
    clinical=read_csv(OUT/"metrics"/"clinical_scores.csv")
    runtime=read_csv(OUT/"runtime"/"runtime_by_stage.csv")
    ga=read_json(OUT/"search"/"ga_result.json")
    best=max(methods,key=lambda r:float(r.get("final_auprc_mean",0))) if methods else {}
    text=preamble()+f"""
\\title{{Federated Optimization on MIMIC-IV ICU Mortality}}
\\author{{Federated Learning Optimization Project}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This report extends the UCI HAR federated optimization project to MIMIC-IV ICU mortality prediction. The experiment keeps the same optimization core: FedAvg convergence, CVaR-style fairness, linear-programming communication shadow prices, and genetic hyperparameter search. The clinical task is more difficult than UCI HAR because it is binary, imbalanced, and naturally non-IID across ICU care units. The final cohort contains {meta.get('rows','73141')} ICU stays, {meta.get('features','796')} engineered features, and {meta.get('clients','9')} federated ICU clients.
\\end{{abstract}}

\\tableofcontents
\\newpage

\\section{{Problem Definition}}
The goal is to predict in-hospital mortality for an ICU stay using information available in the first 24 hours after ICU admission. Each federated client is an ICU care unit identified by \\texttt{{first\\_careunit}}. This is a natural non-IID split because MICU, SICU, CVICU, Neuro ICU, and other units treat different populations with different mortality rates.

\\section{{Preprocessing and Cohort}}
The preprocessing stage joined \\texttt{{icustays}}, \\texttt{{admissions}}, and \\texttt{{patients}} to define the cohort and label. It then joined chart events, lab events, input events, output events, procedure events, prescriptions, diagnoses, procedures, services, and transfers. Time-stamped clinical event tables were restricted to the first 24 hours after ICU admission to reduce leakage. Numeric events were aggregated into per-stay mean, min, max, standard deviation, and count features where appropriate.

{table(ds,['rows','features','clients','positive_label','negative_label','mortality_rate','train_rows','test_rows','prediction_window_hours'],'MIMIC-IV model matrix summary','tab:dataset')}

{table(clients,['client_id','client_name','rows','mortality_rate','deaths','alive'],'Federated ICU client summary','tab:clients')}

\\section{{EDA}}
{figure('../outputs/full_mimic_iv/plots/eda/mortality_label_distribution.png','Mortality label imbalance in the final cohort.','fig:label')}
{figure('../outputs/full_mimic_iv/plots/eda/client_sample_counts.png','ICU stays per federated client.','fig:client-counts')}
{figure('../outputs/full_mimic_iv/plots/eda/client_mortality_rates.png','Mortality rate differs strongly by ICU care unit.','fig:client-mortality')}
{figure('../outputs/full_mimic_iv/plots/eda/client_noniid_js_divergence.png','Non-IID label severity measured by Jensen-Shannon divergence.','fig:noniid')}
{figure('../outputs/full_mimic_iv/plots/eda/pca_by_mortality.png','Two-dimensional PCA visualization by mortality label. PCA is used for EDA only, not as the main training representation.','fig:pca-mortality')}
{figure('../outputs/full_mimic_iv/plots/eda/pca_3d_by_client.png','Three-dimensional PCA visualization by ICU client.','fig:pca-client-3d')}

\\section{{Training Configuration}}
All main validation uses ten seeds: \\texttt{{7,11,19,23,29,31,37,41,43,47}}. The MIMIC model is a tabular MLP with 796 inputs and two output classes. Weighted cross entropy is used because mortality prevalence is about 11.4 percent. Early stopping uses loss with \\texttt{{max\\_rounds=1000}}, \\texttt{{patience=25}}, and \\texttt{{min\\_delta=0.0005}}.

\\section{{Main Results}}
The strongest mean AUPRC in this run is \\texttt{{{esc(best.get('method',''))}}} with mean AUPRC {fnum(best.get('final_auprc_mean',0))}. Accuracy is reported, but clinical interpretation focuses more on AUPRC, sensitivity, specificity, balanced accuracy, and worst-client metrics.

{table(methods,['method','n','final_accuracy_mean','final_auroc_mean','final_auprc_mean','final_balanced_accuracy_mean','final_sensitivity_mean','final_specificity_mean','final_worst_client_recall_mean','total_comm_until_stop_mean','stopped_round_mean'],'Method comparison across seeds','tab:methods')}

{table(clinical,['accuracy','balanced_accuracy','auroc','auprc','sensitivity','specificity','precision_ppv','npv','f1_death','brier'],'Final selected model clinical metrics','tab:clinical')}

\\section{{Training Dynamics}}
{figure('../outputs/full_mimic_iv_training/plots/training/loss_vs_rounds.png','FedAvg loss across communication rounds.','fig:loss')}
{figure('../outputs/full_mimic_iv_training/plots/training/auprc_vs_rounds.png','AUPRC across communication rounds.','fig:auprc-rounds')}
{figure('../outputs/full_mimic_iv_training/plots/training/sensitivity_vs_rounds.png','Mortality sensitivity across communication rounds.','fig:sens-rounds')}
{figure('../outputs/full_mimic_iv_training/plots/training/worst_client_recall_vs_rounds.png','Worst-client mortality recall across rounds.','fig:worst-recall')}

\\section{{Fairness and Client-Level Analysis}}
{figure('../outputs/full_mimic_iv_training/plots/fairness/per_client_auprc.png','Per-client AUPRC for the final selected model.','fig:client-auprc')}
{figure('../outputs/full_mimic_iv_training/plots/fairness/per_client_mortality_recall.png','Per-client mortality recall for the final selected model.','fig:client-recall')}
{figure('../outputs/full_mimic_iv_training/plots/fairness/worst_client_recall_by_method.png','Worst-client recall by method.','fig:worst-method')}

\\section{{Clinical Classification}}
{figure('../outputs/full_mimic_iv_training/plots/classification/confusion_matrix.png','Confusion matrix for the final selected model.','fig:cm')}
{figure('../outputs/full_mimic_iv_training/plots/classification/roc_curve.png','ROC curve.','fig:roc')}
{figure('../outputs/full_mimic_iv_training/plots/classification/precision_recall_curve.png','Precision-recall curve, which is especially important for imbalanced mortality prediction.','fig:pr')}

\\section{{Search and Communication Optimization}}
The genetic search evaluated {ga.get('evaluations',0)} candidate configurations. The best raw vector was \\texttt{{{esc(ga.get('x',''))}}}. Grid and GA candidates were then validated with the same ten-seed protocol as FedAvg and CVaR.

{figure('../outputs/full_mimic_iv_training/plots/search/auprc_vs_communication_scatter.png','Search candidates plotted by AUPRC and communication.','fig:search-scatter')}
{figure('../outputs/full_mimic_iv_training/plots/search/ga_fitness_vs_evaluations.png','Best genetic-search fitness over evaluations.','fig:ga')}
{figure('../outputs/full_mimic_iv_training/plots/search/hyperparameter_3d_fitness.png','Three-dimensional hyperparameter search landscape.','fig:hp3d')}
{figure('../outputs/full_mimic_iv_training/plots/optimization/shadow_price_vs_budget.png','LP communication shadow price by budget.','fig:lambda')}
{figure('../outputs/full_mimic_iv_training/plots/optimization/auprc_fairness_communication_3d.png','Three-way tradeoff between AUPRC, fairness, and communication.','fig:tradeoff3d')}

\\section{{Runtime}}
{table(runtime,['stage','seconds'],'Runtime by pipeline stage','tab:runtime')}

\\section{{Limitations}}
This run is a strong federated optimization baseline, but it should be interpreted carefully. Some administrative features may reflect coding processes rather than pure early physiology, so leakage-sensitive ablations should be used in future work. The local-only baseline is not directly comparable to ten-seed global methods because it trains separate client models. Finally, high AUROC does not guarantee high death recall in every ICU unit, so false-negative analysis remains central.

\\section{{Artifacts}}
The main output folder is \\texttt{{outputs/full\\_mimic\\_iv\\_training}}. It contains raw round logs, method summaries, clinical metrics, predictions, calibration outputs, LP diagnostics, search history, plots, model checkpoints, and report drafts. The preprocessing folder \\texttt{{outputs/full\\_mimic\\_iv}} contains the joined model matrix, EDA outputs, and the saved preprocessor.

\\end{{document}}
"""
    TEX.write_text(text,encoding="utf-8")
    print(TEX)


def preamble()->str:
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
"""


if __name__=="__main__":
    main()
