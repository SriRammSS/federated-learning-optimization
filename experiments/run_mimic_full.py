
import argparse
import platform
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from flopt.analysis import ablation_deltas,communication_efficiency,failure_modes,selected_case_clients,summarize_rows
from flopt.baselines import centralized_train,train_client_model
from flopt.calibration import calibration_bins
from flopt.config import FLConfig
from flopt.duality import solve_policy_lp
from flopt.fedavg import _device,federated_train,predict_clients
from flopt.io import convergence_summary,ensure_dirs,flatten_round_records,write_csv,write_json
from flopt.metrics import binary_clinical_scores,binary_per_client_rows,classification_rows,confusion_rows,pr_curve_rows,roc_curve_rows
from flopt.mimic import load_mimic_iv_arrays
from flopt.models import TabularMLP,count_parameters
from flopt.plots import bar,line_mean_std,scatter,scatter3
from flopt.profiling import timed
from flopt.search import ga_search,grid_search
from flopt.stats import confidence_intervals,correlations,paired_tests


ALPHAS=[0,0.5,0.75,0.9,0.95]
SEEDS=[7,11,19,23,29,31,37,41,43,47]
GRID=[(1,3,0.003),(1,5,0.005),(1,7,0.005),(2,5,0.003),(2,7,0.005),(1,9,0.003),(2,3,0.01),(3,5,0.003)]


def parse_seeds(raw:str) -> list[int]:
    return [int(seed) for seed in raw.split(",") if seed.strip()]


def build_mlp_factory(feature_count:int):
    def factory() -> TabularMLP:
        return TabularMLP(feature_count,2,hidden=(256,128,64),dropout=0.1)

    return factory


def build_base_config(args,bundle,seeds:list[int])->FLConfig:
    return FLConfig(
        rounds=args.max_rounds,
        max_rounds=args.max_rounds,
        local_epochs=args.local_epochs,
        clients_per_round=min(args.clients_per_round,len(bundle.clients)),
        lr=args.lr,
        batch_size=args.batch_size,
        seed=seeds[0],
        cvar_alpha=0.0,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stopping=True,
        monitor="loss",
        class_weights=bundle.class_weights,
        optimizer=args.optimizer,
    )


def choose_best_cvar_alpha(cvar_rows:list[dict])->float:
    summary=summarize_rows(
        cvar_rows,
        "method",
        ["final_worst_client_recall","final_worst_client_auprc","final_auprc","final_loss"],
    )
    best=max(
        summary,
        key=lambda row:(
            row.get("final_worst_client_recall_mean",0),
            row.get("final_worst_client_auprc_mean",0),
        ),
    )["method"]
    return float(str(best).replace("cvar_",""))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mimic-out",default="outputs/full_mimic_iv")
    parser.add_argument("--out",default="outputs/full_mimic_iv_training")
    parser.add_argument("--max-rounds",type=int,default=1000)
    parser.add_argument("--search-rounds",type=int,default=180)
    parser.add_argument("--patience",type=int,default=25)
    parser.add_argument("--min-delta",type=float,default=0.0005)
    parser.add_argument("--seeds",default=",".join(map(str,SEEDS)))
    parser.add_argument("--local-epochs",type=int,default=1)
    parser.add_argument("--clients-per-round",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--lr",type=float,default=0.005)
    parser.add_argument("--optimizer",default="adam")
    parser.add_argument("--ga-maxiter",type=int,default=3)
    parser.add_argument("--ga-popsize",type=int,default=4)
    parser.add_argument("--threads",type=int,default=12)
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--skip-search",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    out=Path(args.out)
    ensure_dirs(out)
    runtime=[]
    seeds=parse_seeds(args.seeds)

    with timed("load_mimic_arrays",runtime):
        bundle=load_mimic(Path(args.mimic_out),seed=seeds[0])
        clients=bundle.clients
    model_factory=build_mlp_factory(len(bundle.feature_names))
    base=build_base_config(args,bundle,seeds)
    if args.debug:
        base=replace(base,max_rounds=min(args.max_rounds,30),rounds=min(args.max_rounds,30),patience=min(args.patience,8))
        args.search_rounds=min(args.search_rounds,30)
        args.ga_maxiter=1
        args.ga_popsize=2
        args.skip_search=True
    search_base=replace(base,max_rounds=args.search_rounds,rounds=args.search_rounds,patience=max(8,min(args.patience//2,15)))
    write_metadata(out,args,bundle,base,seeds,model_factory)
    copy_preprocessor(Path(args.mimic_out),out)

    all_rounds=[]
    conv=[]
    method_seed_rows=[]
    final_models={}

    with timed("fedavg_default",runtime):
        rows,model=run_seeded_method("fedavg_default",clients,base,seeds,out,all_rounds,conv,model_factory)
        method_seed_rows.extend(rows)
        final_models["fedavg_default"]=model

    cvar_rows=[]
    cvar_models={}
    with timed("cvar_sweep",runtime):
        for alpha in ALPHAS:
            cfg=replace(base,cvar_alpha=alpha)
            rows,model=run_seeded_method(f"cvar_{alpha}",clients,cfg,seeds,out,all_rounds,conv,model_factory,alpha=alpha)
            method_seed_rows.extend(rows)
            cvar_rows.extend(rows)
            cvar_models[alpha]=model
    best_alpha=choose_best_cvar_alpha(cvar_rows)
    final_models["cvar_best"]=cvar_models[best_alpha]

    with timed("baselines",runtime):
        cent_rows,cent_model=run_seeded_centralized(clients,base,seeds,out,all_rounds,conv,model_factory)
        method_seed_rows.extend(cent_rows)
        final_models["centralized"]=cent_model
        local_rows,local_seed_rows=run_local_only(clients,replace(base,seed=seeds[0],max_rounds=50,rounds=50),model_factory,bundle.client_names,out)
        write_csv(out/"baselines"/"local_only_clients.csv",local_rows)
        write_csv(out/"raw"/"local_only_detail.csv",local_seed_rows)
        method_seed_rows.append(local_summary(local_rows,seeds[0]))

    grid_rows=[]
    ga={"x":[base.local_epochs,base.clients_per_round,base.lr],"fitness":0.0,"evaluations":0,"history":[]}
    with timed("search",runtime):
        if not args.skip_search:
            grid_rows=grid_search(clients,replace(search_base,seed=seeds[0]),GRID,gamma=1e-8,model_factory=model_factory,score_key="auprc",min_score=0.25)
            write_csv(out/"search"/"grid_search.csv",grid_rows)
            best_grid=grid_rows[0]
            ga=ga_search(clients,replace(search_base,seed=seeds[0]),bounds=[(1,3),(3,len(clients)),(0.001,0.02)],maxiter=args.ga_maxiter,popsize=args.ga_popsize,gamma=1e-8,model_factory=model_factory,score_key="auprc",min_score=0.25)
            write_json(out/"search"/"ga_result.json",ga)
            write_csv(out/"search"/"ga_history.csv",ga["history"])
            grid_cfg=replace(base,local_epochs=int(best_grid["local_epochs"]),clients_per_round=int(best_grid["clients_per_round"]),lr=float(best_grid["lr"]))
            ga_cfg=replace(base,local_epochs=max(1,int(round(ga["x"][0]))),clients_per_round=max(1,int(round(ga["x"][1]))),lr=float(ga["x"][2]))
            grid_valid,grid_model=run_seeded_method("grid_best_validated",clients,grid_cfg,seeds,out,all_rounds,conv,model_factory)
            ga_valid,ga_model=run_seeded_method("ga_best_validated",clients,ga_cfg,seeds,out,all_rounds,conv,model_factory)
            method_seed_rows.extend(grid_valid)
            method_seed_rows.extend(ga_valid)
            final_models["grid_best_validated"]=grid_model
            final_models["ga_best_validated"]=ga_model

    write_csv(out/"raw"/"all_round_metrics.csv",all_rounds)
    write_csv(out/"raw"/"convergence_summary.csv",conv)
    write_csv(out/"metrics"/"method_seed_results.csv",method_seed_rows)
    summary_metrics=["final_loss","final_accuracy","final_worst_client_accuracy","final_auroc","final_auprc","final_balanced_accuracy","final_sensitivity","final_specificity","final_worst_client_recall","final_worst_client_auprc","total_comm_until_stop","stopped_round"]
    method_summary=summarize_rows(method_seed_rows,"method",summary_metrics)
    write_csv(out/"metrics"/"method_summary.csv",method_summary)

    with timed("predictions_metrics",runtime):
        best_model=final_models.get("ga_best_validated") or final_models.get("grid_best_validated") or final_models["cvar_best"]
        pred_rows=predict_clients(best_model,clients)
        add_client_names(pred_rows,bundle.client_names)
        write_csv(out/"metrics"/"predictions.csv",pred_rows)
        write_csv(out/"metrics"/"classification_report.csv",classification_report(pred_rows,bundle.class_names))
        write_csv(out/"metrics"/"confusion_matrix.csv",confusion_table(pred_rows,bundle.class_names))
        write_csv(out/"metrics"/"clinical_scores.csv",[binary_clinical_scores(pred_rows)])
        per_client=client_scores(pred_rows,bundle.client_names)
        write_csv(out/"metrics"/"per_client_clinical_metrics.csv",per_client)
        write_csv(out/"metrics"/"roc_curve.csv",roc_curve(pred_rows))
        write_csv(out/"metrics"/"precision_recall_curve.csv",pr_curve(pred_rows))
        cal_rows,cal_summary=calibration_bins(pred_rows)
        write_csv(out/"calibration"/"calibration_bins.csv",cal_rows)
        write_csv(out/"calibration"/"calibration_summary.csv",[cal_summary])

    with timed("lp_diagnostics",runtime):
        lp_source=grid_rows+[{"loss":r["final_loss"],"comm":r["total_comm_until_stop"]} for r in method_seed_rows if r.get("method") in {"fedavg_default","grid_best_validated","ga_best_validated","cvar_best"}]
        lp_source=[r for r in lp_source if float(r.get("comm",0) or 0)>=0]
        losses=[float(r["loss"]) for r in lp_source] or [1.0]
        costs=[float(r["comm"]) for r in lp_source] or [0.0]
        budgets=np.linspace(min(costs),max(costs),12).tolist() if max(costs)>min(costs) else [max(costs)]
        lp_rows=solve_policy_lp(losses,costs,budgets)
        write_json(out/"lp"/"lp_shadow_price.json",lp_rows)
        write_csv(out/"lp"/"lp_shadow_price.csv",flatten_lp(lp_rows))

    with timed("diagnostics",runtime):
        run_diagnostics(out,method_seed_rows,per_client,conv)

    with timed("plots",runtime):
        run_plots(out,all_rounds,method_summary,method_seed_rows,per_client,grid_rows,ga,lp_rows,pred_rows)

    with timed("artifacts",runtime):
        for name,model in final_models.items():
            torch.save(model.state_dict(),out/"artifacts"/f"{name}.pt")

    write_csv(out/"runtime"/"runtime_by_stage.csv",runtime)
    write_report(out,method_summary,best_alpha,ga,bundle)
    write_csv(out/"reports"/"theoretical_alignment_table.csv",theoretical_rows())
    print(f"wrote MIMIC-IV training artifacts to {out}")


def write_metadata(out,args,bundle,base,seeds,model_factory):
    write_json(out/"run_metadata.json",{
        "dataset":"MIMIC-IV ICU mortality",
        "mimic_out":args.mimic_out,
        "rows":sum(len(c.x_train)+len(c.x_test) for c in bundle.clients),
        "features":len(bundle.feature_names),
        "clients":len(bundle.clients),
        "class_names":bundle.class_names,
        "class_weights":bundle.class_weights,
        "seeds":seeds,
        "threads":args.threads,
        "platform":platform.platform(),
        "torch_version":torch.__version__,
        "model_parameters":count_parameters(model_factory()),
    })


def copy_preprocessor(mimic_out:Path,out:Path):
    src=mimic_out/"artifacts"/"mimic_preprocessor.pkl"
    if src.exists():
        shutil.copy2(src,out/"artifacts"/"mimic_preprocessor.pkl")


def run_seeded_method(name,clients,cfg,seeds,out,all_rounds,conv,model_factory,alpha=None):
    rows=[]
    last_model=None
    for seed in seeds:
        run_cfg=replace(cfg,seed=seed)
        model,records=federated_train(model_factory(),clients,run_cfg,drift=name in {"fedavg_default",f"cvar_{alpha}"})
        last_model=model
        round_rows=round_records_to_csv(records,name,seed,alpha)
        all_rounds.extend(round_rows)
        write_csv(out/"raw"/f"{name}_seed_{seed}_rounds.csv",round_rows)
        c=convergence_summary(records,name,seed,alpha,run_cfg.max_rounds or run_cfg.rounds)
        conv.append(c)
        rows.append({"method":name,"seed":seed,**{k:v for k,v in c.items() if k not in {"run_type","seed","alpha","max_rounds"}}})
        write_drift(out,name,seed,records)
    return rows,last_model


def run_seeded_centralized(clients,cfg,seeds,out,all_rounds,conv,model_factory):
    rows=[]
    last_model=None
    for seed in seeds:
        run_cfg=replace(cfg,seed=seed)
        model,records=centralized_train(model_factory(),clients,run_cfg)
        last_model=model
        rr=round_records_to_csv(records,"centralized",seed)
        all_rounds.extend(rr)
        write_csv(out/"raw"/f"centralized_seed_{seed}_rounds.csv",rr)
        c=convergence_summary(records,"centralized",seed,None,run_cfg.max_rounds or run_cfg.rounds)
        conv.append(c)
        rows.append({"method":"centralized","seed":seed,**{k:v for k,v in c.items() if k not in {"run_type","seed","alpha","max_rounds"}}})
    return rows,last_model


def run_local_only(clients,cfg,model_factory,client_names,out):
    device=_device()
    rows=[]
    detail=[]
    for idx,client in enumerate(clients):
        model=model_factory().to(device)
        train_client_model(model,client,cfg,device)
        pred=predict_clients(model,[client],device)
        add_client_names(pred,client_names)
        scores=binary_clinical_scores(pred)
        cid=client.client_id if client.client_id is not None else idx
        row={"client_id":int(cid),"client_name":client_names.get(int(cid),str(cid)),"train_samples":len(client.x_train),"test_samples":len(client.x_test),**scores}
        rows.append(row)
        detail.append({"run_type":"local_only","seed":cfg.seed,**row})
    return rows,detail


def local_summary(rows,seed):
    return {
        "method":"local_only",
        "seed":seed,
        "final_loss":0.0,
        "final_accuracy":float(np.mean([r["accuracy"] for r in rows])),
        "final_worst_client_accuracy":float(np.min([r["accuracy"] for r in rows])),
        "final_auroc":float(np.mean([r["auroc"] for r in rows])),
        "final_auprc":float(np.mean([r["auprc"] for r in rows])),
        "final_balanced_accuracy":float(np.mean([r["balanced_accuracy"] for r in rows])),
        "final_sensitivity":float(np.mean([r["sensitivity"] for r in rows])),
        "final_specificity":float(np.mean([r["specificity"] for r in rows])),
        "final_worst_client_recall":float(np.min([r["sensitivity"] for r in rows])),
        "final_worst_client_auprc":float(np.min([r["auprc"] for r in rows])),
        "total_comm_until_stop":0,
        "stopped_round":50,
        "stopped_early":False,
    }


def add_client_names(rows,client_names):
    for row in rows:
        row["client_name"]=client_names.get(int(row["client_id"]),str(row["client_id"]))


def write_drift(out,name,seed,records):
    rows=[]
    for r in records:
        ids=r.get("drift_client_ids",[])
        norms=r.get("drift_update_norms",[])
        cos=r.get("drift_cosine_to_mean",[])
        dist=r.get("drift_distance_to_mean",[])
        for i,cid in enumerate(ids):
            rows.append({"method":name,"seed":seed,"round":r["round"],"client_id":cid,"update_norm":norms[i],"cosine_to_mean":cos[i],"distance_to_mean":dist[i]})
    if rows:
        write_csv(out/"drift"/f"{name}_seed_{seed}_drift.csv",rows)


def flatten_lp(rows):
    result=[]
    for r in rows:
        kkt=r.get("kkt",{})
        result.append({"budget":r.get("budget"),"loss":r.get("loss"),"cost":r.get("cost"),"lambda":r.get("lambda"),"status":r.get("status"),**kkt})
    return result


def run_diagnostics(out,method_seed_rows,per_client,conv):
    metrics=["final_accuracy","final_auroc","final_auprc","final_balanced_accuracy","final_sensitivity","final_specificity","final_worst_client_recall","total_comm_until_stop","stopped_round"]
    write_csv(out/"stats"/"confidence_intervals.csv",confidence_intervals(method_seed_rows,"method",metrics))
    write_csv(out/"stats"/"paired_tests.csv",paired_tests(method_seed_rows,"method","seed",metrics,"fedavg_default"))
    write_csv(out/"ablations"/"ablation_deltas.csv",ablation_deltas(method_seed_rows))
    write_csv(out/"efficiency"/"communication_efficiency.csv",communication_efficiency(method_seed_rows,threshold=0.25))
    write_csv(out/"failure_modes"/"failure_modes.csv",failure_modes(method_seed_rows)+clinical_failure_modes(method_seed_rows))
    noniid_path=Path("outputs/full_mimic_iv/noniid/client_distribution_metrics.csv")
    if noniid_path.exists():
        noniid=read_csv_dicts(noniid_path)
        write_csv(out/"noniid"/"noniid_performance_correlations.csv",correlations(noniid,per_client,"client_id",["js_divergence","mortality_rate","samples"],["auprc","sensitivity","balanced_accuracy"]))
    case_ids=selected_case_clients([{"client_id":r["client_id"],"accuracy":r["balanced_accuracy"]} for r in per_client])
    write_csv(out/"case_studies"/"selected_clients.csv",[{"client_id":cid} for cid in case_ids])


def clinical_failure_modes(rows):
    flags=[]
    for row in rows:
        if float(row.get("final_sensitivity",1) or 0)<0.50:
            flags.append({**row,"failure_type":"low_mortality_recall","description":"Mortality sensitivity is below 0.50."})
        if float(row.get("final_auprc",1) or 0)<0.20:
            flags.append({**row,"failure_type":"low_auprc","description":"AUPRC remains low for the imbalanced mortality task."})
    return flags


def read_csv_dicts(path:Path):
    import csv
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_plots(out,all_rounds,method_summary,method_seed_rows,per_client,grid_rows,ga,lp_rows,pred_rows):
    fed=[r for r in all_rounds if r["run_type"]=="fedavg_default"]
    for metric,title in [("loss","FedAvg Loss"),("auroc","FedAvg AUROC"),("auprc","FedAvg AUPRC"),("balanced_accuracy","FedAvg Balanced Accuracy"),("sensitivity","FedAvg Sensitivity"),("worst_client_recall","Worst-Client Recall")]:
        rows=[r for r in fed if r.get(metric) not in {None,""}]
        if rows:
            line_mean_std(rows,metric,str(out/"plots"/"training"/f"{metric}_vs_rounds.png"),title)
    bar(method_summary,"method","final_auprc_mean",str(out/"plots"/"baselines"/"baseline_auprc_comparison.png"),"Baseline AUPRC")
    bar(method_summary,"method","final_sensitivity_mean",str(out/"plots"/"baselines"/"baseline_sensitivity_comparison.png"),"Baseline Mortality Recall")
    bar(method_summary,"method","final_worst_client_recall_mean",str(out/"plots"/"fairness"/"worst_client_recall_by_method.png"),"Worst-Client Recall")
    bar(per_client,"client_name","auprc",str(out/"plots"/"fairness"/"per_client_auprc.png"),"Per-Client AUPRC")
    bar(per_client,"client_name","sensitivity",str(out/"plots"/"fairness"/"per_client_mortality_recall.png"),"Per-Client Mortality Recall")
    plot_curve(read_csv_dicts(out/"metrics"/"roc_curve.csv"),"fpr","tpr",out/"plots"/"classification"/"roc_curve.png","ROC Curve","FPR","TPR")
    plot_curve(read_csv_dicts(out/"metrics"/"precision_recall_curve.csv"),"recall","precision",out/"plots"/"classification"/"precision_recall_curve.png","Precision-Recall Curve","Recall","Precision")
    plot_confusion(out)
    if grid_rows:
        scatter(grid_rows,"comm","auprc",str(out/"plots"/"search"/"auprc_vs_communication_scatter.png"),"AUPRC vs Communication","fitness")
        scatter3(grid_rows,"local_epochs","clients_per_round","lr",str(out/"plots"/"search"/"hyperparameter_3d_fitness.png"),"3D Hyperparameter Fitness","fitness")
    if ga.get("history"):
        write_csv(out/"search"/"ga_history_best_so_far.csv",ga_best_history(ga["history"]))
        line_mean_std(ga_best_history(ga["history"]),"best_fitness",str(out/"plots"/"search"/"ga_fitness_vs_evaluations.png"),"GA Best Fitness")
    flat=flatten_lp(lp_rows)
    if flat:
        scatter(flat,"budget","lambda",str(out/"plots"/"optimization"/"shadow_price_vs_budget.png"),"Shadow Price vs Budget")
        scatter(flat,"budget","loss",str(out/"plots"/"optimization"/"loss_vs_budget.png"),"Loss vs Budget")
    scatter3(method_seed_rows,"final_auprc","final_worst_client_recall","total_comm_until_stop",str(out/"plots"/"optimization"/"auprc_fairness_communication_3d.png"),"AUPRC Fairness Communication")


def plot_curve(rows,x,y,path,title,xlabel,ylabel):
    if not rows:
        return
    plt.figure(figsize=(6,4))
    plt.plot([float(r[x]) for r in rows],[float(r[y]) for r in rows])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(path,dpi=160)
    plt.close()


def plot_confusion(out):
    rows=read_csv_dicts(out/"metrics"/"confusion_matrix.csv")
    if not rows:
        return
    labels=["survived","expired"]
    mat=np.array([[float(row[label]) for label in labels] for row in rows])
    plt.figure(figsize=(5,4))
    plt.imshow(mat,cmap="viridis")
    plt.colorbar()
    plt.xticks(range(2),labels)
    plt.yticks(range(2),labels)
    plt.title("Confusion Matrix")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    path=out/"plots"/"classification"/"confusion_matrix.png"
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(path,dpi=160)
    plt.close()


def ga_best_history(history):
    best=float("inf")
    rows=[]
    for row in history:
        best=min(best,float(row["fitness"]))
        rows.append({"round":row["evaluation"],"best_fitness":best})
    return rows


def write_report(out,method_summary,best_alpha,ga,bundle):
    best=max(method_summary,key=lambda r:r.get("final_auprc_mean",0))
    text=f"""# MIMIC-IV Federated Optimization Report Draft

## Dataset

This run uses the preprocessed MIMIC-IV ICU mortality cohort with `{sum(len(c.x_train)+len(c.x_test) for c in bundle.clients)}` ICU stays, `{len(bundle.feature_names)}` features, and `{len(bundle.clients)}` ICU-unit clients.

## Headline Results

- Best mean AUPRC method: `{best['method']}` with AUPRC `{best.get('final_auprc_mean',0):.4f}`.
- Best CVaR alpha selected by worst-client clinical performance: `{best_alpha}`.
- GA best raw vector: `{ga.get('x')}`.
- GA search evaluations: `{ga.get('evaluations',0)}`.

See the generated LaTeX report for the final formatted explanation.
"""
    (out/"reports"/"report_draft_mimic.md").write_text(text,encoding="utf-8")


def theoretical_rows():
    return [
        {"component":"FedAvg","optimization_concept":"local SGD and federated stochastic optimization"},
        {"component":"Weighted cross entropy","optimization_concept":"cost-sensitive empirical risk minimization"},
        {"component":"CVaR weighting","optimization_concept":"tail-risk and fairness-aware optimization"},
        {"component":"LP shadow price","optimization_concept":"duality and resource-constrained optimization"},
        {"component":"Differential evolution","optimization_concept":"global black-box hyperparameter search"},
        {"component":"Early stopping","optimization_concept":"empirical convergence criterion"},
    ]


if __name__=="__main__":
    main()
