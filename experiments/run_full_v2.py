from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from flopt.analysis import ablation_rows,communication_efficiency_rows,fairness_gap_rows,failure_mode_rows,selected_case_clients,summarize_rows
from flopt.baselines import centralized_train,local_only_summary
from flopt.calibration import calibration_bins
from flopt.config import FLConfig
from flopt.data import load_uci_har
from flopt.duality import solve_policy_lp
from flopt.eda import eda_tables,noniid_rows
from flopt.fedavg import federated_train,predict_clients
from flopt.io import convergence_summary,ensure_dirs,flatten_round_records,write_csv,write_json
from flopt.metrics import aggregate_scores,classification_rows,confusion_rows,per_class_error_rows,per_client_rows,top_confusions
from flopt.models import HARMLP,count_parameters
from flopt.plots import bar,grouped_bar,heatmap,line_mean_std,pca_plots,scatter,scatter3
from flopt.profiling import timed
from flopt.search import ga_search,grid_search
from flopt.stats import confidence_rows,correlation_rows,paired_tests


ALPHAS=[0,0.5,0.75,0.9,0.95]
SEEDS=[7,11,19,23,29,31,37,41,43,47]
GRID=[(1,5,0.01),(1,10,0.03),(2,10,0.03),(4,10,0.02),(2,15,0.01),(4,15,0.02),(5,10,0.02),(3,12,0.04)]


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="outputs/full_uci_v2")
    parser.add_argument("--max-rounds",type=int,default=1000)
    parser.add_argument("--search-rounds",type=int,default=120)
    parser.add_argument("--patience",type=int,default=30)
    parser.add_argument("--min-delta",type=float,default=0.001)
    parser.add_argument("--seeds",default=",".join(map(str,SEEDS)))
    parser.add_argument("--ga-maxiter",type=int,default=4)
    parser.add_argument("--ga-popsize",type=int,default=6)
    parser.add_argument("--threads",type=int,default=12)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    out=Path(args.out)
    ensure_dirs(out)
    runtime=[]
    seeds=[int(s) for s in args.seeds.split(",") if s.strip()]

    with timed("load_data",runtime):
        bundle=load_uci_har(Path("data"),seed=seeds[0])
        clients=bundle.clients
    activity_names=bundle.activity_names
    base=FLConfig(rounds=args.max_rounds,max_rounds=args.max_rounds,local_epochs=2,clients_per_round=min(10,len(clients)),lr=0.03,batch_size=32,early_stopping=True,patience=args.patience,min_delta=args.min_delta)
    search_base=replace(base,max_rounds=args.search_rounds,rounds=args.search_rounds,patience=max(10,args.patience//2))

    write_json(out/"run_metadata.json",{
        "max_rounds":args.max_rounds,
        "search_rounds":args.search_rounds,
        "patience":args.patience,
        "min_delta":args.min_delta,
        "seeds":seeds,
        "threads":args.threads,
        "platform":platform.platform(),
        "torch_version":torch.__version__,
        "model_parameters":count_parameters(HARMLP()),
    })
    with (out/"artifacts"/"scaler.pkl").open("wb") as f:
        pickle.dump(bundle.scaler,f)

    with timed("eda",runtime):
        run_eda(bundle,out)

    all_rounds=[]
    conv=[]
    method_seed_rows=[]
    final_models={}

    with timed("fedavg_default",runtime):
        rows,model=run_seeded_method("fedavg_default",clients,base,seeds,out,all_rounds,conv)
        method_seed_rows.extend(rows)
        final_models["fedavg_default"]=model

    cvar_summaries=[]
    cvar_models={}
    with timed("cvar_sweep",runtime):
        for alpha in ALPHAS:
            cfg=replace(base,cvar_alpha=alpha)
            rows,model=run_seeded_method(f"cvar_{alpha}",clients,cfg,seeds,out,all_rounds,conv,alpha=alpha)
            method_seed_rows.extend(rows)
            cvar_summaries.extend(rows)
            cvar_models[alpha]=model
    best_alpha=max(summarize_rows(cvar_summaries,"method",["final_accuracy","final_worst_client_accuracy"]),key=lambda r:r.get("final_worst_client_accuracy_mean",0))["method"]
    best_alpha_value=float(str(best_alpha).replace("cvar_",""))
    final_models["cvar_best"]=cvar_models[best_alpha_value]

    with timed("baselines",runtime):
        cent_rows,cent_model=run_seeded_centralized(clients,base,seeds,out,all_rounds,conv)
        method_seed_rows.extend(cent_rows)
        final_models["centralized"]=cent_model
        local_rows,local_detail=local_only_summary(HARMLP,clients,replace(base,seed=seeds[0],max_rounds=50,rounds=50))
        write_csv(out/"baselines"/"local_only_clients.csv",local_rows)
        write_csv(out/"raw"/"local_only_detail.csv",local_detail)
        method_seed_rows.append({"method":"local_only","seed":seeds[0],"final_loss":float(np.mean([r["loss"] for r in local_rows])),"final_accuracy":float(np.mean([r["accuracy"] for r in local_rows])),"final_worst_client_accuracy":float(np.min([r["accuracy"] for r in local_rows])),"total_comm_until_stop":0,"stopped_round":50,"stopped_early":False})

    with timed("search",runtime):
        grid_rows=grid_search(clients,replace(search_base,seed=seeds[0]),GRID)
        write_csv(out/"search"/"grid_search.csv",grid_rows)
        best_grid=grid_rows[0]
        ga=ga_search(clients,replace(search_base,seed=seeds[0]),maxiter=args.ga_maxiter,popsize=args.ga_popsize)
        write_json(out/"search"/"ga_result.json",ga)
        write_csv(out/"search"/"ga_history.csv",ga["history"])
        grid_cfg=replace(base,local_epochs=int(best_grid["local_epochs"]),clients_per_round=int(best_grid["clients_per_round"]),lr=float(best_grid["lr"]))
        ga_cfg=replace(base,local_epochs=max(1,int(round(ga["x"][0]))),clients_per_round=max(1,int(round(ga["x"][1]))),lr=float(ga["x"][2]))
        grid_valid,grid_model=run_seeded_method("grid_best_validated",clients,grid_cfg,seeds,out,all_rounds,conv)
        ga_valid,ga_model=run_seeded_method("ga_best_validated",clients,ga_cfg,seeds,out,all_rounds,conv)
        method_seed_rows.extend(grid_valid)
        method_seed_rows.extend(ga_valid)
        final_models["grid_best_validated"]=grid_model
        final_models["ga_best_validated"]=ga_model

    write_csv(out/"raw"/"all_round_metrics.csv",all_rounds)
    write_csv(out/"raw"/"convergence_summary.csv",conv)
    write_csv(out/"metrics"/"method_seed_results.csv",method_seed_rows)
    method_summary=summarize_rows(method_seed_rows,"method",["final_loss","final_accuracy","final_worst_client_accuracy","total_comm_until_stop","stopped_round"])
    write_csv(out/"metrics"/"method_summary.csv",method_summary)

    with timed("predictions_metrics",runtime):
        best_model=final_models["ga_best_validated"]
        pred_rows=predict_clients(best_model,clients)
        write_csv(out/"metrics"/"predictions.csv",pred_rows)
        write_csv(out/"metrics"/"classification_report.csv",classification_rows(pred_rows,activity_names))
        write_csv(out/"metrics"/"confusion_matrix.csv",confusion_rows(pred_rows,activity_names))
        write_csv(out/"metrics"/"confusion_matrix_normalized.csv",confusion_rows(pred_rows,activity_names,normalize=True))
        per_client=per_client_rows(pred_rows,clients)
        write_csv(out/"metrics"/"per_client_metrics.csv",per_client)
        write_csv(out/"errors"/"top_confusions.csv",top_confusions(pred_rows,activity_names))
        write_csv(out/"errors"/"per_class_error_rates.csv",per_class_error_rows(pred_rows,activity_names))
        cal_rows,cal_summary=calibration_bins(pred_rows)
        write_csv(out/"calibration"/"calibration_bins.csv",cal_rows)
        write_csv(out/"calibration"/"calibration_summary.csv",[cal_summary])
        write_json(out/"metrics"/"aggregate_scores.json",aggregate_scores(pred_rows))

    with timed("lp_diagnostics",runtime):
        lp_source=grid_rows+[{"loss":r["final_loss"],"comm":r["total_comm_until_stop"]} for r in method_seed_rows if r["method"] in {"fedavg_default","grid_best_validated","ga_best_validated"}]
        losses=[float(r["loss"]) for r in lp_source]
        costs=[float(r["comm"]) for r in lp_source]
        budgets=np.linspace(min(costs),max(costs),12).tolist()
        lp_rows=solve_policy_lp(losses,costs,budgets)
        write_json(out/"lp"/"lp_shadow_price.json",lp_rows)
        write_csv(out/"lp"/"lp_shadow_price.csv",flatten_lp(lp_rows))

    with timed("diagnostics",runtime):
        run_diagnostics(out,method_seed_rows,per_client,bundle,pred_rows,conv)

    with timed("plots",runtime):
        run_plots(out,bundle,all_rounds,method_summary,method_seed_rows,per_client,grid_rows,ga,lp_rows,pred_rows,activity_names)

    with timed("artifacts",runtime):
        for name,model in final_models.items():
            torch.save(model.state_dict(),out/"artifacts"/f"{name}.pt")

    write_csv(out/"runtime"/"runtime_by_stage.csv",runtime)
    write_report(out,method_summary,best_alpha_value,ga)
    print(f"wrote v2 artifacts to {out}")


def run_seeded_method(name,clients,cfg,seeds,out,all_rounds,conv,alpha=None):
    rows=[]
    last_model=None
    for seed in seeds:
        run_cfg=replace(cfg,seed=seed)
        model,records=federated_train(HARMLP(),clients,run_cfg,track_drift=name in {"fedavg_default",f"cvar_{alpha}"})
        last_model=model
        round_rows=flatten_round_records(records,name,seed,alpha)
        all_rounds.extend(round_rows)
        write_csv(out/"raw"/f"{name}_seed_{seed}_rounds.csv",round_rows)
        c=convergence_summary(records,name,seed,alpha,run_cfg.max_rounds or run_cfg.rounds)
        conv.append(c)
        rows.append({"method":name,"seed":seed,**{k:v for k,v in c.items() if k not in {"run_type","seed","alpha","max_rounds"}}})
        write_drift(out,name,seed,records)
    return rows,last_model


def run_seeded_centralized(clients,cfg,seeds,out,all_rounds,conv):
    rows=[]
    last_model=None
    for seed in seeds:
        run_cfg=replace(cfg,seed=seed)
        model,records=centralized_train(HARMLP(),clients,run_cfg)
        last_model=model
        rr=flatten_round_records(records,"centralized",seed)
        all_rounds.extend(rr)
        write_csv(out/"raw"/f"centralized_seed_{seed}_rounds.csv",rr)
        c=convergence_summary(records,"centralized",seed,None,run_cfg.max_rounds or run_cfg.rounds)
        conv.append(c)
        rows.append({"method":"centralized","seed":seed,**{k:v for k,v in c.items() if k not in {"run_type","seed","alpha","max_rounds"}}})
    return rows,last_model


def run_eda(bundle,out):
    tables=eda_tables(bundle)
    for name,rows in tables.items():
        write_csv(out/"eda"/f"{name}.csv",rows)
    niid=noniid_rows(bundle)
    write_csv(out/"noniid"/"client_distribution_metrics.csv",niid)
    class_rows=tables["class_distribution"]
    bar(class_rows,"activity","total_count",str(out/"plots"/"eda"/"class_distribution.png"),"Class Distribution")
    grouped_bar(class_rows,"activity",["train_count","test_count"],str(out/"plots"/"eda"/"train_test_class_distribution.png"),"Train/Test Class Distribution")
    bar(tables["client_sample_counts"],"client_id","total_samples",str(out/"plots"/"eda"/"client_sample_counts.png"),"Samples Per Client")
    bar(tables["client_sample_counts"],"client_id","label_entropy",str(out/"plots"/"eda"/"client_entropy.png"),"Client Label Entropy")
    matrix=np.array([[r["proportion"] for r in tables["client_label_distribution"] if r["client_id"]==cid] for cid in sorted({r["client_id"] for r in tables["client_label_distribution"]})])
    heatmap(matrix,bundle.activity_names,[str(cid) for cid in sorted({r["client_id"] for r in tables["client_label_distribution"]})],str(out/"plots"/"eda"/"client_label_heatmap.png"),"Client Label Distribution","activity","client")
    corr=np.corrcoef(bundle.x_train_raw[:,:30],rowvar=False)
    heatmap(corr,[str(i) for i in range(30)],[str(i) for i in range(30)],str(out/"plots"/"eda"/"feature_correlation_subset.png"),"Feature Correlation Subset")
    x=np.vstack([bundle.x_train_raw,bundle.x_test_raw])
    y=np.concatenate([bundle.y_train,bundle.y_test])
    s=np.concatenate([bundle.subject_train,bundle.subject_test])
    idx=np.linspace(0,len(x)-1,min(len(x),2500),dtype=int)
    pca_plots(x[idx],y[idx].tolist(),bundle.activity_names,str(out/"plots"/"eda"/"pca_by_activity.png"),str(out/"plots"/"eda"/"pca_3d_by_activity.png"),"Activity")
    pca_plots(x[idx],s[idx].tolist(),[],str(out/"plots"/"eda"/"pca_by_client.png"),str(out/"plots"/"eda"/"pca_3d_by_client.png"),"Client")


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
    out=[]
    for r in rows:
        kkt=r.get("kkt",{})
        out.append({"budget":r.get("budget"),"loss":r.get("loss"),"cost":r.get("cost"),"lambda":r.get("lambda"),"status":r.get("status"),**kkt})
    return out


def run_diagnostics(out,method_seed_rows,per_client,bundle,pred_rows,conv):
    metrics=["final_accuracy","final_worst_client_accuracy","final_loss","total_comm_until_stop","stopped_round"]
    write_csv(out/"stats"/"confidence_intervals.csv",confidence_rows(method_seed_rows,"method",metrics))
    write_csv(out/"stats"/"paired_tests.csv",paired_tests(method_seed_rows,"method","seed",metrics,"fedavg_default"))
    write_csv(out/"ablations"/"ablation_deltas.csv",ablation_rows(method_seed_rows))
    write_csv(out/"efficiency"/"communication_efficiency.csv",communication_efficiency_rows(method_seed_rows))
    write_csv(out/"failure_modes"/"failure_modes.csv",failure_mode_rows(method_seed_rows))
    noniid=noniid_rows(bundle)
    write_csv(out/"noniid"/"noniid_performance_correlations.csv",correlation_rows(noniid,per_client,"client_id",["js_divergence","label_entropy","samples"],["accuracy","error_rate"]))
    case_ids=selected_case_clients(per_client)
    write_csv(out/"case_studies"/"selected_clients.csv",[{"client_id":cid} for cid in case_ids])
    theoretical=[
        {"component":"FedAvg","optimization_concept":"local SGD and stochastic optimization"},
        {"component":"LP shadow price","optimization_concept":"duality and constrained optimization"},
        {"component":"KKT residuals","optimization_concept":"optimality diagnostics"},
        {"component":"CVaR weighting","optimization_concept":"robust and fairness-aware optimization"},
        {"component":"Differential evolution","optimization_concept":"global search"},
        {"component":"Early stopping","optimization_concept":"empirical convergence criterion"},
    ]
    write_csv(out/"reports"/"theoretical_alignment_table.csv",theoretical)


def run_plots(out,bundle,all_rounds,method_summary,method_seed_rows,per_client,grid_rows,ga,lp_rows,pred_rows,activity_names):
    fed=[r for r in all_rounds if r["run_type"]=="fedavg_default"]
    if fed:
        line_mean_std(fed,"loss",str(out/"plots"/"training"/"loss_vs_rounds_mean_std.png"),"FedAvg Loss")
        line_mean_std(fed,"accuracy",str(out/"plots"/"training"/"accuracy_vs_rounds_mean_std.png"),"FedAvg Accuracy")
        line_mean_std(fed,"worst_client_accuracy",str(out/"plots"/"training"/"worst_client_accuracy_vs_rounds.png"),"Worst-Client Accuracy")
        line_mean_std(fed,"best_loss_so_far",str(out/"plots"/"training"/"best_loss_so_far_vs_rounds.png"),"Best Loss So Far")
        line_mean_std(fed,"rounds_since_improvement",str(out/"plots"/"training"/"rounds_since_improvement.png"),"Rounds Since Improvement")
        scatter(fed,"round","total_comm_bytes",str(out/"plots"/"training"/"communication_vs_rounds.png"),"Communication Per Round")
        scatter(fed,"total_comm_bytes","accuracy",str(out/"plots"/"training"/"accuracy_vs_cumulative_communication.png"),"Accuracy vs Communication")
    bar(method_summary,"method","final_accuracy_mean",str(out/"plots"/"baselines"/"baseline_accuracy_comparison.png"),"Baseline Accuracy")
    bar(method_summary,"method","final_worst_client_accuracy_mean",str(out/"plots"/"baselines"/"baseline_worst_client_accuracy.png"),"Worst-Client Accuracy")
    bar(method_summary,"method","total_comm_until_stop_mean",str(out/"plots"/"baselines"/"baseline_communication_cost.png"),"Communication Cost")
    bar(method_summary,"method","stopped_round_mean",str(out/"plots"/"baselines"/"baseline_rounds_to_convergence.png"),"Rounds To Convergence")
    bar(per_client,"client_id","accuracy",str(out/"plots"/"fairness"/"per_client_accuracy_fedavg.png"),"Per-Client Accuracy")
    scatter(per_client,"train_samples","accuracy",str(out/"plots"/"noniid"/"sample_count_vs_accuracy.png"),"Sample Count vs Accuracy")
    cm=np.array([[r[name] for name in activity_names] for r in confusion_rows(pred_rows,activity_names)])
    heatmap(cm,activity_names,activity_names,str(out/"plots"/"classification"/"confusion_matrix_raw.png"),"Confusion Matrix","predicted","true")
    cmn=np.array([[r[name] for name in activity_names] for r in confusion_rows(pred_rows,activity_names,normalize=True)])
    heatmap(cmn,activity_names,activity_names,str(out/"plots"/"classification"/"confusion_matrix_normalized.png"),"Normalized Confusion Matrix","predicted","true")
    bar(per_class_error_rows(pred_rows,activity_names),"label","error_rate",str(out/"plots"/"errors"/"per_class_error_rate.png"),"Per-Class Error Rate")
    bar(top_confusions(pred_rows,activity_names),"true_label","count",str(out/"plots"/"errors"/"top_confused_pairs.png"),"Top Confusions")
    plot_lp(out,lp_rows)
    scatter(grid_rows,"comm","accuracy",str(out/"plots"/"search"/"accuracy_vs_communication_scatter.png"),"Accuracy vs Communication","fitness")
    scatter3(grid_rows,"local_epochs","clients_per_round","lr",str(out/"plots"/"search"/"hyperparameter_3d_fitness.png"),"3D Hyperparameter Fitness","fitness")
    write_csv(out/"search"/"ga_history_best_so_far.csv",ga_best_history(ga["history"]))
    if ga["history"]:
        line_mean_std(ga_best_history(ga["history"]),"best_fitness",str(out/"plots"/"search"/"ga_fitness_vs_evaluations.png"),"GA Best Fitness")
    eff=communication_efficiency_rows(method_seed_rows)
    bar(summarize_rows(eff,"method",["accuracy_per_mb","worst_accuracy_per_mb"]),"method","accuracy_per_mb_mean",str(out/"plots"/"efficiency"/"accuracy_per_mb.png"),"Accuracy per MB")
    scatter3(method_seed_rows,"final_accuracy","final_worst_client_accuracy","total_comm_until_stop",str(out/"plots"/"optimization"/"accuracy_fairness_communication_3d.png"),"Accuracy Fairness Communication")


def plot_lp(out,lp_rows):
    flat=flatten_lp(lp_rows)
    scatter(flat,"budget","lambda",str(out/"plots"/"optimization"/"shadow_price_vs_budget.png"),"Shadow Price vs Budget")
    scatter(flat,"budget","loss",str(out/"plots"/"optimization"/"loss_vs_budget.png"),"Loss vs Budget")
    scatter(flat,"budget","stationarity_residual",str(out/"plots"/"optimization"/"kkt_residuals.png"),"KKT Residual")
    scatter3(flat,"budget","loss","lambda",str(out/"plots"/"optimization"/"lp_budget_loss_lambda_3d.png"),"LP Budget Loss Lambda")


def ga_best_history(history):
    best=float("inf")
    rows=[]
    for row in history:
        best=min(best,float(row["fitness"]))
        rows.append({"round":row["evaluation"],"best_fitness":best})
    return rows


def write_report(out,method_summary,best_alpha,ga):
    best=max(method_summary,key=lambda r:r.get("final_accuracy_mean",0))
    text=f"""# Federated Learning Optimization V2 Report Draft

## Main Upgrade

This run uses early-stopped federated learning with organized diagnostics, EDA, baselines, search validation, LP shadow pricing, calibration, and extensive visualization.

## Headline Results

- Best mean accuracy method: `{best['method']}` with accuracy `{best.get('final_accuracy_mean',0):.4f}`.
- Best CVaR alpha selected from sweep: `{best_alpha}`.
- GA best raw vector: `{ga['x']}`.
- GA search evaluations: `{ga['evaluations']}`.

See `reports/overleaf_report_v2.tex` for the final formatted report after report generation.
"""
    (out/"reports"/"report_draft_v2.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    main()
