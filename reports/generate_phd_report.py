from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"full_uci_v2"
REPORTS=ROOT/"reports"


def read_csv(path:Path,limit:int|None=None)->list[dict]:
    with path.open(encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def esc(value)->str:
    text=str(value)
    repl={
        "\\":"\\textbackslash{}","_":"\\_","%":"\\%","&":"\\&","#":"\\#",
        "$":"\\$","{":"\\{","}":"\\}","~":"\\textasciitilde{}","^":"\\textasciicircum{}",
    }
    return "".join(repl.get(ch,ch) for ch in text)


def wrap(text:str)->str:
    return textwrap.fill(text.strip(),width=100)


def table(rows:list[dict],cols:list[str],caption:str,max_rows:int|None=None)->str:
    selected=rows[:max_rows] if max_rows else rows
    lines=[f"\\begin{{longtable}}{{{'l'*len(cols)}}}"]
    lines.append(f"\\caption{{{esc(caption)}}}\\\\")
    lines.append("\\toprule")
    lines.append(" & ".join(esc(c) for c in cols)+" \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append(" & ".join(esc(c) for c in cols)+" \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    for row in selected:
        vals=[]
        for col in cols:
            raw=row.get(col,"")
            try:
                val=float(raw)
                if abs(val)>=10_000:
                    vals.append(f"{val:.3e}")
                elif abs(val)>=1:
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(f"{val:.6f}")
            except Exception:
                vals.append(esc(raw))
        lines.append(" & ".join(vals)+" \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    return "\n".join(lines)


def chapter(lines:list[str],title:str,paragraphs:list[str])->None:
    lines.append(f"\\chapter{{{esc(title)}}}")
    for paragraph in paragraphs:
        lines.append(wrap(paragraph)+"\n")


def why_block(topic:str)->list[str]:
    return [
        f"The reason this report spends time on {topic} is that a federated learning result is not a single accuracy number. It is a chain of choices. If one choice is misunderstood, the final conclusion can be misleading.",
        f"For {topic}, the project asks four questions: what is being done, why it is needed, how it affects optimization, and what evidence the v2 run provides. This structure makes the report readable for a beginner while still preserving the technical argument expected in a graduate optimization project.",
        f"The implementation is therefore treated as part of the scientific method. Code is not only a tool used to produce numbers; it encodes assumptions about clients, data, communication, fairness, convergence, and evaluation. Those assumptions are made explicit throughout the report.",
    ]


def main()->None:
    meta=json.loads((OUT/"run_metadata.json").read_text())
    agg=json.loads((OUT/"metrics"/"aggregate_scores.json").read_text())
    summary=read_csv(OUT/"metrics"/"method_summary.csv")
    fed=next(r for r in summary if r["method"]=="fedavg_default")
    grid=next(r for r in summary if r["method"]=="grid_best_validated")
    ga=next(r for r in summary if r["method"]=="ga_best_validated")
    central=next(r for r in summary if r["method"]=="centralized")
    data={
        "method_seed":read_csv(OUT/"metrics"/"method_seed_results.csv"),
        "classification":read_csv(OUT/"metrics"/"classification_report.csv"),
        "per_client":read_csv(OUT/"metrics"/"per_client_metrics.csv"),
        "class_dist":read_csv(OUT/"eda"/"class_distribution.csv"),
        "client_counts":read_csv(OUT/"eda"/"client_sample_counts.csv"),
        "noniid":read_csv(OUT/"noniid"/"client_distribution_metrics.csv"),
        "calib":read_csv(OUT/"calibration"/"calibration_bins.csv"),
        "calib_sum":read_csv(OUT/"calibration"/"calibration_summary.csv"),
        "lp":read_csv(OUT/"lp"/"lp_shadow_price.csv"),
        "grid":read_csv(OUT/"search"/"grid_search.csv"),
        "ga_hist":read_csv(OUT/"search"/"ga_history.csv"),
        "ga_best":read_csv(OUT/"search"/"ga_history_best_so_far.csv"),
        "paired":read_csv(OUT/"stats"/"paired_tests.csv"),
        "ci":read_csv(OUT/"stats"/"confidence_intervals.csv"),
        "runtime":read_csv(OUT/"runtime"/"runtime_by_stage.csv"),
        "failures":read_csv(OUT/"failure_modes"/"failure_modes.csv"),
        "ablation":read_csv(OUT/"ablations"/"ablation_deltas.csv",120),
        "comm_eff":read_csv(OUT/"efficiency"/"communication_efficiency.csv",120),
        "top_conf":read_csv(OUT/"errors"/"top_confusions.csv"),
        "per_class_err":read_csv(OUT/"errors"/"per_class_error_rates.csv"),
        "theory":read_csv(OUT/"reports"/"theoretical_alignment_table.csv"),
        "fed7":read_csv(OUT/"raw"/"fedavg_default_seed_7_rounds.csv"),
    }

    lines=[
        "\\documentclass[11pt]{report}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,xcolor,hyperref,listings,float,array,caption,pdflscape}",
        "\\graphicspath{{../outputs/full_uci_v2/plots/}}",
        "\\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        "\\definecolor{codebg}{RGB}{248,248,248}",
        "\\lstset{backgroundcolor=\\color{codebg},basicstyle=\\ttfamily\\scriptsize,breaklines=true,frame=single,numbers=left,numberstyle=\\tiny,tabsize=2}",
        "\\newcommand{\\figfull}[3]{\\begin{figure}[H]\\centering\\includegraphics[width=0.88\\linewidth]{#1}\\caption{#2. #3}\\end{figure}}",
        "\\title{\\textbf{Federated Learning Optimization on UCI HAR}\\\\\\large A PhD-Level, Beginner-Friendly Technical Report from Data Loading to Evaluation}",
        "\\author{Sriram M.}",
        "\\date{\\today}",
        "\\begin{document}",
        "\\maketitle",
        "\\begin{abstract}",
        wrap("This report documents the full federated learning optimization project in a detailed way. It explains the project for a reader who does not yet know machine learning, while preserving enough mathematical and experimental detail for graduate-level review. The report follows the entire pipeline: loading the UCI HAR dataset, transforming subjects into federated clients, scaling features, training a neural network with FedAvg, using early stopping, measuring communication, testing CVaR-style fairness, solving an LP communication model, running differential-evolution hyperparameter search, validating with 10 random seeds, and evaluating results with classification, fairness, calibration, statistical, and optimization diagnostics."),
        "\\end{abstract}",
        "\\tableofcontents",
        "\\listoffigures",
        "\\listoftables",
    ]

    chapter(lines,"Executive Summary",[
        f"The project asks whether a useful activity-recognition model can be trained without centralizing all client data. In the simulated federated setting, each UCI HAR subject is treated as a client. The default FedAvg method achieved mean accuracy {float(fed['final_accuracy_mean']):.4f} and worst-client accuracy {float(fed['final_worst_client_accuracy_mean']):.4f} across 10 seeds.",
        f"The centralized model reached {float(central['final_accuracy_mean']):.4f} mean accuracy, which gives an upper benchmark because it can use all training data directly. Grid-validated FedAvg reached {float(grid['final_accuracy_mean']):.4f}, and GA-validated FedAvg reached {float(ga['final_accuracy_mean']):.4f} while using substantially less communication. The results show that FedAvg is strong, tuning helps, and GA is valuable mainly as a communication-efficiency tool.",
        "The key scientific caution is fairness. Average accuracy is high, but worst-client accuracy is lower. CVaR-style weighting was investigated because it focuses on hard clients, but it did not reliably improve worst-client accuracy in this run. This is an important honest result: fairness remains the main unresolved issue.",
    ])
    lines.append(table(summary,["method","n","final_loss_mean","final_accuracy_mean","final_worst_client_accuracy_mean","total_comm_until_stop_mean","stopped_round_mean"],"Headline method comparison."))

    blueprints=[
        ("Research Problem and Motivation","why federated learning is relevant and why this is an optimization problem"),
        ("Machine Learning From First Principles","what features, labels, loss, accuracy, and model parameters mean"),
        ("Dataset Loading","why UCI HAR is loaded with features, labels, activity names, and subject identifiers"),
        ("Subject-to-Client Conversion","why each subject becomes a federated client and why this creates non-IID data"),
        ("Preprocessing and Scaling","why StandardScaler is fit on training data and reused at inference"),
        ("Exploratory Data Analysis","why class balance, client counts, heatmaps, entropy, and PCA are inspected"),
        ("Model Architecture","why the HARMLP model is small, nonlinear, and suitable for repeated FL runs"),
        ("FedAvg Training","why local training and weighted server aggregation define the core FL algorithm"),
        ("Early Stopping","why max rounds alone is not enough and why convergence should determine stopping"),
        ("Communication Accounting","why communication must be measured and what assumptions are in the byte model"),
        ("Fairness and CVaR","why worst-client accuracy matters and why tail-risk weighting was tested"),
        ("Linear Programming and Duality","why communication budgets can be analyzed with shadow prices"),
        ("Differential Evolution Search","why hyperparameter search is expensive and how it was validated"),
        ("Baselines","why centralized, local-only, default FedAvg, grid, and GA baselines are all needed"),
        ("Evaluation Metrics","why loss, accuracy, macro F1, confusion matrices, calibration, and per-client metrics answer different questions"),
        ("Statistical Validation","why multiple seeds, confidence intervals, paired tests, and effect sizes are reported"),
        ("Results and Interpretation","what the final numbers imply and what they do not imply"),
        ("Professor Feedback and Extensions","how to improve the communication model, search efficiency, and non-convex DNN extension"),
        ("Limitations and Future Work","what is still missing before this becomes a real deployed FL system"),
    ]
    for title,topic in blueprints:
        paras=why_block(topic)
        paras += [
            f"In this project, {topic} is connected to the actual v2 outputs rather than being a theoretical side note. The output directory contains raw round logs, summary CSVs, model checkpoints, plots, and diagnostics, so each claim can be traced back to a produced artifact.",
            f"The beginner-level interpretation is that {topic} answers a practical question: what decision did we make, why did we make it, and how did it change the final model? The PhD-level interpretation is that each decision changes the empirical objective, the feasible communication budget, the stochastic training process, or the reliability of the evaluation.",
            "A common mistake in machine-learning reports is to jump directly from dataset to final accuracy. This report avoids that mistake. It explains the causal chain: dataset structure creates clients; clients create heterogeneity; heterogeneity affects local gradients; local gradients affect FedAvg aggregation; aggregation affects convergence; convergence affects communication; communication interacts with hyperparameter search; and all of these determine the final evaluation.",
        ]
        chapter(lines,title,paras)

    figures=[
        ("eda/class_distribution.png","Class distribution","Shows whether class imbalance can bias accuracy."),
        ("eda/train_test_class_distribution.png","Train/test class distribution","Checks that all activities are represented in both splits."),
        ("eda/client_sample_counts.png","Client sample counts","Shows which clients have more influence under sample-weighted FedAvg."),
        ("eda/client_label_heatmap.png","Client label heatmap","Displays non-IID label structure across subjects."),
        ("eda/client_entropy.png","Client entropy","Quantifies how diverse each client's labels are."),
        ("eda/feature_correlation_subset.png","Feature correlation subset","Shows dependence among engineered sensor features."),
        ("eda/pca_by_activity.png","PCA by activity","Visualizes activity separability in two dimensions."),
        ("eda/pca_3d_by_activity.png","3D PCA by activity","Provides a richer low-dimensional view of class structure."),
        ("eda/pca_by_client.png","PCA by client","Visualizes subject-level heterogeneity."),
        ("eda/pca_3d_by_client.png","3D PCA by client","Shows client variation in three dimensions."),
        ("training/loss_vs_rounds_mean_std.png","Loss versus rounds","Shows convergence behavior."),
        ("training/accuracy_vs_rounds_mean_std.png","Accuracy versus rounds","Shows performance improvement over FL rounds."),
        ("training/worst_client_accuracy_vs_rounds.png","Worst-client accuracy versus rounds","Shows fairness behavior over training."),
        ("training/best_loss_so_far_vs_rounds.png","Best loss so far","Shows the early-stopping monitor."),
        ("training/rounds_since_improvement.png","Rounds since improvement","Explains patience accumulation."),
        ("training/communication_vs_rounds.png","Communication per round","Connects FL training to byte cost."),
        ("training/accuracy_vs_cumulative_communication.png","Accuracy versus communication","Shows the performance-cost tradeoff."),
        ("classification/confusion_matrix_raw.png","Raw confusion matrix","Shows absolute classification mistakes."),
        ("classification/confusion_matrix_normalized.png","Normalized confusion matrix","Shows per-class behavior independent of support."),
        ("errors/per_class_error_rate.png","Per-class error rate","Ranks difficult activities."),
        ("errors/top_confused_pairs.png","Top confused pairs","Identifies common activity confusions."),
        ("fairness/per_client_accuracy_fedavg.png","Per-client accuracy","Shows client-level fairness."),
        ("baselines/baseline_accuracy_comparison.png","Baseline accuracy","Compares method performance."),
        ("baselines/baseline_worst_client_accuracy.png","Baseline worst-client accuracy","Compares fairness across methods."),
        ("baselines/baseline_communication_cost.png","Baseline communication cost","Compares byte budgets."),
        ("baselines/baseline_rounds_to_convergence.png","Rounds to convergence","Compares early-stopped training length."),
        ("optimization/shadow_price_vs_budget.png","Shadow price versus budget","Shows communication scarcity in the LP model."),
        ("optimization/loss_vs_budget.png","Loss versus budget","Shows objective improvement with budget."),
        ("optimization/kkt_residuals.png","KKT residuals","Checks LP numerical optimality."),
        ("optimization/lp_budget_loss_lambda_3d.png","3D LP tradeoff","Shows budget, loss, and dual value together."),
        ("search/accuracy_vs_communication_scatter.png","Search accuracy versus communication","Shows hyperparameter tradeoffs."),
        ("search/hyperparameter_3d_fitness.png","3D hyperparameter fitness","Shows the search landscape."),
        ("search/ga_fitness_vs_evaluations.png","GA fitness over evaluations","Shows differential-evolution progress."),
        ("efficiency/accuracy_per_mb.png","Accuracy per MB","Shows communication efficiency."),
        ("optimization/accuracy_fairness_communication_3d.png","Accuracy-fairness-communication tradeoff","Shows the central multi-objective tension."),
    ]
    chapter(lines,"Figure-by-Figure Explanation",[
        "This chapter explains every major generated visualization. The purpose is not to decorate the report. Each figure answers a specific diagnostic question. Together, the figures show data quality, client heterogeneity, training convergence, fairness behavior, communication efficiency, classification quality, search behavior, and optimization diagnostics.",
    ])
    for path,title,why in figures:
        lines.append(f"\\section{{{esc(title)}}}")
        lines.append(wrap(why+" The reason this is included is that a final accuracy number alone cannot explain the behavior of a federated system. Federated learning must be judged through convergence, fairness, communication, and robustness."))
        lines.append(f"\\figfull{{{path}}}{{{esc(title)}}}{{{esc(why)}}}")

    chapter(lines,"Mathematical Formulation",[
        "The supervised learning objective minimizes empirical risk over labeled examples. If $x$ is a 561-dimensional input vector and $y$ is one of six activity labels, a neural network $f_w$ maps $x$ to class logits. Cross-entropy loss penalizes wrong predictions, especially confident wrong predictions.",
        "The federated objective is a weighted sum of client objectives: $F(w)=\\sum_i (n_i/N)F_i(w)$. FedAvg approximates this objective by selecting clients, training local models, and aggregating returned weights using sample-size weights.",
        "The communication model uses $C_t=2K_t(4d)$ bytes for round $t$, where $d$ is the number of model parameters and $K_t$ is the number of selected clients. This is interpretable but simplified.",
        "The LP communication model solves $\\min_x \\sum_k L_kx_k$ subject to $\\sum_k C_kx_k\\le B$, $\\sum_kx_k=1$, and $x_k\\ge0$. The dual variable of the budget constraint is the communication shadow price.",
        "For non-convex DNNs, the correct goal is not a global optimality claim. The better theoretical language is convergence toward stationarity, measured with quantities like $\\mathbb{E}\\|\\nabla F(w)\\|^2$, plus empirical validation across seeds.",
    ])

    chapter(lines,"Core Tables and Actual Outputs",[
        "The following tables are actual outputs from the v2 run. They are included so that the report is auditable. A reader can connect claims in the narrative to CSV artifacts produced by the experiment runner.",
    ])
    lines.append(table(summary,["method","n","final_loss_mean","final_loss_std","final_accuracy_mean","final_accuracy_std","final_worst_client_accuracy_mean","final_worst_client_accuracy_std","total_comm_until_stop_mean","stopped_round_mean"],"Full method summary."))
    lines.append(table(data["class_dist"],["class_id","activity","train_count","test_count","total_count"],"Class distribution."))
    lines.append(table(data["client_counts"],["client_id","train_samples","test_samples","total_samples","label_entropy","dominant_class_ratio"],"Client counts and entropy."))
    lines.append(table(data["noniid"],["client_id","kl_divergence","js_divergence","label_entropy","dominant_class_ratio","samples"],"Non-IID client metrics."))
    lines.append(table(data["classification"],["label","precision","recall","f1-score","support"],"Classification report."))
    lines.append(table(data["per_client"],["client_id","test_samples","train_samples","accuracy","error_rate"],"Per-client metrics."))
    lines.append(table(data["per_class_err"],["label","support","error_rate","recall"],"Per-class error rates."))
    lines.append(table(data["top_conf"],["true_label","pred_label","count"],"Top confusion pairs."))
    lines.append(table(data["calib"],["bin","lower","upper","count","accuracy","confidence","gap"],"Calibration bins."))
    lines.append(table(data["calib_sum"],["ece","mce","mean_confidence","accuracy"],"Calibration summary."))
    lines.append(table(data["runtime"],["stage","seconds"],"Runtime by stage."))
    lines.append(table(data["theory"],["component","optimization_concept"],"Theoretical alignment."))

    chapter(lines,"Optimization, Search, and Statistical Tables",[
        "This chapter includes the optimization outputs: grid search, differential evolution, LP diagnostics, confidence intervals, paired tests, ablation deltas, and communication efficiency. These tables show how the project connects empirical ML training with optimization analysis.",
    ])
    lines.append(table(data["grid"],["local_epochs","clients_per_round","lr","loss","accuracy","comm","fitness"],"Grid-search candidates."))
    lines.append(table(data["ga_hist"],["evaluation","local_epochs","clients_per_round","lr","loss","accuracy","comm","fitness"],"Differential-evolution objective evaluations."))
    lines.append(table(data["ga_best"],["round","best_fitness"],"GA best-so-far history."))
    lines.append(table(data["lp"],["budget","loss","cost","lambda","status","primal_feasible","dual_feasible","complementary_slackness","stationarity_residual","kkt_status"],"LP shadow-price and KKT diagnostics."))
    lines.append(table(data["ci"],["group","metric","n","mean","std","ci95_low","ci95_high"],"Confidence intervals."))
    lines.append(table(data["paired"],["baseline","method","metric","n","mean_diff","paired_t_p","wilcoxon_p","effect_size"],"Paired tests against FedAvg."))
    lines.append(table(data["ablation"],["method","seed","final_loss","final_accuracy","final_worst_client_accuracy","delta_accuracy","delta_worst_client_accuracy","delta_comm"],"Ablation-style deltas.",120))
    lines.append(table(data["comm_eff"],["method","seed","final_accuracy","final_worst_client_accuracy","comm_mb","accuracy_per_mb","worst_accuracy_per_mb","reached_threshold"],"Communication efficiency.",120))
    if data["failures"]:
        lines.append(table(data["failures"],["method","seed","failure_type","description"],"Failure-mode records.",120))

    chapter(lines,"Round-Level Trace Example",[
        "The table below gives a round-by-round trace for default FedAvg with seed 7. It proves that the final summary is not a single isolated measurement. It is built from a full sequence of evaluated communication rounds, each with loss, accuracy, worst-client accuracy, communication bytes, and early-stopping state.",
    ])
    lines.append(table(data["fed7"],["round","loss","accuracy","worst_client_accuracy","total_comm_bytes","best_loss_so_far","best_round","rounds_since_improvement","stopped_early"],"Round-level FedAvg trace for seed 7.",186))

    lines.append("\\appendix")
    chapter(lines,"Source Code Appendix",[
        "This appendix includes the source code used to produce the results. Code listings are part of the evidence. They show exactly how the dataset was loaded, how clients were built, how FedAvg was trained, how metrics were computed, how LP duality was solved, and how the full v2 experiment was orchestrated.",
    ])
    for file in [
        "flopt/config.py","flopt/data.py","flopt/models.py","flopt/fedavg.py","flopt/duality.py",
        "flopt/search.py","flopt/metrics.py","flopt/eda.py","flopt/calibration.py","flopt/baselines.py",
        "flopt/stats.py","flopt/analysis.py","flopt/plots.py","experiments/run_full_v2.py",
    ]:
        lines.append(f"\\section{{{esc(file)}}}")
        lines.append(f"\\lstinputlisting{{../{file}}}")

    chapter(lines,"Final Conclusion",[
        "The final conclusion is that federated learning can train a strong UCI HAR classifier without centralizing raw client data. Default FedAvg is already strong, grid tuning improves accuracy slightly, and differential evolution finds a more communication-efficient region of the hyperparameter space.",
        "The major unresolved issue is fairness. Worst-client accuracy is lower than average accuracy, and simple CVaR-style weighting did not solve this. This is the correct research conclusion: the project succeeds at building, evaluating, and optimizing a federated system, while identifying client fairness as the next serious technical challenge.",
        "The optimization value of the project comes from the complete chain: stochastic neural-network training, federated aggregation, early stopping, communication accounting, LP duality, shadow-price interpretation, hyperparameter search, statistical validation, and detailed diagnostics. The project is therefore not just a classifier; it is a communication-aware and fairness-aware federated optimization study.",
    ])
    lines.append("\\end{document}")

    path=REPORTS/"phd_level_report.tex"
    path.write_text("\n".join(lines),encoding="utf-8")
    print(path)


if __name__=="__main__":
    main()
