"""Generate every figure for the NeurIPS inference report directly from project CSVs.

Outputs go to reports/inference_neurips/figures/. Every figure is grounded in a
specific source CSV/JSON listed at the top of its function. Matplotlib uses the
'Agg' backend so this works headless.
"""

from __future__ import annotations
import json
import os
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/sriramm/Cursor_Projects/federated-learning-optimization")
OUT = ROOT / "outputs" / "full_mimic_iv_training"
PROP = ROOT / "outputs" / "full_mimic_iv_proposal_alignment"
PRE = ROOT / "outputs" / "full_mimic_iv"
FIG = ROOT / "reports" / "inference_neurips" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def fig_client_mortality_and_size():
    """Source: outputs/full_mimic_iv/eda/client_summary.csv"""
    df = pd.read_csv(PRE / "eda" / "client_summary.csv")
    df = df.sort_values("rows", ascending=False).reset_index(drop=True)

    short = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Neuro Intermediate": "Neuro Int",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
        "Neuro Stepdown": "Neuro Step",
    }
    names = [short[n] for n in df["client_name"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(names, df["rows"], color="#3b6fb5")
    axes[0].set_ylabel("ICU stays (n)")
    axes[0].set_title("(a) Client size distribution: 15.7$\\times$ spread")
    axes[0].tick_params(axis="x", rotation=35)
    for i, v in enumerate(df["rows"]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(names, df["mortality_rate"] * 100, color="#c0392b")
    axes[1].set_ylabel("24h mortality rate (\\%)")
    axes[1].set_title("(b) Per-client mortality: 8.2$\\times$ spread")
    axes[1].tick_params(axis="x", rotation=35)
    for i, v in enumerate(df["mortality_rate"]):
        axes[1].text(i, v * 100, f"{v*100:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save(fig, "fig_client_mortality_and_size.png")


def fig_size_vs_mortality_scatter():
    """Source: outputs/full_mimic_iv/eda/client_summary.csv"""
    df = pd.read_csv(PRE / "eda" / "client_summary.csv")
    short = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Neuro Intermediate": "Neuro Int",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
        "Neuro Stepdown": "Neuro Step",
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["rows"], df["mortality_rate"] * 100, s=80, c="#2c3e50", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(
            short[row["client_name"]],
            (row["rows"], row["mortality_rate"] * 100),
            xytext=(6, 4), textcoords="offset points", fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("ICU stays (log scale)")
    ax.set_ylabel("24h mortality rate (\\%)")
    ax.set_title("Client size vs. mortality rate. The smallest units carry the most extreme rates.")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_size_vs_mortality_scatter.png")


def fig_fedprox_mu_sweep():
    """Source: outputs/full_mimic_iv_proposal_alignment/metrics/fedprox_vs_fedavg_summary.csv"""
    df = pd.read_csv(PROP / "metrics" / "fedprox_vs_fedavg_summary.csv")
    fdf = df[df["method"].str.startswith("fedprox_mu_")].copy()
    mu_map = {
        "fedprox_mu_0p0": 0.0, "fedprox_mu_0p001": 0.001,
        "fedprox_mu_0p01": 0.01, "fedprox_mu_0p1": 0.1,
    }
    fdf["mu"] = fdf["method"].map(mu_map)
    fdf = fdf.sort_values("mu").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    axes[0].errorbar(fdf["mu"], fdf["final_auprc_mean"],
                     yerr=fdf["final_auprc_std"], marker="o",
                     color="#1f77b4", capsize=4, lw=2, markersize=8)
    axes[0].set_xscale("symlog", linthresh=1e-4)
    axes[0].set_xlabel("FedProx $\\mu$")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title("(a) AUPRC: monotone increase with $\\mu$")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(fdf["mu"], fdf["final_worst_client_recall_mean"],
                     yerr=fdf["final_worst_client_recall_std"], marker="s",
                     color="#d62728", capsize=4, lw=2, markersize=8)
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_xlabel("FedProx $\\mu$")
    axes[1].set_ylabel("Worst-client recall")
    axes[1].set_title("(b) Worst-recall: $0.19 \\to 0.50$")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(fdf["mu"], fdf["final_accuracy_mean"],
                     yerr=fdf["final_accuracy_std"], marker="^",
                     color="#2ca02c", capsize=4, lw=2, markersize=8)
    axes[2].set_xscale("symlog", linthresh=1e-4)
    axes[2].set_xlabel("FedProx $\\mu$")
    axes[2].set_ylabel("Mean accuracy")
    axes[2].set_title("(c) Accuracy: gentle drop ($0.89 \\to 0.84$)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("FedProx $\\mu$ sweep on MLP backbone (10 seeds, mean$\\pm$std).", y=1.02)
    fig.tight_layout()
    save(fig, "fig_fedprox_mu_sweep.png")


def fig_cvar_alpha_collapse():
    """Source: outputs/full_mimic_iv_training/metrics/method_summary.csv"""
    df = pd.read_csv(OUT / "metrics" / "method_summary.csv")
    cdf = df[df["method"].str.startswith("cvar_")].copy()
    alpha_map = {"cvar_0": 0.0, "cvar_0.5": 0.5, "cvar_0.75": 0.75,
                 "cvar_0.9": 0.9, "cvar_0.95": 0.95}
    cdf["alpha"] = cdf["method"].map(alpha_map)
    cdf = cdf.sort_values("alpha").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].errorbar(cdf["alpha"], cdf["final_auprc_mean"],
                     yerr=cdf["final_auprc_std"], marker="o",
                     color="#1f77b4", capsize=4, lw=2, markersize=8)
    axes[0].set_xlabel("CVaR $\\alpha$")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title("(a) AUPRC vs $\\alpha$: $\\alpha\\!\\geq\\!0.75$ all collapse to identical means")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvspan(0.74, 0.96, alpha=0.15, color="#d62728",
                    label="Tail collapse region")
    axes[0].legend()

    axes[1].errorbar(cdf["alpha"], cdf["final_worst_client_recall_mean"],
                     yerr=cdf["final_worst_client_recall_std"], marker="s",
                     color="#d62728", capsize=4, lw=2, markersize=8)
    axes[1].set_xlabel("CVaR $\\alpha$")
    axes[1].set_ylabel("Worst-client recall")
    axes[1].set_title("(b) Worst-recall vs $\\alpha$: same collapse")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvspan(0.74, 0.96, alpha=0.15, color="#d62728")

    fig.suptitle("CVaR $\\alpha$ sweep on MLP, 9 natural clients, $C\\!=\\!5$ per round.", y=1.02)
    fig.tight_layout()
    save(fig, "fig_cvar_alpha_collapse.png")


def fig_pareto_auprc_vs_comm():
    """Sources: outputs/full_mimic_iv_training/metrics/method_summary.csv +
    outputs/full_mimic_iv_proposal_alignment/metrics/proposal_method_summary.csv"""
    old = pd.read_csv(OUT / "metrics" / "method_summary.csv")
    prop = pd.read_csv(PROP / "metrics" / "proposal_method_summary.csv")

    fig, ax = plt.subplots(figsize=(8.5, 5))

    families = [
        ("MLP FedAvg / CVaR", old[~old["method"].isin(["centralized", "local_only"])
                                   & old["method"].str.startswith(("fedavg", "cvar"))],
         "o", "#1f77b4", 60),
        ("MLP search-best", old[old["method"].isin(["grid_best_validated", "ga_best_validated"])],
         "*", "#9467bd", 200),
        ("MLP centralized/local-only", old[old["method"].isin(["centralized", "local_only"])],
         "X", "#7f7f7f", 110),
        ("MLP FedProx ($\\mu$ sweep)", prop[prop["method"].str.startswith("fedprox_mu_")],
         "P", "#d62728", 100),
        ("Logistic FedAvg / CVaR / FedProx", prop[prop["method"].str.startswith("logreg_")],
         "s", "#2ca02c", 50),
        ("Dirichlet $\\beta$ controls", prop[prop["method"].str.startswith("dirichlet_")],
         "D", "#ff7f0e", 40),
    ]
    for label, sub, marker, color, size in families:
        if sub.empty:
            continue
        x = sub["total_comm_until_stop_mean"].clip(lower=1.0)
        y = sub["final_auprc_mean"]
        ax.scatter(x, y, label=label, marker=marker, c=color, s=size,
                   alpha=0.85, edgecolors="black", linewidth=0.5, zorder=3)

    fp01 = prop[prop["method"] == "fedprox_mu_0p1"].iloc[0]
    ax.annotate(
        "MLP FedProx $\\mu\\!=\\!0.1$\nAUPRC 0.665, recall$_{\\min}$ 0.50",
        xy=(fp01["total_comm_until_stop_mean"], fp01["final_auprc_mean"]),
        xytext=(8e7, 0.685), fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )
    lp01 = prop[prop["method"] == "logreg_fedprox_mu_0p1"].iloc[0]
    ax.annotate(
        "Logistic FedProx $\\mu\\!=\\!0.1$\n148$\\times$ cheaper, AUPRC 0.612",
        xy=(lp01["total_comm_until_stop_mean"], lp01["final_auprc_mean"]),
        xytext=(2.5e6, 0.555), fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Total upload$+$download bytes until early-stop (log scale)")
    ax.set_ylabel("AUPRC (mean of 10 seeds)")
    ax.set_title("Pareto frontier: AUPRC vs.\\ communication. The frontier is bimodal.")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, ncol=1)
    fig.tight_layout()
    save(fig, "fig_pareto_auprc_vs_comm.png")


def fig_fairness_frontier():
    """Sources: method_summary.csv + proposal_method_summary.csv"""
    old = pd.read_csv(OUT / "metrics" / "method_summary.csv")
    prop = pd.read_csv(PROP / "metrics" / "proposal_method_summary.csv")

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    families = [
        ("MLP FedAvg / CVaR / Centralized / Local",
         old[~old["method"].str.startswith(("dirichlet_",))], "o", "#1f77b4", 60),
        ("MLP FedProx",
         prop[prop["method"].str.startswith("fedprox_mu_")], "P", "#d62728", 90),
        ("Logistic methods",
         prop[prop["method"].str.startswith("logreg_")], "s", "#2ca02c", 50),
        ("Dirichlet $\\beta\\!=\\!0.1$ (failure)",
         prop[prop["method"].str.startswith("dirichlet_beta_0p1_")], "v", "#8c564b", 70),
        ("Dirichlet $\\beta\\!=\\!\\infty$ (IID ceiling)",
         prop[prop["method"].str.startswith("dirichlet_beta_infinity_")], "^", "#17becf", 70),
    ]
    for label, sub, marker, color, size in families:
        if sub.empty:
            continue
        ax.scatter(sub["final_accuracy_mean"], sub["final_worst_client_recall_mean"],
                   label=label, marker=marker, c=color, s=size,
                   alpha=0.85, edgecolors="black", linewidth=0.5, zorder=3)

    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Worst-client recall (death sensitivity at the worst ICU)")
    ax.set_title("Fairness frontier. The two corners are not simultaneously reachable.")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=8)

    ax.annotate("Centralized:\nhigh-acc, low-fairness",
                xy=(0.908, 0.16), xytext=(0.86, 0.05),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    ax.annotate("FedProx $\\mu\\!=\\!0.1$:\nthe balance corner",
                xy=(0.842, 0.50), xytext=(0.74, 0.55),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    ax.annotate("Dirichlet $\\beta\\!=\\!\\infty$:\nfairness ceiling",
                xy=(0.813, 0.726), xytext=(0.74, 0.78),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    ax.annotate("Dirichlet $\\beta\\!=\\!0.1$:\ncatastrophic floor",
                xy=(0.66, 0.0), xytext=(0.59, 0.07),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    fig.tight_layout()
    save(fig, "fig_fairness_frontier.png")


def fig_dirichlet_trends():
    """Source: proposal_method_summary.csv (dirichlet rows)"""
    df = pd.read_csv(PROP / "metrics" / "proposal_method_summary.csv")
    df = df[df["method"].str.startswith("dirichlet_")].copy()

    def _beta(s):
        if "0p1" in s: return 0.1
        if "0p5" in s: return 0.5
        if "1p0" in s: return 1.0
        return float("inf")

    def _algo(s):
        if "fedprox" in s: return "FedProx"
        if "cvar" in s: return "CVaR"
        return "FedAvg"

    df["beta"] = df["method"].apply(_beta)
    df["algo"] = df["method"].apply(_algo)
    df = df.sort_values(["algo", "beta"]).reset_index(drop=True)
    df["beta_x"] = df["beta"].replace(np.inf, 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {"FedAvg": "#1f77b4", "FedProx": "#d62728", "CVaR": "#2ca02c"}
    metrics = [
        ("final_auprc_mean", "AUPRC", "(a) AUPRC vs $\\beta$"),
        ("final_worst_client_recall_mean", "Worst-client recall",
         "(b) Worst-recall: 0 at $\\beta\\!=\\!0.1$, 0.73 at $\\beta\\!=\\!\\infty$"),
        ("final_accuracy_mean", "Mean accuracy",
         "(c) Accuracy peaks near $\\beta\\!=\\!1$"),
    ]
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for algo in ["FedAvg", "FedProx", "CVaR"]:
            sub = df[df["algo"] == algo]
            if sub.empty:
                continue
            ax.plot(sub["beta_x"], sub[col], "o-", color=colors[algo],
                    label=algo, lw=2, markersize=8)
        ax.set_xticks([0.1, 0.5, 1.0, 5.0])
        ax.set_xticklabels(["0.1", "0.5", "1.0", "$\\infty$"])
        ax.set_xlabel("Dirichlet $\\beta$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Dirichlet study (K=30 synthetic clients). Heterogeneity dominates everything.", y=1.02)
    fig.tight_layout()
    save(fig, "fig_dirichlet_trends.png")


def fig_ga_convergence():
    """Sources: full_mimic_iv_training/search/ga_history.csv +
    proposal_alignment/search/proposal_ga_best_so_far.csv +
    grid_search.csv (for grid baselines)"""
    old_ga_full = pd.read_csv(OUT / "search" / "ga_history.csv")
    old_ga_full["best_so_far"] = old_ga_full["fitness"].cummin()
    old_grid = pd.read_csv(OUT / "search" / "grid_search.csv")
    old_grid_best = old_grid["fitness"].min()

    prop_best = pd.read_csv(PROP / "search" / "proposal_ga_best_so_far.csv")
    prop_grid = pd.read_csv(PROP / "search" / "proposal_grid_search.csv")
    prop_grid_best = prop_grid["fitness"].min()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(old_ga_full["evaluation"], old_ga_full["best_so_far"],
                 "-o", color="#1f77b4", lw=2, markersize=4, label="GA best so far")
    axes[0].axhline(old_grid_best, color="#d62728", lw=2, ls="--",
                    label=f"Grid best ({old_grid_best:.3f})")
    axes[0].set_xlabel("Evaluations")
    axes[0].set_ylabel("Fitness (lower = better)")
    axes[0].set_title("(a) Old run (MLP, 180-round search)\nGA reaches 1.824 in 31 evals")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(prop_best["round"], prop_best["best_fitness"],
                 "-o", color="#1f77b4", lw=2, markersize=4, label="GA best so far")
    axes[1].axhline(prop_grid_best, color="#d62728", lw=2, ls="--",
                    label=f"Grid best ({prop_grid_best:.4f})")
    axes[1].set_xlabel("Evaluations")
    axes[1].set_ylabel("Fitness (lower = better)")
    axes[1].set_title("(b) Proposal run (logistic, 30-round search)\nGA beats grid by eval $\\sim$73")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle("Differential-evolution search convergence vs. grid baseline.", y=1.02)
    fig.tight_layout()
    save(fig, "fig_ga_convergence.png")


def fig_sparsity_compression():
    """Source: outputs/full_mimic_iv_proposal_alignment/lp/sparsity_summary.csv"""
    df = pd.read_csv(PROP / "lp" / "sparsity_summary.csv")
    pivot = df.pivot_table(index="topk_fraction", columns="mu",
                           values="compression_ratio_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    fractions = sorted(df["topk_fraction"].unique())
    mus = sorted(df["mu"].unique())
    width = 0.18
    x = np.arange(len(fractions))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(mus)))
    for i, mu in enumerate(mus):
        vals = [pivot.loc[f, mu] for f in fractions]
        axes[0].bar(x + i * width, vals, width=width, color=cmap[i],
                    label=f"$\\mu={mu}$", edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x + width * (len(mus) - 1) / 2)
    axes[0].set_xticklabels([f"{int(f*100)}\\%" for f in fractions])
    axes[0].set_xlabel("Top-$k$ fraction kept")
    axes[0].set_ylabel("Sparse / dense byte ratio (lower = more compression)")
    axes[0].set_title("(a) Compression ratio by top-$k$ and $\\mu$.\n5\\% keep $\\Rightarrow$ 10$\\times$ cheaper.")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].axhline(1.0, color="black", lw=0.7, ls=":")

    nonzero = df.drop_duplicates("mu")[["mu", "l0_fraction_nonzero_mean"]]
    nonzero = nonzero.sort_values("mu")
    axes[1].bar(nonzero["mu"].astype(str), nonzero["l0_fraction_nonzero_mean"] * 100,
                color="#c0392b", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("FedProx $\\mu$")
    axes[1].set_ylabel("\\% of update entries that are nonzero")
    axes[1].set_title("(b) Updates are 95.8--97.7\\% dense.\n$\\mu$ slightly densifies updates.")
    axes[1].set_ylim(94, 99)
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(nonzero["l0_fraction_nonzero_mean"] * 100):
        axes[1].text(i, v + 0.05, f"{v:.2f}\\%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_sparsity_compression.png")


def fig_lp_shadow_price():
    """Sources: full_mimic_iv_training/lp/lp_shadow_price.csv (dense, MLP) +
    full_mimic_iv_proposal_alignment/lp/sparsity_lp_shadow_price.csv (sparse-aware)"""
    dense = pd.read_csv(OUT / "lp" / "lp_shadow_price.csv")
    sparse = pd.read_csv(PROP / "lp" / "sparsity_lp_shadow_price.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(dense["budget"], dense["loss"], "-o", color="#1f77b4",
                 lw=2, markersize=6, label="Dense LP (old, MLP)")
    axes[0].plot(sparse["budget"], sparse["loss"], "-s", color="#d62728",
                 lw=2, markersize=6, label="Sparsity-aware LP (proposal)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Communication budget (bytes, log)")
    axes[0].set_ylabel("LP-optimal feasible loss")
    axes[0].set_title("(a) Loss--budget Pareto front.\nLoss drops only inside the active region.")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(dense["budget"], dense["lambda"].clip(lower=1e-22), "-o",
                 color="#1f77b4", lw=2, markersize=6, label="Dense LP $\\lambda$")
    axes[1].plot(sparse["budget"], sparse["lambda"].clip(lower=1e-22), "-s",
                 color="#d62728", lw=2, markersize=6, label="Sparsity-aware LP $\\lambda$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Communication budget (bytes, log)")
    axes[1].set_ylabel("Shadow price $\\lambda$ (log)")
    axes[1].set_title("(b) Shadow price collapses by $\\sim$13 orders of magnitude\nas budget exits the active region.")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "fig_lp_shadow_price.png")


def fig_loss_landscape_1d():
    """Sources: outputs/full_mimic_iv_proposal_alignment/landscape/{logreg,mlp}_1d_loss_curve.csv"""
    log = pd.read_csv(PROP / "landscape" / "logreg_1d_loss_curve.csv")
    mlp = pd.read_csv(PROP / "landscape" / "mlp_1d_loss_curve.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log["alpha"], log["loss"], "-o", color="#2ca02c", lw=2,
            markersize=4, label=f"Logistic (min $\\approx$ {log['loss'].min():.3f} at $\\alpha\\!=\\!{log.loc[log['loss'].idxmin(),'alpha']:.2f}$)")
    ax.plot(mlp["alpha"], mlp["loss"], "-s", color="#d62728", lw=2,
            markersize=4, label=f"MLP (min $\\approx$ {mlp['loss'].min():.3f} at $\\alpha\\!=\\!{mlp.loc[mlp['loss'].idxmin(),'alpha']:.2f}$)")
    ax.axvline(0.0, color="black", lw=0.6, ls=":")
    ax.text(0.0, ax.get_ylim()[1] * 0.95, " init", fontsize=8, ha="left", va="top")
    ax.axvline(1.0, color="black", lw=0.6, ls=":")
    ax.text(1.0, ax.get_ylim()[1] * 0.95, " final", fontsize=8, ha="left", va="top")
    ax.set_xlabel("Interpolation $\\alpha$ (init $=0$, final $=1$)")
    ax.set_ylabel("Loss on stratified 5,000-row validation")
    ax.set_title("1D linear-interpolation loss curves.\nLogistic is convex; MLP is non-convex but the basin is smooth and wide.")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_loss_landscape_1d.png")


def fig_loss_landscape_2d():
    """Sources: full_mimic_iv_proposal_alignment/landscape/{logreg,mlp}_2d_loss_surface.csv"""
    log = pd.read_csv(PROP / "landscape" / "logreg_2d_loss_surface.csv")
    mlp = pd.read_csv(PROP / "landscape" / "mlp_2d_loss_surface.csv")

    def _grid(df):
        col_x = next((c for c in ["alpha1", "d1", "x"] if c in df.columns), None)
        col_y = next((c for c in ["alpha2", "d2", "y"] if c in df.columns), None)
        d1 = sorted(df[col_x].unique())
        d2 = sorted(df[col_y].unique())
        Z = df.pivot_table(index=col_y, columns=col_x, values="loss").values
        return np.array(d1), np.array(d2), Z

    d1l, d2l, Zl = _grid(log)
    d1m, d2m, Zm = _grid(mlp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im0 = axes[0].pcolormesh(d1l, d2l, Zl, cmap="viridis", shading="auto")
    axes[0].set_title(f"(a) Logistic 2D landscape\n(min loss $=${Zl.min():.3f}, max $=${Zl.max():.3f})")
    axes[0].set_xlabel("Direction 1 ($\\alpha_1$)")
    axes[0].set_ylabel("Direction 2 ($\\alpha_2$)")
    axes[0].plot(0, 0, "rx", markersize=10, mew=2, label="trained model")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im0, ax=axes[0], label="Loss")

    im1 = axes[1].pcolormesh(d1m, d2m, Zm, cmap="viridis", shading="auto")
    axes[1].set_title(f"(b) MLP 2D landscape\n(min loss $=${Zm.min():.3f}, max $=${Zm.max():.3f})")
    axes[1].set_xlabel("Direction 1 ($\\alpha_1$)")
    axes[1].set_ylabel("Direction 2 ($\\alpha_2$)")
    axes[1].plot(0, 0, "rx", markersize=10, mew=2, label="trained model")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=axes[1], label="Loss")
    fig.tight_layout()
    save(fig, "fig_loss_landscape_2d.png")


def fig_per_client_clinical():
    """Source: outputs/full_mimic_iv_training/metrics/per_client_clinical_metrics.csv"""
    df = pd.read_csv(OUT / "metrics" / "per_client_clinical_metrics.csv")
    short = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Neuro Intermediate": "Neuro Int",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
        "Neuro Stepdown": "Neuro Step",
    }
    df["short"] = df["client_name"].map(short)
    df = df.sort_values("auprc", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].bar(df["short"], df["auprc"], color="#3b6fb5", edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title("(a) Per-client AUPRC, 2.3$\\times$ spread")
    axes[0].tick_params(axis="x", rotation=35)
    for i, v in enumerate(df["auprc"]):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(df["short"], df["sensitivity"], color="#c0392b", edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Sensitivity (death recall)")
    axes[1].set_title("(b) Per-client recall on deaths\nNeuro Int floor: 0.40")
    axes[1].tick_params(axis="x", rotation=35)
    for i, v in enumerate(df["sensitivity"]):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    axes[2].bar(df["short"], df["accuracy"], color="#27ae60", edgecolor="black", linewidth=0.4)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("(c) Per-client accuracy\nDriven by class imbalance")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set_ylim(0.7, 1.0)
    for i, v in enumerate(df["accuracy"]):
        axes[2].text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Per-client clinical metrics on the best validated MLP (best_validated row).", y=1.02)
    fig.tight_layout()
    save(fig, "fig_per_client_clinical.png")


def fig_local_only_vs_federated():
    """Sources: outputs/full_mimic_iv_training/metrics/per_client_clinical_metrics.csv (federated)
    and outputs/full_mimic_iv_training/baselines/local_only_clients.csv (local-only)"""
    fed = pd.read_csv(OUT / "metrics" / "per_client_clinical_metrics.csv")
    loc = pd.read_csv(OUT / "baselines" / "local_only_clients.csv")
    short = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Neuro Intermediate": "Neuro Int",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
        "Neuro Stepdown": "Neuro Step",
    }
    fed = fed.copy(); loc = loc.copy()
    fed["short"] = fed["client_name"].map(short)
    loc["short"] = loc["client_name"].map(short)
    join = fed.merge(loc, on="short", suffixes=("_fed", "_local"))
    join = join.sort_values("auprc_fed", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(join))
    w = 0.38
    ax.bar(x - w/2, join["auprc_fed"], w, label="Federated (best validated)", color="#3b6fb5", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, join["auprc_local"], w, label="Local-only", color="#aaaaaa", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(join["short"], rotation=35)
    ax.set_ylabel("AUPRC")
    ax.set_title("Federated vs.\\ local-only AUPRC per client. Smaller ICUs gain the most.")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, (f, l) in enumerate(zip(join["auprc_fed"], join["auprc_local"])):
        delta = f - l
        ax.annotate(f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}",
                    xy=(i, max(f, l) + 0.015), ha="center", fontsize=7,
                    color="#1f77b4" if delta > 0 else "#c0392b")
    fig.tight_layout()
    save(fig, "fig_local_vs_fed_per_client.png")


def fig_confusion_matrix():
    """Source: outputs/full_mimic_iv_training/metrics/confusion_matrix.csv"""
    df = pd.read_csv(OUT / "metrics" / "confusion_matrix.csv")
    mat = df.set_index("true_label")[["survived", "expired"]].values
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(mat[i,j]):,}", ha="center", va="center",
                    color="white" if mat[i,j] > mat.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted survived", "Predicted expired"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual survived", "Actual expired"])
    ax.set_title("Confusion matrix on best validated MLP.\n13,733 TN, 2,470 FP, 451 FN, 1,634 TP.")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    save(fig, "fig_confusion_matrix.png")


def fig_roc_pr_curves():
    """Sources: outputs/full_mimic_iv_training/metrics/roc_curve.csv +
    outputs/full_mimic_iv_training/metrics/precision_recall_curve.csv"""
    roc = pd.read_csv(OUT / "metrics" / "roc_curve.csv")
    pr = pd.read_csv(OUT / "metrics" / "precision_recall_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(roc["fpr"], roc["tpr"], color="#1f77b4", lw=2)
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    axes[0].set_xlabel("False positive rate (1 $-$ specificity)")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    axes[0].set_title("(a) ROC curve, AUROC $=$ 0.905")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pr["recall"], pr["precision"], color="#c0392b", lw=2)
    axes[1].axhline(0.114, color="grey", lw=0.8, ls="--", label="Base rate (11.4\\%)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) PR curve, AUPRC $=$ 0.653")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    save(fig, "fig_roc_pr_curves.png")


def fig_method_grouped_bars():
    """Sources: method_summary.csv + proposal_method_summary.csv"""
    rows = [
        ("MLP FedAvg",      "fedavg_default",      OUT),
        ("MLP CVaR$_{0}$",   "cvar_0",              OUT),
        ("MLP CVaR$_{0.5}$", "cvar_0.5",            OUT),
        ("MLP CVaR$_{0.75}$","cvar_0.75",           OUT),
        ("MLP Centralized", "centralized",         OUT),
        ("MLP Local-only",  "local_only",          OUT),
        ("MLP Grid-best",   "grid_best_validated", OUT),
        ("MLP GA-best",     "ga_best_validated",   OUT),
        ("MLP FedProx 0",   "fedprox_mu_0p0",      PROP),
        ("MLP FedProx 0.01","fedprox_mu_0p01",     PROP),
        ("MLP FedProx 0.1", "fedprox_mu_0p1",      PROP),
        ("Logistic FedAvg", "logreg_fedavg",       PROP),
        ("Logistic FedProx 0.1", "logreg_fedprox_mu_0p1", PROP),
    ]
    sumO = pd.read_csv(OUT / "metrics" / "method_summary.csv")
    sumP = pd.read_csv(PROP / "metrics" / "proposal_method_summary.csv")

    data = []
    for label, m, src in rows:
        df = sumO if src == OUT else sumP
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        data.append((label,
                     float(r["final_accuracy_mean"]),
                     float(r["final_auroc_mean"]),
                     float(r["final_auprc_mean"]),
                     float(r["final_worst_client_recall_mean"])))

    labels = [d[0] for d in data]
    acc    = [d[1] for d in data]
    auroc  = [d[2] for d in data]
    auprc  = [d[3] for d in data]
    rec    = [d[4] for d in data]

    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 1.5*w, acc,   w, label="Accuracy",    color="#27ae60", edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5*w, auroc, w, label="AUROC",       color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5*w, auprc, w, label="AUPRC",       color="#9467bd", edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5*w, rec,   w, label="Worst-recall",color="#c0392b", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Method head-to-head: four metrics tell four different stories.")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, "fig_method_grouped_bars.png")


def fig_resource_timeseries():
    """Source: monitoring/resource_timeseries.csv (best-effort)"""
    p = PROP / "monitoring" / "resource_timeseries.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["t"] = pd.to_datetime(df["timestamp"])
        df["t_min"] = (df["t"] - df["t"].min()).dt.total_seconds() / 60.0
    elif "elapsed_seconds" in df.columns:
        df["t_min"] = df["elapsed_seconds"] / 60.0
    else:
        df["t_min"] = np.arange(len(df)) * 0.5
    mem_col = next((c for c in ["memory_used_gb", "rss_gb", "memory_gb"]
                    if c in df.columns), None)
    if mem_col is None:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["t_min"], df[mem_col], color="#1f77b4", lw=1.2)
    ax.axhline(34, color="orange", ls="--", lw=0.8, label="warn (34 GB)")
    ax.axhline(38, color="red",   ls="--", lw=0.8, label="pause (38 GB)")
    ax.axhline(42, color="black", ls="--", lw=0.8, label="stop (42 GB)")
    ax.set_xlabel("Wall-clock minutes")
    ax.set_ylabel("Memory used (GB)")
    ax.set_title(f"M4 Max memory usage during proposal-alignment run.\nPeak {df[mem_col].max():.2f} GB; never approached the warn threshold.")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_resource_timeseries.png")


def fig_per_seed_auprc():
    """Source: outputs/full_mimic_iv_proposal_alignment/metrics/proposal_method_seed_results.csv"""
    df = pd.read_csv(PROP / "metrics" / "proposal_method_seed_results.csv")
    keep = ["fedprox_mu_0p0", "fedprox_mu_0p001", "fedprox_mu_0p01", "fedprox_mu_0p1",
            "logreg_fedavg", "logreg_fedprox_mu_0p1",
            "dirichlet_beta_infinity_fedprox", "dirichlet_beta_0p1_fedprox"]
    df = df[df["method"].isin(keep)].copy()
    label_map = {
        "fedprox_mu_0p0":   "MLP $\\mu\\!=\\!0$",
        "fedprox_mu_0p001": "MLP $\\mu\\!=\\!0.001$",
        "fedprox_mu_0p01":  "MLP $\\mu\\!=\\!0.01$",
        "fedprox_mu_0p1":   "MLP $\\mu\\!=\\!0.1$",
        "logreg_fedavg":    "Log FedAvg",
        "logreg_fedprox_mu_0p1": "Log FedProx 0.1",
        "dirichlet_beta_infinity_fedprox": "Dir $\\beta\\!=\\!\\infty$ FedProx",
        "dirichlet_beta_0p1_fedprox":      "Dir $\\beta\\!=\\!0.1$ FedProx",
    }
    df["label"] = df["method"].map(label_map)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    order = list(label_map.values())
    data = [df[df["label"] == lbl]["final_auprc"].values for lbl in order]
    ax.boxplot(data, labels=order, showmeans=True,
               medianprops=dict(color="black", lw=1.2),
               meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="red"))
    ax.set_ylabel("AUPRC across 10 seeds")
    ax.set_title("Per-seed AUPRC distribution. Tighter box = more reliable method.")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_per_seed_auprc.png")


def fig_drift_norms():
    """Source: outputs/full_mimic_iv_training/drift/fedavg_default_seed_7_drift.csv (representative seed)"""
    p = OUT / "drift" / "fedavg_default_seed_7_drift.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "round" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    grouped = df.groupby("round")
    if "update_norm" in df.columns:
        col = "update_norm"
    elif "update_l2" in df.columns:
        col = "update_l2"
    else:
        col = df.select_dtypes(include="number").columns.tolist()[2]
    means = grouped[col].mean()
    maxs = grouped[col].max()
    axes[0].plot(means.index, means.values, color="#1f77b4", lw=1.6, label="mean")
    axes[0].plot(maxs.index, maxs.values, color="#c0392b", lw=1.6, ls="--", label="max")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel(f"{col}")
    axes[0].set_title("(a) Per-round client update norms (FedAvg, seed 7)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    cos_col = next((c for c in ["cosine_to_mean", "cos_to_mean", "cosine"]
                    if c in df.columns), None)
    if cos_col:
        cm = grouped[cos_col].mean()
        cmin = grouped[cos_col].min()
        axes[1].plot(cm.index, cm.values, color="#2ca02c", lw=1.6, label="mean")
        axes[1].plot(cmin.index, cmin.values, color="#9467bd", lw=1.6, ls="--", label="min")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel(cos_col)
        axes[1].set_title("(b) Cosine alignment of client updates to consensus")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis("off")
    fig.tight_layout()
    save(fig, "fig_drift_norms.png")


def fig_search_history_scatter():
    """Source: full_mimic_iv_proposal_alignment/search/proposal_ga_history.csv"""
    df = pd.read_csv(PROP / "search" / "proposal_ga_history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(df["local_epochs"], df["fitness"], c="#1f77b4", s=18, alpha=0.6)
    axes[0].set_xlabel("local\\_epochs")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("(a) GA evaluations by local\\_epochs")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(df["clients_per_round"], df["fitness"], c="#d62728", s=18, alpha=0.6)
    axes[1].set_xlabel("clients\\_per\\_round")
    axes[1].set_ylabel("Fitness")
    axes[1].set_title("(b) GA evaluations by clients\\_per\\_round")
    axes[1].grid(True, alpha=0.3)
    axes[2].scatter(df["lr"], df["fitness"], c="#2ca02c", s=18, alpha=0.6)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("lr (log)")
    axes[2].set_ylabel("Fitness")
    axes[2].set_title("(c) GA evaluations by lr")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Differential-evolution sampling pattern (proposal run, 90 evals).", y=1.02)
    fig.tight_layout()
    save(fig, "fig_search_history_scatter.png")


if __name__ == "__main__":
    fns = [
        fig_client_mortality_and_size,
        fig_size_vs_mortality_scatter,
        fig_method_grouped_bars,
        fig_fedprox_mu_sweep,
        fig_cvar_alpha_collapse,
        fig_pareto_auprc_vs_comm,
        fig_fairness_frontier,
        fig_dirichlet_trends,
        fig_ga_convergence,
        fig_sparsity_compression,
        fig_lp_shadow_price,
        fig_loss_landscape_1d,
        fig_loss_landscape_2d,
        fig_per_client_clinical,
        fig_local_only_vs_federated,
        fig_confusion_matrix,
        fig_roc_pr_curves,
        fig_resource_timeseries,
        fig_per_seed_auprc,
        fig_drift_norms,
        fig_search_history_scatter,
    ]
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - guard for missing optional CSVs
            print(f"  SKIP {fn.__name__}: {exc}")
    print("Done.")
