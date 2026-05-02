"""
Statistical tests for UCI HAR homogeneity across subjects (FL clients).

Tests:
  1. Chi-squared test of homogeneity on label distributions (K x C contingency)
  2. Kruskal-Wallis H-test on top PCA components (feature distribution)
  3. PERMANOVA on raw feature space (multivariate group centroids)

Run: python experiments/uci_homogeneity_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, kruskal
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from flopt.data import load_uci_har

ALPHA=0.05
N_PCA=10     # PCA components for Kruskal-Wallis
PERMANOVA_PERMS=999

# ── data ──────────────────────────────────────────────────────────────────────

bundle=load_uci_har(Path("data"))
subjects=sorted(set(bundle.subject_train) | set(bundle.subject_test))
x_all=np.concatenate([bundle.x_train_raw, bundle.x_test_raw])
y_all=np.concatenate([bundle.y_train,     bundle.y_test])
s_all=np.concatenate([bundle.subject_train, bundle.subject_test])
n_classes=len(bundle.activity_names)

print(f"Dataset: {len(subjects)} subjects, {n_classes} activities, {x_all.shape[0]} samples, {x_all.shape[1]} features")
print("=" * 70)

# ── 1. Chi-squared test of homogeneity on label distributions ─────────────────

contingency=np.zeros((len(subjects), n_classes), dtype=int)
for i, subj in enumerate(subjects):
    mask=s_all == subj
    for c in range(n_classes):
        contingency[i, c]=int((y_all[mask] == c).sum())

chi2, p_chi2, dof, expected=chi2_contingency(contingency)

print("\n[1] Chi-squared test of homogeneity (label distributions)")
print(f"    H0: activity proportions are identical across all {len(subjects)} subjects")
print(f"    χ²={chi2:.2f}, df={dof}, p={p_chi2:.4f}")
if p_chi2 > ALPHA:
    print(f"    → FAIL TO REJECT H0 (p > {ALPHA}) — label distributions are homogeneous")
else:
    print(f"    → REJECT H0 (p ≤ {ALPHA}) — label distributions differ across subjects")

# Per-subject proportions for inspection
print("\n    Per-subject activity proportions (rows=subjects, cols=activities):")
print("    " + "  ".join(f"{name[:6]:>6}" for name in bundle.activity_names))
props=contingency / contingency.sum(axis=1, keepdims=True)
for i, subj in enumerate(subjects):
    row="  ".join(f"{props[i,c]:.3f}" for c in range(n_classes))
    print(f"    S{subj:02d}: {row}")

# ── 2. Kruskal-Wallis on PCA components ───────────────────────────────────────

from sklearn.preprocessing import StandardScaler
x_scaled=StandardScaler().fit_transform(x_all)
pca=PCA(n_components=N_PCA, random_state=42)
x_pca=pca.fit_transform(x_scaled)
explained=pca.explained_variance_ratio_.sum()

print(f"\n[2] Kruskal-Wallis H-test on top {N_PCA} PCA components")
print(f"    (components explain {explained*100:.1f}% of variance)")
print(f"    H0: each PC has the same distribution across all {len(subjects)} subjects")

kw_results=[]
for pc in range(N_PCA):
    groups=[x_pca[s_all == subj, pc] for subj in subjects]
    h, p=kruskal(*groups)
    kw_results.append((pc + 1, h, p))
    flag="" if p > ALPHA else "  ← REJECT"
    print(f"    PC{pc+1:>2}: H={h:7.2f}, p={p:.4f}{flag}")

n_reject=sum(1 for _, _, p in kw_results if p <= ALPHA)
print(f"\n    {N_PCA - n_reject}/{N_PCA} components: fail to reject H0 (homogeneous)")
print(f"    {n_reject}/{N_PCA} components: reject H0 (heterogeneous)")

# ── 3. PERMANOVA (pseudo-F on Euclidean distances) ────────────────────────────

print(f"\n[3] PERMANOVA ({PERMANOVA_PERMS} permutations, Euclidean distance, PCA space)")
print(f"    H0: group centroids are indistinguishable across subjects")

group_labels=np.array([np.where(np.array(subjects) == s)[0][0] for s in s_all])

def pseudo_f(x: np.ndarray, labels: np.ndarray) -> float:
    n=len(x)
    k=len(np.unique(labels))
    grand_mean=x.mean(axis=0)
    # SS_between
    ss_b=sum(
        ((x[labels == g].mean(axis=0) - grand_mean) ** 2).sum() * (labels == g).sum()
        for g in np.unique(labels)
    )
    # SS_within
    ss_w=sum(
        ((x[labels == g] - x[labels == g].mean(axis=0)) ** 2).sum()
        for g in np.unique(labels)
    )
    return (ss_b / (k - 1)) / (ss_w / (n - k))

observed_f=pseudo_f(x_pca, group_labels)

rng=np.random.default_rng(42)
null_f=np.array([
    pseudo_f(x_pca, rng.permutation(group_labels))
    for _ in range(PERMANOVA_PERMS)
])
p_permanova=(null_f >= observed_f).mean()

print(f"    Observed pseudo-F = {observed_f:.4f}")
print(f"    p-value = {p_permanova:.4f} (permutation)")
if p_permanova > ALPHA:
    print(f"    → FAIL TO REJECT H0 (p > {ALPHA}) — multivariate distributions are homogeneous")
else:
    print(f"    → REJECT H0 (p ≤ {ALPHA}) — group centroids differ across subjects")

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print(f"  Chi-squared (label dist):       p={p_chi2:.4f}  → {'homogeneous' if p_chi2 > ALPHA else 'heterogeneous'}")
n_homo_kw=N_PCA - n_reject
print(f"  Kruskal-Wallis (features):      {n_homo_kw}/{N_PCA} PCs homogeneous")
print(f"  PERMANOVA (multivariate):       p={p_permanova:.4f}  → {'homogeneous' if p_permanova > ALPHA else 'heterogeneous'}")
