# MIMIC-IV Proposal Alignment Report

This report records the proposal-faithful additions to the existing MIMIC-IV clinical FL project.

## Non-overwrite guarantee

All outputs are under `outputs/full_mimic_iv_proposal_alignment`. Existing `outputs/full_mimic_iv` and `outputs/full_mimic_iv_training` are read-only baselines.

## Compliance summary

Completed checklist items: 7/7.

See:

- `reports/proposal_alignment_checklist.csv`
- `reports/proposal_compliance_matrix.csv`
- `metrics/proposal_method_summary.csv`
- `lp/sparsity_lp_shadow_price.csv`
- `landscape/*loss_surface.csv`

## Interpretation

The existing MIMIC MLP experiment remains the main clinical extension. This alignment suite adds FedProx, convex logistic controls, Dirichlet beta non-IID controls, sparsity communication LP, stronger GA search, loss landscapes, and resource monitoring to directly address the original MSML604 proposal.
