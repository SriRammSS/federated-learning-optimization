# MIMIC-IV Federated Optimization Report Draft

## Dataset

This run uses the preprocessed MIMIC-IV ICU mortality cohort with `73141` ICU stays, `796` features, and `9` ICU-unit clients.

## Headline Results

- Best mean AUPRC method: `fedavg_default` with AUPRC `0.6389`.
- Best CVaR alpha selected by worst-client clinical performance: `0.0`.
- GA best raw vector: `[1, 5, 0.005]`.
- GA search evaluations: `0`.

See the generated LaTeX report for the final formatted explanation.
