# MIMIC-IV Federated Optimization Report Draft

## Dataset

This run uses the preprocessed MIMIC-IV ICU mortality cohort with `73141` ICU stays, `1021` features, and `9` ICU-unit clients.

## Headline Results

- Best mean AUPRC method: `fedavg_default` with AUPRC `0.6316`.
- Best CVaR alpha selected by worst-client clinical performance: `0.0`.
- GA best raw vector: `[1.2660918762649054, 3.4847698845669535, 0.005769185832593745]`.
- GA search evaluations: `48`.

See the generated LaTeX report for the final formatted explanation.
