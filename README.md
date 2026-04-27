# Federated Learning Optimization

This project studies federated learning as an optimization problem: convergence under non-IID clients, communication shadow prices through LP duality, CVaR-style fairness, and global search over training configuration.

## Core Question

Can a federated learning system be accurate, communication-efficient, and fair to weak clients at the same time?

## What This Builds

- A compact FedAvg simulator in PyTorch.
- Natural UCI HAR client experiments where each subject is one federated client.
- Per-round logging for loss, accuracy, worst-client accuracy, and communication cost.
- CVaR-style aggregation that gives more weight to high-loss clients.
- A communication-budget LP over measured training policies.
- KKT diagnostics and `lambda*` shadow-price curves.
- Grid search and differential-evolution hooks for `(E,K,lr)`.

## Quick Start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/run_full.py --source uci --rounds 30 --seeds 7,11,19 --out outputs/full_uci --skip-ga
```

Outputs are written to `outputs/full_uci`.

## Project Structure

```text
flopt/
  config.py      experiment configuration
  data.py        UCI HAR loader
  fedavg.py      FedAvg training and evaluation
  models.py      HAR MLP and logistic baseline
  duality.py     communication-budget LP and KKT checks
  search.py      grid search and differential evolution
  plots.py       report-ready plots
experiments/
  run_full.py    full UCI HAR experiment
```

## Report Framing

The LP is framed over candidate FL training policies, not over neural-network weights. Each policy has measured validation loss and communication cost. The LP selects the best policy mixture under a bandwidth budget, and the dual variable gives the marginal value of relaxing that budget.

