# Federated Learning Optimization Report Draft

## Setup

Dataset source: `uci`  
Clients: `30`  
Rounds: `30`  
Seeds: `[7, 11, 19]`  
Model: `HARMLP`, a 561-feature activity classifier with two hidden layers.

## What Was Trained

We trained one global federated model with FedAvg. Each round selected clients, trained local copies on each client's activity-recognition data, and aggregated local weights into the next global model.

## Main Results

- Mean FedAvg accuracy: `0.9339` with worst-client accuracy `0.6914`.
- Best CVaR setting by worst-client accuracy: `alpha=0.9`, worst-client accuracy `0.7037`.
- Best grid policy: `E=4`, `K=10`, `lr=0.02`, fitness `0.9082`.
- Differential evolution result: raw `x=[4.893972128846123, 4.154513594150248, 0.0716965180222601]`, fitness `0.3993` after `36` evaluations.
- LP budgets with positive communication shadow price: `5` out of `10`.

## Figures

- `convergence_mean.png`: mean convergence across seeds with an `O(1/sqrt(T))` reference.
- `cvar_pareto.png`: average-client accuracy vs worst-client accuracy.
- `shadow_price.png`: LP dual variable `lambda*` vs communication budget.
- `ga_vs_grid.png`: best grid-search fitness vs differential-evolution fitness.

## Interpretation

The UCI HAR experiment now uses real naturally federated clients: each subject acts as one client. The convergence plot supports the FedAvg learning story. The CVaR sweep tests whether emphasizing high-loss clients improves the worst-client outcome. The shadow-price plot translates communication limits into an optimization quantity: when `lambda*` is positive, bandwidth is binding and extra communication has measurable value.
