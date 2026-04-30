# Shakespeare Federated Learning Experiment

**CharTransformer + FedAvg + CVaR Fairness + LP Duality + GA/Grid Search**

GPU-optimized, fully self-contained. No external project dependencies.

---

## Quick Start (HPC)

```bash
# 1. Upload and unzip
unzip shakespeare.zip
cd shakespeare

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (use your HPC's CUDA-compatible PyTorch)
# Option A: If the HPC has a module system
module load cuda/12.x python/3.10+
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Option B: Plain pip (auto-detects CUDA)
pip install -r requirements.txt

# 4. Run the full experiment
python -m shakespeare.run_experiment
```

---

## Run Order

| Step | What runs | Script |
|------|-----------|--------|
| 1 | Data download + tokenization + client partitioning | `data_loader.py` (auto) |
| 2 | Exploratory Data Analysis (13 plots) | `eda.py` (auto) |
| 3 | Multi-seed training: 10 seeds x 6 methods | `training.py` (auto) |
| 4 | Hyperparameter search: Grid (27) + GA (DE) | `search.py` (auto) |
| 5 | LP shadow price analysis | `utils.py` (auto) |
| 6 | Aggregation + statistical tests + all plots | `utils.py` (auto) |

**Everything is orchestrated by a single command:**

```bash
python -m shakespeare.run_experiment
```

---

## SLURM Submission Script

Save as `submit.sh` in the shakespeare folder:

```bash
#!/bin/bash
#SBATCH --job-name=shakespeare-fl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%j.log

module load cuda python
source venv/bin/activate
python -m shakespeare.run_experiment 2>&1 | tee experiment.log
```

Submit with: `sbatch submit.sh`

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | CharTransformer (295K params) |
| d_model / nhead / layers / dim_ff | 128 / 4 / 2 / 256 |
| Seeds | 10 (0-9) |
| FL rounds | 100 (early stopping, patience=15) |
| Clients per round | 10 |
| Batch size | 256 (training), 512 (centralized), 8192 (eval) |
| Learning rate | 0.8 (SGD) |
| CVaR alphas | 0.0, 0.1, 0.3, 0.5 |
| Mixed precision (AMP) | Yes (fp16 on CUDA) |
| DataLoader workers | 4 |

---

## Methods (6 per seed)

1. **FedAvg** (alpha=0.0) — standard federated averaging
2. **CVaR 0.1** — fairness-weighted aggregation (10th percentile)
3. **CVaR 0.3** — fairness-weighted aggregation (30th percentile)
4. **CVaR 0.5** — fairness-weighted aggregation (50th percentile)
5. **Centralized** — all data pooled, single model (upper bound)
6. **Local-only** — each client trains independently (lower bound)

---

## Output Structure

```
shakespeare/outputs/
├── eda/                    # 13 EDA plots + eda_summary.json
├── seed_0/ ... seed_9/     # Per-seed CSVs and JSONs
│   ├── fedavg_rounds.csv
│   ├── fedavg_summary.json
│   ├── cvar_0.1_rounds.csv
│   ├── cvar_0.3_rounds.csv
│   ├── cvar_0.5_rounds.csv
│   ├── centralized_rounds.csv
│   └── local_only.csv
├── summary/
│   ├── all_round_rows.csv          # Every round from every run
│   ├── convergence_summary.csv     # One row per (method, seed)
│   ├── confidence_intervals.csv    # 95% CIs per method
│   ├── paired_tests.csv            # Paired t-test / Wilcoxon vs FedAvg
│   ├── grid_search.csv
│   ├── ga_search.json
│   ├── lp_shadow.json
│   └── per_client_accuracy.csv
├── plots/
│   ├── fedavg_loss_mean_std.png
│   ├── fedavg_acc_mean_std.png
│   ├── fedavg_convergence_seed0.png
│   ├── centralized_*.png
│   ├── cvar_tradeoff.png
│   ├── shadow_price.png
│   ├── search_comparison.png
│   ├── method_accuracy_bar.png
│   ├── method_loss_bar.png
│   └── per_client_accuracy.png
└── artifacts/
    ├── vocab.json
    ├── client_names.json
    ├── config.json
    └── experiment_meta.json
```

---

## GPU Optimizations

- **Mixed precision (AMP)**: `torch.cuda.amp.autocast` + `GradScaler` — ~2x speedup, halves VRAM usage
- **pin_memory**: CPU→GPU transfer overlaps with compute
- **non_blocking transfers**: Async `.to(device)` calls
- **cuDNN benchmark**: `torch.backends.cudnn.benchmark = True` for stable input sizes
- **Large eval batches**: 8192-sample chunks for maximum GPU utilization during evaluation
- **Persistent DataLoader workers**: Avoids worker respawn overhead
- **set_to_none=True**: Faster gradient zeroing

---

## Estimated Runtime

| GPU | Estimated Time |
|-----|---------------|
| A100 (80GB) | ~1-2 hours |
| V100 (32GB) | ~2-3 hours |
| RTX 3090 (24GB) | ~2-4 hours |
| T4 (16GB) | ~4-6 hours |

---

## File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | `ShakespeareConfig` dataclass with all hyperparameters |
| `models.py` | `CharTransformer` — lightweight Transformer for next-char prediction |
| `data_loader.py` | Downloads/caches Shakespeare dataset, partitions by character |
| `training.py` | FedAvg, centralized, local-only training loops (AMP-enabled) |
| `search.py` | Grid search + Differential Evolution (GA) for hyperparameter tuning |
| `eda.py` | 13 EDA plots (distributions, heatmaps, t-SNE, non-IID metrics) |
| `metrics.py` | Perplexity, per-client summary, top-k accuracy |
| `utils.py` | Self-contained: ClientData, IO, stats, LP duality, plotting |
| `run_experiment.py` | Main orchestrator — runs everything in sequence |
| `requirements.txt` | Python dependencies |

---

## Data

The `data/shakespeare.csv` file (482 MB) is included. If missing, it auto-downloads from HuggingFace on first run.

After first run, tokenized data is cached in `data/processed.npz` + `data/meta.json` for instant loading on subsequent runs.

**Dataset**: `flwrlabs/shakespeare` — character-level next-character prediction, naturally partitioned by Shakespeare character (424 clients after filtering ≥2000 samples).
