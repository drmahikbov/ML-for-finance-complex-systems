# ML for Finance — Complex Systems

Master's project code for training **implicit neural-network policies** on
optimal-control problems in quantitative finance, using
**Jacobian-Free Backpropagation (JFB)** and a comparison baseline
**full autodiff (Full-AD)**.

The main result reproduced here is a **20-asset Almgren–Chriss portfolio
liquidation** problem with **stochastic price dynamics**
(`dS = −κ u dt + σ_S dW`), trained and evaluated end-to-end from this
repository.

## Acknowledgements

This project builds on the **Jacobian-Free Backpropagation** codebase
from Gelphman, Verma, Yang, Osher, and Wu Fung (ICML / arXiv 2025–2026).
See [ORIGINAL-AUTHORS.md](ORIGINAL-AUTHORS.md) for citation details and
the upstream examples (quadcopter, multi-bicycle, consumption–savings).

For architecture notes and how to add new problems, see [DEVELOPERS.md](DEVELOPERS.md).

---

## What you need

| Requirement | Notes |
|-------------|-------|
| **Python 3.12** | Version pinned in `.python-version` |
| **~4 GB disk** | Dependencies + generated plots/checkpoints |
| **GPU (optional)** | Speeds up training; CPU works but is slower |
| **Git** | To clone the repository |

Estimated runtime for the reference simulation below:
**several hours on CPU**, much faster with CUDA. The script trains **two**
policies (JFB + Full-AD), **300 epochs** each.

---

## Installation

```bash
git clone git@github.com:akatsukey/ML-for-finance-complex-systems.git
cd ML-for-finance-complex-systems

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Repository layout

```
ML-for-finance-complex-systems/
├── README.md                      ← this file (reference simulation)
├── ORIGINAL-AUTHORS.md            ← upstream ICML JFB code & citations
├── DEVELOPERS.md                  ← architecture / add-new-model recipe
├── requirements.txt
├── jfb-for-implicit-oc/           ← all Python code lives here
│   ├── models/LiquidationPortfolio.py
│   ├── examples/explicit_ustar/plot_liquidation_jfb.py   ← main runner
│   └── results/                   ← created at run time (git-ignored)
│       └── LiquidationPortfolioOC/
│           ├── training/          ← checkpoints, loss curves
│           └── benchmark/         ← comparison & diagnostic figures
└── presentation/                  ← Slidev slides (optional)
```

Training outputs are **not** committed to git. They appear under
`jfb-for-implicit-oc/results/` after you run the script.

---

## What the reference simulation does

The command at the end of this file runs
`examples/explicit_ustar/plot_liquidation_jfb.py`, which:

1. **Builds** a 20-asset liquidation problem with stochastic prices
   (`σ_S = 0.02`), power-law trading costs (`γ = 1.9`), and cross-asset
   inventory risk (`σ = 0.01414`).
2. **Trains** an implicit JFB policy for 300 epochs
   (`--tag FULL-STOCHASTIC-VS-DET-20-ASSETS`).
3. **Trains** a second Full-AD policy for comparison (`--full-ad`).
4. **Rolls out** the best JFB policy over **256 Monte-Carlo paths**
   (`--n-paths 256`) and plots mean ± 1 std bands on prices and cash.
5. **Writes diagnostic figures** (inner fixed-point residuals, costates, etc.).

Key flags in this run:

| Flag | Value | Meaning |
|------|-------|---------|
| `--n-assets 20` | 20 | Portfolio size |
| `--sigma-S 0.02` | 0.02 | Price volatility (stochastic SDE) |
| `--n-hutch 4` | 4 | Hutchinson probes for the stochastic HJB trace term |
| `--n-paths 256` | 256 | MC paths for rollout bands |
| `--full-ad` | on | Also train a Full-AD baseline |
| `--no-aa` | on | Plain fixed-point inner solver (no Anderson acceleration) |
| `--seed 42` / `--noise-seed 42` | 42 | Reproducible training and Brownian paths |

---

## Expected outputs

After a successful run, look under:

```
jfb-for-implicit-oc/results/LiquidationPortfolioOC/
├── training/
│   ├── best_policy_FULL-STOCHASTIC-VS-DET-20-ASSETS_<timestamp>.pth
│   ├── best_policy_FULL-STOCHASTIC-VS-DET-20-ASSETS-fullAD_<timestamp>.pth
│   ├── history_*.csv
│   └── loss_curve_*.png
└── benchmark/
    ├── jfb_multiasset_rollout_n20_FULL-STOCHASTIC-VS-DET-20-ASSETS.png
    ├── best_policy_same_axes_FULL-STOCHASTIC-VS-DET-20-ASSETS.png   ← main result figure
    ├── jfb_diagnostics_FULL-STOCHASTIC-VS-DET-20-ASSETS.png
    ├── jfb_diagnostics_FULL-STOCHASTIC-VS-DET-20-ASSETS-fullAD.png
    └── training_curves_jfb_vs_ad_FULL-STOCHASTIC-VS-DET-20-ASSETS.png
```

The script prints the absolute path of each figure as it is written.
The most important plot for a quick check is
`best_policy_same_axes_FULL-STOCHASTIC-VS-DET-20-ASSETS.png`
(inventory, trading rates, prices with uncertainty bands, and cash).

---

## Run the reference simulation

```bash
cd jfb-for-implicit-oc

python examples/explicit_ustar/plot_liquidation_jfb.py \
  --n-assets 20 --t-final 2.0 --nt 100 \
  --kappa 1e-5 --sigma 0.01414 --eta 0.5 \
  --sigma-S 0.02 --n-hutch 4 --n-paths 256 --noise-seed 42 \
  --gamma 1.9 --epsilon 1e-2 --alpha 30.0 \
  --q0-min 3 --q0-max 8 --S0 2.0 --X0 0.0 \
  --train-epochs 300 --batch-size 256 --lr 1e-3 \
  --fp-max-iters 200 --fp-tol 1e-6 \
  --phi-arch default --seed 42 \
  --tag FULL-STOCHASTIC-VS-DET-20-ASSETS \
  --no-aa --fp-alpha 0.9 --learned-costate-overlay off \
  --full-ad
```

To use a GPU when available, add `--device cuda` to the command above.
