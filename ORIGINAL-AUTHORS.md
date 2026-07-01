# Original authors — JFB / implicit Hamiltonians (ICML)

This document preserves the README from the upstream **Jacobian-Free
Backpropagation (JFB)** codebase by Gelphman, Verma, Yang, Osher, and Wu Fung.
Our master's project extends that framework with finance applications
(Almgren–Chriss liquidation, stochastic price dynamics, RL extensions).

The code described below lives under `jfb-for-implicit-oc/`. The upstream
layout was refactored into subpackages; the mapping is:

| Original (flat) file | Current location |
| -------------------- | ---------------- |
| `ImplicitNets.py` | `jfb-for-implicit-oc/core/ImplicitNets.py` |
| `ImplicitOC.py` | `jfb-for-implicit-oc/core/ImplicitOC.py` |
| `CVXPolicy.py` | `jfb-for-implicit-oc/core/CVXPolicy.py` |
| `DirectControlNets.py` | `jfb-for-implicit-oc/core/DirectControlNets.py` |
| `OptimalControlTrainer.py` | `jfb-for-implicit-oc/core/OptimalControlTrainer.py` |
| `utils.py` | `jfb-for-implicit-oc/core/utils.py` |
| `MultiBicycle.py` | `jfb-for-implicit-oc/models/MultiBicycle.py` |
| `Quadcopter.py` | `jfb-for-implicit-oc/models/Quadcopter.py` |
| `Consumption.py` | `jfb-for-implicit-oc/models/Consumption.py` |
| `example_multibicycle.py` | `jfb-for-implicit-oc/examples/example_multibicycle.py` |
| `example_multi_quadcopter.py` | `jfb-for-implicit-oc/examples/example_multi_quadcopter.py` |
| `example_multiConsumption.py` | `jfb-for-implicit-oc/examples/example_multiConsumption.py` |

Run all commands below from `jfb-for-implicit-oc/`:

```bash
cd jfb-for-implicit-oc
```

---

## Overview

Code for training and evaluating optimal control policies using implicit
neural networks with **Jacobian-Free Backpropagation (JFB)** and
**Jacobian-Based Backpropagation (JBB/CVX)** for the three examples
presented in the ICML paper.

### Core implementation

- `core/ImplicitNets.py` — implicit neural network architectures (JFB)
- `core/ImplicitOC.py` — implicit optimal control layer with HJB conditions
- `core/CVXPolicy.py` — CVXPY-based policies (JBB)
- `core/DirectControlNets.py` — direct transcription baselines
- `core/OptimalControlTrainer.py` — unified training framework
- `core/utils.py` — utilities

### Problem definitions (ICML examples)

- `models/MultiBicycle.py` — multi-agent bicycle optimal control
- `models/Quadcopter.py` — single and multi-agent quadcopter control
- `models/Consumption.py` — multi-agent consumption–savings control

### Training scripts

| Problem | Script | Agents |
| ------- | ------ | ------ |
| Multi-bicycle | `examples/example_multibicycle.py` | 100 |
| Quadrotor | `examples/example_multi_quadcopter.py` | 1, 6, or 100 |
| Consumption–savings | `examples/example_multiConsumption.py` | 100 |

---

## Requirements

See the repository root `requirements.txt`. Upstream minimum versions:

```
torch>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
cvxpy>=1.2.0
cvxpylayers>=0.1.5
```

---

## Usage

### 1. Multi-bicycle (100 agents)

```bash
python examples/example_multibicycle.py
```

Key parameters in the script:

- `batch_size` — default 100
- `nt` — time steps (default 60)
- `t_final` — horizon (default 4.0)
- `n_b` — number of bicycles (fixed at 100)
- `alphaG` — terminal cost weight (default 500.0)
- `epochs` — training epochs (default 500)

### 2. Quadrotor (1, 6, or 100 agents)

```bash
python examples/example_multi_quadcopter.py
```

**GPU:**

```bash
python examples/example_multi_quadcopter.py --device cuda
python examples/example_multi_quadcopter.py --device cuda:0
```

**Agent count:**

```bash
python examples/example_multi_quadcopter.py --num_quadcopters 1
python examples/example_multi_quadcopter.py --num_quadcopters 6
```

**JBB (CVXPyLayers):**

```bash
python examples/example_multi_quadcopter.py --train_jbb
python examples/example_multi_quadcopter.py --train_jfb --train_jbb
python examples/example_multi_quadcopter.py --no_train_jfb --train_jbb
```

**Other options:**

```bash
python examples/example_multi_quadcopter.py --num_quadcopters 100 --epochs 1000 --lr 0.005 --device cuda:0
```

Key parameters:

- `batch_size` — default 50
- `nt` — default 160
- `t_final` — default 4.5
- `num_quadcopters` — 1, 6, or 100 (default 100)
- `alphaG` — default 1000.0
- `epochs` — default 500

### 3. Consumption–savings (100 agents)

```bash
python examples/example_multiConsumption.py
```

Key parameters:

- `batch_size` — default 128
- `nt` — default 100
- `t_final` — default 2.0
- `m` — number of agents (fixed at 100)
- `epochs` — default 500

---

## Numerical verification

During training, `compute_loss()` with `save_history=True` writes CSV
history files containing:

- Total loss (running cost + terminal cost)
- Running and terminal cost separately
- Optimality violations (`cHJB`, `cHJBfin`, `cadj`, `cadjfin`)
- Gradient metrics (`max_grad_H`, `avg_grad_H`)
- **Contractivity**: `max_grad_T_u`
- **M_theta conditioning**: `smallest_M_sdval`, `largest_M_sdval`
- **Descent direction**: angle between expected gradients

These CSVs reproduce the numerical verification plots from the paper.

---

## Output files

Training produces:

- `best_policy_*.pth` — model weights
- `history_*.csv` — full training history
- `*_run.log` — training logs
- Trajectory plots under `results/<ProblemClassName>/`

---

## Training tips

1. **GPU** — set `device='cuda'` or `device='cuda:X'` in the config.
2. **Hyperparameters** — defaults are tuned per problem.
3. **Convergence** — monitor loss, optimality violations (`cHJB`, `cadj`), and gradient norms.
4. **Memory** — large batch sizes need substantial GPU memory.
5. **Trials** — run multiple trials (`n_trials` in `main()`) for statistics.

---

## Key configuration parameters

### Common

- `batch_size`, `nt`, `t_final`, `alphaG`
- `alphaHJB` — `[cHJB_weight, cHJBfin_weight]`
- `lr` — default `1e-3` (JFB), `5e-4` (direct control)
- `epochs`

### Method-specific

- **JFB** — `max_iters`, `tol`, `tracked_iters`, `alpha` (inner FP solver)
- **JBB/CVX** — `tol` (CVXPY)
- **Direct transcription** — `weight_decay` (explicit regularization)

---

## JFB vs direct transcription

Direct transcription (control sequences without optimality constraints) typically needs:

- **10× smaller learning rate** (`5e-4` vs `5e-3`)
- **Explicit weight decay**
- **100× smaller weight initialization**

JFB enforces `∇_u H = 0`, providing implicit regularization; direct
transcription lacks that and needs explicit regularization to match performance.

---

## Citation

If you use the upstream JFB code or build on this repository, please cite:

```
@misc{gelphman2025jfb,
      title={End-to-End Training of High-Dimensional Optimal Control with Implicit Hamiltonians via Jacobian-Free Backpropagation},
      author={Eric Gelphman and Deepanshu Verma and Nicole Tianjiao Yang and Stanley Osher and Samy Wu Fung},
      year={2025},
      eprint={2510.00359},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2510.00359},
}

@misc{gelphman2026convergence,
      title={On the Convergence of Jacobian-Free Backpropagation for Optimal Control Problems with Implicit Hamiltonians},
      author={Eric Gelphman and Deepanshu Verma and Nicole Tianjiao Yang and Stanley Osher and Samy Wu Fung},
      year={2026},
      eprint={2602.00921},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.00921},
}
```

Paper links:

- [JFB training (2025)](https://arxiv.org/abs/2510.00359)
- [JFB convergence (2026)](https://arxiv.org/abs/2602.00921)
