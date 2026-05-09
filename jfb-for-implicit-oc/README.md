# Implicit Hamiltonians with Jacobian-Free/Jacobian-Based Backpropagation

This repository contains code for training and evaluating optimal control policies using implicit neural networks with Jacobian-Free Backpropagation (JFB) and Jacobian-Based Backpropagation (JBB/CVX) for the three examples presented in our ICML paper.

## Repository Structure

### Core Implementation Files

- `ImplicitNets.py`: Implicit neural network architectures (JFB method)
- `ImplicitOC.py`: Implicit optimal control layer with HJB optimality conditions
- `CVXPolicy.py`: CVXPY-based policies (JBB method)
- `DirectControlNets.py`: Direct transcription baseline policies (for comparison)
- `OptimalControlTrainer.py`: Unified training framework for all policy types
- `utils.py`: Utility functions

### Problem Definitions

- `MultiBicycle.py`: Multi-agent bicycle optimal control problem
- `Quadcopter.py`: Single and multi-agent quadcopter optimal control
- `Consumption.py`: Multi-agent consumption-savings optimal control

### Training Scripts

#### Multi-Bicycle (100 agents)

- `example_multibicycle.py`: Train JFB policy on multi-bicycle problem

#### Quadrotor (1, 6, or 100 agents)

- `example_multi_quadcopter.py`: Train JFB and JBB policies on quadrotor problem

#### Consumption-Savings (100 agents)

- `example_multiConsumption.py`: Train JFB policy on multi-agent consumption problem

## Requirements

```bash
torch>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
cvxpy>=1.2.0
cvxpylayers>=0.1.5
```

## Installation

```bash
pip install torch numpy pandas matplotlib cvxpy cvxpylayers
```

## Usage

### Training Examples

#### 1. Multi-Bicycle Problem (100 agents)

Train with JFB:

```python
python example_multibicycle.py
```

Configuration parameters in the script:

- `batch_size`: Batch size for training (default: 100)
- `nt`: Number of time steps (default: 60)
- `t_final`: Final time horizon (default: 4.0)
- `n_b`: Number of bicycles (fixed at 100)
- `alphaG`: Terminal cost weight (default: 500.0)
- `epochs`: Number of training epochs (default: 500)

#### 2. Quadrotor Problem (1, 6, or 100 agents)

Train with JFB (and optionally JBB):

```python
python example_multi_quadcopter.py
```

The script supports training with different numbers of quadrotors and training methods via command-line arguments:

**Basic usage (default: 100 quadrotors, JFB training, CPU device):**

```bash
python example_multi_quadcopter.py
```

**Using GPU:**

```bash
python example_multi_quadcopter.py --device cuda
# or specify GPU device
python example_multi_quadcopter.py --device cuda:0
```

**Single Quadrotor (1 agent):**

```bash
python example_multi_quadcopter.py --num_quadcopters 1
```

**6 Quadrotors:**

```bash
python example_multi_quadcopter.py --num_quadcopters 6
```

**Train with JBB (CVXPyLayers):**

```bash
python example_multi_quadcopter.py --train_jbb
```

**Train with both JFB and JBB:**

```bash
python example_multi_quadcopter.py --train_jfb --train_jbb
```

**Disable JFB training:**

```bash
python example_multi_quadcopter.py --no_train_jfb --train_jbb
```

**Other useful arguments:**

```bash
python example_multi_quadcopter.py --num_quadcopters 100 --epochs 1000 --lr 0.005 --device cuda:0
```

Configuration parameters:

- `batch_size`: Batch size for training (default: 50)
- `nt`: Number of time steps (default: 160)
- `t_final`: Final time horizon (default: 4.5)
- `num_quadcopters`: Number of quadrotors - 1, 6, or 100 (default: 100)
- `alphaG`: Terminal cost weight (default: 1000.0)
- `epochs`: Number of training epochs (default: 500)

#### 3. Consumption-Savings Problem (100 agents)

Train with JFB:

```python
python example_multiConsumption.py
```

Configuration parameters:

- `batch_size`: Batch size for training (default: 128)
- `nt`: Number of time steps (default: 100)
- `t_final`: Final time horizon (default: 2.0)
- `m`: Number of agents (fixed at 100)
- `epochs`: Number of training epochs (default: 500)

### Numerical Verification

Numerical verification is performed automatically during training through the `compute_loss()` function when the `save_history=True` flag is set in the training configuration. This generates CSV files containing:

- Total loss (control objective = running_cost + alphaG * terminal_cost)
- Running cost and terminal cost (separate)
- Optimality condition violations (cHJB, cHJBfin, cadj, cadjfin)
- Gradient metrics (max_grad_H, avg_grad_H)
- **Contractivity verification**: max_grad_T_u (maximum gradient of T_theta operator)
- **M_theta conditioning**: smallest_M_sdval, largest_M_sdval (singular values of M_theta matrix)
- **Descent direction**: angle between expected gradients

The training scripts save these history CSV files (e.g., `history_best_policy_JFB_*.csv`) which contain all numerical verification metrics reported in the paper. These CSVs can be loaded and analyzed to reproduce the numerical verification plots shown in the paper.

## Output Files

Training scripts generate:

- `best_policy_*.pth`: Saved model weights
- `history_*.csv`: Training history with all numerical verification metrics (loss, costs, optimality violations, gradient norms, contractivity checks, M_theta singular values, descent angles)
- `*_run.log`: Training logs
- Trajectory plots in `results_*/` directories

## Training Tips

1. **GPU Usage**: Set `device='cuda'` or `device='cuda:X'` in config for GPU acceleration
2. **Hyperparameters**: The provided hyperparameters are tuned for each problem
3. **Convergence**: Monitor training logs for:
   - Loss convergence
   - Optimality condition violations (cHJB, cadj)
   - Gradient norms
4. **Memory**: Large batch sizes may require significant GPU memory
5. **Multiple Trials**: Run multiple trials (set `n_trials` in main()) for statistical significance

## Key Configuration Parameters

### Common to All Problems

- `batch_size`: Number of initial conditions per training batch
- `nt`: Discretization time steps
- `t_final`: Time horizon
- `alphaG`: Weight on terminal cost
- `alphaHJB`: Weights for optimality condition penalties [cHJB_weight, cHJBfin_weight]
- `lr`: Learning rate (default: 1e-3 for JFB, 5e-4 for Direct Control)
- `epochs`: Number of training epochs

### Method-Specific

- **JFB**: `max_iters`, `tol`, `tracked_iters`, `alpha` for fixed-point solver
- **JBB/CVX**: `tol` for CVXPY solver tolerance
- **Direct Transcription**: `weight_decay` for explicit regularization (required due to lack of optimality constraints)

## Comparison: JFB vs Direct Transcription

The comparison scripts demonstrate a key insight: Direct transcription policies (which directly optimize control sequences without enforcing optimality conditions) require:

- **10x smaller learning rate** (5e-4 vs 5e-3)
- **Explicit weight decay** for regularization
- **100x smaller weight initialization**

These differences arise because JFB enforces optimality conditions (∇_u H = 0), providing implicit regularization. Direct transcription lacks these constraints and requires explicit regularization to match performance.

## Citation

If you use this code, please cite our paper:

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

Run this command to get somewhat good graphs
```
python examples/explicit_ustar/plot_liquidation_jfb.py \                    
  --fp-alpha 1e-3 --fp-max-iters 200 --fp-tol 1e-4 --no-aa \
  --clamp-u --u-min 0 --u-max 10 \
  --t-final 10.0 --nt 100 \
  --tag JFB-old-solver-regime
```


ESSAIE CA FRANCOIS 

python examples/explicit_ustar/plot_liquidation_jfb.py \
    --n-assets 1 --t-final 1.0 --nt 50 \
    --sigma 0.01414 --kappa 1e-5 --eta 0.5 \
    --gamma 2.0 --epsilon 1e-2 --alpha 49.0 \
    --q0-min 0.95 --q0-max 1.05 --S0 1.0 --X0 0.0 \
    --train-epochs 500 --batch-size 256 --lr 1e-3 \
    --fp-max-iters 50 --fp-tol 1e-6 \
    --phi-arch default \
    --seed 42 --tag FRANCOIS-REMPLACE-CA --no-aa --fp-alpha 0.9