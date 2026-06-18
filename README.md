# ML-for-Finance-Complex-Systems

Training implicit-Hamiltonian optimal control policies via Jacobian-Free Backpropagation (JFB). The core idea: instead of backpropagating through an entire forward rollout (BPTT), the network is parameterised as a fixed-point operator T(u) = u − α∇_u H, and gradients flow only through the last few fixed-point iterations. This makes training tractable at time horizons and state dimensions where full autodiff would be prohibitive. The repo covers both the known-dynamics setting and an RL extension where the agent never accesses the dynamics directly — only a learned local Jacobian estimate from rollouts.

Code accompanies: arXiv 2510.00359 and arXiv 2602.00921.

## Setup

```bash
pip install -r requirements.txt
```

Python version is pinned in `.python-version`. The main code lives under `jfb-for-implicit-oc/`; all example runners assume that as the working directory.

## Repository layout

```
jfb-for-implicit-oc/
  core/            Abstract OC pipeline for known dynamics
  core_RL/         RL extension: dynamics hidden, estimated via RLS
  benchmarking/    Shared plotting and trajectory service (used by both pipelines)
  models/          Concrete OC problems (known-dynamics and RL variants)
  examples/        Runners for known-dynamics problems
  examples-RL/     Runners for RL problems
  results/         Output artifacts — auto-generated, not committed

presentation/      Slidev slide deck (slides.md, slides-part2-rl.md)
latex/             Notes and reference figures
```

## The two pipelines

### Known-dynamics pipeline (`core/` + `models/` + `examples/`)

A problem subclasses `ImplicitOC` (`core/ImplicitOC.py`) and implements the math contract: `compute_lagrangian`, `compute_grad_lagrangian`, `compute_f`, `compute_grad_f_u`, `compute_grad_f_z`, `compute_G`, `compute_grad_G_z`, and `sample_initial_condition`. It also provides `panels()` and `to_trajectory()` for the plotting service.

The policy is an `ImplicitNetOC` (in `core/ImplicitNets.py`): a ResNN-based network that runs fixed-point iteration T(u) = u − α∇_u H at each time step. Backpropagation flows only through the last `tracked_iters` iterations (the JFB approximation). `CVXPolicy` (CvxpyLayer QP) and `DirectControlNets` are alternative baselines that bypass the fixed-point structure.

`OptimalControlTrainer` (`core/OptimalControlTrainer.py`) owns the training loop: forward rollout via `compute_f`, loss from `compute_lagrangian` + `compute_G`, backward via JFB surrogate, optimizer step, periodic plotting, and checkpoint saving. It never touches file paths directly — those go through `RunIO`.

`RunIO` (`core/run_io.py`) is the single source of truth for artifact filenames. It owns `tag` (e.g. `"JFB"` or `"FullAD"`) and `run_id` (timestamp), and derives every `.pth`, `.csv`, `.png` path from them. `Paths` (`core/paths.py`) creates the directory tree under `results/<ProblemClassName>/training/` and `results/<ProblemClassName>/rollouts/`.

To run an example:

```bash
cd jfb-for-implicit-oc
python examples/example_liquidationportfolio.py
```

Each run produces a deterministic six-artifact bundle:

```
results/LiquidationPortfolioOC/
  training/
    best_policy_<tag>_<timestamp>.pth
    history_<tag>_<timestamp>.csv
    loss_curve_<tag>_<timestamp>.png
    training-plots/rollout_<tag>_<timestamp>_NNNN.png
  rollouts/
    policy_rollout_<tag>_<timestamp>.png
    trajectory_<tag>_<timestamp>.pth
```

Re-running never overwrites because `run_id` is a timestamp. Passing `tag="FullAD"` to `OptimalControlTrainer` lets you compare two training strategies in the same `results/` tree without collisions.

### RL pipeline (`core_RL/` + `models/` + `examples-RL/`)

The RL pipeline mirrors the known-dynamics one, but the agent never calls `compute_f`. Instead it interacts with an `Environment` (`core_RL/Environment.py`), which exposes only a `step(z, u, t)` interface. For experiments where the dynamics are actually known, `AnalyticalEnvironment` wraps a `f_callable` and detaches all autograd through it, so the agent can't cheat.

`JacobianEstimator` (`core_RL/JacobianEstimator.py`) estimates the local per-step Jacobians a_k = ∂f/∂z and b_k = ∂f/∂u from rollout data. Two implementations: `RLSJacobianEstimator` runs block recursive least-squares with a forgetting factor; `OracleJacobianEstimator` cheats using the analytical Jacobians and is used as a sanity-check baseline.

`ImplicitOC_RL` (`core_RL/ImplicitOC_RL.py`) is the abstract problem base for the RL setting. Like `ImplicitOC` but without `compute_f` in the abstract contract — the cost and gradient methods are the same; dynamics are hidden. `ImplicitNetOC_RL` (`core_RL/ImplicitNets_RL.py`) extends the known-dynamics network: T is overridden to use b_k, which must be injected via `policy.set_step_jacobian(b_k)` before every forward call.

The training loop in `OptimalControlTrainer_RL` (`core_RL/OptimalControlTrainer_RL.py`) owns both the Environment and the JacobianEstimator. Per-epoch: roll out via `env.step`, update the RLS estimates, build the JFB surrogate S(θ) explicitly from the detached a_k, b_k, and adjoint p_{k+1}, then call `surrogate.backward()`. The adjoint is computed analytically using `compute_grad_lagrangian_z` and `compute_grad_G_z`, not via autograd through the dynamics.

`core/paths.py`, `core/run_io.py`, `core/log_format.py`, and the entire `benchmarking/` package are shared unchanged between the two pipelines.

To run a VdP benchmark:

```bash
cd jfb-for-implicit-oc
python examples-RL/vanderpol_comparison.py      # standard VdP: JFB-RL/RLS vs Oracle vs BPTT
python examples-RL/hard_vdp_comparison.py       # same with exponential control cost
python examples-RL/hard_gain_vdp.py             # adds state-dependent gain β·tanh(x₁)
```

For the portfolio problem:

```bash
python examples-RL/portfolio_optimization_RL.py   # train
python examples-RL/evaluate_portfolio_rl.py        # evaluate a saved checkpoint
python examples-RL/plot_portfolio_analytical.py    # plot analytical baseline
```

## Key files

`**core/ImplicitOC.py**` — abstract base class for all problems. Defines the interface that `OptimalControlTrainer` calls into and that model authors implement.

`**core/ImplicitNets.py**` — `ResNN` (residual MLP), `Phi` (the policy backbone), `ImplicitNetOC` (wraps Phi into the fixed-point operator). The `tracked_iters` parameter controls the JFB approximation depth.

`**core/run_io.py**` — all filename decisions go here. If you find yourself writing `os.path.join` in a model or example file, stop and add a method to `RunIO` instead.

`**benchmarking/trajectory.py**` — `Trajectory` dataclass holding `t`, `z`, `u` numpy arrays and plot style. Every model's `to_trajectory()` returns one of these.

`**benchmarking/plotter.py**` — `Panel` (declarative subplot spec) and `BenchmarkPlotter` (renders a figure from a list of Trajectories against a list of Panels). Models define what to plot via `panels()`; the plotter executes it.

`**core_RL/JacobianEstimator.py**` — shapes matter: a_k is `(B, n, n)` with `a_k[b, i, j] = ∂f_i/∂z_j`; b_k is `(B, m, n)` with `b_k[b, i, j] = ∂f_j/∂u_i` (transposed convention, consistent with `core/ImplicitOC.compute_grad_H_u`). If the adjoint pass gives wrong signs, check this layout first.

`**core_RL/ImplicitOC_RL.py**` — `compute_grad_H_u_estimated` and `compute_loss_RL`. The sign convention for the adjoint (p_k += Δt·(aₖᵀ p_{k+1} + ∇_z L)) is documented at the top of the file.

## Models

**Known dynamics** (in `models/`, used by `examples/`):

- `LiquidationPortfolio.py` — Almgren-Chriss liquidation; has a closed-form γ=2 BVP solution used for overlay comparison
- `Consumption.py` — 100-agent consumption-savings problem
- `MultiBicycle.py` — 100-agent bicycle stabilisation
- `Quadcopter.py` — 1, 6, or 100 quadrotors; uses bespoke `plot_position_trajectories` (legacy, predates the Panel/Trajectory service)

**RL setting** (in `models/`, used by `examples-RL/`):

- `VanDerPolOC_RL.py` — standard VdP; L = x₁² + x₂² + 0.5u², ∇_u H = u + p₂ is trivially solvable (useful as a baseline since the implicit iteration is unnecessary)
- `Hard_VDP_RL.py` — same dynamics, exponential cost L = x₁² + x₂² + λ(eᵘ² − 1); ∇_u H is transcendental, so the fixed-point iteration is genuinely required
- `Hard_Gain_VDP_RL.py` — adds state-dependent control gain ẋ₂ = (1−x₁²)x₂ − x₁ + (1 + β·tanh(x₁))·u; harder for RLS to identify b_k
- `PortfolioOC_RL.py` — Merton portfolio with exponential risk penalty; the agent sees only wealth, never μ or r

## Adding a new model

Copy `examples/example_TEMPLATE.py` to `examples/example_myproblem.py` and resolve the two `TODO` blocks (import + constructor). Create `models/MyProblemOC.py` subclassing `ImplicitOC`. Run it with:

```bash
cd jfb-for-implicit-oc
python examples/example_myproblem.py
```

The six-artifact bundle will appear under `results/MyProblemOC/` automatically. For a comparison against a reference solution, see `examples/explicit_ustar/liquidation_benchmark.py` and `plot_liquidation_jfb.py` as the pattern to follow — those write to `results/<cls>/benchmark/`, distinct from the trainer's rollout output.

For a new RL problem, subclass `ImplicitOC_RL`, implement the cost and gradient methods but not `compute_f`, and wire it to `AnalyticalEnvironment` + one of the `JacobianEstimator` implementations. See `examples-RL/vanderpol_comparison.py` for the wiring pattern.

## Sanity-check workflow for the RL pipeline

The most useful diagnostic when something breaks: train on the same seed with (A) the known-dynamics `OptimalControlTrainer` and (B) `OptimalControlTrainer_RL` with `OracleJacobianEstimator`. The loss curves should match up to numerical noise. If they diverge, the bug is in the surrogate construction. Once A ≈ B, swap Oracle for RLS — the gap is then the cost of estimating Jacobians from data rather than using the analytical ones.