"""
examples-RL.evaluate_portfolio_rl
----------------------------------
Load the latest JFB-RL checkpoint and compare against the analytical
Merton optimum on a fresh batch of initial conditions. No retraining.

Run from the repo root:
    python jfb-for-implicit-oc/examples-RL/evaluate_portfolio_rl.py
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# sys.path bootstrap (same pattern as portfolio_optimization_RL.py)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (
    _ROOT,
    os.path.join(_ROOT, "core"),
    os.path.join(_ROOT, "core_RL"),
    os.path.join(_ROOT, "models"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.ImplicitNets import Phi
from core_RL.Environment import AnalyticalEnvironment
from core_RL.ImplicitNets_RL import ImplicitNetOC_RL
from core_RL.JacobianEstimator import RLSJacobianEstimator
from models.PortfolioOC_RL import PortfolioOC_RL

# ===========================================================================
# Config — must match the hyperparameters used during training exactly
# ===========================================================================
DEVICE = "cpu"
N_EVAL = 512        # fresh ICs — never seen during training
N_WARMUP = 10       # rollouts used to warm up the fresh jac_est
WARMUP_STD = 0.4    # exploration noise at warmup start (decays each rollout)
WARMUP_DECAY = 0.6  # multiplicative decay per warmup rollout
SEED = 42

# ===========================================================================
# 1. Rebuild problem + environment (identical hyperparameters to training)
# ===========================================================================
torch.manual_seed(SEED)
np.random.seed(SEED)

prob = PortfolioOC_RL(
    mu_true=0.10, r_true=0.03, lam=0.1,
    W_ref=10, W0_min=0.8, W0_max=1.2,
    batch_size=N_EVAL,
    t_initial=0.0, t_final=2.0, nt=50,
    alphaL=1.0, alphaG=5.0,
    device=DEVICE,
)

env = AnalyticalEnvironment(
    state_dim=prob.state_dim,
    control_dim=prob.control_dim,
    t_initial=prob.t_initial,
    t_final=prob.t_final,
    nt=prob.nt,
    f_callable=prob.compute_f,
    device=DEVICE,
)

# Fresh RLS estimator — will be warmed up with the trained policy below.
jac_est = RLSJacobianEstimator(
    nt=prob.nt,
    state_dim=prob.state_dim,
    control_dim=prob.control_dim,
    dt=prob.h,
    alpha_rls=0.9,
    q0=1.0,
    device=DEVICE,
)

phi = Phi(3, 50, prob.state_dim, dev=DEVICE)

inn = ImplicitNetOC_RL(
    prob.state_dim, prob.control_dim,
    alpha=0.25, max_iters=300, tol=1e-4,
    p_net=phi, oc_problem=prob,
    u_min=-2.0, u_max=2.0, use_control_limits=True,
    dev=DEVICE,
).to(DEVICE)

# ===========================================================================
# 2. Load latest checkpoint
# ===========================================================================
ckpt_dir = os.path.join(_ROOT, "results", "PortfolioOC_RL", "training")
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "best_policy_JFB-RL_RLS_*.pth")))
if not ckpts:
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
ckpt_path = ckpts[-1]
print(f"Checkpoint : {os.path.basename(ckpt_path)}")

inn.load_state_dict(
    torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
)
inn.eval()

# ===========================================================================
# 3. Sample fresh initial conditions
# ===========================================================================
z0_eval = prob.sample_initial_condition()   # (N_EVAL, 1)

# ===========================================================================
# 4. Warm up jac_est with N_WARMUP rollouts of the trained policy + noise.
#
#    Problem without noise: b_k starts at 0 → policy outputs π = 0 → jac_est
#    never sees control variation → b_k stays at 0 (circular deadlock).
#    Fix: add decaying exploration noise so the RLS has enough control
#    excitation to identify b_k = (μ−r)·W from the first rollout.
#    Noise decays to ~0 by the last few warm-up rollouts so the final
#    jac_est estimates reflect the clean policy trajectory.
# ===========================================================================
print(f"Warming up jac_est ({N_WARMUP} rollouts × {N_EVAL} samples, "
      f"exploration std {WARMUP_STD:.2f} → 0) …", end=" ")
with torch.no_grad():
    std = WARMUP_STD
    for _ in range(N_WARMUP):
        z = z0_eval.clone()
        t = prob.t_initial
        for k in range(prob.nt):
            _, b_k = jac_est.AB(k)
            inn.set_step_jacobian(b_k)
            u_k = inn(z, t).view(N_EVAL, prob.control_dim)
            # add exploration noise and clamp to valid range
            u_exc = (u_k + std * torch.randn_like(u_k)).clamp(inn.u_min, inn.u_max)
            z_next = env.step(z, u_exc, t)
            jac_est.update(k, z, u_exc, z_next)
            z = z_next
            t += prob.h
        std *= WARMUP_DECAY   # decay exploration each rollout
print("done.")

# ===========================================================================
# 5. Evaluation rollout — trained JFB-RL policy
# ===========================================================================
with torch.no_grad():
    def _setter(k):
        _, b_k = jac_est.AB(k)
        inn.set_step_jacobian(b_k)

    z_traj_pol, u_traj_pol = env.rollout(
        inn, z0_eval, jac_setter=_setter, return_full_trajectory=True,
    )

# ===========================================================================
# 6. Analytical optimum — exact discrete-time Merton optimal
#
#    Conservation law:  W(t)·p(t) = -αG  (for any π)
#    Optimality:        αL · 2λ π exp(π²) = αG · (μ−r)
#    → π* is constant, solves π exp(π²) = αG(μ−r) / (2λ αL)
# ===========================================================================
rhs = prob.alphaG * (prob.mu_true - prob.r_true) / (2.0 * prob.lam * prob.alphaL)
pi_star = brentq(lambda pi: pi * np.exp(pi**2) - rhs, 1e-9, 2.0)
print(f"Analytical π* = {pi_star:.6f}")

u_opt_const = torch.full(
    (N_EVAL, prob.control_dim), pi_star, dtype=z0_eval.dtype, device=DEVICE,
)
z_traj_opt = torch.zeros(N_EVAL, prob.state_dim, prob.nt + 1, device=DEVICE, dtype=z0_eval.dtype)
u_traj_opt = torch.zeros(N_EVAL, prob.control_dim, prob.nt, device=DEVICE, dtype=z0_eval.dtype)
z_traj_opt[:, :, 0] = z0_eval
z = z0_eval.clone()
t = prob.t_initial

with torch.no_grad():
    for k in range(prob.nt):
        z_next = env.step(z, u_opt_const, t)
        z_traj_opt[:, :, k + 1] = z_next
        u_traj_opt[:, :, k] = u_opt_const
        z = z_next
        t += prob.h

# ===========================================================================
# 7. Cost computation
# ===========================================================================
def total_cost(z_traj, u_traj):
    """Returns (total, running, terminal) as numpy arrays, shape (N_EVAL,)."""
    B = z_traj.shape[0]
    running = torch.zeros(B, device=DEVICE, dtype=z_traj.dtype)
    with torch.no_grad():
        for k in range(prob.nt):
            t_k = prob.t_initial + k * prob.h
            running += prob.h * prob.compute_lagrangian(
                t_k, z_traj[:, :, k], u_traj[:, :, k]
            )
    terminal = prob.compute_G(z_traj[:, :, -1])
    total = prob.alphaL * running + prob.alphaG * terminal
    return (
        total.cpu().numpy(),
        running.cpu().numpy(),
        terminal.cpu().numpy(),
    )

cost_pol, run_pol, term_pol = total_cost(z_traj_pol, u_traj_pol)
cost_opt, run_opt, term_opt = total_cost(z_traj_opt, u_traj_opt)

print()
print(f"{'':>22} {'JFB-RL (ours)':>15} {'Analytical':>12}  {'gap':>8}")
print("-" * 62)
for label, a, b in [
    ("Mean total cost",    cost_pol.mean(), cost_opt.mean()),
    ("  Mean running",     run_pol.mean(),  run_opt.mean()),
    ("  Mean terminal",    term_pol.mean(), term_opt.mean()),
    ("Std  total cost",    cost_pol.std(),  cost_opt.std()),
]:
    print(f"{label:>22} {a:>15.4f} {b:>12.4f}  {a-b:>+8.4f}")

suboptimality = (cost_pol.mean() - cost_opt.mean()) / abs(cost_opt.mean()) * 100
print(f"\nSuboptimality gap : {cost_pol.mean() - cost_opt.mean():.4f}"
      f"  ({suboptimality:+.2f}%)")

# ===========================================================================
# 8. Plots
# ===========================================================================
t_grid = np.linspace(prob.t_initial, prob.t_final, prob.nt + 1)
t_ctrl = t_grid[:-1]

W_pol = z_traj_pol[:, 0, :].cpu().numpy()    # (N_EVAL, nt+1)
W_opt = z_traj_opt[:, 0, :].cpu().numpy()
pi_pol = u_traj_pol[:, 0, :].cpu().numpy()   # (N_EVAL, nt)

COL_POL = "#d6604d"
COL_OPT = "#4393c3"
ALF = 0.20

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"JFB-RL vs Analytical Merton Optimal  —  N = {N_EVAL} fresh ICs\n"
    f"Mean cost:  JFB-RL = {cost_pol.mean():.4f}  |  "
    f"Optimal = {cost_opt.mean():.4f}  |  "
    f"Gap = {cost_pol.mean()-cost_opt.mean():+.4f}  "
    f"({suboptimality:+.2f}%)",
    fontsize=12,
)

# ── Panel 1 : Wealth W(t) ────────────────────────────────────────────
ax = axes[0, 0]
for W, col, label in [(W_pol, COL_POL, "JFB-RL"), (W_opt, COL_OPT, "Optimal")]:
    lo, hi = np.percentile(W, 10, axis=0), np.percentile(W, 90, axis=0)
    ax.fill_between(t_grid, lo, hi, color=col, alpha=ALF)
    ls = "-" if col == COL_POL else "--"
    ax.plot(t_grid, W.mean(axis=0), color=col, lw=2, ls=ls, label=f"{label} (mean ± p10/p90)")
ax.set_xlabel("t"); ax.set_ylabel("W(t)")
ax.set_title("Wealth trajectory W(t)")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

# ── Panel 2 : Portfolio fraction π(t) ────────────────────────────────
ax = axes[0, 1]
lo, hi = np.percentile(pi_pol, 10, axis=0), np.percentile(pi_pol, 90, axis=0)
ax.fill_between(t_ctrl, lo, hi, color=COL_POL, alpha=ALF)
ax.plot(t_ctrl, pi_pol.mean(axis=0), color=COL_POL, lw=2, label="JFB-RL (mean ± p10/p90)")
ax.axhline(pi_star, color=COL_OPT, lw=2, ls="--", label=f"Analytical π* = {pi_star:.4f}")
ax.set_xlabel("t"); ax.set_ylabel("π(t)")
ax.set_title("Portfolio fraction π(t)")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

# ── Panel 3 : Terminal wealth W(T) distribution ───────────────────────
ax = axes[1, 0]
all_W_T = np.concatenate([W_pol[:, -1], W_opt[:, -1]])
bins = np.linspace(all_W_T.min() * 0.97, all_W_T.max() * 1.03, 45)
for W_T, col, label in [(W_pol[:, -1], COL_POL, "JFB-RL"),
                         (W_opt[:, -1], COL_OPT, "Optimal")]:
    ax.hist(W_T, bins=bins, color=col, alpha=0.5, density=True,
            label=f"{label}  (μ={W_T.mean():.3f})")
    ax.axvline(W_T.mean(), color=col, lw=1.5, ls="--")
ax.set_xlabel("W(T)"); ax.set_ylabel("density")
ax.set_title("Terminal wealth W(T)")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

# ── Panel 4 : Total cost J distribution ─────────────────────────────
ax = axes[1, 1]
all_cost = np.concatenate([cost_pol, cost_opt])
bins = np.linspace(all_cost.min() * 0.99, all_cost.max() * 1.01, 45)
for cost, col, label in [(cost_pol, COL_POL, "JFB-RL"),
                          (cost_opt, COL_OPT, "Optimal")]:
    ax.hist(cost, bins=bins, color=col, alpha=0.5, density=True,
            label=f"{label}  (μ={cost.mean():.4f})")
    ax.axvline(cost.mean(), color=col, lw=1.5, ls="--")
ax.set_xlabel("Total cost J"); ax.set_ylabel("density")
ax.set_title("Total cost distribution")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()

out_dir = os.path.join(_ROOT, "results", "PortfolioOC_RL", "evaluation")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "eval_vs_optimal.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()
