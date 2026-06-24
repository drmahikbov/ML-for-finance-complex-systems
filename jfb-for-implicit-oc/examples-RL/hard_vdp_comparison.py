"""
examples-RL.hard_vdp_comparison
---------------------------------
Three-way benchmark on the Hard Van der Pol oscillator (HardVanDerPolOC_RL):
JFB-RL / RLS, JFB-RL / Oracle Jacobians, and Autodiff-BPTT. The hard variant
adds a control regularisation term (lambda_u). Same structure as
vanderpol_comparison.py.

Run from the repo root:
    python jfb-for-implicit-oc/examples-RL/hard_vdp_comparison.py
"""

from __future__ import annotations

import os
import sys
import time
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# sys.path bootstrap
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
from core_RL.JacobianEstimator import RLSJacobianEstimator, OracleJacobianEstimator
from models.Hard_VDP_RL import HardVanDerPolOC_RL

# ===========================================================================
# Config
# ===========================================================================
DEVICE = "cpu"
SEED   = 1

BATCH      = 64      # training batch size
N_EPOCHS   = 300     # epochs per method
LR         = 3e-3    # Adam learning rate for both methods
GRAD_CLIP  = 1.0     # gradient norm clip (applied to both)


# JFB-RL specific
EXPLORE_STD   = 0.3   # initial exploration noise standard deviation
EXPLORE_DECAY = 0.99  # multiplicative decay per epoch

#modified hyperparam to adapt to new loss
FP_ALPHA      = 0.05   # fixed-point step size  (< 2/α_L = 2 for this problem)
FP_MAX_ITERS  = 80    # maximum FP iterations
FP_TOL        = 1e-4  # FP convergence tolerance
U_MIN, U_MAX  = -2.0, 2.0
LAMBDA_U      = 0.05

# Evaluation
N_EVAL        = 128   # fresh test ICs (never seen during training)
N_WARMUP      = 8     # jac_est warm-up rollouts for JFB-RL evaluation
WARMUP_STD    = 0.4
WARMUP_DECAY  = 0.6

# Smoothing for loss curves in the plot
SMOOTH_WIN = 15       # moving-average window (epochs)
SAVE_EVERY = 25

torch.manual_seed(SEED)
np.random.seed(SEED)

# ===========================================================================
# Problem
# ===========================================================================
prob = HardVanDerPolOC_RL(
    x10_min=1.5, x10_max=2.5,
    x20_min=-0.5, x20_max=0.5,
    batch_size=BATCH,
    t_initial=0.0, t_final=3.0, nt=60,
    alphaL=1.0, alphaG=5.0,
    lambda_u= LAMBDA_U,
    device=DEVICE,
)
print(f"Problem  : {prob.oc_problem_name}")
print(f"State    : {prob.state_dim}-D  |  Control: {prob.control_dim}-D")
print(f"Horizon  : T={prob.t_final}, nt={prob.nt}, h={prob.h:.3f}")
print(f"ICs      : x1~U[{prob.x10_min},{prob.x10_max}], x2~U[{prob.x20_min},{prob.x20_max}]")
print(f"Batch    : {BATCH}  |  Epochs: {N_EPOCHS}  |  LR: {LR}")
print("-" * 60)

# ===========================================================================
# A. JFB-RL setup
# ===========================================================================
def make_jfb_policy():
    phi = Phi(3, 50, prob.state_dim, dev=DEVICE)

    inn = ImplicitNetOC_RL(
        prob.state_dim,
        prob.control_dim,
        alpha=FP_ALPHA,
        max_iters=FP_MAX_ITERS,
        tol=FP_TOL,
        p_net=phi,
        oc_problem=prob,
        u_min=U_MIN,
        u_max=U_MAX,
        use_control_limits=True,
        dev=DEVICE,
    ).to(DEVICE)

    opt = torch.optim.Adam(inn.parameters(), lr=LR)
    return inn, opt

inn, opt_jfb = make_jfb_policy()
initial_jfb_state = copy.deepcopy(inn.state_dict())

def make_rls_jacobian_estimator():
    return RLSJacobianEstimator(
        nt=prob.nt,
        state_dim=prob.state_dim,
        control_dim=prob.control_dim,
        dt=prob.h,
        alpha_rls=0.9,
        q0=1.0,
        device=DEVICE,
    )


def make_oracle_jacobian_estimator():
    return OracleJacobianEstimator(
        nt=prob.nt,
        state_dim=prob.state_dim,
        control_dim=prob.control_dim,
        dt=prob.h,
        grad_f_z=prob.compute_grad_f_z,
        grad_f_u=prob.compute_grad_f_u,
        schedule_t=lambda k: prob.t_initial + k * prob.h,
        device=DEVICE,
    )

jac_est = make_rls_jacobian_estimator()

def prime_oracle_jacobian(jac_est, z0):
    """
    Initializes OracleJacobianEstimator caches so that AB(k) can be called
    before the first training update.

    For Van der Pol, this is safe because B = df/du is constant.
    The A values will be overwritten during the clean rollout.
    """
    with torch.no_grad():
        z = z0.detach().clone()
        t = prob.t_initial

        for k in range(prob.nt):
            u = torch.zeros(z.shape[0], prob.control_dim, device=DEVICE)
            z_next = env.step(z, u, t)
            jac_est.update(k, z, u, z_next)

            z = z_next
            t += prob.h

env = AnalyticalEnvironment(
    state_dim=prob.state_dim,
    control_dim=prob.control_dim,
    t_initial=prob.t_initial,
    t_final=prob.t_final,
    nt=prob.nt,
    f_callable=prob.compute_f,
    device=DEVICE,
)

out_dir = os.path.join(_ROOT, "results", "HardVanDerPolOC_RL", f"seed_{SEED}")
ckpt_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)


# ===========================================================================
# B. Autodiff-BPTT setup — explicit MLP policy trained with BPTT
#
#    The MLP maps (z, t) -> u directly.  The training loop keeps the full
#    computation graph through the Euler rollout, so PyTorch autograd
#    propagates ∂J/∂θ through all nt dynamics steps.  This is only possible
#    because the dynamics are differentiable (we call prob.compute_f with
#    grad enabled).  In a real RL setting with a simulator black box, BPTT
#    would not be available.
# ===========================================================================

class MLPPolicy(nn.Module):
    """Explicit policy  π_θ(z, t) → u.  Input: (x1, x2, t), output: scalar u."""

    def __init__(self, state_dim: int, control_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),        nn.Tanh(),
            nn.Linear(hidden, control_dim),
        )
        self.u_min = U_MIN
        self.u_max = U_MAX

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        t_feat = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype) * t
        x = torch.cat([z, t_feat], dim=-1)
        return self.net(x).clamp(self.u_min, self.u_max)


mlp = MLPPolicy(prob.state_dim, prob.control_dim, hidden=64).to(DEVICE)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=LR)

# ===========================================================================
# Training helper — Autodiff-BPTT step
# ===========================================================================

def autodiff_step(policy: MLPPolicy, prob: HardVanDerPolOC_RL, opt) -> dict:
    """One BPTT training step — differentiable Euler rollout."""
    policy.train()
    opt.zero_grad()

    z0 = prob.sample_initial_condition()
    z = z0.clone()           # starts with no grad; graph builds from first u_k
    running = torch.zeros(prob.batch_size, device=DEVICE)

    for k in range(prob.nt):
        t_k = prob.t_initial + k * prob.h
        u_k = policy(z, t_k)                          # (B, 1)
        running = running + prob.h * prob.compute_lagrangian(t_k, z, u_k)
        dz = prob.compute_f(t_k, z, u_k)             # differentiable!
        z = z + prob.h * dz                           # graph grows nt steps deep

    terminal = prob.compute_G(z)
    loss = (prob.alphaL * running + prob.alphaG * terminal).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    opt.step()

    return {
        "loss":          loss.item(),
        "running_cost":  (prob.alphaL * running.mean()).item(),
        "terminal_cost": (prob.alphaG * terminal.mean()).item(),
    }


# ===========================================================================
# Train A: JFB-RL
# ===========================================================================
def train_jfb_variant(
    name: str,
    inn: ImplicitNetOC_RL,
    opt,
    jac_est,
    n_epochs: int,
    exploration_std0: float,
    exploration_decay: float,
    ckpt_prefix: str | None = None,
):
    print(f"Training {name} …")

    hist = {"loss": [], "running_cost": [], "terminal_cost": [], "lin_residual": []}
    explore_std = exploration_std0
    t0 = time.time()

    # Save initial policy before training.
    if ckpt_prefix is not None:
        torch.save(
            inn.state_dict(),
            os.path.join(ckpt_dir, f"{ckpt_prefix}_epoch_0000.pth"),
        )

    for epoch in range(1, n_epochs + 1):
        z0 = prob.sample_initial_condition()

        out = prob.compute_loss_RL(
            policy=inn,
            env=env,
            jac_est=jac_est,
            z0=z0,
            exploration_std=explore_std,
        )

        opt.zero_grad()
        out["surrogate"].backward()
        torch.nn.utils.clip_grad_norm_(inn.parameters(), GRAD_CLIP)
        opt.step()

        explore_std *= exploration_decay

        hist["loss"].append(out["total_cost"])
        hist["running_cost"].append(out["running_cost"])
        hist["terminal_cost"].append(out["terminal_cost"])
        hist["lin_residual"].append(out["lin_residual"])

        if ckpt_prefix is not None and epoch % SAVE_EVERY == 0:
            torch.save(
                inn.state_dict(),
                os.path.join(ckpt_dir, f"{ckpt_prefix}_epoch_{epoch:04d}.pth"),
            )

        if epoch % 50 == 0:
            print(
                f"  [{epoch:4d}/{n_epochs}]  "
                f"loss={out['total_cost']:.4f}  "
                f"L={out['running_cost']:.3f}  "
                f"G={out['terminal_cost']:.3f}  "
                f"lin_res={out['lin_residual']:.3e}  "
                f"explore_std={explore_std:.3f}"
            )

    print(f"{name} training done in {time.time() - t0:.1f}s")
    print("-" * 60)

    return hist

# =======================================================================
# Train A1: JFB-RL with RLS Jacobians
# =======================================================================

inn_rls, opt_rls = make_jfb_policy()
inn_rls.load_state_dict(initial_jfb_state)

jac_est_rls = make_rls_jacobian_estimator()

hist_jfb_rls = train_jfb_variant(
    name="JFB-RL / RLS Jacobians",
    inn=inn_rls,
    opt=opt_rls,
    jac_est=jac_est_rls,
    n_epochs=N_EPOCHS,
    exploration_std0=EXPLORE_STD,
    exploration_decay=EXPLORE_DECAY,
    ckpt_prefix="hard_jfb_rl_rls",
)

# =======================================================================
# Train A2: JFB-RL with Oracle Jacobians
# =======================================================================

inn_oracle, opt_oracle = make_jfb_policy()
inn_oracle.load_state_dict(initial_jfb_state)

jac_est_oracle = make_oracle_jacobian_estimator()

# Prime once before the first AB(k) call.
z0_prime = prob.sample_initial_condition()
prime_oracle_jacobian(jac_est_oracle, z0_prime)

hist_jfb_oracle = train_jfb_variant(
    name="JFB-RL / Oracle Jacobians",
    inn=inn_oracle,
    opt=opt_oracle,
    jac_est=jac_est_oracle,
    n_epochs=N_EPOCHS,
    exploration_std0=0.0,
    exploration_decay=1.0,
)

# ===========================================================================
# Train B: Autodiff-BPTT
# ===========================================================================
print("Training Autodiff-BPTT …")
hist_mlp = {"loss": [], "running_cost": [], "terminal_cost": []}
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    out = autodiff_step(mlp, prob, opt_mlp)

    hist_mlp["loss"].append(out["loss"])
    hist_mlp["running_cost"].append(out["running_cost"])
    hist_mlp["terminal_cost"].append(out["terminal_cost"])

    if epoch % 50 == 0:
        print(f"  [{epoch:4d}/{N_EPOCHS}]  loss={out['loss']:.4f}  "
              f"L={out['running_cost']:.3f}  G={out['terminal_cost']:.3f}")

print(f"Autodiff-BPTT training done in {time.time()-t0:.1f}s")
print("-" * 60)

# ===========================================================================
# Save checkpoints
# ===========================================================================

torch.save(inn_rls.state_dict(),     os.path.join(ckpt_dir, "hard_jfb_rl_rls.pth"))
torch.save(inn_oracle.state_dict(),  os.path.join(ckpt_dir, "hard_jfb_rl_oracle.pth"))
torch.save(mlp.state_dict(),         os.path.join(ckpt_dir, "hard_autodiff.pth"))
print(f"Checkpoints saved to {ckpt_dir}")

# ===========================================================================
# Evaluation — fresh ICs never seen during training
# ===========================================================================
def total_cost_traj(z_traj, u_traj):
    B = z_traj.shape[0]
    running = torch.zeros(B, device=DEVICE)
    with torch.no_grad():
        for k in range(prob.nt):
            t_k = prob.t_initial + k * prob.h
            running += prob.h * prob.compute_lagrangian(t_k, z_traj[:, :, k], u_traj[:, :, k])
    terminal = prob.compute_G(z_traj[:, :, -1])
    total = prob.alphaL * running + prob.alphaG * terminal
    return total.cpu().numpy(), running.cpu().numpy(), terminal.cpu().numpy()

def evaluate_implicit_policy(
    name: str,
    inn: ImplicitNetOC_RL,
    jac_est,
    z0_eval: torch.Tensor,
    warmup_rls: bool = False,
):
    inn.eval()
    N = z0_eval.shape[0]

    if warmup_rls:
        print(f"  Warming up {name} jacobian estimator …", end=" ")
        with torch.no_grad():
            std = WARMUP_STD
            for _ in range(N_WARMUP):
                z = z0_eval.clone()
                t = prob.t_initial

                for k in range(prob.nt):
                    _, b_k = jac_est.AB(k)
                    inn.set_step_jacobian(b_k)

                    u_k = inn(z, t).view(N, prob.control_dim)
                    u_exc = (u_k + std * torch.randn_like(u_k)).clamp(U_MIN, U_MAX)

                    z_next = env.step(z, u_exc, t)
                    jac_est.update(k, z, u_exc, z_next)

                    z = z_next
                    t += prob.h

                std *= WARMUP_DECAY
        print("done.")

    else:
        # Needed for OracleJacobianEstimator.
        if isinstance(jac_est, OracleJacobianEstimator):
            prime_oracle_jacobian(jac_est, z0_eval)

    with torch.no_grad():
        z = z0_eval.clone()
        t = prob.t_initial

        z_traj = torch.zeros(N, prob.state_dim, prob.nt + 1, device=DEVICE)
        u_traj = torch.zeros(N, prob.control_dim, prob.nt, device=DEVICE)
        z_traj[:, :, 0] = z

        for k in range(prob.nt):
            _, b_k = jac_est.AB(k)
            inn.set_step_jacobian(b_k)

            u_k = inn(z, t).view(N, prob.control_dim)
            z_next = env.step(z, u_k, t)

            # For oracle, update cache on the actual clean trajectory.
            if isinstance(jac_est, OracleJacobianEstimator):
                jac_est.update(k, z, u_k, z_next)

            u_traj[:, :, k] = u_k
            z_traj[:, :, k + 1] = z_next

            z = z_next
            t += prob.h

    cost, running, terminal = total_cost_traj(z_traj, u_traj)

    print(
        f"{name:>24}: "
        f"mean={cost.mean():.4f}, std={cost.std():.4f}, "
        f"running={running.mean():.4f}, terminal={terminal.mean():.4f}"
    )

    return z_traj, u_traj, cost, running, terminal

print(f"\nEvaluating on N_EVAL={N_EVAL} fresh ICs …")

# Sample fixed test ICs for a fair comparison.
torch.manual_seed(SEED + 1)
prob.batch_size = N_EVAL
z0_eval = prob.sample_initial_condition()
prob.batch_size = BATCH   # restore

# ------ JFB-RL evaluation ---------------------------------------------------
# Build a fresh jac_est for the test set and warm it up with decaying
# exploration noise (same technique as evaluate_portfolio_rl.py).
# JFB-RL / RLS

jac_est_rls_eval = make_rls_jacobian_estimator()

z_traj_rls, u_traj_rls, cost_rls, run_rls, term_rls = evaluate_implicit_policy(
    name="JFB-RL / RLS",
    inn=inn_rls,
    jac_est=jac_est_rls_eval,
    z0_eval=z0_eval,
    warmup_rls=True,
)

# JFB-RL / Oracle
jac_est_oracle_eval = make_oracle_jacobian_estimator()

z_traj_oracle, u_traj_oracle, cost_oracle, run_oracle, term_oracle = evaluate_implicit_policy(
    name="JFB-RL / Oracle",
    inn=inn_oracle,
    jac_est=jac_est_oracle_eval,
    z0_eval=z0_eval,
    warmup_rls=False,
)

# ------ Autodiff-BPTT evaluation --------------------------------------------
mlp.eval()
with torch.no_grad():
    z_ad = z0_eval.clone()
    t_ad = prob.t_initial
    z_traj_ad = torch.zeros(N_EVAL, prob.state_dim, prob.nt + 1, device=DEVICE)
    u_traj_ad = torch.zeros(N_EVAL, prob.control_dim, prob.nt, device=DEVICE)
    z_traj_ad[:, :, 0] = z_ad
    for k in range(prob.nt):
        u_k = mlp(z_ad, t_ad).view(N_EVAL, prob.control_dim)
        z_next = env.step(z_ad, u_k, t_ad)
        u_traj_ad[:, :, k] = u_k
        z_traj_ad[:, :, k + 1] = z_next
        z_ad = z_next
        t_ad += prob.h

# ------ Cost computation ----------------------------------------------------



cost_rls, run_rls, term_rls = total_cost_traj(z_traj_rls, u_traj_rls)
cost_oracle, run_oracle, term_oracle = total_cost_traj(z_traj_oracle, u_traj_oracle)
cost_ad,  run_ad,  term_ad  = total_cost_traj(z_traj_ad,  u_traj_ad)

print("\nSummary")
print("-" * 70)
print(f"{'JFB-RL / RLS':>24}: {cost_rls.mean():.4f}")
print(f"{'JFB-RL / Oracle':>24}: {cost_oracle.mean():.4f}")
print(f"{'Autodiff-BPTT':>24}: {cost_ad.mean():.4f}")

print("\nGaps")
print("-" * 70)
print(f"RLS - Oracle       : {cost_rls.mean() - cost_oracle.mean():+.4f}")
print(f"Oracle - Autodiff  : {cost_oracle.mean() - cost_ad.mean():+.4f}")
print(f"RLS - Autodiff     : {cost_rls.mean() - cost_ad.mean():+.4f}")

# ===========================================================================
# Plots
# ===========================================================================
t_grid = np.linspace(prob.t_initial, prob.t_final, prob.nt + 1)
t_ctrl = t_grid[:-1]

# RLS-JFB
x1_rls = z_traj_rls[:, 0, :].cpu().numpy()
x2_rls = z_traj_rls[:, 1, :].cpu().numpy()
u_rls  = u_traj_rls[:, 0, :].cpu().numpy()

# Oracle-JFB
x1_oracle = z_traj_oracle[:, 0, :].cpu().numpy()
x2_oracle = z_traj_oracle[:, 1, :].cpu().numpy()
u_oracle  = u_traj_oracle[:, 0, :].cpu().numpy()

# Autodiff-BPTT
x1_ad = z_traj_ad[:, 0, :].cpu().numpy()
x2_ad = z_traj_ad[:, 1, :].cpu().numpy()
u_ad  = u_traj_ad[:, 0, :].cpu().numpy()

COL_RLS    = "#d6604d"   # red
COL_ORACLE = "#1b9e77"   # green
COL_AD     = "#4393c3"   # blue
ALF        = 0.18


def smooth(arr, w):
    """Simple moving-average smoothing."""
    arr = np.asarray(arr)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


gap_rls_ad = cost_rls.mean() - cost_ad.mean()
gap_oracle_ad = cost_oracle.mean() - cost_ad.mean()
gap_rls_oracle = cost_rls.mean() - cost_oracle.mean()

gap_rls_ad_pct = 100.0 * gap_rls_ad / cost_ad.mean()
gap_oracle_ad_pct = 100.0 * gap_oracle_ad / cost_ad.mean()


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "Hard Van der Pol stabilisation — RLS-JFB vs Oracle-JFB vs Autodiff-BPTT\n"
    f"N = {N_EVAL} fresh ICs | "
    f"RLS = {cost_rls.mean():.3f}, "
    f"Oracle = {cost_oracle.mean():.3f}, "
    f"Autodiff = {cost_ad.mean():.3f}\n"
    f"RLS-Oracle gap = {gap_rls_oracle:+.3f} | "
    f"Oracle-Autodiff gap = {gap_oracle_ad:+.3f} ({gap_oracle_ad_pct:+.1f}%) | "
    f"RLS-Autodiff gap = {gap_rls_ad:+.3f} ({gap_rls_ad_pct:+.1f}%)",
    fontsize=11,
)

# ── Panel 1: Training loss curves ───────────────────────────────────────────
ax = axes[0, 0]

losses_rls_smooth = smooth(hist_jfb_rls["loss"], SMOOTH_WIN)
losses_oracle_smooth = smooth(hist_jfb_oracle["loss"], SMOOTH_WIN)
losses_ad_smooth = smooth(hist_mlp["loss"], SMOOTH_WIN)

if len(losses_rls_smooth) == N_EPOCHS:
    ep_sm = np.arange(1, N_EPOCHS + 1)
else:
    ep_sm = np.arange(SMOOTH_WIN, N_EPOCHS + 1)

ax.plot(hist_jfb_rls["loss"], color=COL_RLS, alpha=0.22, lw=0.8)
ax.plot(hist_jfb_oracle["loss"], color=COL_ORACLE, alpha=0.22, lw=0.8)
ax.plot(hist_mlp["loss"], color=COL_AD, alpha=0.22, lw=0.8)

ax.plot(
    ep_sm,
    losses_rls_smooth,
    color=COL_RLS,
    lw=2,
    label=f"JFB-RL / RLS (final={hist_jfb_rls['loss'][-1]:.3f})",
)
ax.plot(
    ep_sm,
    losses_oracle_smooth,
    color=COL_ORACLE,
    lw=2,
    ls="-.",
    label=f"JFB-RL / Oracle (final={hist_jfb_oracle['loss'][-1]:.3f})",
)
ax.plot(
    ep_sm,
    losses_ad_smooth,
    color=COL_AD,
    lw=2,
    ls="--",
    label=f"Autodiff-BPTT (final={hist_mlp['loss'][-1]:.3f})",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Total cost J")
ax.set_title("Training convergence")
ax.legend(fontsize=8)
ax.grid(True, ls="--", alpha=0.4)


# ── Panel 2: Phase portrait ─────────────────────────────────────────────────
ax = axes[0, 1]
n_show = min(20, N_EVAL)
rng = np.random.default_rng(0)
idx = rng.choice(N_EVAL, n_show, replace=False)

for i in idx:
    ax.plot(x1_rls[i], x2_rls[i], color=COL_RLS, alpha=0.35, lw=0.8)
    ax.plot(x1_oracle[i], x2_oracle[i], color=COL_ORACLE, alpha=0.35, lw=0.8, ls="-.")
    ax.plot(x1_ad[i], x2_ad[i], color=COL_AD, alpha=0.35, lw=0.8, ls="--")

rep = idx[0]
ax.plot(x1_rls[rep], x2_rls[rep], color=COL_RLS, lw=2, label="JFB-RL / RLS")
ax.plot(x1_oracle[rep], x2_oracle[rep], color=COL_ORACLE, lw=2, ls="-.", label="JFB-RL / Oracle")
ax.plot(x1_ad[rep], x2_ad[rep], color=COL_AD, lw=2, ls="--", label="Autodiff-BPTT")
ax.plot(0, 0, "k*", ms=10, label="target")

ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_title("Phase portrait (20 test trajectories)")
ax.legend(fontsize=8)
ax.grid(True, ls="--", alpha=0.4)


# ── Panel 3: Control u(t), mean ± p10/p90 ───────────────────────────────────
ax = axes[1, 0]

for u_arr, col, ls, label in [
    (u_rls, COL_RLS, "-", "JFB-RL / RLS"),
    (u_oracle, COL_ORACLE, "-.", "JFB-RL / Oracle"),
    (u_ad, COL_AD, "--", "Autodiff-BPTT"),
]:
    lo = np.percentile(u_arr, 10, axis=0)
    hi = np.percentile(u_arr, 90, axis=0)
    ax.fill_between(t_ctrl, lo, hi, color=col, alpha=ALF)
    ax.plot(
        t_ctrl,
        u_arr.mean(axis=0),
        color=col,
        lw=2,
        ls=ls,
        label=f"{label} (mean ± p10/p90)",
    )

ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("t")
ax.set_ylabel("u(t)")
ax.set_title("Control signal u(t)")
ax.legend(fontsize=8)
ax.grid(True, ls="--", alpha=0.4)


# ── Panel 4: Total cost distribution ────────────────────────────────────────
ax = axes[1, 1]

all_cost = np.concatenate([cost_rls, cost_oracle, cost_ad])
bins = np.linspace(all_cost.min() * 0.97, all_cost.max() * 1.03, 40)

for cost, col, ls, label in [
    (cost_rls, COL_RLS, "-", "JFB-RL / RLS"),
    (cost_oracle, COL_ORACLE, "-.", "JFB-RL / Oracle"),
    (cost_ad, COL_AD, "--", "Autodiff-BPTT"),
]:
    ax.hist(
        cost,
        bins=bins,
        color=col,
        alpha=0.38,
        density=True,
        label=f"{label} (μ={cost.mean():.3f})",
    )
    ax.axvline(cost.mean(), color=col, lw=1.8, ls=ls)

ax.set_xlabel("Total cost J")
ax.set_ylabel("density")
ax.set_title("Cost distribution on test set")
ax.legend(fontsize=8)
ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()

fig_path = os.path.join(out_dir, "hard_comparison_oracle_vs_rls_seed_{SEED}.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {fig_path}")
plt.show()