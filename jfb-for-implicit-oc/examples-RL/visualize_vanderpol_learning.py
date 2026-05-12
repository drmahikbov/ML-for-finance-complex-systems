from __future__ import annotations

import os
import sys
import re
import glob

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

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
from models.VanDerPolOC_RL import VanDerPolOC_RL


# ===========================================================================
# Config
# ===========================================================================
DEVICE = "cpu"
SEED = 123

BATCH = 64
N_EVAL = 64
N_SHOW = 20

FP_ALPHA = 0.5
FP_MAX_ITERS = 30
FP_TOL = 1e-4
U_MIN, U_MAX = -3.0, 3.0

GRID_SIZE = 60
GIF_FPS = 3

METHOD = "rls"  # "rls" or "oracle"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Problem and environment
# ===========================================================================
prob = VanDerPolOC_RL(
    x10_min=1.5, x10_max=2.5,
    x20_min=-0.5, x20_max=0.5,
    batch_size=BATCH,
    t_initial=0.0,
    t_final=3.0,
    nt=60,
    alphaL=1.0,
    alphaG=5.0,
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


# ===========================================================================
# Builders
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

    return inn


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


def prime_oracle_jacobian(jac_est, z0):
    with torch.no_grad():
        z = z0.detach().clone()
        t = prob.t_initial

        for k in range(prob.nt):
            u = torch.zeros(z.shape[0], prob.control_dim, device=DEVICE)
            z_next = env.step(z, u, t)
            jac_est.update(k, z, u, z_next)
            z = z_next
            t += prob.h


# ===========================================================================
# Cost and rollout
# ===========================================================================
def total_cost_traj(z_traj, u_traj):
    B = z_traj.shape[0]
    running = torch.zeros(B, device=DEVICE)

    with torch.no_grad():
        for k in range(prob.nt):
            t_k = prob.t_initial + k * prob.h
            running += prob.h * prob.compute_lagrangian(
                t_k,
                z_traj[:, :, k],
                u_traj[:, :, k],
            )

    terminal = prob.compute_G(z_traj[:, :, -1])
    total = prob.alphaL * running + prob.alphaG * terminal
    return total.cpu().numpy(), running.cpu().numpy(), terminal.cpu().numpy()


def warmup_rls_jacobian(jac_est, policy, z0_eval, n_warmup=6, std0=0.4, decay=0.6):
    with torch.no_grad():
        std = std0

        for _ in range(n_warmup):
            z = z0_eval.clone()
            t = prob.t_initial

            for k in range(prob.nt):
                _, b_k = jac_est.AB(k)
                policy.set_step_jacobian(b_k)

                u_k = policy(z, t).view(z.shape[0], prob.control_dim)
                u_exc = (u_k + std * torch.randn_like(u_k)).clamp(U_MIN, U_MAX)

                z_next = env.step(z, u_exc, t)
                jac_est.update(k, z, u_exc, z_next)

                z = z_next
                t += prob.h

            std *= decay


def rollout_policy(policy, jac_est, z0_eval, method="rls"):
    policy.eval()
    N = z0_eval.shape[0]

    with torch.no_grad():
        z = z0_eval.clone()
        t = prob.t_initial

        z_traj = torch.zeros(N, prob.state_dim, prob.nt + 1, device=DEVICE)
        u_traj = torch.zeros(N, prob.control_dim, prob.nt, device=DEVICE)
        z_traj[:, :, 0] = z

        for k in range(prob.nt):
            if method == "oracle":
                u_probe = torch.zeros(N, prob.control_dim, device=DEVICE)
                z_probe_next = env.step(z, u_probe, t)
                jac_est.update(k, z, u_probe, z_probe_next)

            _, b_k = jac_est.AB(k)
            policy.set_step_jacobian(b_k)

            u_k = policy(z, t).view(N, prob.control_dim)
            z_next = env.step(z, u_k, t)

            if method == "oracle":
                jac_est.update(k, z, u_k, z_next)

            u_traj[:, :, k] = u_k
            z_traj[:, :, k + 1] = z_next

            z = z_next
            t += prob.h

    return z_traj, u_traj


# ===========================================================================
# Policy heatmap
# ===========================================================================
def policy_heatmap(policy, jac_est, method="rls", t=0.0):
    x1 = np.linspace(-0.5, 2.7, GRID_SIZE)
    x2 = np.linspace(-1.2, 0.6, GRID_SIZE)

    X1, X2 = np.meshgrid(x1, x2)
    z_grid = torch.tensor(
        np.stack([X1.ravel(), X2.ravel()], axis=1),
        dtype=torch.float32,
        device=DEVICE,
    )

    N = z_grid.shape[0]

    with torch.no_grad():
        if method == "oracle":
            u_probe = torch.zeros(N, prob.control_dim, device=DEVICE)
            z_next = env.step(z_grid, u_probe, t)
            jac_est.update(0, z_grid, u_probe, z_next)

        _, b0 = jac_est.AB(0)
        policy.set_step_jacobian(b0)

        u_grid = policy(z_grid, t).view(-1).cpu().numpy()

    return X1, X2, u_grid.reshape(GRID_SIZE, GRID_SIZE)


# ===========================================================================
# Main
# ===========================================================================
def extract_epoch(path):
    match = re.search(r"epoch_(\d+)\.pth$", path)
    return int(match.group(1)) if match else -1


def main():
    out_dir = os.path.join(_ROOT, "results", "VanDerPolOC_RL")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    if METHOD == "rls":
        pattern = os.path.join(ckpt_dir, "jfb_rl_rls_epoch_*.pth")
    elif METHOD == "oracle":
        pattern = os.path.join(ckpt_dir, "jfb_rl_oracle_epoch_*.pth")
    else:
        raise ValueError("METHOD must be 'rls' or 'oracle'.")

    ckpt_paths = sorted(glob.glob(pattern), key=extract_epoch)

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found with pattern:\n{pattern}\n"
            "You need to rerun training after adding checkpoint saving."
        )

    print(f"Found {len(ckpt_paths)} checkpoints.")

    frames_dir = os.path.join(out_dir, f"learning_frames_{METHOD}")
    os.makedirs(frames_dir, exist_ok=True)

    # Fixed evaluation ICs for all frames.
    torch.manual_seed(SEED + 1)
    prob.batch_size = N_EVAL
    z0_eval = prob.sample_initial_condition()
    prob.batch_size = BATCH

    rng = np.random.default_rng(0)
    show_idx = rng.choice(N_EVAL, min(N_SHOW, N_EVAL), replace=False)

    t_grid = np.linspace(prob.t_initial, prob.t_final, prob.nt + 1)
    t_ctrl = t_grid[:-1]

    frame_paths = []

    for frame_id, ckpt_path in enumerate(ckpt_paths):
        epoch = extract_epoch(ckpt_path)
        print(f"Rendering epoch {epoch:04d}...")

        policy = make_jfb_policy()
        policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        policy.eval()

        if METHOD == "rls":
            jac_est = make_rls_jacobian_estimator()
            warmup_rls_jacobian(jac_est, policy, z0_eval)
        else:
            jac_est = make_oracle_jacobian_estimator()
            prime_oracle_jacobian(jac_est, z0_eval)

        z_traj, u_traj = rollout_policy(policy, jac_est, z0_eval, method=METHOD)
        cost, running, terminal = total_cost_traj(z_traj, u_traj)

        x1 = z_traj[:, 0, :].cpu().numpy()
        x2 = z_traj[:, 1, :].cpu().numpy()
        u = u_traj[:, 0, :].cpu().numpy()

        X1, X2, U = policy_heatmap(policy, jac_est, method=METHOD, t=0.0)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Van der Pol — JFB-RL/{METHOD.upper()} learning visualization\n"
            f"Epoch {epoch:04d} | mean cost = {cost.mean():.3f}",
            fontsize=12,
        )

        # Panel 1: phase portrait
        ax = axes[0, 0]
        for i in show_idx:
            ax.plot(x1[i], x2[i], color="tab:red", alpha=0.35, lw=1.0)
        ax.plot(0, 0, "k*", ms=10, label="target")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Closed-loop trajectories")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        # Panel 2: policy heatmap
        ax = axes[0, 1]
        im = ax.contourf(X1, X2, U, levels=40, cmap="coolwarm")
        plt.colorbar(im, ax=ax, label="u(t=0,x)")
        ax.plot(0, 0, "k*", ms=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Policy heatmap at t=0")
        ax.grid(True, ls="--", alpha=0.25)

        # Panel 3: control signal
        ax = axes[1, 0]
        lo = np.percentile(u, 10, axis=0)
        hi = np.percentile(u, 90, axis=0)
        ax.fill_between(t_ctrl, lo, hi, color="tab:red", alpha=0.2)
        ax.plot(t_ctrl, u.mean(axis=0), color="tab:red", lw=2)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.set_title("Control signal, mean ± p10/p90")
        ax.grid(True, ls="--", alpha=0.4)

        # Panel 4: cost distribution
        ax = axes[1, 1]
        ax.hist(cost, bins=25, density=True, alpha=0.6, color="tab:red")
        ax.axvline(cost.mean(), color="k", lw=2, ls="--", label=f"mean={cost.mean():.3f}")
        ax.set_xlabel("Total cost J")
        ax.set_ylabel("density")
        ax.set_title("Cost distribution")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        frame_path = os.path.join(frames_dir, f"frame_epoch_{epoch:04d}.png")
        plt.savefig(frame_path, dpi=130)
        plt.close(fig)

        frame_paths.append(frame_path)

    gif_path = os.path.join(out_dir, f"vanderpol_learning_{METHOD}.gif")

    base_img = Image.open(frame_paths[0]).convert("RGB")
    base_size = base_img.size  # (width, height)

    images = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")

        if img.size != base_size:
            img = img.resize(base_size)

        images.append(np.array(img))

    imageio.mimsave(gif_path, images, fps=GIF_FPS)

    print(f"\nSaved GIF to:\n{gif_path}")


if __name__ == "__main__":
    main()