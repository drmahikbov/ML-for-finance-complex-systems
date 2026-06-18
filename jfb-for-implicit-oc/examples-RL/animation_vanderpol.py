"""
examples-RL.animation_vanderpol
---------------------------------
Animate the Hard Gain Van der Pol policy improving over training epochs.

Loads per-epoch checkpoints from results/HardGainVanDerPolOC_RL/checkpoints/,
rolls out a fixed initial condition under each, and assembles the results into
a two-panel animation (physical view + phase portrait). Saves as a GIF when
SAVE_GIF is True.
"""
from __future__ import annotations

import os
import sys
import re
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
from core_RL.JacobianEstimator import RLSJacobianEstimator
from models.Hard_Gain_VDP_RL import HardGainVanDerPolOC_RL


# ===========================================================================
# Config
# ===========================================================================
DEVICE = "cpu"
SEED = 123

# Problem config (must match training)
BATCH = 64
FP_ALPHA = 0.05
FP_MAX_ITERS = 80
FP_TOL = 1e-4
U_MIN, U_MAX = -2.0, 2.0

# Animation config
CHECKPOINT_PATTERN = "hardg_jfb_rl_rls_epoch_*.pth"
FPS = 20
PAUSE_FRAMES = 20         # pause after each episode
TAIL_LENGTH = None        # None = draw full past trajectory, or integer
N_WARMUP = 6
WARMUP_STD = 0.4
WARMUP_DECAY = 0.6

# Save either GIF or not
SAVE_GIF = True
GIF_NAME = "hardgvanderpol_rl_learning_style.gif"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Problem / env
# ===========================================================================
prob = HardGainVanDerPolOC_RL(
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
# Helpers
# ===========================================================================
def extract_epoch(path: str) -> int:
    match = re.search(r"epoch_(\d+)\.pth$", path)
    return int(match.group(1)) if match else -1


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


def warmup_rls_jacobian(jac_est, policy, z0_eval, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        std = WARMUP_STD

        for _ in range(N_WARMUP):
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

            std *= WARMUP_DECAY


def rollout_single(policy, jac_est, z0_single):
    """
    Rollout one trajectory starting from a single initial condition.
    Returns:
        z_traj: (nt+1, 2)
        u_traj: (nt, 1)
        total_cost: scalar
    """
    policy.eval()

    with torch.no_grad():
        z = z0_single.clone().view(1, prob.state_dim)
        t = prob.t_initial

        z_traj = torch.zeros(prob.nt + 1, prob.state_dim, device=DEVICE)
        u_traj = torch.zeros(prob.nt, prob.control_dim, device=DEVICE)
        z_traj[0] = z[0]

        for k in range(prob.nt):
            _, b_k = jac_est.AB(k)
            policy.set_step_jacobian(b_k)

            u_k = policy(z, t).view(1, prob.control_dim)
            z_next = env.step(z, u_k, t)

            u_traj[k] = u_k[0]
            z_traj[k + 1] = z_next[0]

            z = z_next
            t += prob.h

    # Compute total cost for this one rollout
    z_batch = z_traj.unsqueeze(0).transpose(1, 2)  # (1, 2, nt+1)
    u_batch = u_traj.unsqueeze(0).transpose(1, 2)  # (1, 1, nt)
    total, _, _ = total_cost_traj(z_batch, u_batch)

    return z_traj.cpu().numpy(), u_traj.cpu().numpy(), float(total[0])


# ===========================================================================
# Load all checkpoint rollouts
# ===========================================================================
def build_rollout_database():
    out_dir = os.path.join(_ROOT, "results", "HardGainVanDerPolOC_RL")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    pattern = os.path.join(ckpt_dir, CHECKPOINT_PATTERN)
    ckpt_paths = sorted(glob.glob(pattern), key=extract_epoch)

    if not ckpt_paths:
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé avec le pattern:\n{pattern}\n"
            "Il faut d'abord relancer l'entraînement avec la sauvegarde des checkpoints."
        )

    print(f"{len(ckpt_paths)} checkpoints found.")

    # Fixed initial condition for all episodes
    torch.manual_seed(SEED + 1)
    prob.batch_size = 1
    z0_single = prob.sample_initial_condition()[0].to(DEVICE)
    prob.batch_size = BATCH

    print(f"Used initial conditions for the : {z0_single.tolist()}")

    episodes = []

    for ckpt_path in ckpt_paths:
        epoch = extract_epoch(ckpt_path)
        print(f"Preparing rollout epoch {epoch:04d}...")

        policy = make_jfb_policy()
        policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        policy.eval()

        jac_est = make_rls_jacobian_estimator()

        # Warm-up on the SAME initial condition repeated
        z0_warm = z0_single.view(1, -1).repeat(BATCH, 1)
        warmup_rls_jacobian(jac_est, policy, z0_warm, seed=SEED)

        z_traj, u_traj, total_cost = rollout_single(policy, jac_est, z0_single)

        episodes.append(
            {
                "epoch": epoch,
                "z_traj": z_traj,   # (nt+1, 2)
                "u_traj": u_traj,   # (nt, 1)
                "cost": total_cost,
            }
        )

    return episodes, z0_single.cpu().numpy(), out_dir


# ===========================================================================
# Animation
# ===========================================================================
def main():
    episodes, z0_single, out_dir = build_rollout_database()

    # Time structure
    frames_per_episode = prob.nt + 1
    total_frames = len(episodes) * (frames_per_episode + PAUSE_FRAMES)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_phys = axes[0]
    ax_phase = axes[1]

    # ---------------- Left panel: "physical" view ----------------
    ax_phys.set_title("Simplified physical view")
    ax_phys.set_xlim(-0.5, 2.8)
    ax_phys.set_ylim(-1.0, 1.0)
    ax_phys.set_aspect("equal", adjustable="box")
    ax_phys.grid(True, ls="--", alpha=0.3)

    # rail
    rail_y = 0.0
    ax_phys.plot([-0.4, 2.7], [rail_y, rail_y], color="black", lw=2)
    ax_phys.plot(0.0, rail_y, "k*", ms=12, label="target")
    ax_phys.legend(loc="upper right")

    ball, = ax_phys.plot([], [], "o", color="crimson", ms=16)
    arrow = None

    epoch_text = ax_phys.text(
        0.02, 0.95, "", transform=ax_phys.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold"
    )
    cost_text = ax_phys.text(
        0.02, 0.87, "", transform=ax_phys.transAxes,
        ha="left", va="top", fontsize=11
    )
    time_text = ax_phys.text(
        0.02, 0.79, "", transform=ax_phys.transAxes,
        ha="left", va="top", fontsize=11
    )

    # ---------------- Right panel: phase portrait ----------------
    ax_phase.set_title("Phase space")
    ax_phase.set_xlim(-0.2, 2.7)
    ax_phase.set_ylim(-1.2, 0.6)
    ax_phase.grid(True, ls="--", alpha=0.3)
    ax_phase.set_xlabel("x1")
    ax_phase.set_ylabel("x2")
    ax_phase.plot(0, 0, "k*", ms=12)

    traj_line, = ax_phase.plot([], [], color="crimson", lw=2)
    current_point, = ax_phase.plot([], [], "o", color="navy", ms=8)

    # Optional initial-condition marker
    ax_phase.plot(z0_single[0], z0_single[1], "o", color="gray", alpha=0.5, ms=6)

    # Figure title
    supt = fig.suptitle("", fontsize=14)

    def frame_to_episode_and_step(frame_idx):
        block = frames_per_episode + PAUSE_FRAMES
        ep_idx = frame_idx // block
        inner = frame_idx % block

        step = min(inner, frames_per_episode - 1)
        return ep_idx, step

    def init():
        nonlocal arrow
        ball.set_data([], [])
        traj_line.set_data([], [])
        current_point.set_data([], [])
        epoch_text.set_text("")
        cost_text.set_text("")
        time_text.set_text("")
        supt.set_text("Van der Pol — Stabilisation learning")

        if arrow is not None:
            arrow.remove()
            arrow = None

        return ball, traj_line, current_point, epoch_text, cost_text, time_text, supt

    def update(frame_idx):
        nonlocal arrow

        ep_idx, step = frame_to_episode_and_step(frame_idx)
        ep = episodes[ep_idx]

        z = ep["z_traj"]
        u = ep["u_traj"]
        epoch = ep["epoch"]
        cost = ep["cost"]

        x1 = z[:, 0]
        x2 = z[:, 1]

        # Current state
        x1_now = x1[step]
        x2_now = x2[step]

        # For control, use last available control
        if step == 0:
            u_now = u[0, 0]
        elif step <= prob.nt:
            u_now = u[min(step - 1, prob.nt - 1), 0]
        else:
            u_now = u[-1, 0]

        # ----- Left panel -----
        ball.set_data([x1_now], [rail_y])

        if arrow is not None:
            arrow.remove()
            arrow = None

        # Scale arrow for display
        arrow_scale = 0.18
        dx = arrow_scale * u_now
        arrow = ax_phys.arrow(
            x1_now, -0.25, dx, 0.0,
            width=0.03,
            head_width=0.12,
            head_length=0.08,
            color="tab:blue",
            length_includes_head=True,
            alpha=0.85,
        )

        epoch_text.set_text(f"Epoch {epoch:04d}")
        cost_text.set_text(f"Rollout cost = {cost:.3f}")
        time_text.set_text(f"t = {min(step, prob.nt) * prob.h:.2f}")

        # ----- Right panel -----
        if TAIL_LENGTH is None:
            start_idx = 0
        else:
            start_idx = max(0, step - TAIL_LENGTH)

        traj_line.set_data(x1[start_idx:step + 1], x2[start_idx:step + 1])
        current_point.set_data([x1_now], [x2_now])

        supt.set_text(
            "Van der Pol — Stabilisation learning\n"
            f"Same initial condition | episode {ep_idx + 1}/{len(episodes)}"
        )

        artists = [
            ball, traj_line, current_point,
            epoch_text, cost_text, time_text, supt
        ]
        if arrow is not None:
            artists.append(arrow)
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if SAVE_GIF:
        gif_path = os.path.join(out_dir, GIF_NAME)
        print(f"Sauvegarde GIF → {gif_path}")
        ani.save(gif_path, writer=PillowWriter(fps=FPS))
        print("GIF sauvegardé.")

    plt.show()


if __name__ == "__main__":
    main()