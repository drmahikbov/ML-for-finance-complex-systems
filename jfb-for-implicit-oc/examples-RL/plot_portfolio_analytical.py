#!/usr/bin/env python3
"""
examples-RL.plot_portfolio_analytical
--------------------------------------
Plot the analytical Merton optimum (π*) optionally overlaid with a learned
JFB-RL rollout from results/. π* solves αL·2λπ exp(π²) = αG·(μ−r).

    python jfb-for-implicit-oc/examples-RL/plot_portfolio_analytical.py
    python ... --checkpoint <path.pth>   # explicit checkpoint
    python ... --no-learned              # analytical only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

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
    os.path.join(_ROOT, "core-RL"),
    os.path.join(_ROOT, "models"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarking import BenchmarkPlotter, Trajectory
from core.ImplicitNets import Phi
from core_RL.Environment import AnalyticalEnvironment
from core_RL.ImplicitNets_RL import ImplicitNetOC_RL
from core_RL.JacobianEstimator import RLSJacobianEstimator
from models.PortfolioOC_RL import PortfolioOC_RL

# Match portfolio_optimization_RL.py training seed / batch.
TRAINING_SEED = 420
TRAINING_BATCH_SIZE = 32


# Defaults match portfolio_optimization_RL.py / evaluate_portfolio_rl.py
DEFAULT_MU = 0.10
DEFAULT_R = 0.03
DEFAULT_LAM = 0.1
DEFAULT_W_REF = 10.0
DEFAULT_W0_MIN = 0.8
DEFAULT_W0_MAX = 1.2
DEFAULT_T_FINAL = 2.0
DEFAULT_NT = 50
DEFAULT_ALPHA_L = 1.0
DEFAULT_ALPHA_G = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Policy .pth checkpoint. Default: latest best_policy_JFB-RL_*.pth in results.",
    )
    p.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Saved rollout .pth from training (z_traj only). Used when no checkpoint is given.",
    )
    p.add_argument(
        "--no-learned",
        action="store_true",
        help="Plot the analytical solution only (skip learned overlay).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_BATCH_SIZE,
        help="Batch size (default: 32, same as training).",
    )
    p.add_argument("--w0", type=float, default=None, help="Single initial wealth W0 (overrides batch sampling).")
    p.add_argument(
        "--path-index",
        type=int,
        default=0,
        help="Which batch path to display (default: 0, same as trainer plots).",
    )
    p.add_argument(
        "--eval-mode",
        choices=("trainer", "rollout"),
        default="trainer",
        help=(
            "'trainer' reproduces OptimalControlTrainer_RL._plot_rollout (rollout to "
            "prime b_k, then to_trajectory). 'rollout' uses per-step controls from "
            "env.rollout with a fresh RLS estimator."
        ),
    )
    p.add_argument(
        "--use-saved-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a saved trajectory_*.pth exists, plot its wealth path (default: on).",
    )
    p.add_argument("--output", type=str, default=None, help="Output PNG path.")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto).")
    p.add_argument("--seed", type=int, default=TRAINING_SEED)
    p.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="RLS warm-up rollouts before evaluation (default: 10).",
    )
    return p.parse_args()


def build_problem(args: argparse.Namespace, device: str) -> PortfolioOC_RL:
    return PortfolioOC_RL(
        mu_true=DEFAULT_MU,
        r_true=DEFAULT_R,
        lam=DEFAULT_LAM,
        W_ref=DEFAULT_W_REF,
        W0_min=DEFAULT_W0_MIN,
        W0_max=DEFAULT_W0_MAX,
        batch_size=args.batch_size,
        t_initial=0.0,
        t_final=DEFAULT_T_FINAL,
        nt=DEFAULT_NT,
        alphaL=DEFAULT_ALPHA_L,
        alphaG=DEFAULT_ALPHA_G,
        device=device,
    )


def build_environment(prob: PortfolioOC_RL, device: str) -> AnalyticalEnvironment:
    return AnalyticalEnvironment(
        state_dim=prob.state_dim,
        control_dim=prob.control_dim,
        t_initial=prob.t_initial,
        t_final=prob.t_final,
        nt=prob.nt,
        f_callable=prob.compute_f,
        device=device,
    )


def compute_pi_star(prob: PortfolioOC_RL) -> float:
    """Constant optimal π* from the Pontryagin optimality condition."""
    rhs = prob.alphaG * (prob.mu_true - prob.r_true) / (2.0 * prob.lam * prob.alphaL)
    return float(brentq(lambda pi: pi * np.exp(pi**2) - rhs, 1e-9, 2.0))


def sample_initial_conditions(prob: PortfolioOC_RL, w0: float | None) -> torch.Tensor:
    if w0 is not None:
        return torch.full(
            (prob.batch_size, prob.state_dim),
            w0,
            dtype=torch.float32,
            device=prob.device,
        )
    return prob.sample_initial_condition()


def rollout_constant_policy(
    prob: PortfolioOC_RL,
    env: AnalyticalEnvironment,
    z0: torch.Tensor,
    pi: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out a constant control π along the true dynamics."""
    batch = z0.shape[0]
    u_const = torch.full(
        (batch, prob.control_dim),
        pi,
        dtype=z0.dtype,
        device=z0.device,
    )
    z_traj = torch.zeros(batch, prob.state_dim, prob.nt + 1, device=z0.device, dtype=z0.dtype)
    u_traj = torch.zeros(batch, prob.control_dim, prob.nt, device=z0.device, dtype=z0.dtype)
    z_traj[:, :, 0] = z0
    z = z0.clone()
    t = prob.t_initial

    with torch.no_grad():
        for k in range(prob.nt):
            z_next = env.step(z, u_const, t)
            z_traj[:, :, k + 1] = z_next
            u_traj[:, :, k] = u_const
            z = z_next
            t += prob.h
    return z_traj, u_traj


def pack_trajectory(
    prob: PortfolioOC_RL,
    z_traj: torch.Tensor,
    u_traj: torch.Tensor | None,
    *,
    label: str,
    style: dict,
    path_index: int | None = None,
) -> Trajectory:
    """Convert torch rollouts to a :class:`Trajectory`.

    When ``path_index`` is set, extract a single deterministic path (trainer style).
    Otherwise keep the full batch (batch-mean line in the plotter).
    """
    t_np = np.linspace(prob.t_initial, prob.t_final, prob.nt + 1)
    z_np = z_traj.detach().cpu().numpy()  # (B, state_dim, nt+1)
    batch, _, _ = z_np.shape

    if path_index is not None:
        if not 0 <= path_index < batch:
            raise IndexError(f"path_index={path_index} out of range for batch={batch}")
        z_out = z_np[path_index].T  # (nt+1, state_dim)
        u_out = None
        if u_traj is not None:
            u_out = u_traj[path_index].T.detach().cpu().numpy()  # (nt, control_dim)
    elif batch == 1:
        z_out = z_np[0].T
        u_out = u_traj[0].T.detach().cpu().numpy() if u_traj is not None else None
    else:
        z_out = np.transpose(z_np, (0, 2, 1))
        u_out = (
            np.transpose(u_traj.detach().cpu().numpy(), (0, 2, 1))
            if u_traj is not None
            else None
        )

    return Trajectory(t=t_np, z=z_out, u=u_out, label=label, style=style)


def find_latest_checkpoint() -> str | None:
    ckpt_dir = os.path.join(_ROOT, "results", "PortfolioOC_RL", "training")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "best_policy_JFB-RL_*.pth")))
    return ckpts[-1] if ckpts else None


def find_latest_trajectory() -> str | None:
    roll_dir = os.path.join(_ROOT, "results", "PortfolioOC_RL", "rollouts")
    paths = sorted(glob.glob(os.path.join(roll_dir, "trajectory_JFB-RL_*.pth")))
    return paths[-1] if paths else None


def build_learned_policy(prob: PortfolioOC_RL, device: str) -> ImplicitNetOC_RL:
    phi = Phi(3, 50, prob.state_dim, dev=device)
    return ImplicitNetOC_RL(
        prob.state_dim,
        prob.control_dim,
        alpha=0.25,
        max_iters=300,
        tol=1e-4,
        p_net=phi,
        oc_problem=prob,
        u_min=-2.0,
        u_max=2.0,
        use_control_limits=True,
        dev=device,
    ).to(device)


def warmup_jacobian_estimator(
    prob: PortfolioOC_RL,
    env: AnalyticalEnvironment,
    jac_est: RLSJacobianEstimator,
    policy: ImplicitNetOC_RL,
    z0: torch.Tensor,
    *,
    n_warmup: int,
    warmup_std: float = 0.4,
    warmup_decay: float = 0.6,
) -> None:
    """Warm up RLS so b_k ≈ (μ−r) W before the evaluation rollout."""
    batch = z0.shape[0]
    with torch.no_grad():
        std = warmup_std
        for _ in range(n_warmup):
            z = z0.clone()
            t = prob.t_initial
            for k in range(prob.nt):
                _, b_k = jac_est.AB(k)
                policy.set_step_jacobian(b_k)
                u_k = policy(z, t).view(batch, prob.control_dim)
                u_exc = (u_k + std * torch.randn_like(u_k)).clamp(policy.u_min, policy.u_max)
                z_next = env.step(z, u_exc, t)
                jac_est.update(k, z, u_exc, z_next)
                z = z_next
                t += prob.h
            std *= warmup_decay


def make_rls_jacobian_estimator(prob: PortfolioOC_RL) -> RLSJacobianEstimator:
    return RLSJacobianEstimator(
        nt=prob.nt,
        state_dim=prob.state_dim,
        control_dim=prob.control_dim,
        dt=prob.h,
        alpha_rls=0.9,
        q0=1.0,
        device=prob.device,
    )


def rollout_with_jacobian(
    prob: PortfolioOC_RL,
    env: AnalyticalEnvironment,
    policy: ImplicitNetOC_RL,
    jac_est: RLSJacobianEstimator,
    z0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        def _setter(k: int) -> None:
            _, b_k = jac_est.AB(k)
            policy.set_step_jacobian(b_k)

        return env.rollout(
            policy, z0, jac_setter=_setter, return_full_trajectory=True,
        )


def rollout_learned_policy(
    prob: PortfolioOC_RL,
    env: AnalyticalEnvironment,
    policy: ImplicitNetOC_RL,
    z0: torch.Tensor,
    *,
    n_warmup: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    jac_est = make_rls_jacobian_estimator(prob)
    warmup_jacobian_estimator(
        prob, env, jac_est, policy, z0, n_warmup=n_warmup,
    )
    return rollout_with_jacobian(prob, env, policy, jac_est, z0)


def trainer_style_learned_trajectory(
    prob: PortfolioOC_RL,
    env: AnalyticalEnvironment,
    policy: ImplicitNetOC_RL,
    z0: torch.Tensor,
    z_plot: torch.Tensor,
    *,
    n_warmup: int,
    path_index: int,
    label: str = "JFB-RL (learned)",
) -> Trajectory:
    """Match :meth:`OptimalControlTrainer_RL._plot_rollout`.

    The trainer rolls out to refresh ``b_k`` on the policy, then calls
    ``to_trajectory(z_traj, policy)`` which reconstructs ``π(t)`` using the
    **final** ``b_k`` at every time step. That is what appears in
    ``results/.../policy_rollout_*.png``.
    """
    jac_est = make_rls_jacobian_estimator(prob)
    warmup_jacobian_estimator(
        prob, env, jac_est, policy, z0, n_warmup=n_warmup,
    )
    rollout_with_jacobian(prob, env, policy, jac_est, z0)
    return prob.to_trajectory(
        z_plot.detach(),
        policy,
        path_index=path_index,
        label=label,
    )


def load_trajectory_tensors(path: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    z_traj = data["z_traj"]
    if not isinstance(z_traj, torch.Tensor):
        z_traj = torch.as_tensor(z_traj, dtype=torch.float32)
    z0_saved = data.get("z0")
    if z0_saved is not None and not isinstance(z0_saved, torch.Tensor):
        z0_saved = torch.as_tensor(z0_saved, dtype=torch.float32)
    return z_traj, z0_saved


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prob = build_problem(args, device)
    env = build_environment(prob, device)

    pi_star = compute_pi_star(prob)
    growth_rate = prob.r_true + pi_star * (prob.mu_true - prob.r_true)
    print(
        f"Analytical optimum:  μ={prob.mu_true:.3f},  r={prob.r_true:.3f},  "
        f"λ={prob.lam:.3g},  αG/αL={prob.alphaG/prob.alphaL:.1f}"
    )
    print(f"  π* = {pi_star:.6f}   (wealth growth rate r + π*(μ−r) = {growth_rate:.6f})")

    learned_traj: Trajectory | None = None
    z0_for_analytical: torch.Tensor | None = None
    z_plot_saved: torch.Tensor | None = None
    z0_eval = sample_initial_conditions(prob, args.w0)

    traj_path = args.trajectory or find_latest_trajectory()
    if args.use_saved_z and traj_path is not None and os.path.isfile(traj_path):
        z_plot_saved, z0_saved = load_trajectory_tensors(traj_path)
        z_plot_saved = z_plot_saved.to(device=prob.device, dtype=torch.float32)
        if z0_saved is not None:
            z0_for_analytical = z0_saved.to(device=prob.device, dtype=torch.float32)
        print(f"Using saved wealth path: {os.path.basename(traj_path)}")

    if not args.no_learned:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint()

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            policy = build_learned_policy(prob, device)
            policy.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True),
            )
            policy.eval()
            print(f"Learned policy: {os.path.basename(ckpt_path)}")
            print(f"Evaluation mode: {args.eval_mode}")

            z0_roll = z0_for_analytical if z0_for_analytical is not None else z0_eval

            if args.eval_mode == "trainer":
                z_for_plot = z_plot_saved
                if z_for_plot is None:
                    z_for_plot, _ = rollout_learned_policy(
                        prob, env, policy, z0_roll, n_warmup=args.n_warmup,
                    )
                learned_traj = trainer_style_learned_trajectory(
                    prob,
                    env,
                    policy,
                    z0_roll,
                    z_for_plot,
                    n_warmup=args.n_warmup,
                    path_index=args.path_index,
                )
            else:
                z_roll, u_roll = rollout_learned_policy(
                    prob, env, policy, z0_roll, n_warmup=args.n_warmup,
                )
                z_src = z_plot_saved if z_plot_saved is not None else z_roll
                u_src = None if z_plot_saved is not None else u_roll
                learned_traj = pack_trajectory(
                    prob,
                    z_src,
                    u_src,
                    label="JFB-RL (learned)",
                    style={"color": "#d6604d", "lw": 2.0, "ls": "-"},
                    path_index=args.path_index,
                )
            if z0_for_analytical is None:
                z0_for_analytical = z0_roll
        elif args.checkpoint:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        else:
            print("No learned checkpoint found — analytical plot only.")

    z0 = z0_for_analytical if z0_for_analytical is not None else z0_eval
    z_opt, u_opt = rollout_constant_policy(prob, env, z0, pi_star)
    trajectories = [
        pack_trajectory(
            prob,
            z_opt,
            u_opt,
            label=f"Analytical  (π*={pi_star:.4f})",
            style={"color": "#2166ac", "lw": 2.0, "ls": "--"},
            path_index=args.path_index,
        ),
    ]
    if learned_traj is not None:
        trajectories.append(learned_traj)

    w0_note = f"W₀={args.w0:g}" if args.w0 is not None else f"W₀ ~ U[{prob.W0_min}, {prob.W0_max}]"
    eval_note = f"eval={args.eval_mode}, path={args.path_index}"
    title = (
        f"Merton portfolio — analytical vs learned\n"
    )
    if learned_traj is None:
        title = (
            f"Merton portfolio — analytical optimum\n"
        )

    out_path = args.output or os.path.join(
        _ROOT,
        "results",
        "PortfolioOC_RL",
        "benchmark",
        "analytical_vs_learned.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    BenchmarkPlotter(prob.panels(), ncols=2).plot(
        trajectories,
        save_path=out_path,
        title=title,
    )
    print(f"Figure written to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
