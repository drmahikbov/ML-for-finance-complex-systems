#!/usr/bin/env python3
"""
Plot JFB (implicit-net) rollouts for :class:`MultiBicycleOC`.

Mirrors :mod:`plot_liquidation_jfb` for the multi-bicycle problem. Unlike
the Almgren-Chriss case there is **no closed-form / BVP reference**
ready to ship in :mod:`benchmarking.solvers`, so this script only
overlays JFB rollouts (and optionally a full-AD overlay) on
:func:`benchmarking.plotter.bicycle_panels`. Per agent the figure shows:

    1. Parametric position trajectory  (x_i, y_i)
    2. Heading       theta_i(t)
    3. Speed         v_i(t)
    4. Steering      delta_i(t)   (control)
    5. Acceleration  a_i(t)       (control)

Run from the repository (cwd = ``jfb-for-implicit-oc`` or project root)::

    cd jfb-for-implicit-oc
    python examples/explicit_ustar/plot_bicycle_jfb.py --train-epochs 50

Use a checkpoint saved by :class:`OptimalControlTrainer` (``state_dict``)::

    python examples/explicit_ustar/plot_bicycle_jfb.py \\
        --checkpoint results/MultiBicycleOC/training/best_policy_<tag>_<run_id>.pth \\
        --num-bicycles 1 --nt 40 --t-final 4.0

You **must** use the same :class:`MultiBicycleOC` hyperparameters as
training when loading a checkpoint (defaults below mirror
``example_multibicycle.main()`` so a fresh run reproduces it).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List

import numpy as np
import torch

# Path bootstrap: identical layout to plot_liquidation_jfb.py. core/ and
# models/ use flat imports, the project root is needed for `core.paths`.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../examples/explicit_ustar
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # .../jfb-for-implicit-oc
for _p in (
    _HERE,
    _ROOT,
    os.path.join(_ROOT, "core"),
    os.path.join(_ROOT, "models"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ImplicitNets import ImplicitNetOC_MB, Phi
from MultiBicycle import MultiBicycleOC
from OptimalControlTrainer import OptimalControlTrainer
from core.paths import results_dir
from benchmarking import (
    BenchmarkPlotter,
    JFBPolicyRollout,
    Trajectory,
    bicycle_panels,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to policy .pth (state_dict). If omitted, train with --train-epochs.",
    )
    p.add_argument("--train-epochs", type=int, default=50,
                   help="Adam epochs when no checkpoint (default 50).")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Adam learning rate when training (default 5e-4).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Training batch size / sample_initial_condition size.")
    p.add_argument("--n-show", type=int, default=3,
                   help="Max trajectories overlaid in the final figure.")
    p.add_argument("--output", type=str, default=None,
                   help="Output PNG path. Default: results/MultiBicycleOC/benchmark/"
                        "bicycle_state_evolution_<tag>.png.")
    p.add_argument("--device", type=str, default=None,
                   help="cpu or cuda (default: auto).")
    p.add_argument("--tag", type=str, default="JFB",
                   help="Run tag passed to OptimalControlTrainer (artifact suffix).")
    p.add_argument("--seed", type=int, default=420,
                   help="Seed applied to torch and numpy at the start of main() "
                        "(default 420). Ignored when --no-seed is set.")
    p.add_argument("--no-seed", action="store_true",
                   help="Skip RNG seeding entirely (every run gets a fresh init).")
    # ------------------------------------------------------------------ #
    # Inner fixed-point solver knobs (ImplicitNetOC_MB).                 #
    # ------------------------------------------------------------------ #
    p.add_argument("--fp-alpha", type=float, default=5e-5,
                   help="Inner fixed-point step size α (default 5e-5, the value "
                        "used by example_multibicycle for comparable runs).")
    p.add_argument("--fp-max-iters", type=int, default=999,
                   help="Inner FP iteration cap (default 999, like example_multibicycle).")
    p.add_argument("--fp-tol", type=float, default=0.1,
                   help="Inner FP residual tolerance (default 0.1, matches "
                        "example_multibicycle's 'tol=0.1').")
    p.add_argument("--use-aa", dest="use_aa", action="store_true", default=False,
                   help="Enable Anderson acceleration on the inner FP solver "
                        "(default OFF for bicycle, matching the working baseline).")
    p.add_argument("--no-aa", dest="use_aa", action="store_false",
                   help="Disable Anderson acceleration (default).")
    p.add_argument("--aa-beta", type=float, default=0.5,
                   help="Anderson damping coefficient β (only used when --use-aa).")
    p.add_argument("--tracked-iters", type=int, default=1,
                   help="Number of inner FP iterations differentiated through "
                        "with full autograd (default 1; matches example_multibicycle).")
    # ------------------------------------------------------------------ #
    # Steering-angle clamps (handlebars only; acceleration is unclamped  #
    # by ImplicitNetOC_MB.apply_control_limits).                         #
    # ------------------------------------------------------------------ #
    p.add_argument("--steering-min", type=float, default=-(0.5 * np.pi - 0.25),
                   help="Lower bound on steering angle δ (default ≈ -1.32 rad).")
    p.add_argument("--steering-max", type=float, default=0.5 * np.pi - 0.25,
                   help="Upper bound on steering angle δ (default ≈ +1.32 rad).")
    # ------------------------------------------------------------------ #
    # Full-AD overlay (mirrors the liquidation script).                  #
    # ------------------------------------------------------------------ #
    p.add_argument("--full-ad", dest="full_ad", action="store_true", default=False,
                   help="Also train a second policy with full autograd through every "
                        "inner-FP iteration (track_all_fp_iters=True). The benchmark "
                        "figure overlays JFB-trained and AD-trained rollouts.")
    # ------------------------------------------------------------------ #
    # Optimality-condition loss weights.                                 #
    # ------------------------------------------------------------------ #
    p.add_argument("--alpha-hjb-run", type=float, default=1e-4,
                   help="Running-time HJB residual weight (default 1e-4, matches "
                        "example_multibicycle).")
    p.add_argument("--alpha-hjb-fin", type=float, default=1e-2,
                   help="Terminal HJB residual weight (default 1e-2, matches "
                        "example_multibicycle).")
    p.add_argument("--verify", action="store_true",
                   help="Enable per-epoch numerical verification (max_grad_T_u, "
                        "M_theta singular values, optimality residuals → CSV). "
                        "Slow; use sparingly.")
    # ------------------------------------------------------------------ #
    # Problem parameters (must match training when using --checkpoint).  #
    # ------------------------------------------------------------------ #
    p.add_argument("--num-bicycles", type=int, default=1,
                   help="Number of bicycles N (default 1, matching example_multibicycle).")
    p.add_argument("--t-final", type=float, default=4.0,
                   help="Time horizon (default 4.0, matches example_multibicycle).")
    p.add_argument("--nt", type=int, default=40,
                   help="Number of Euler steps (default 40, matches example_multibicycle).")
    p.add_argument("--alpha-G", type=float, default=500.0,
                   help="Terminal-cost weight α_G (default 500.0, matches "
                        "example_multibicycle).")
    p.add_argument("--alpha-interaction", type=float, default=5.0,
                   help="Inter-agent repulsion weight (default 5.0, matches "
                        "example_multibicycle).")
    p.add_argument("--ic-mean", type=float, default=0.0,
                   help="Initial-condition Gaussian mean (default 0.0).")
    p.add_argument("--ic-var", type=float, default=0.1,
                   help="Initial-condition Gaussian std (default 0.1).")
    p.add_argument("--phi-hidden", type=int, default=100,
                   help="Phi backbone hidden width (default 100, matches "
                        "example_multibicycle).")
    p.add_argument("--phi-layers", type=int, default=3,
                   help="Phi backbone ResNet depth (default 3, matches "
                        "example_multibicycle).")
    p.add_argument("--no-pen-pos", dest="pen_pos", action="store_false", default=True,
                   help="Disable the position-only terminal penalty (uses the full "
                        "state-target deviation when off). Default ON.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_problem(args: argparse.Namespace, device: str) -> MultiBicycleOC:
    return MultiBicycleOC(
        batch_size=args.batch_size,
        t_initial=0.0,
        t_final=args.t_final,
        nt=args.nt,
        n_b=args.num_bicycles,
        alphaL=1.0,
        alphaG=args.alpha_G,
        alphaHJB=[args.alpha_hjb_run, args.alpha_hjb_fin],
        alpha_interaction=args.alpha_interaction,
        pen_pos=args.pen_pos,
        ic_mean=args.ic_mean,
        ic_var=args.ic_var,
        device=device,
    )


def build_policy(prob: MultiBicycleOC, args: argparse.Namespace,
                 device: str) -> ImplicitNetOC_MB:
    phi = Phi(args.phi_layers, args.phi_hidden, prob.state_dim, dev=device).to(device)
    # Inner FP cap is reduced by `tracked_iters` so the total work matches
    # the legacy `example_multibicycle` exactly (max_itr = 1000 - tracked).
    max_iters = max(1, args.fp_max_iters - args.tracked_iters)
    inn = ImplicitNetOC_MB(
        prob.state_dim, prob.control_dim,
        alpha=args.fp_alpha, max_iters=max_iters, tol=args.fp_tol,
        tracked_iters=args.tracked_iters,
        oc_problem=prob,
        use_control_limits=True,
        u_min=args.steering_min, u_max=args.steering_max,
        p_net=phi, dev=device,
        use_aa=args.use_aa, beta=args.aa_beta,
    ).to(device)
    return inn


def _train_or_load(
    prob: MultiBicycleOC,
    args: argparse.Namespace,
    device: str,
    *,
    full_ad: bool,
    trainer_tag: str,
) -> ImplicitNetOC_MB:
    """Build a fresh policy and either load weights or train from scratch.

    Mirrors :func:`plot_liquidation_jfb._train_or_load` so the script
    behaves the same way: ``--full-ad`` flips ``track_all_fp_iters`` for
    the duration of training, and ``--checkpoint`` short-circuits the
    training loop.
    """
    inn = build_policy(prob, args, device)

    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt):
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        inn.load_state_dict(state, strict=True)
        print(f"Loaded policy weights from: {ckpt}")
        inn.eval()
        return inn

    if args.train_epochs <= 0:
        raise SystemExit("Provide --checkpoint or set --train-epochs > 0.")

    if full_ad and args.use_aa:
        print(
            "[warn] --full-ad with Anderson acceleration ON is heavy on memory "
            "and gradient noise; consider --no-aa for the AD pass."
        )

    prob.track_all_fp_iters = full_ad

    opt = torch.optim.Adam(inn.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-8,
    )
    trainer = OptimalControlTrainer(
        inn, prob, opt, scheduler=sched, device=device, tag=trainer_tag,
        ver=args.verify,
    )
    trainer.set_mode("standard")
    print(
        f"Training {'full-AD' if full_ad else 'JFB (analytic implicit grad)'} "
        f"policy [trainer_tag={trainer_tag!r}] for {args.train_epochs} epochs..."
    )
    z0_train = prob.sample_initial_condition()
    trainer.train(
        z0_train,
        num_epochs=args.train_epochs,
        verbose=True,
        plot_frequency=max(10, args.train_epochs // 5),
    )
    best_path = trainer.run_io.policy_path()
    if os.path.isfile(best_path):
        print(f"Best policy stored at: {os.path.abspath(best_path)}")

    prob.track_all_fp_iters = False
    inn.eval()
    return inn


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Distinct palette for overlaid policies (JFB / full-AD / future overlays).
# Reuse the liquidation palette so figures across problems are visually
# consistent (red = JFB primary, blue = secondary reference).
_PALETTE = ("#d6604d", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00")


def _rollout_policies(
    prob: MultiBicycleOC,
    policies: Dict[str, Any],
    z0_batch: torch.Tensor,
    n_show: int,
) -> List[Trajectory]:
    """Roll out each labelled policy from up to ``n_show`` initial conditions.

    Each ``(label, policy)`` pair yields up to ``n_show`` trajectories
    (one per row of ``z0_batch``). The first trajectory in each group
    keeps the policy's display label; subsequent ones get ``label=None``
    so the legend stays compact.

    ``MultiBicycleOC.compute_f`` allocates its output buffer using
    ``self.batch_size`` rather than reading the caller's first axis, so
    we temporarily pin ``prob.batch_size = 1`` for the duration of the
    rollouts (the only context in which a single z0 is fed in) and
    restore the original value afterwards. Without this guard, training
    runs with ``--batch-size > 1`` would silently broadcast the
    single-z0 trajectory back to ``batch_size`` rows.
    """
    batch = min(z0_batch.shape[0], n_show)
    z0 = z0_batch[:batch].to(prob.device)
    trajectories: List[Trajectory] = []
    saved_bs = prob.batch_size
    prob.batch_size = 1
    try:
        for i, (label, pol) in enumerate(policies.items()):
            color = _PALETTE[i % len(_PALETTE)]
            roller = JFBPolicyRollout(prob, pol)
            for b in range(batch):
                tr = roller.solve(z0[b])
                tr = _dc_replace(
                    tr,
                    label=label if b == 0 else None,
                    style={"color": color, "lw": 1.8, "alpha": 0.85},
                )
                trajectories.append(tr)
    finally:
        prob.batch_size = saved_bs
    return trajectories


def _plot_bicycle_rollout(
    prob: MultiBicycleOC,
    policies: Dict[str, Any],
    z0_batch: torch.Tensor,
    save_path: str,
    n_show: int,
    title: str | None = None,
) -> None:
    """Render the per-agent panel grid for the supplied policies."""
    trajectories = _rollout_policies(prob, policies, z0_batch, n_show)
    panels = bicycle_panels(prob.num_agents)
    # 5 panels per agent; one row per agent reads cleanly. Fall back to a
    # 3-column grid for many agents so the figure stays printable.
    ncols = 5 if prob.num_agents <= 3 else 3
    BenchmarkPlotter(panels, ncols=ncols).plot(
        trajectories, save_path=save_path, title=title,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.no_seed:
        print("RNG seeding disabled (--no-seed); run will not be reproducible.")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Seeded torch and numpy with seed={args.seed}.")

    prob = build_problem(args, device)
    print(
        f"Problem: num_agents={prob.num_agents}, state_dim={prob.state_dim}, "
        f"control_dim={prob.control_dim}, T={prob.t_final:g}, nt={prob.nt}"
    )
    print(
        "Inner FP solver: "
        f"alpha={args.fp_alpha:.3g}  max_iters={args.fp_max_iters - args.tracked_iters}  "
        f"tol={args.fp_tol:.1e}  tracked_iters={args.tracked_iters}  "
        + (f"Anderson(beta={args.aa_beta:.2f})" if args.use_aa else "no Anderson")
    )
    print(
        "Loss weights: "
        f"alphaG={prob.alphaG:g}  alphaHJB=({args.alpha_hjb_run:g}, {args.alpha_hjb_fin:g})  "
        f"alpha_interaction={prob.alpha_interaction:g}"
    )
    print(
        f"Steering clamp: δ ∈ [{args.steering_min:.3f}, {args.steering_max:.3f}] rad  "
        f"(acceleration unclamped)"
    )
    print(
        "Backprop regime: "
        + ("JFB + full-AD (overlay)" if args.full_ad else "JFB only")
    )

    inn_jfb = _train_or_load(
        prob, args, device, full_ad=False, trainer_tag=args.tag,
    )

    inn_ad: ImplicitNetOC_MB | None = None
    if args.full_ad:
        if args.checkpoint:
            print(
                "[warn] --full-ad ignored: --checkpoint loads a single policy "
                "(no second AD-trained model is produced)."
            )
        else:
            inn_ad = _train_or_load(
                prob, args, device, full_ad=True,
                trainer_tag=f"{args.tag}-fullAD",
            )

    z0_plot = prob.sample_initial_condition()

    out = args.output or os.path.join(
        results_dir(type(prob).__name__, "benchmark"),
        f"bicycle_state_evolution_{args.tag}.png",
    )

    policies: Dict[str, Any] = {"JFB (analytic)": inn_jfb}
    if inn_ad is not None:
        policies["JFB (full AD)"] = inn_ad

    title = (
        f"MultiBicycle (N={prob.num_agents}) — {' vs '.join(policies.keys())}  "
        f"[T={prob.t_final:g}, nt={prob.nt}, α_G={prob.alphaG:g}]  "
        f"[{args.tag}]"
    )
    _plot_bicycle_rollout(
        prob, policies, z0_plot, save_path=out, n_show=args.n_show, title=title,
    )
    print(f"Figure written to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
