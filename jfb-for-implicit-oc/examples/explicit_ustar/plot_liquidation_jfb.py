#!/usr/bin/env python3
"""
examples.explicit_ustar.plot_liquidation_jfb
---------------------------------------------
Plot JFB policy rollouts against the exact Almgren-Chriss BVP for
LiquidationPortfolioOC. Trains a fresh policy by default or loads a checkpoint.

    python plot_liquidation_jfb.py
    python plot_liquidation_jfb.py --checkpoint results/.../best_policy_JFB_*.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Local imports: script may be run from project root or this directory.
# core/ and models/ still use flat imports, so they need to be on sys.path
# in addition to the package root (which is needed for `core.paths`, etc.).
# This script lives at jfb-for-implicit-oc/examples/explicit_ustar/, so we
# need to climb two directories up to reach the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../examples/explicit_ustar
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # .../jfb-for-implicit-oc
for _p in (
    _HERE,                                                   # liquidation_benchmark.py
    _ROOT,                                                   # `core.paths` package import
    os.path.join(_ROOT, "core"),                             # ImplicitNets, OptimalControlTrainer
    os.path.join(_ROOT, "models"),                           # LiquidationPortfolio
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ImplicitNets import ImplicitNetOC, Phi
from LiquidationPortfolio import LiquidationPortfolioOC
from OptimalControlTrainer import OptimalControlTrainer
from liquidation_benchmark import LiquidationBenchmark, benchmark_png_path
from core.paths import results_dir
from benchmarking import (
    BenchmarkPlotter,
    diagnostic_rollout,
    diagnostic_panels,
    attach_bvp_costate_to_meta,
    liquidation_costate_vs_bvp_panels,
)
from benchmarking.solvers import AlmgrenChrissBVPSolver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy .pth (torch.save(state_dict)). If omitted, train with --train-epochs.",
    )
    p.add_argument("--train-epochs", type=int, default=40, help="Adam epochs when no checkpoint (default: 40).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate when training.")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size / sample_initial_condition size.")
    p.add_argument("--n-show", type=int, default=5, help="Max trajectories overlaid in the figure.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Default: results/<ProblemClassName>/benchmark/jfb_vs_exactbvp_benchmark.png",
    )
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto).")
    p.add_argument(
        "--tag",
        type=str,
        default="JFB",
        help="Run tag passed to OptimalControlTrainer (becomes part of every artifact filename).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=420,
        help="Seed applied to torch and numpy at the start of main() for "
             "reproducibility (default: 420). Ignored when --no-seed is set.",
    )
    p.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip RNG seeding entirely (every run gets a fresh init).",
    )
    # ------------------------------------------------------------------ #
    # Inner fixed-point solver knobs (ImplicitNetOC).                    #
    # ------------------------------------------------------------------ #
    p.add_argument(
        "--fp-alpha", type=float, default=1.0,
        help="Inner fixed-point step size (gradient descent on H). Newton "
             "step for liquidation γ=2 is 1/(2η). Default 1.0 is robust "
             "with Anderson acceleration; drop to 1e-2 if AA is off.",
    )
    p.add_argument(
        "--fp-max-iters", type=int, default=50,
        help="Inner fixed-point iteration cap (default 50, used heavily by AA).",
    )
    p.add_argument(
        "--fp-tol", type=float, default=1e-6,
        help="Inner fixed-point residual tolerance (default 1e-6).",
    )
    p.add_argument(
        "--use-aa", dest="use_aa", action="store_true", default=True,
        help="Enable Anderson acceleration on the inner FP solver (default ON).",
    )
    p.add_argument(
        "--no-aa", dest="use_aa", action="store_false",
        help="Disable Anderson acceleration; fall back to plain gradient-descent FP.",
    )
    p.add_argument(
        "--aa-beta", type=float, default=0.5,
        help="Anderson damping coefficient β (default 0.5).",
    )
    p.add_argument(
        "--diagnostics", dest="diagnostics", action="store_true", default=True,
        help="Also write the inner-FP / costate diagnostic figure "
             "(default ON). Uses benchmarking.diagnostic_panels.",
    )
    p.add_argument(
        "--no-diagnostics", dest="diagnostics", action="store_false",
        help="Skip the diagnostic figure.",
    )
    # ------------------------------------------------------------------ #
    # Control-bound knobs.                                               #
    # The original script clamped u to [0, 10]. The lower bound u_min=0  #
    # was the dominant failure mode in earlier runs: with the BC         #
    # p_q(T) = 2 alpha q(T) > 0 the unclamped optimum u* often goes      #
    # negative, the clamp pegs it at 0, and the policy gets no gradient. #
    # Default is now NO clamp; turn it back on with --clamp-u.           #
    # ------------------------------------------------------------------ #
    p.add_argument(
        "--clamp-u", dest="clamp_u", action="store_true", default=False,
        help="Hard-clamp the policy output to [u_min, u_max]. Off by default; "
             "the lower bound was the main collapse mechanism in prior runs.",
    )
    p.add_argument("--u-min", type=float, default=-1.0e6,
                   help="Lower bound when --clamp-u is set (default effectively -inf).")
    p.add_argument("--u-max", type=float, default=1.0e6,
                   help="Upper bound when --clamp-u is set (default effectively +inf).")
    # ------------------------------------------------------------------ #
    # Optimality-condition loss weights (pass-through to ImplicitOC).    #
    # alphaHJB = [running, terminal]   penalty on the HJB residual.      #
    # alphaadj = [running, terminal]   penalty on the adjoint residual.  #
    # Default 0 keeps the legacy "loss-only" objective; set them > 0 to  #
    # actually train p_theta to satisfy PMP.                             #
    # ------------------------------------------------------------------ #
    p.add_argument("--alpha-hjb-run", type=float, default=0.0,
                   help="Running-time HJB residual weight (default 0).")
    p.add_argument("--alpha-hjb-fin", type=float, default=0.0,
                   help="Terminal HJB residual weight (default 0).")
    p.add_argument("--alpha-adj-run", type=float, default=0.0,
                   help="Running-time adjoint residual weight (default 0).")
    p.add_argument("--alpha-adj-fin", type=float, default=0.0,
                   help="Terminal adjoint residual weight (default 0).")
    # Problem parameters (must match training when using --checkpoint)
    p.add_argument("--t-final", type=float, default=2.0)
    p.add_argument("--nt", type=int, default=100)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--kappa", type=float, default=1e-4)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=30.0)
    p.add_argument("--q0-min", type=float, default=0.5)
    p.add_argument("--q0-max", type=float, default=1.5)
    p.add_argument("--S0", type=float, default=1.0)
    p.add_argument("--X0", type=float, default=0.0)
    return p.parse_args()


def build_problem(args: argparse.Namespace, device: str) -> LiquidationPortfolioOC:
    return LiquidationPortfolioOC(
        batch_size=args.batch_size,
        t_initial=0.0,
        t_final=args.t_final,
        nt=args.nt,
        sigma=args.sigma,
        kappa=args.kappa,
        eta=args.eta,
        gamma=args.gamma,
        epsilon=args.epsilon,
        alpha=args.alpha,
        q0_min=args.q0_min,
        q0_max=args.q0_max,
        S0=args.S0,
        X0=args.X0,
        device=device,
        alphaHJB=(args.alpha_hjb_run, args.alpha_hjb_fin),
        alphaadj=(args.alpha_adj_run, args.alpha_adj_fin),
    )


def build_policy(prob: LiquidationPortfolioOC, device: str,
                 fp_alpha: float = 1.0,
                 fp_max_iters: int = 50,
                 fp_tol: float = 1e-6,
                 use_aa: bool = True,
                 aa_beta: float = 0.5,
                 clamp_u: bool = False,
                 u_min: float = -1.0e6,
                 u_max: float = 1.0e6) -> ImplicitNetOC:
    phi = Phi(3, 50, prob.state_dim, dev=device)
    return ImplicitNetOC(
        prob.state_dim,
        prob.control_dim,
        alpha=fp_alpha,
        max_iters=fp_max_iters,
        tol=fp_tol,
        p_net=phi,
        oc_problem=prob,
        u_min=u_min,
        u_max=u_max,
        use_control_limits=clamp_u,
        use_aa=use_aa,
        beta=aa_beta,
        dev=device,
    ).to(device)


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
    inn = build_policy(
        prob, device,
        fp_alpha=args.fp_alpha,
        fp_max_iters=args.fp_max_iters,
        fp_tol=args.fp_tol,
        use_aa=args.use_aa,
        aa_beta=args.aa_beta,
        clamp_u=args.clamp_u,
        u_min=args.u_min,
        u_max=args.u_max,
    )
    print(
        "Inner FP solver: "
        f"alpha={args.fp_alpha:.3g}  max_iters={args.fp_max_iters}  "
        f"tol={args.fp_tol:.1e}  "
        + (f"Anderson(beta={args.aa_beta:.2f})" if args.use_aa else "no Anderson")
    )
    if args.clamp_u:
        print(f"Control clamp: u in [{args.u_min:g}, {args.u_max:g}]")
    else:
        print("Control clamp: OFF (unbounded u)")
    print(
        "Loss weights: "
        f"alphaHJB=({args.alpha_hjb_run:g}, {args.alpha_hjb_fin:g})  "
        f"alphaadj=({args.alpha_adj_run:g}, {args.alpha_adj_fin:g})"
    )

    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt):
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        inn.load_state_dict(state, strict=True)
        print(f"Loaded policy weights from: {ckpt}")
    else:
        if args.train_epochs <= 0:
            raise SystemExit("Provide --checkpoint or set --train-epochs > 0.")
        opt = torch.optim.Adam(inn.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=8
        )
        trainer = OptimalControlTrainer(
            inn, prob, opt, scheduler=sched, device=device, tag=args.tag,
        )
        trainer.set_mode("standard")
        z0_train = prob.sample_initial_condition()
        trainer.train(
            z0_train,
            num_epochs=args.train_epochs,
            verbose=True,
            plot_frequency=10,
        )
        # The trainer already reloaded the best checkpoint into `inn` during
        # finalize(). Keep the path around in case downstream tooling wants it.
        best_path = trainer.run_io.policy_path()
        if os.path.isfile(best_path):
            print(f"Best policy stored at: {os.path.abspath(best_path)}")

    inn.eval()
    bench = LiquidationBenchmark(prob)
    z0_plot = prob.sample_initial_condition()

    out = args.output or os.path.join(
        results_dir(type(prob).__name__, "benchmark"),
        "jfb_vs_exactbvp_benchmark.png",
    )
    bench.plot_comparison(inn, z0_plot, save_path=out, n_show=args.n_show)
    print(f"Figure written to: {os.path.abspath(out)}")

    if args.diagnostics:
        # Single z0 for the diagnostic rollout: pick the first sample from
        # the same batch we just plotted, so the diagnostic curves correspond
        # to a trajectory visible in the comparison figure.
        z0_diag = z0_plot[0].detach().cpu().numpy().reshape(-1)
        diag_traj = diagnostic_rollout(
            prob, inn,
            torch.as_tensor(z0_diag, dtype=torch.float32, device=prob.device),
            label="JFB",
            record_trace_at_t0=True,
        )
        # Optional: overlay learned vs exact-BVP costates (γ=2 only).
        traj_for_plot = diag_traj
        extra_panels = []
        if abs(float(prob.gamma) - 2.0) < 1e-6:
            try:
                traj_for_plot = attach_bvp_costate_to_meta(
                    diag_traj, prob, np.asarray(z0_diag),
                )
                extra_panels = liquidation_costate_vs_bvp_panels()
            except Exception as exc:
                print(f"  [warn] BVP costate overlay skipped: {exc}")

        diag_panels = diagnostic_panels(state_components=(0, 1)) + extra_panels
        diag_out = os.path.join(
            results_dir(type(prob).__name__, "benchmark"),
            f"jfb_diagnostics_{args.tag}.png",
        )
        BenchmarkPlotter(diag_panels, ncols=2).plot(
            [traj_for_plot], save_path=diag_out,
            title=(
                f"JFB diagnostics — α_fp={args.fp_alpha:.2g}, "
                f"max_iters={args.fp_max_iters}, tol={args.fp_tol:.0e}, "
                f"AA={'on' if args.use_aa else 'off'}"
            ),
        )
        print(f"Diagnostics figure written to: {os.path.abspath(diag_out)}")


if __name__ == "__main__":
    main()
