#!/usr/bin/env python3
"""
Plot JFB (implicit-net) rollouts for :class:`LiquidationPortfolioOC`.

Run **from the repository** (either cwd = ``jfb-for-implicit-oc`` or project root):

    cd jfb-for-implicit-oc
    python plot_liquidation_jfb.py

    # or
    python jfb-for-implicit-oc/plot_liquidation_jfb.py

**Default behaviour:** trains a fresh policy for ``--train-epochs`` (small run),
then saves a six-panel figure (JFB vs exact BVP when ``γ=2``) via
:class:`liquidation_benchmark.LiquidationBenchmark`.

**Use a checkpoint** saved by :class:`OptimalControlTrainer` (``state_dict`` only):

    python plot_liquidation_jfb.py --checkpoint results/LiquidationPortfolioOC/training/best_policy_JFB_<run_id>.pth

You must use the **same** ``LiquidationPortfolioOC`` hyperparameters as training
when loading a checkpoint (defaults below match ``liquidation_benchmark`` smoke
settings so the exact reference lines up).

Reduced-state note (post X-out-of-state refactor)
-------------------------------------------------
``LiquidationPortfolioOC`` now has ``state_dim = 2 * n_assets`` (no cash
``X`` in the OC state). The closed-form optimum is

    u* = (S + p_q + κ p_S) / (2 η)

with **no** ``p_X`` term, so the legacy ``--learned-costate-overlay``
``learned`` / ``pinned`` / ``both`` distinction has collapsed to
``on`` / ``off``. Cash ``X(t)`` is integrated as a parallel observer
inside :class:`benchmarking.solvers.JFBPolicyRollout` and packed at
``Trajectory.z[:, 2*n_assets]`` so the existing six-panel layout keeps
displaying ``q, u, S, X`` unchanged.

Inner FP step ``--fp-alpha`` defaults to ``auto``: for ``γ=2`` liquidation
this resolves to ``1/(η_max + η_min)`` (same minimax rule as
``example_liquidationportfolio.py``), i.e. ``1/(2η)`` when all ``η`` match,
for fast inner fixed-point convergence. Pass a numeric ``--fp-alpha`` to
override.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

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
    JFBPolicyRollout,
    LearnedCostatePolicy,
    diagnostic_rollout,
    diagnostic_panels,
    attach_bvp_costate_to_meta,
    liquidation_costate_vs_bvp_panels,
    liquidation_panels,
)
from benchmarking.diagnostics import liquidation_u_decomposition_panel
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
        "--fp-alpha",
        type=_parse_fp_alpha_cli,
        default="auto",
        help="Inner fixed-point step α on ∇_u H. Default 'auto' uses the "
             "minimax-optimal scalar 1/(η_max+η_min) for reduced liquidation "
             "at γ=2 (same as example_liquidationportfolio.py) — collapses "
             "to 1/(2η) when all η match, giving fast / one-shot inner FP. "
             "Pass a positive float to override. With Anderson on, large α "
             "is still often fine; with --no-aa, avoid α ≫ 1/(2η_min) without "
             "reason.",
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
    # Full-AD vs JFB switch.                                             #
    # When set, the script trains TWO policies on the same problem:       #
    #   1. analytic JFB        (track_all_fp_iters=False, default).      #
    #   2. full autograd       (track_all_fp_iters=True).                #
    # Both are then overlaid in the final benchmark figure against the   #
    # exact BVP reference so the JFB approximation can be compared       #
    # head-to-head with the unrolled-AD ground truth. The flag itself    #
    # is intentionally NOT folded into ``--tag``; the per-run trainer     #
    # tag for the AD pass is suffixed internally with ``-fullAD`` purely  #
    # so the two trainings do not overwrite each other's checkpoints.    #
    # ------------------------------------------------------------------ #
    p.add_argument(
        "--full-ad", dest="full_ad", action="store_true", default=False,
        help="Also train a second policy with full autograd through every "
             "inner-FP iteration (track_all_fp_iters=True). When set, the "
             "benchmark figure overlays the JFB-trained u(t) and the "
             "AD-trained u(t) against the exact BVP reference.",
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
    # phi(t, z) architecture selector.                                   #
    #   default  -> generic Phi(3, 50, state_dim) (legacy behaviour).    #
    #   anchored -> prob.make_p_net(): a TerminalAnchoredPhi wrapping    #
    #               the same generic Phi backbone, hard-anchoring        #
    #               phi(T, z) = G(z) by architecture.                    #
    # ------------------------------------------------------------------ #
    p.add_argument("--phi-arch", type=str, default="anchored",
                   choices=["default", "anchored"],
                   help="Value-function network: 'default' = generic Phi, "
                        "'anchored' = TerminalAnchoredPhi enforcing "
                        "phi(T, z) = G(z) by construction (default).")
    # ------------------------------------------------------------------ #
    # Learned-costate diagnostic overlay (the green "JFB u*(p_θ)" curve). #
    # Modes:                                                              #
    #   off -> do NOT add the overlay at all (cleanest plot).             #
    #   on  -> evaluate u*(p_θ) along the JFB rollout. In the reduced     #
    #          formulation the costate has shape (B, 2n) and the formula  #
    #          u* = (S + p_q + κ p_S) / (2η) does NOT divide by p_X       #
    #          (no p_X exists), so the legacy 'pinned' fail-safe is gone. #
    # ------------------------------------------------------------------ #
    p.add_argument("--learned-costate-overlay", type=str, default="on",
                   choices=["off", "on"],
                   help="Closed-form u*(p_theta) overlay on the JFB-vs-BVP "
                        "figure. 'off' disables it. Has no effect for "
                        "problems without a closed-form u*.")
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
    p.add_argument("--verify", action="store_true",
                   help="Enable per-epoch numerical verification (computes "
                        "max_grad_T_u, M_theta singular values, optimality "
                        "residuals, etc. and writes them to the history CSV). "
                        "Significantly slower per epoch; intended for "
                        "diagnostic runs, not full training.")
    # ------------------------------------------------------------------ #
    # Problem parameters (must match training when using --checkpoint).  #
    # Per-asset arrays accept ``nargs='+'``: pass a single value to       #
    # broadcast across all assets, or n_assets values for heterogeneous   #
    # parameters. The asset count is set with --n-assets (default 1 to    #
    # keep the legacy single-asset benchmark path active).                #
    # ------------------------------------------------------------------ #
    p.add_argument("--n-assets", type=int, default=1,
                   help="Number of assets in the liquidation portfolio "
                        "(default 1 = legacy single-asset). Multi-asset runs "
                        "(>=2) currently use identity covariance: "
                        "L = ½ Σᵢ σᵢ² qᵢ².")
    p.add_argument("--t-final", type=float, default=2.0)
    p.add_argument("--nt", type=int, default=100)
    p.add_argument("--sigma", type=float, nargs="+", default=[0.02],
                   help="σ per asset; scalar broadcasts to all assets.")
    p.add_argument("--kappa", type=float, nargs="+", default=[1e-4],
                   help="κ (linear permanent impact) per asset.")
    p.add_argument("--eta",   type=float, nargs="+", default=[0.1],
                   help="η (nonlinear impact / friction) per asset.")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--q0-min", type=float, nargs="+", default=[0.5])
    p.add_argument("--q0-max", type=float, nargs="+", default=[1.5])
    p.add_argument("--S0",     type=float, nargs="+", default=[1.0])
    p.add_argument("--X0",     type=float, default=0.0)
    return p.parse_args()


def _parse_fp_alpha_cli(value: str) -> float | str:
    """``argparse`` type for ``--fp-alpha``: positive float or the literal ``auto``."""
    v = value.strip().lower()
    if v == "auto":
        return "auto"
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--fp-alpha must be a positive float or 'auto', got {value!r}"
        ) from exc
    if x <= 0.0:
        raise argparse.ArgumentTypeError(f"--fp-alpha must be positive, got {x}")
    return x


def resolve_liquidation_fp_alpha(prob: Any, fp_alpha: float | str) -> float:
    """Scalar inner fixed-point step for :class:`ImplicitNetOC`.

    When ``fp_alpha == \"auto\"`` and the problem is the reduced
    liquidation model at ``γ = 2`` (constant per-asset Hessian ``2η`` in
    ``u``), use the same minimax-optimal scalar as
    ``example_liquidationportfolio.py``:

        α_fp = 1 / (η_max + η_min)
        worst contraction  max_i |1 − α_fp · 2 η_i|

    For homogeneous ``η`` this is ``1/(2η)`` — one Newton step on ``H``,
    typically much faster inner-FP convergence than a generic ``α = 1``
    with Anderson off. For other problems or ``γ ≠ 2``, ``auto`` falls
    back to ``1.0`` with a one-line ``[info]`` print.
    """
    if isinstance(fp_alpha, str) and fp_alpha.strip().lower() == "auto":
        if not getattr(prob, "has_closed_form_u_star", lambda: False)():
            print(
                "[info] --fp-alpha auto: no closed-form u*; using alpha_fp=1.0"
            )
            return 1.0
        if abs(float(prob.gamma) - 2.0) >= 1e-6:
            print(
                f"[info] --fp-alpha auto: gamma={float(prob.gamma):g} != 2; "
                "using alpha_fp=1.0"
            )
            return 1.0
        eta = prob.eta
        if not hasattr(eta, "max"):
            print("[info] --fp-alpha auto: missing prob.eta; using alpha_fp=1.0")
            return 1.0
        eta_max = float(eta.max().item())
        eta_min = float(eta.min().item())
        if eta_max <= 0.0 or eta_min <= 0.0:
            print(
                "[warn] --fp-alpha auto: non-positive eta; using alpha_fp=1.0"
            )
            return 1.0
        alpha_fp = 1.0 / (eta_max + eta_min)
        contraction = float(torch.abs(1.0 - alpha_fp * 2.0 * eta).max().item())
        print(
            "FP step (--fp-alpha auto): "
            f"alpha_fp = 1/(eta_max + eta_min) = {alpha_fp:.4g}  "
            f"worst |1 − alpha_fp·2η| = {contraction:.3e}  "
            "(0 ⇒ one-shot exact when all η are equal; <1 ⇒ strict contraction)"
        )
        return alpha_fp
    return float(fp_alpha)


def _broadcast(values: list[float], n_assets: int, name: str) -> list[float]:
    """Broadcast a CLI per-asset list to length ``n_assets``.

    A single scalar is repeated; a length-``n_assets`` list passes through;
    anything else is rejected.
    """
    if len(values) == 1:
        return values * n_assets
    if len(values) == n_assets:
        return list(values)
    raise SystemExit(
        f"--{name} expected 1 or {n_assets} values (n_assets={n_assets}); "
        f"got {len(values)}: {values}"
    )


def build_problem(args: argparse.Namespace, device: str) -> LiquidationPortfolioOC:
    n = int(args.n_assets)
    if n < 1:
        raise SystemExit(f"--n-assets must be >= 1, got {n}")
    sigma  = _broadcast(args.sigma,  n, "sigma")
    kappa  = _broadcast(args.kappa,  n, "kappa")
    eta    = _broadcast(args.eta,    n, "eta")
    q0_min = _broadcast(args.q0_min, n, "q0-min")
    q0_max = _broadcast(args.q0_max, n, "q0-max")
    S0     = _broadcast(args.S0,     n, "S0")
    return LiquidationPortfolioOC(
        batch_size=args.batch_size,
        t_initial=0.0,
        t_final=args.t_final,
        nt=args.nt,
        n_assets=n,
        sigma=tuple(sigma),
        kappa=tuple(kappa),
        eta=tuple(eta),
        gamma=args.gamma,
        epsilon=args.epsilon,
        alpha=args.alpha,
        q0_min=tuple(q0_min),
        q0_max=tuple(q0_max),
        S0=tuple(S0),
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
                 u_max: float = 1.0e6,
                 phi_arch: str = "anchored") -> ImplicitNetOC:
    if phi_arch == "anchored":
        phi = prob.make_p_net(hidden_dim=50, n_resnet_layers=3, device=device)
    elif phi_arch == "default":
        phi = Phi(3, 50, prob.state_dim, dev=device)
    else:
        raise ValueError(
            f"Unknown phi_arch={phi_arch!r}; expected 'default' or 'anchored'."
        )
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


def _train_or_load(
    prob: LiquidationPortfolioOC,
    args: argparse.Namespace,
    device: str,
    *,
    full_ad: bool,
    trainer_tag: str,
) -> ImplicitNetOC:
    """Build a fresh ImplicitNetOC and either load weights from
    ``args.checkpoint`` or train from scratch.

    Parameters
    ----------
    full_ad
        Sets ``prob.track_all_fp_iters`` for the duration of training so the
        backward pass differentiates through every inner-FP iteration.  At
        eval time the flag is irrelevant (forward always runs the FP loop
        under ``no_grad``), so we leave it untouched after training.
    trainer_tag
        Tag handed to ``OptimalControlTrainer``.  This is what shows up in
        artifact filenames; ``args.tag`` is left untouched here so the
        public ``--tag`` value is never silently mutated by ``--full-ad``.
    """
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
        phi_arch=args.phi_arch,
    )

    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt):
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        inn.load_state_dict(state, strict=True)
        print(f"Loaded policy weights from: {ckpt}")
        inn.eval()
        return inn

    if args.train_epochs <= 0:
        raise SystemExit("Provide --checkpoint or set --train-epochs > 0.")

    # Anderson + full autograd is incompatible: the inner linalg.solve in
    # `anderson_direct` builds an autograd tape per iteration and quickly
    # explodes in memory.  Warn the user but keep the run going so the
    # comparison still happens.
    if full_ad and args.use_aa:
        print(
            "[warn] --full-ad with Anderson acceleration ON is heavy on memory "
            "and gradient noise; consider --no-aa for the AD pass."
        )

    prob.track_all_fp_iters = full_ad

    opt = torch.optim.Adam(inn.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8
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
        plot_frequency=10,
    )
    best_path = trainer.run_io.policy_path()
    if os.path.isfile(best_path):
        print(f"Best policy stored at: {os.path.abspath(best_path)}")

    # Reset the flag so subsequent rollouts/eval are deterministic.
    prob.track_all_fp_iters = False
    inn.eval()
    return inn


def _plot_multiasset_rollout(
    prob: LiquidationPortfolioOC,
    policies: dict[str, Any],
    z0_batch: torch.Tensor,
    save_path: str,
    n_show: int,
    title: str | None = None,
    tag: str | None = None,
) -> None:
    """Multi-asset analogue of :meth:`LiquidationBenchmark.plot_comparison`.

    Rolls out each labeled policy with :class:`JFBPolicyRollout` for the
    first ``n_show`` initial conditions and overlays them on the
    ``liquidation_panels(n_assets)`` grid. No exact reference is drawn:
    the closed-form BVP solver only handles the single-asset case, so for
    ``n_assets > 1`` we fall back to JFB-only.
    """
    from dataclasses import replace as _dc_replace
    palette = ("#d6604d", "#4daf4a", "#984ea3", "#ff7f00", "#377eb8")
    batch = min(z0_batch.shape[0], n_show)
    z0 = z0_batch[:batch].to(prob.device)
    trajectories = []
    for i, (label, pol) in enumerate(policies.items()):
        color = palette[i % len(palette)]
        roller = JFBPolicyRollout(prob, pol)
        for b in range(batch):
            tr = roller.solve(z0[b])
            tr = _dc_replace(
                tr,
                label=label if b == 0 else None,
                style={"color": color, "lw": 2.0, "alpha": 0.75},
            )
            trajectories.append(tr)

    panels = liquidation_panels(prob.n_assets)
    ncols = 3 if prob.n_assets > 1 else 2
    if title is None:
        tag_str = f"  [{tag}]" if tag else ""
        title = (
            f"Liquidation portfolio (n_assets={prob.n_assets}) — "
            f"{' vs '.join(policies.keys())}  "
            f"(γ={prob.gamma:.1f}){tag_str}"
        )
    BenchmarkPlotter(panels, ncols=ncols).plot(
        trajectories, save_path=save_path, title=title,
    )


def _write_diagnostics(
    prob: LiquidationPortfolioOC,
    policy: ImplicitNetOC,
    z0_diag: np.ndarray,
    args: argparse.Namespace,
    *,
    label: str,
    out_filename: str,
) -> None:
    diag_traj = diagnostic_rollout(
        prob, policy,
        torch.as_tensor(z0_diag, dtype=torch.float32, device=prob.device),
        label=label,
        record_trace_at_t0=True,
    )
    traj_for_plot = diag_traj
    extra_panels = []
    # The exact-BVP costate overlay is single-asset only; skip it cleanly
    # whenever n_assets > 1 (the costate panels then just plot p_θ alone).
    if abs(float(prob.gamma) - 2.0) < 1e-6 and getattr(prob, "n_assets", 1) == 1:
        try:
            traj_for_plot = attach_bvp_costate_to_meta(
                diag_traj, prob, np.asarray(z0_diag),
            )
            extra_panels = liquidation_costate_vs_bvp_panels()
        except Exception as exc:
            print(f"  [warn] BVP costate overlay skipped: {exc}")
    elif getattr(prob, "n_assets", 1) > 1:
        print("  [info] BVP costate overlay disabled for multi-asset runs.")

    # For multi-asset, plot the inventory costates p_{q_1}, p_{q_2} (still
    # the most informative pair; for n_assets >= 2 these are indices 0, 1).
    n_assets = int(getattr(prob, "n_assets", 1))
    state_components = (0, 1) if n_assets >= 2 else (0, 1)
    diag_panels = diagnostic_panels(state_components=state_components) + extra_panels
    # Closed-form u*(p_θ) along the JFB rollout: gap to the trajectory's
    # own u(t) isolates the inner-FP convergence error (no re-rollout).
    # In the reduced formulation u* = (S + p_q + κ p_S) / (2η) — no
    # division by p_X, no pinning needed.
    if prob.has_closed_form_u_star() and args.learned_costate_overlay == "on":
        try:
            diag_panels = diag_panels + [
                liquidation_u_decomposition_panel(prob)
            ]
        except Exception as exc:
            print(f"  [warn] u*(p_θ) decomposition panel skipped: {exc}")
    diag_out = os.path.join(
        results_dir(type(prob).__name__, "benchmark"),
        out_filename,
    )
    BenchmarkPlotter(diag_panels, ncols=2).plot(
        [traj_for_plot], save_path=diag_out,
        title=(
            f"{label} diagnostics — α_fp={args.fp_alpha:.6g}, "
            f"max_iters={args.fp_max_iters}, tol={args.fp_tol:.0e}, "
            f"AA={'on' if args.use_aa else 'off'}"
        ),
    )
    print(f"Diagnostics figure written to: {os.path.abspath(diag_out)}")


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
    args.fp_alpha = resolve_liquidation_fp_alpha(prob, args.fp_alpha)
    print(
        f"Problem: n_assets={prob.n_assets}, state_dim={prob.state_dim}, "
        f"control_dim={prob.control_dim}, gamma={prob.gamma:.3g}"
    )
    print(
        "Inner FP solver: "
        f"alpha={args.fp_alpha:.6g}  max_iters={args.fp_max_iters}  "
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
    print(
        "Backprop regime: "
        + ("JFB + full-AD (overlay)" if args.full_ad else "JFB only")
    )

    # Always train (or load) the analytic-JFB policy.  When --full-ad is set
    # we also train a second policy through full autograd.  The user's
    # ``--tag`` is preserved verbatim for the JFB run; the AD run gets an
    # internal ``-fullAD`` suffix so file artifacts don't collide.
    inn_jfb = _train_or_load(
        prob, args, device, full_ad=False, trainer_tag=args.tag,
    )

    inn_ad: ImplicitNetOC | None = None
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

    # Output path is always tag-stamped so back-to-back runs don't clobber
    # each other; the regime (single- vs multi-asset) only changes the
    # filename stem so the two layouts remain easy to find on disk.
    default_name = (
        f"jfb_vs_exactbvp_benchmark_{args.tag}.png" if prob.n_assets == 1
        else f"jfb_multiasset_rollout_n{prob.n_assets}_{args.tag}.png"
    )
    out = args.output or os.path.join(
        results_dir(type(prob).__name__, "benchmark"),
        default_name,
    )

    policies: dict[str, Any] = {"JFB (analytic)": inn_jfb}

    def _add_learned_costate_overlays(label_prefix: str, p_net) -> None:
        """Append a single LearnedCostatePolicy overlay when enabled.

        Gated on ``prob.has_closed_form_u_star()`` so problems without a
        closed-form argmin_u H silently no-op. Reduced LiquidationPortfolio
        has no p_X to pin and no division by p_X in the closed form, so
        the legacy ``learned`` / ``pinned`` / ``both`` distinction is
        gone — only ``on`` / ``off``.
        """
        if not prob.has_closed_form_u_star():
            return
        if args.learned_costate_overlay == "off":
            return
        policies[f"{label_prefix}  u*(p_θ)"] = LearnedCostatePolicy(
            p_net, prob,
        )

    _add_learned_costate_overlays("JFB", inn_jfb.p_net)
    if inn_ad is not None:
        policies["JFB (full AD)"] = inn_ad
        _add_learned_costate_overlays("AD", inn_ad.p_net)

    if prob.n_assets == 1:
        # Single-asset: keep the legacy 6-panel BVP-vs-JFB comparison figure.
        # Tag is injected into the figure title for parity with the
        # multi-asset path.
        bench = LiquidationBenchmark(prob)
        policy_str = "JFB" if len(policies) == 1 else " vs ".join(policies.keys())
        bench_title = (
            f"LiquidationPortfolio — {policy_str} vs Exact BVP  "
            f"(γ={float(prob.gamma):.1f}, η={float(prob.eta):g}, "
            f"κ={float(prob.kappa):.0e})  [{args.tag}]"
        )
        if len(policies) == 1:
            bench.plot_comparison(inn_jfb, z0_plot, save_path=out,
                                  n_show=args.n_show, title=bench_title)
        else:
            bench.plot_comparison(
                policies, z0_plot, save_path=out, n_show=args.n_show,
                title=bench_title,
            )
    else:
        # Multi-asset: closed-form BVP solver doesn't generalise yet, so we
        # produce a JFB-only roll-out figure with one (q_i, u_i, S_i) row
        # per asset plus a shared X(t) panel.
        print(
            "[info] Multi-asset run (n_assets > 1): exact BVP reference is "
            "single-asset only and is being skipped. The figure shows the "
            "JFB rollout(s) on the per-asset panels."
        )
        _plot_multiasset_rollout(
            prob, policies, z0_plot, save_path=out, n_show=args.n_show,
            tag=args.tag,
        )
    print(f"Figure written to: {os.path.abspath(out)}")

    if args.diagnostics:
        z0_diag = z0_plot[0].detach().cpu().numpy().reshape(-1)
        _write_diagnostics(
            prob, inn_jfb, z0_diag, args,
            label="JFB",
            out_filename=f"jfb_diagnostics_{args.tag}.png",
        )
        if inn_ad is not None:
            _write_diagnostics(
                prob, inn_ad, z0_diag, args,
                label="JFB (full AD)",
                out_filename=f"jfb_diagnostics_{args.tag}-fullAD.png",
            )


if __name__ == "__main__":
    main()
