#!/usr/bin/env python3
"""
Overlay the per-asset break-even trading rate on the 20-asset trading-rate
panel for the IGNA-20-ASSETS-FINAL liquidation run.

Break-even rate (per asset) is the u at which the cash flow
    dX_i/dt = S_i u_i - eta_i (u_i^2 + eps)^(gamma/2)
crosses zero.  Ignoring the small eps regularisation,
    S_i u = eta_i u^gamma  =>  u_be_i(t) = (S_i(t) / eta_i)^(1/(gamma-1)),
which decreases as the impacted price S_i(t) is pushed down by selling.

Reproduces the trained policy from the saved checkpoint (same hyperparameters
as the original CLI run) and plots all 20 u_i(t) plus the dashed-black
break-even curve(s).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # .../jfb-for-implicit-oc
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LiquidationPortfolio import LiquidationPortfolioOC
from ImplicitNets import Phi, ImplicitNetOC
from benchmarking import JFBPolicyRollout

# ----------------------------------------------------------------------
# Exact hyperparameters of the IGNA-20-ASSETS-FINAL run.
# ----------------------------------------------------------------------
N_ASSETS = 20
T_FINAL = 2.0
NT = 100
KAPPA = 1e-5
SIGMA = 0.01414
ETA = 0.5
GAMMA = 1.9
EPSILON = 1e-2
ALPHA = 30.0
Q0_MIN = 3.0
Q0_MAX = 8.0
S0 = 2.0
X0 = 0.0
FP_ALPHA = 0.9
FP_MAX_ITERS = 200
FP_TOL = 1e-6
SEED = 42

CKPT = os.path.join(
    _ROOT,
    "results/LiquidationPortfolioOC/training/"
    "best_policy_IGNA-20-ASSETS-FINAL_20260515_190434.pth",
)
OUT = os.path.join(
    _ROOT,
    "results/LiquidationPortfolioOC/benchmark/"
    "trading_rate_with_breakeven_IGNA-20-ASSETS-FINAL.png",
)

ASSET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def main() -> None:
    device = "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    prob = LiquidationPortfolioOC(
        batch_size=256, t_initial=0.0, t_final=T_FINAL, nt=NT,
        n_assets=N_ASSETS,
        sigma=SIGMA, kappa=KAPPA, eta=ETA, gamma=GAMMA, epsilon=EPSILON,
        alpha=ALPHA, q0_min=Q0_MIN, q0_max=Q0_MAX, S0=S0, X0=X0,
        device=device, alphaHJB=(0.0, 0.0), alphaadj=(0.0, 0.0),
    )

    phi = Phi(3, 50, prob.state_dim, dev=device)
    inn = ImplicitNetOC(
        prob.state_dim, prob.control_dim,
        alpha=FP_ALPHA, max_iters=FP_MAX_ITERS, tol=FP_TOL,
        p_net=phi, oc_problem=prob,
        use_aa=False, beta=0.0,
        use_control_limits=False,
        dev=device,
    ).to(device)

    state = torch.load(CKPT, map_location=device, weights_only=True)
    inn.load_state_dict(state, strict=True)
    inn.eval()

    # Same path selection as plot_best_policy_same_axes: fresh sample, path 0.
    z0 = prob.sample_initial_condition()
    z0_one = z0[0]

    roller = JFBPolicyRollout(prob, inn)
    tr = roller.solve(z0_one)

    t = tr.t                       # (nt+1,)
    z = tr.z                       # (nt+1, 2n + 1)  -> [q, S, X]
    u = tr.u                       # (nt,   n)
    n = prob.n_assets
    S = z[:, n:2 * n]              # (nt+1, n)

    eta = float(ETA)
    gamma = float(GAMMA)
    eps = float(EPSILON)

    # Exact break-even (solve S u = eta (u^2 + eps)^(gamma/2) by bisection),
    # evaluated per asset on the price path (drop the last time node so it
    # aligns with u, which is defined on t[:-1]).
    S_u = S[:-1]                   # (nt, n)

    def breakeven(Si: np.ndarray) -> np.ndarray:
        # Vectorised bisection for u in (0, u_hi].
        lo = np.full_like(Si, 1e-6)
        hi = np.full_like(Si, 1e3)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            g = eta * (mid ** 2 + eps) ** (gamma / 2.0) - Si * mid
            # g < 0 => impact below revenue => break-even is higher => go right
            lo = np.where(g < 0, mid, lo)
            hi = np.where(g < 0, hi, mid)
        return 0.5 * (lo + hi)

    u_be = breakeven(S_u)          # (nt, n)
    u_be_approx = (S_u / eta) ** (1.0 / (gamma - 1.0))

    print(f"S range over horizon: [{S.min():.6f}, {S.max():.6f}]")
    print(f"break-even u range  : [{u_be.min():.4f}, {u_be.max():.4f}] "
          f"(exact),  approx (S/eta)^(1/(g-1)) at t=0 = {u_be_approx[0].mean():.4f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in range(n):
        c = ASSET_COLORS[i % len(ASSET_COLORS)]
        ax.plot(t[:-1], u[:, i], color=c, lw=1.8, alpha=0.85)

    # Per-asset break-even (dashed black). They nearly coincide because
    # kappa is tiny, so they render as a single dashed-black band.
    for i in range(n):
        ax.plot(
            t[:-1], u_be[:, i], color="black", ls="--", lw=1.4, alpha=0.9,
            label="break-even $u^{be}_i(t)=(S_i/\\eta)^{1/(\\gamma-1)}$"
            if i == 0 else None,
        )

    ax.set_title(f"Trading rate $u_i(t)$ with break-even, "
                 f"n_assets={n}, $\\gamma$={gamma:g}")
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
