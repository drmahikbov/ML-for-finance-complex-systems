"""
examples-RL.portfolio_optimization_RL
--------------------------------------
Train a JFB-RL implicit Hamiltonian policy on the Merton portfolio problem.
True dynamics are hidden behind AnalyticalEnvironment; the agent only sees
env.step. Set USE_ORACLE_JACOBIAN = True to swap RLS for analytical Jacobians
as a sanity check (should reproduce the known-dynamics JFB result).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# sys.path bootstrap — same pattern as the existing examples/ runners,
# extended to include core-RL/ and the examples-RL/ folder itself.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../examples-RL
_ROOT = os.path.dirname(_HERE)                                # .../jfb-for-implicit-oc
for _p in (
    _ROOT,
    os.path.join(_ROOT, "core"),
    os.path.join(_ROOT, "core-RL"),
    os.path.join(_ROOT, "models"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# core/ (unchanged) — value-function network, RunIO infrastructure
from core.ImplicitNets import Phi                                   # core/

# core-RL/ — new RL-aware components
from core_RL.Environment import AnalyticalEnvironment                  # core-RL/
from core_RL.JacobianEstimator import (                                # core-RL/
    RLSJacobianEstimator,
    OracleJacobianEstimator,
)
from core_RL.ImplicitNets_RL import ImplicitNetOC_RL                   # core-RL/
from core_RL.OptimalControlTrainer_RL import OptimalControlTrainer_RL  # core-RL/

# models/ — the Merton problem
from models.PortfolioOC_RL import PortfolioOC_RL                      # models/


# ===========================================================================
# Toggle: oracle baseline vs RLS estimator. See module docstring.
# ===========================================================================
USE_ORACLE_JACOBIAN = False


def run_portfolio_rl(
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    plot_frequency: int = 25,
    device: str = "cpu",
) -> OptimalControlTrainer_RL:
    """Train a JFB-RL portfolio policy and return the trainer.

    The trainer writes its full six-artifact bundle under
    ``results/PortfolioOC_RL/``.
    """

    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############     Merton Portfolio (RL) with INN     ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()

    # ----------------------------------------------------------------- #
    # 1. Problem.                                                       #
    # ----------------------------------------------------------------- #
    # μ_true = 0.10, r_true = 0.03 are *hidden* from the agent. They are
    # used only by the simulator (env.step) and (optionally) by the
    # OracleJacobianEstimator baseline.
    prob = PortfolioOC_RL(
        mu_true=0.10,
        r_true=0.03,
        lam=0.1,            # reduced from 0.5 so running-cost gradient no longer
                            # dominates the terminal-cost signal in the bracket
        W_ref=10,
        W0_min=0.8,
        W0_max=1.2,
        batch_size=32,
        t_initial=0.0,
        t_final=2.0,
        nt=50,
        alphaL=1.0,
        alphaG=5.0,         # amplify terminal-cost weight in adjoint and bracket
        device=device,
    )

    # ----------------------------------------------------------------- #
    # 2. Environment — wraps true f behind env.step. The agent never    #
    # queries prob.compute_f directly.                                  #
    # ----------------------------------------------------------------- #
    env = AnalyticalEnvironment(
        state_dim=prob.state_dim,
        control_dim=prob.control_dim,
        t_initial=prob.t_initial,
        t_final=prob.t_final,
        nt=prob.nt,
        f_callable=prob.compute_f,
        device=device,
    )

    # ----------------------------------------------------------------- #
    # 3. Jacobian estimator.                                            #
    # ----------------------------------------------------------------- #
    if USE_ORACLE_JACOBIAN:
        # Oracle baseline: cheats by querying analytical Jacobians.
        # If the rest of the RL pipeline is implemented correctly, this
        # should reproduce the known-dynamics JFB result.
        jac_est = OracleJacobianEstimator(
            nt=prob.nt,
            state_dim=prob.state_dim,
            control_dim=prob.control_dim,
            dt=prob.h,
            grad_f_z=prob.compute_grad_f_z,
            grad_f_u=prob.compute_grad_f_u,
            schedule_t=lambda k: prob.t_initial + k * prob.h,
            device=device,
        )
        tag_suffix = "Oracle"
    else:
        # PDF §5.3 recommends α_rls ∈ [0.8, 0.95] for nearly-stationary
        # dynamics. ``q0`` (initial precision regularisation) is set high
        # so the first few updates don't overcommit to noisy data.
        jac_est = RLSJacobianEstimator(
            nt=prob.nt,
            state_dim=prob.state_dim,
            control_dim=prob.control_dim,
            dt=prob.h,
            alpha_rls=0.9,
            q0=1.0,
            device=device,
        )
        tag_suffix = "RLS"

    # ----------------------------------------------------------------- #
    # 4. Value-function network and implicit policy.                    #
    # ----------------------------------------------------------------- #
    # Phi(nTh=3, hidden=50, d=state_dim) — same width pattern as the
    # liquidation example. Smooth activations (anti-derivative of tanh)
    # are required because we differentiate Phi twice (once analytically
    # for ∇_z φ, once via autograd for ∂(surrogate)/∂θ).
    phi = Phi(3, 50, prob.state_dim, dev=device)

    # Implicit policy. With lam=0.1 and alphaL=1, the effective curvature
    # of H w.r.t. u near π=0 is 2*αL*lam = 0.2. Setting alpha=0.25 gives
    # FP contraction rate |1 - 0.25*0.2| = 0.95 — same as the original
    # (lam=0.5, alpha=0.05) setting, so max_iters=300 is still sufficient.
    inn = ImplicitNetOC_RL(
        prob.state_dim, prob.control_dim,
        alpha=0.25,
        max_iters=300,
        tol=1e-4,
        p_net=phi,
        oc_problem=prob,
        u_min=-2.0, u_max=2.0,
        use_control_limits=True,
        dev=device,
    ).to(device)

    # ----------------------------------------------------------------- #
    # 5. Optimiser, scheduler, trainer.                                 #
    # ----------------------------------------------------------------- #
    opt = torch.optim.Adam(inn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15,
    )

    trainer = OptimalControlTrainer_RL(
        policy_net=inn,
        oc_problem=prob,
        env=env,
        jac_est=jac_est,
        optimizer=opt,
        scheduler=scheduler,
        device=device,
        tag=f"JFB-RL_{tag_suffix}",
    )

    # Sample a single batch of initial conditions and reuse it across
    # epochs (matches the convention of the existing examples — the
    # policy is trained semi-globally on this distribution).
    z0 = prob.sample_initial_condition()

    trainer.train(z0, num_epochs=epochs, plot_frequency=plot_frequency)
    return trainer


def main():
    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_portfolio_rl(
        epochs=200,
        lr=1e-3,
        plot_frequency=25,
        device=device,
    )


if __name__ == "__main__":
    main()