"""
examples.example_liquidationportfolio
--------------------------------------
Train the JFB implicit policy on the Almgren-Chriss liquidation problem.
Artifacts go to results/LiquidationPortfolioOC/ via RunIO. Set full_AD=True
to train with full autodiff (BPTT) as a baseline.
"""

import os
import sys

import numpy as np
import torch

# Make the reorganised package importable when running this file directly:
# core/ and models/ still use flat imports (e.g. `from ImplicitOC import ...`),
# so they need to be on sys.path themselves; the project root is needed for
# `core.paths`.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../jfb-for-implicit-oc/examples
_ROOT = os.path.dirname(_HERE)                               # .../jfb-for-implicit-oc
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LiquidationPortfolio  import LiquidationPortfolioOC   # models/
from ImplicitNets          import Phi, ImplicitNetOC       # core/
from OptimalControlTrainer import OptimalControlTrainer    # core/


# Pinned experiment configuration matching the CLI invocation
#
#   python plot_liquidation_jfb.py --t-final 10 --kappa 1e-3 --epsilon 1e-3 \
#       --alpha 20 --q0-min 3 --q0-max 5 --S0 5 --train-epochs 50 --gamma 2 \
#       --tag JFB_g05_alpha20_q3to5-gamma2
#
# All values not set on the CLI fall back to defaults: nt=100, sigma=0.02,
# eta=0.1, X0=0.0, batch_size=64, lr=1e-3, plot_frequency=10. alphaHJB /
# alphaadj match the CLI's silent default of (0.0, 0.0); they are still
# spelled out explicitly in the constructor below so a reader doesn't have
# to dig into ``LiquidationPortfolioOC.__init__`` to discover they exist.
# This keeps the example bit-for-bit consistent with the CLI configuration
# (modulo the RNG seed, which the CLI script does not set but this runner
# does).
EXPERIMENT_TAG_SUFFIX = "example-run-gamma2"


def run_liquidation_jfb(
    *,
    full_AD: bool = False,
    epochs: int = 50,
    lr: float = 1e-3,
    plot_frequency: int = 10,
    device: str = "cpu",
) -> OptimalControlTrainer:
    """Train a JFB liquidation policy and return the trainer (so the caller
    can read ``trainer.run_io`` if it needs to look up artifact paths)."""

    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############     Liquidation Portfolio with INN     ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()

    lp = LiquidationPortfolioOC(
        batch_size=64, t_initial=0.0, t_final=10.0, nt=100,
        sigma=0.02, kappa=1e-3, eta=0.1, gamma=2, epsilon=1e-3,
        alpha=20, q0_min=3.0, q0_max=5.0, S0=5.0, X0=0.0,
        device=device,
        # HJB / adjoint consistency-loss weights. Default = 0, i.e. the JFB
        # objective is only running cost + terminal cost. Bump to e.g.
        # ``[1.0, 1.0]`` to penalise HJB / adjoint residuals during training.
        alphaHJB=[0.0, 0.0],
        alphaadj=[0.0, 0.0],
    )
    lp.track_all_fp_iters = full_AD

    phi = Phi(3, 50, lp.state_dim, dev=device)
    inn = ImplicitNetOC(
        lp.state_dim, lp.control_dim,
        # Anderson-accelerated FP solver: large step + small cap is fine
        # because AA adapts; lower tol than the legacy 1e-4 means we
        # actually converge to a real fixed point of T at inference.
        alpha=1.0, max_iters=50, tol=1e-6,
        use_aa=True, beta=0.5,
        p_net=phi, oc_problem=lp,
        u_min=0, u_max=10, use_control_limits=True,
        dev=device,
    ).to(device)

    opt = torch.optim.Adam(inn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10,
    )

    tag_prefix = "FullAD" if full_AD else "JFB"
    tag = f"{tag_prefix}_{EXPERIMENT_TAG_SUFFIX}"
    trainer = OptimalControlTrainer(
        inn, lp, opt, scheduler=scheduler, device=device, tag=tag,
    )
    trainer.set_mode("standard")  # JFB = standard

    z0 = lp.sample_initial_condition()
    trainer.train(z0, num_epochs=epochs, plot_frequency=plot_frequency)
    return trainer


def main():
    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_liquidation_jfb(full_AD=False, epochs=50, lr=1e-3,
                        plot_frequency=10, device=device)


if __name__ == "__main__":
    main()
