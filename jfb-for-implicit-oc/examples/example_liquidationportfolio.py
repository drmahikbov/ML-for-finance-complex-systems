"""
Almgren-Chriss style liquidation: train the JFB policy and let the trainer
write every artifact to ``results/LiquidationPortfolioOC/``.

This file is the *declarative* layer of the workflow. It only:

1. instantiates a :class:`LiquidationPortfolioOC` problem with concrete
   parameters,
2. wires up the :class:`ImplicitNetOC` policy and an optimizer / scheduler,
3. hands them to :class:`OptimalControlTrainer` and calls ``train``.

Everything path- or filename-related lives in
:class:`core.run_io.RunIO` and :func:`core.paths.results_dir`. Every plot
is produced by :class:`benchmarking.BenchmarkPlotter` via the model's
``panels()`` and ``to_trajectory()`` methods. Adding a new model means
copying this file, swapping the problem class, and trusting the trainer
to write the standard six-artifact bundle.
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
from OptimalControlTrainer import OptimalControlTrainer    # core/*
from utils import GradientTester



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
    sigma_S=0.0,
    n_hutch: int = 1,
    n_paths_per_ic: int = 1,
    noise_seed=None,
) -> OptimalControlTrainer:
    """Train a JFB liquidation policy and return the trainer (so the caller
    can read ``trainer.run_io`` if it needs to look up artifact paths).

    Stochastic extension
    --------------------
    Set ``sigma_S > 0`` to switch the price dynamics from the deterministic
    ``dS = -kappa u dt`` to the SDE ``dS = -kappa u dt + sigma_S dW``. When
    enabled:

    * the forward rollout becomes Euler-Maruyama;
    * the HJB residual diagnostic picks up the Hutchinson estimate of
      ``1/2 Tr(Sigma_S Hess phi)`` (``n_hutch`` probe vectors);
    * ``n_paths_per_ic`` replicates each initial condition so the training
      objective is an explicit Monte-Carlo expectation over Brownian paths;
    * ``noise_seed`` makes the Brownian increments reproducible.

    With ``sigma_S = 0`` (the default) every code path collapses to the
    original deterministic behaviour.
    """

    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############     Liquidation Portfolio with INN     ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()

    lp = LiquidationPortfolioOC(
        batch_size=64, t_initial=0.0, t_final=10.0, nt=100,n_assets=2,
        sigma=(0.02, 0.04),
        kappa=(1e-4, 1e-3),
        eta=(0.1, 0.3),
        gamma=2,
        epsilon=1e-2,
        alpha=30,
        q0_min=(1, 1.5),
        q0_max=(1.5, 2),
        S0=(1.0, 1.5),
        X0=0.0,
        device=device,
        # HJB / adjoint consistency-loss weights. Default = 0, i.e. the JFB
        # objective is only running cost + terminal cost. Bump to e.g.
        # ``[1.0, 1.0]`` to penalise HJB / adjoint residuals during training.
        alphaHJB=[0.0, 0.0],
        alphaadj=[0.0, 0.0],
        # Stochastic price extension (0 => deterministic, original behaviour).
        sigma_S=sigma_S,
        n_hutch=n_hutch,
        noise_seed=noise_seed,
    )
    lp.track_all_fp_iters = full_AD
    # Monte-Carlo paths per initial condition for the (stochastic) objective.
    lp.n_paths_per_ic = int(n_paths_per_ic)

    # Reduced LiquidationPortfolioOC has ``∂²H'/∂u² = 2η`` (problem constant
    # per asset). Per asset i the FP iteration ``u_i ← u_i − α (∂_u H')_i``
    # has contraction factor ``|1 − α · 2 η_i|``. ImplicitNetOC uses a
    # SCALAR α in the residual norm (cannot pass a per-asset α), so we
    # pick the minimax-optimal scalar:
    #
    #     α_fp = 1 / (η_max + η_min)
    #         ⇒ |1 − α 2 η_i| ∈ [|η_max − η_min| / (η_max + η_min)]  (worst case)
    #
    # For a homogeneous-η problem (η_max = η_min = η) this collapses to
    # ``α_fp = 1/(2 η)`` — exactly the one-shot Newton step used by the
    # reference jfb-new-copy AlmgrenChriss training loop. For
    # heterogeneous η it is the smallest worst-case contraction factor
    # achievable with a scalar step.
    eta_max = float(lp.eta.max().item())
    eta_min = float(lp.eta.min().item())
    alpha_fp = 1.0 / (eta_max + eta_min)
    contraction_worst = float(
        torch.abs(1.0 - alpha_fp * 2.0 * lp.eta).max().item()
    )
    print(
        f"FP step: alpha_fp = 1/(eta_max + eta_min) = {alpha_fp:.4g}  "
        f"worst contraction factor = {contraction_worst:.3e}  "
        f"(0 ⇒ one-shot exact, < 1 ⇒ strict contraction)"
    )

    phi = Phi(3, 50, lp.state_dim, dev=device)
    inn = ImplicitNetOC(
        lp.state_dim, lp.control_dim,
        # Minimax-optimal scalar Newton step on the constant Hessian 2η.
        # One-shot exact for homogeneous-η problems, strict contraction
        # for heterogeneous ones — no Anderson required either way.
        alpha=alpha_fp, max_iters=50, tol=1e-6,
        use_aa=False, beta=0.0,
        p_net=phi, oc_problem=lp,
        u_min=0, u_max=10, use_control_limits=False,
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


def run_liquidation_jfb_stochastic(
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    plot_frequency: int = 10,
    device: str = "cpu",
    sigma_S=0.05,
    n_hutch: int = 4,
    n_paths_per_ic: int = 8,
    noise_seed: int = 7,
    n_band_paths: int = 256,
):
    """Train a *stochastic* AC liquidation policy and draw an MC band rollout.

    Demonstrates the full stochastic pipeline end to end: Euler-Maruyama
    rollout, Hutchinson HJB-trace diagnostic, and a Monte-Carlo objective,
    followed by a mean +/- std band over ``n_band_paths`` simulated paths.
    """
    import matplotlib.pyplot as plt
    from benchmarking import monte_carlo_policy_band

    trainer = run_liquidation_jfb(
        full_AD=False, epochs=epochs, lr=lr, plot_frequency=plot_frequency,
        device=device, sigma_S=sigma_S, n_hutch=n_hutch,
        n_paths_per_ic=n_paths_per_ic, noise_seed=noise_seed,
    )

    prob = trainer.oc_problem
    policy = trainer.policy
    z0 = prob.sample_initial_condition()[0]   # single IC -> MC band over paths

    band = monte_carlo_policy_band(
        prob, policy, z0, n_paths=n_band_paths, seed=noise_seed,
        label="JFB MC band",
    )
    print(
        f"[stochastic demo] MC band: n_paths={band.meta.get('n_paths')}, "
        f"z.shape={band.z.shape} (mean +/- std over paths)"
    )

    # Self-contained mean +/- 1 std band figure for asset 0 (inventory q_0 and
    # price S_0). ``band.z`` has shape (n_paths, nt+1, state_dim[+cash]).
    try:
        t = band.t
        n = prob.n_assets
        z = band.z                                  # (P, nt+1, ...)
        q0 = z[:, :, 0]                             # inventory of asset 0
        S0 = z[:, :, n]                            # price of asset 0
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, data, name in ((axes[0], q0, "q_0 (inventory)"),
                               (axes[1], S0, "S_0 (price)")):
            mu = data.mean(axis=0)
            sd = data.std(axis=0)
            ax.plot(t, mu, color="#4393c3", lw=2.0, label="mean")
            ax.fill_between(t, mu - sd, mu + sd, color="#4393c3", alpha=0.25,
                            label="+/- 1 std")
            ax.set_title(f"Stochastic AC rollout: {name}")
            ax.set_xlabel("t")
            ax.legend()
        fig.tight_layout()
        save_path = os.path.join(trainer.run_io.benchmark_dir,
                                 "stochastic_mc_band.png")
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"[stochastic demo] wrote MC band figure -> {save_path}")
    except Exception as exc:   # plotting is best-effort in the demo
        print(f"[stochastic demo] band plot skipped: {exc!r}")

    return trainer


def main():
    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_liquidation_jfb(full_AD=False, epochs=30, lr=1e-3,
                        plot_frequency=10, device=device)

    # Stochastic extension demo (price volatility on S). Comment out to skip.
    run_liquidation_jfb_stochastic(epochs=30, lr=1e-3, plot_frequency=10,
                                   device=device)


if __name__ == "__main__":
    main()
