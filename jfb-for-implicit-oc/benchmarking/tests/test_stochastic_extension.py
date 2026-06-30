"""Tests for the stochastic Almgren-Chriss extension.

Covers the four validation milestones of the stochastic-dynamics plan:

(a) ``compute_sigma`` shape / block contract -- the diffusion lives only on
    the price (S) block of the state and is control-independent;
(b) Hutchinson trace estimator is unbiased for ``Tr(Sigma Hess phi)`` --
    checked against an exact dense-Hessian evaluation on a tiny network;
(c) ``sigma_S = 0`` continuity -- with the diffusion off, Euler-Maruyama
    collapses to explicit Euler and the BVP reference is recovered, so all
    deterministic behaviour is byte-for-byte preserved;
(d) the multi-path rollout produces a stochastic (3D) ``Trajectory`` whose
    paths genuinely differ, ready for mean +/- std band plots.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _make_prob(
    *,
    sigma=0.02, kappa=1e-4, eta=0.1, gamma=2.0,
    epsilon=1e-2, alpha=30.0,
    t_initial=0.0, t_final=2.0, nt=40,
    n_assets=1, sigma_S=0.0, n_hutch=1, noise_seed=None,
    q0_min=0.5, q0_max=1.5, S0=1.0,
):
    """Build a (single- or multi-asset) stochastic ``LiquidationPortfolioOC``."""
    from LiquidationPortfolio import LiquidationPortfolioOC
    return LiquidationPortfolioOC(
        batch_size=4, t_initial=t_initial, t_final=t_final, nt=nt,
        n_assets=n_assets, sigma=sigma, kappa=kappa, eta=eta, gamma=gamma,
        epsilon=epsilon, alpha=alpha, q0_min=q0_min, q0_max=q0_max, S0=S0,
        sigma_S=sigma_S, n_hutch=n_hutch, noise_seed=noise_seed,
    )


def _make_policy(prob):
    """Minimal ImplicitNetOC policy wired to ``prob`` (gamma=2, scalar FP step)."""
    from ImplicitNets import Phi, ImplicitNetOC
    phi = Phi(2, 16, prob.state_dim, dev=prob.device)
    eta_max = float(prob.eta.max().item())
    eta_min = float(prob.eta.min().item())
    alpha_fp = 1.0 / (eta_max + eta_min)
    return ImplicitNetOC(
        prob.state_dim, prob.control_dim,
        alpha=alpha_fp, max_iters=30, tol=1e-6,
        use_aa=False, beta=0.0, p_net=phi, oc_problem=prob,
        use_control_limits=False, dev=prob.device,
    )


# ------------------------------------------------------------------
# (a) compute_sigma shape / block contract
# ------------------------------------------------------------------

@pytest.mark.parametrize("n_assets", [1, 3])
def test_compute_sigma_shape_and_blocks(n_assets):
    prob = _make_prob(
        n_assets=n_assets,
        sigma=tuple([0.02] * n_assets) if n_assets > 1 else 0.02,
        kappa=tuple([1e-4] * n_assets) if n_assets > 1 else 1e-4,
        eta=tuple([0.1] * n_assets) if n_assets > 1 else 0.1,
        q0_min=tuple([0.5] * n_assets) if n_assets > 1 else 0.5,
        q0_max=tuple([1.5] * n_assets) if n_assets > 1 else 1.5,
        S0=tuple([1.0] * n_assets) if n_assets > 1 else 1.0,
        sigma_S=0.05,
    )
    B = 5
    z = torch.randn(B, prob.state_dim)
    u = torch.randn(B, prob.control_dim)
    sigma = prob.compute_sigma(0.0, z, u)

    assert sigma.shape == (B, prob.state_dim, prob.n_brownian)
    # q-block (rows 0..n-1) must be identically zero.
    assert torch.allclose(sigma[:, :n_assets, :], torch.zeros_like(sigma[:, :n_assets, :]))
    # S-block must be nonzero (the price diffusion factor).
    assert torch.any(sigma[:, n_assets:2 * n_assets, :] != 0)
    # Control-independence: changing u does not change sigma.
    sigma2 = prob.compute_sigma(0.0, z, u + 3.14)
    assert torch.allclose(sigma, sigma2)


def test_has_diffusion_flag():
    assert _make_prob(sigma_S=0.0).has_diffusion() is False
    assert _make_prob(sigma_S=0.1).has_diffusion() is True


# ------------------------------------------------------------------
# (b) Hutchinson trace unbiasedness vs exact Hessian
# ------------------------------------------------------------------

def test_hutchinson_trace_matches_exact_hessian():
    """The Hutchinson estimate of Tr(Sigma Hess phi) is unbiased.

    Averaging many probes must converge to the exact dense-Hessian trace.
    """
    torch.manual_seed(0)
    prob = _make_prob(n_assets=1, sigma_S=0.3, nt=10)
    policy = _make_policy(prob)
    p_net = policy.p_net

    B = 3
    z = torch.randn(B, prob.state_dim)
    u = torch.zeros(B, prob.control_dim)

    # Exact: Sigma = sigma sigma^T, H = Hessian of getPhi wrt z, per sample.
    sigma = prob.compute_sigma(0.0, z, u)              # (B, D, n_bm)
    Sigma = torch.bmm(sigma, sigma.transpose(1, 2))    # (B, D, D)

    exact = torch.zeros(B)
    for b in range(B):
        zb = z[b : b + 1].clone().requires_grad_(True)

        def phi_scalar(zz):
            return p_net.getPhi(0.0, zz).sum()

        H = torch.autograd.functional.hessian(phi_scalar, zb)
        H = H.reshape(prob.state_dim, prob.state_dim)
        exact[b] = torch.trace(Sigma[b] @ H)

    gen = torch.Generator().manual_seed(123)
    est = prob.compute_trace_sigma_hess_phi(
        0.0, z, p_net, u=u, n_hutch=4000, generator=gen, create_graph=False,
    )

    assert est.shape == (B,)
    # Monte-Carlo: loose tolerance, but must track the exact trace.
    np.testing.assert_allclose(
        est.detach().numpy(), exact.detach().numpy(), atol=5e-2, rtol=5e-2,
    )


def test_trace_zero_when_deterministic():
    prob = _make_prob(sigma_S=0.0)
    policy = _make_policy(prob)
    z = torch.randn(3, prob.state_dim)
    est = prob.compute_trace_sigma_hess_phi(0.0, z, policy.p_net, n_hutch=8)
    assert torch.allclose(est, torch.zeros(3))


# ------------------------------------------------------------------
# (c) sigma_S = 0 continuity / determinism preservation
# ------------------------------------------------------------------

def test_diffusion_increment_zero_when_deterministic():
    prob = _make_prob(sigma_S=0.0)
    z = torch.randn(4, prob.state_dim)
    u = torch.randn(4, prob.control_dim)
    incr = prob.diffusion_increment(0.0, z, u, prob.h)
    assert torch.allclose(incr, torch.zeros_like(z))


def test_euler_maruyama_reduces_to_euler_when_sigma_zero():
    """generate_trajectory with sigma_S=0 equals plain explicit Euler."""
    prob = _make_prob(sigma_S=0.0, nt=20)
    policy = _make_policy(prob)
    z0 = prob.sample_initial_condition()

    traj = prob.generate_trajectory(policy, z0, prob.nt, return_full_trajectory=True)

    # Reference explicit-Euler rollout.
    z = z0.clone()
    t = prob.t_initial
    with torch.no_grad():
        for _ in range(prob.nt):
            u = policy(z, t)
            z = z + prob.h * prob.compute_f(t, z, u)
            t += prob.h
    assert torch.allclose(traj[:, :, -1], z, atol=1e-5)


def test_bvp_recovery_unaffected_by_sigma_S():
    """sigma_S enters only the SDE, never the deterministic BVP reference."""
    from benchmarking.solvers import AlmgrenChrissBVPSolver

    prob_det = _make_prob(sigma_S=0.0)
    prob_stoch = _make_prob(sigma_S=0.5)

    z0 = np.array([1.0, 1.0])
    a = AlmgrenChrissBVPSolver(prob_det, n_bvp_nodes=200, bvp_tol=1e-9).solve(z0)
    b = AlmgrenChrissBVPSolver(prob_stoch, n_bvp_nodes=200, bvp_tol=1e-9).solve(z0)

    np.testing.assert_allclose(a.z, b.z, atol=1e-8)
    np.testing.assert_allclose(a.u, b.u, atol=1e-8)


def test_optimal_u_unchanged_by_sigma_S():
    """The closed-form u* depends only on (S, p_q, p_S, eta, kappa)."""
    prob_det = _make_prob(sigma_S=0.0)
    prob_stoch = _make_prob(sigma_S=0.7)
    z = torch.randn(4, prob_det.state_dim)
    p = torch.randn(4, prob_det.state_dim)
    u_det = prob_det.optimal_u_from_costate(0.0, z, p)
    u_stoch = prob_stoch.optimal_u_from_costate(0.0, z, p)
    assert torch.allclose(u_det, u_stoch)


# ------------------------------------------------------------------
# (d) multi-path stochastic rollout / band readiness
# ------------------------------------------------------------------

def test_jfb_policy_rollout_multipath_is_stochastic():
    from benchmarking.solvers import JFBPolicyRollout

    prob = _make_prob(sigma_S=0.2, nt=20)
    policy = _make_policy(prob)
    z0 = prob.sample_initial_condition()[0]

    n_paths = 16
    traj = JFBPolicyRollout(prob, policy).solve(
        z0.detach().numpy(), n_paths=n_paths, noise_seed=0,
    )

    assert traj.is_stochastic
    # (n_paths, nt+1, state_dim + cash_observer_col)
    assert traj.z.shape[0] == n_paths
    assert traj.z.shape[1] == prob.nt + 1
    # Paths must genuinely differ (price diffusion injects spread).
    spread = traj.z[:, -1, prob.n_assets].std()
    assert spread > 0.0


def test_monte_carlo_band_helper_smoke():
    from benchmarking import monte_carlo_policy_band

    prob = _make_prob(sigma_S=0.15, nt=15)
    policy = _make_policy(prob)
    z0 = prob.sample_initial_condition()[0]

    band = monte_carlo_policy_band(prob, policy, z0, n_paths=12, seed=1)
    assert band.is_stochastic
    assert band.meta["n_paths"] == 12
    assert band.label == "MC band"


def test_deterministic_rollout_collapses_to_single_path():
    """n_paths>1 on a deterministic problem still yields a single 2D path."""
    from benchmarking.solvers import JFBPolicyRollout

    prob = _make_prob(sigma_S=0.0, nt=10)
    policy = _make_policy(prob)
    z0 = prob.sample_initial_condition()[0]

    traj = JFBPolicyRollout(prob, policy).solve(
        z0.detach().numpy(), n_paths=8,
    )
    assert not traj.is_stochastic
    assert traj.z.ndim == 2
