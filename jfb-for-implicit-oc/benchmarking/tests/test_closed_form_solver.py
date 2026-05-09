"""Tests for :class:`benchmarking.solvers.AlmgrenChrissClosedForm`.

These tests pin the closed-form solver against:

(a) the existing :class:`AlmgrenChrissBVPSolver` -- both should agree to
    BVP tolerance on every state and control component;
(b) PMP terminal stationarity ``2 eta u(T) = S(T) + 2 alpha q(T)`` --
    a property the closed form must satisfy by construction;
(c) ``sigma -> 0`` continuity between the analytical formula branch and
    the constant-rate fallback branch;
(d) the limit ``alpha -> infinity`` driving terminal inventory to zero,
    recovering the classic Almgren-Chriss solution.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_prob(
    *,
    sigma=0.02, kappa=1e-4, eta=0.1, gamma=2.0,
    epsilon=1e-2, alpha=30.0,
    t_initial=0.0, t_final=2.0, nt=100,
):
    """Build a single-asset :class:`LiquidationPortfolioOC` for tests."""
    from LiquidationPortfolio import LiquidationPortfolioOC
    return LiquidationPortfolioOC(
        batch_size=1, t_initial=t_initial, t_final=t_final, nt=nt,
        n_assets=1, sigma=sigma, kappa=kappa, eta=eta, gamma=gamma,
        epsilon=epsilon, alpha=alpha, q0_min=0.5, q0_max=1.5,
        S0=1.0,
    )


# ------------------------------------------------------------------
# (a) Cross-check vs the BVP reference
# ------------------------------------------------------------------

def test_closed_form_matches_bvp_reference():
    """Closed-form trajectory matches the BVP solver to BVP tolerance.

    The BVP collocation residual is ``bvp_tol = 1e-10`` here, so allowing
    ``atol = 1e-6`` on every state component leaves several orders of
    magnitude of headroom while still catching algebraic mistakes in the
    closed form.
    """
    from benchmarking.solvers import (
        AlmgrenChrissBVPSolver, AlmgrenChrissClosedForm,
    )

    prob = _make_prob()
    bvp = AlmgrenChrissBVPSolver(prob, n_bvp_nodes=500, bvp_tol=1e-10)
    cf = AlmgrenChrissClosedForm(prob, n_grid=prob.nt + 1)

    z0 = np.array([1.0, 1.0, 0.0])
    traj_bvp = bvp.solve(z0)
    traj_cf = cf.solve(z0)

    # The BVP solver picks its own node grid -- interpolate it onto the
    # closed-form grid before comparing.
    t_cf = traj_cf.t
    q_bvp = np.interp(t_cf, traj_bvp.t, traj_bvp.z[:, 0])
    S_bvp = np.interp(t_cf, traj_bvp.t, traj_bvp.z[:, 1])
    X_bvp = np.interp(t_cf, traj_bvp.t, traj_bvp.z[:, 2])

    np.testing.assert_allclose(traj_cf.z[:, 0], q_bvp, atol=1e-6)
    np.testing.assert_allclose(traj_cf.z[:, 1], S_bvp, atol=1e-6)
    # X(t) is integrated trapezoidally; the BVP integrates with the
    # midpoint rule, so allow a slightly looser bound on cash.
    np.testing.assert_allclose(traj_cf.z[:, 2], X_bvp, atol=1e-5)
    np.testing.assert_allclose(traj_cf.u[0, 0], traj_bvp.u[0, 0], atol=1e-6)

    assert traj_cf.cost is not None and traj_bvp.cost is not None
    np.testing.assert_allclose(traj_cf.cost, traj_bvp.cost, atol=1e-5)


# ------------------------------------------------------------------
# (b) Terminal stationarity (PMP)
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "sigma,alpha",
    [(0.0, 30.0), (0.02, 1.0), (0.05, 30.0), (0.2, 100.0), (0.5, 5.0)],
)
def test_terminal_stationarity(sigma, alpha):
    """``2 eta u(T) == S(T) + 2 alpha q(T)`` for the closed-form solution.

    This is the right-endpoint stationarity ``dH/du|_{t=T} = 0`` derived
    from PMP with ``p_X = -1, p_S(T) = 0, p_q(T) = 2 alpha q(T)``.  Any
    valid closed form must satisfy it exactly -- it's the equation that
    fixes ``D``.
    """
    from benchmarking.solvers import AlmgrenChrissClosedForm

    prob = _make_prob(sigma=sigma, alpha=alpha)
    cf = AlmgrenChrissClosedForm(prob, n_grid=prob.nt + 1)
    traj = cf.solve(np.array([1.0, 1.0, 0.0]))

    eta = float(prob.eta.reshape(-1)[0])
    q_T = float(traj.z[-1, 0])
    S_T = float(traj.z[-1, 1])
    lam = float(traj.meta["lam"])
    D = float(traj.meta["D"])
    Q0 = float(traj.meta["Q0"])
    if lam == 0.0:
        u_T = float(traj.u[0, 0])
    else:
        T = float(traj.meta["T"]) - float(traj.meta["t0"])
        u_T = -lam * (Q0 * np.sinh(lam * T) + D * np.cosh(lam * T))

    np.testing.assert_allclose(
        2.0 * eta * u_T, S_T + 2.0 * alpha * q_T,
        atol=1e-10, rtol=1e-10,
    )


# ------------------------------------------------------------------
# (c) sigma -> 0 continuity between branches
# ------------------------------------------------------------------

def test_sigma_zero_continuity():
    """The two ``sigma`` branches must agree near the threshold.

    With ``sigma = 1e-8`` we evaluate via the general analytical formula;
    with ``sigma = 0`` we hit the constant-rate fallback.  The two should
    be numerically indistinguishable on every component.
    """
    from benchmarking.solvers import AlmgrenChrissClosedForm

    cf_eps = AlmgrenChrissClosedForm(_make_prob(sigma=1e-8))
    cf_zero = AlmgrenChrissClosedForm(_make_prob(sigma=0.0))

    z0 = np.array([1.0, 1.0, 0.0])
    a, b = cf_eps.solve(z0), cf_zero.solve(z0)

    for k in range(3):
        np.testing.assert_allclose(a.z[:, k], b.z[:, k], atol=1e-6)
    np.testing.assert_allclose(a.u[:, 0], b.u[:, 0], atol=1e-6)


def test_sigma_zero_branch_returns_constant_u():
    """The fallback branch reduces to the analytical TWAP-with-penalty.

    For ``sigma = 0``,
    ``u*(t) = (S0 + 2 alpha Q0) / (2 eta + (kappa + 2 alpha) T)`` is
    constant in ``t``; the trajectory's u column should be exactly that
    scalar everywhere, and the meta ``D`` should be NaN (book-keeping
    flag for the degenerate branch).
    """
    from benchmarking.solvers import AlmgrenChrissClosedForm

    prob = _make_prob(sigma=0.0, alpha=10.0, kappa=1e-4, eta=0.1)
    cf = AlmgrenChrissClosedForm(prob, n_grid=prob.nt + 1)
    traj = cf.solve(np.array([1.0, 1.0, 0.0]))

    T = prob.t_final - prob.t_initial
    eta = float(prob.eta.reshape(-1)[0])
    kappa = float(prob.kappa.reshape(-1)[0])
    alpha = float(prob.alpha)
    expected = (1.0 + 2.0 * alpha) / (2.0 * eta + (kappa + 2.0 * alpha) * T)

    np.testing.assert_allclose(traj.u[:, 0], expected, atol=1e-12)
    assert np.isnan(traj.meta["D"])


# ------------------------------------------------------------------
# (d) Large alpha drives q(T) -> 0
# ------------------------------------------------------------------

def test_large_alpha_drives_terminal_inventory_to_zero():
    """``alpha -> infinity`` recovers classic Almgren-Chriss with q(T)=0.

    With a very large terminal-inventory penalty ``D`` saturates at
    ``-Q0 coth(lambda T)`` and ``q(T) = Q0 sinh(lambda(T-t)) /
    sinh(lambda T)`` evaluated at ``t = T`` gives 0.
    """
    from benchmarking.solvers import AlmgrenChrissClosedForm

    traj = AlmgrenChrissClosedForm(
        _make_prob(sigma=0.5, alpha=1.0e6)
    ).solve(np.array([1.0, 1.0, 0.0]))

    assert abs(traj.z[-1, 0]) < 1e-4


# ------------------------------------------------------------------
# Construction validation
# ------------------------------------------------------------------

def test_rejects_non_gamma_two():
    """``gamma != 2`` must be rejected at construction time."""
    from benchmarking.solvers import AlmgrenChrissClosedForm

    with pytest.raises(ValueError, match="2"):
        AlmgrenChrissClosedForm(_make_prob(gamma=1.5))


def test_trajectory_shape_contract():
    """Output shapes match the same contract as the BVP solver."""
    from benchmarking.solvers import AlmgrenChrissClosedForm

    prob = _make_prob(nt=50)
    cf = AlmgrenChrissClosedForm(prob, n_grid=prob.nt + 1)
    traj = cf.solve(np.array([1.0, 1.0, 0.0]))

    assert traj.t.shape == (prob.nt + 1,)
    assert traj.z.shape == (prob.nt + 1, 3)
    assert traj.u.shape == (prob.nt, 1)
    assert traj.label == "Exact CF"
    assert traj.style.get("ls") == "--"
    np.testing.assert_allclose(
        traj.z[0], np.array([1.0, 1.0, 0.0]), atol=1e-12,
    )


# ------------------------------------------------------------------
# (e) Reduced LiquidationPortfolio terminal-cost shape contract
# ------------------------------------------------------------------

def test_reduced_state_terminal_cost_shape():
    """Reduced ``LiquidationPortfolioOC`` has ``state_dim = 2 * n_assets``
    and ``∂G'/∂z = (2α q, 0)`` per asset.

    Replaces the legacy ``∂G/∂X = -1`` assertion (cash ``X`` is no
    longer an OC state, so there is no terminal-X gradient slot to
    check). Verifies the reduced ``G' = α‖q‖²`` gradient instead, and
    pins ``state_dim`` so future refactors can't silently re-inflate
    the OC state.
    """
    import torch

    prob = _make_prob()
    n = int(getattr(prob, "n_assets", 1))

    assert prob.state_dim == 2 * n, (
        f"state_dim should be 2 * n_assets = {2 * n}, got {prob.state_dim}"
    )

    q_batch = torch.tensor([[0.5], [1.5]], dtype=torch.float32)
    S_batch = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    z = torch.cat([q_batch, S_batch], dim=1)            # (2, 2n) — n_assets=1

    grad = prob.compute_grad_G_z(z).detach().cpu().numpy()
    alpha = float(prob.alpha)

    # ∂G'/∂q_i = 2 α q_i
    np.testing.assert_allclose(grad[:, :n], 2.0 * alpha * q_batch.numpy(), atol=0)
    # ∂G'/∂S_i = 0
    np.testing.assert_allclose(grad[:, n:2 * n], 0.0, atol=0)
