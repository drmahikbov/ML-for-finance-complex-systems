"""
benchmarking.solvers
--------------------
Reference trajectory solvers.

A :class:`ReferenceSolver` turns an initial condition into a
:class:`~benchmarking.trajectory.Trajectory`.  Two concrete implementations
are shipped:

* :class:`AlmgrenChrissBVPSolver` -- closed-form γ=2 two-point BVP solver
  for :class:`LiquidationPortfolioOC`.
* :class:`JFBPolicyRollout` -- explicit-Euler rollout of a trained JFB
  policy.

Both return a single-path (deterministic) Trajectory.

Adding a new problem
~~~~~~~~~~~~~~~~~~~~
Subclass :class:`ReferenceSolver` and implement ``solve(z0, ...)``.
For example, a multi-asset Almgren-Chriss reference solver's
``solve`` signature would look like::

    class MultiAssetAlmgrenChrissBVPSolver(ReferenceSolver):
        def solve(self, z0: np.ndarray, **kwargs) -> Trajectory:
            # z0 layout: [q_1, ..., q_n, S_1, ..., S_n, X]  (length 2n+1)
            # internal state for the BVP is [q, S, p_q, p_S]  (size 4n)
            # return a Trajectory with z.shape == (N, 2n+1) and
            # u.shape == (N-1, n).
            ...

No change to :class:`benchmarking.trajectory.Trajectory`,
:class:`benchmarking.plotter.BenchmarkPlotter` or
:mod:`benchmarking.metrics` is required -- they all work off the generic
shape conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
import warnings
from scipy.integrate import solve_bvp

from .trajectory import Trajectory


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Coerce ``numpy`` or ``torch`` input into a detached numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ReferenceSolver(ABC):
    """Abstract base for anything that produces a Trajectory from ``z0``."""

    @abstractmethod
    def solve(self, z0: ArrayLike, **kwargs: Any) -> Trajectory:
        """Solve for a single initial condition.

        Parameters
        ----------
        z0 : array-like
            The initial state, shape ``(state_dim,)``.  Torch tensors are
            accepted and moved to CPU automatically.

        Returns
        -------
        Trajectory
            A deterministic trajectory.  (Stochastic solvers, when
            eventually added, may return a trajectory with a leading
            ``n_paths`` axis.)
        """
        raise NotImplementedError


# =============================================================================
# Almgren-Chriss Multi-asset BVP solver (γ=2) (can use n_assets=1 for single-asset)
# =============================================================================

_DEFAULT_EXACT_STYLE = {"color": "#2166ac", "ls": "--", "lw": 2.0}

class AlmgrenChrissBVPSolver(ReferenceSolver):
    """BVP reference solver for Almgren-Chriss liquidation with γ = 2.

    This class now supports both:
        - n_assets = 1: same behaviour as the original single-asset solver.
        - n_assets > 1: diagonal multi-asset case.

    Supports:
        - n_assets = 1
        - n_assets > 1
        - diagonal or full risk matrix sigma
        - diagonal or full permanent-impact matrix kappa

    Convention:
        - scalar/vector sigma is interpreted as volatility and squared into a covariance matrix;
        - matrix sigma is interpreted directly as the covariance/risk matrix;
        - scalar/vector kappa becomes diagonal impact;
        - matrix kappa is used directly as cross-impact.

    Internal BVP state:
        y = [q, S, p_q, p_S]  where each block has length n_assets.
    The returned Trajectory state is:
        z = [q_1, ..., q_n, S_1, ..., S_n, X]
    where X is reconstructed after solving the BVP.
    """

    def __init__(
        self,
        prob: Any,
        n_bvp_nodes: int = 500,
        bvp_tol: float = 1e-9,
    ):
        self.prob = prob
        self.n_bvp_nodes = n_bvp_nodes
        self.bvp_tol = bvp_tol

        # Number of assets.
        self.n_assets = int(getattr(prob, "n_assets", 1))

        # The helper _as_vec accepts:
        #   - a scalar, repeated across all assets;
        #   - a tensor / list / array of length n_assets.
        self.sigma = self._to_square_matrix(prob.sigma, self.n_assets, "sigma")
        self.kappa = self._to_square_matrix(prob.kappa, self.n_assets, "kappa")
        self.eta = self._to_asset_vector(prob.eta, self.n_assets, "eta")

        # These parameters are scalar.
        self.gamma = float(prob.gamma)
        self.epsilon = float(prob.epsilon)
        self.alpha = float(prob.alpha)
        self.t0 = float(prob.t_initial)
        self.T = float(prob.t_final)

        # This solver is only valid for gamma = 2 since for gamma != 2, the Hamiltonian is implicit
        if abs(self.gamma - 2.0) >= 1e-6:
            raise ValueError(
                f"AlmgrenChrissBVPSolver requires γ=2 (±1e-6); "
                f"got γ={self.gamma:.6f}."
            )

        # function to convert scalar or vector parameters into asset-aligned vectors of shape (n_assets,) on the correct device
    def _to_asset_vector(self, x, n_assets, name):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x_t = np.asarray(x, dtype=float)

        if x_t.ndim == 0:
            return np.full(n_assets, float(x_t))

        x_t = x_t.reshape(-1)

        if x_t.size == n_assets:
            return x_t

        raise ValueError(
            f"{name} must be a scalar or a vector of length {n_assets}; "
            f"got shape {x_t.shape}"
        )

    #function to convert scalar, vector, or matrix parameters into square matrices of shape (n_assets, n_assets) on the correct device
    def _to_square_matrix(self, x, n_assets, name):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x_t = np.asarray(x, dtype=float)

        if x_t.ndim == 0:
            if name == "sigma":
                return float(x_t) ** 2 * np.eye(n_assets)
            else:
                return float(x_t) * np.eye(n_assets)

        if x_t.ndim == 1:
            x_t = x_t.reshape(-1)

            if x_t.size == 1:
                if name == "sigma":
                    return float(x_t[0]) ** 2 * np.eye(n_assets)
                else:
                    return float(x_t[0]) * np.eye(n_assets)

            if x_t.size == n_assets:
                if name == "sigma":
                    return np.diag(x_t ** 2)
                else:
                    return np.diag(x_t)

        if x_t.ndim == 2 and x_t.shape == (n_assets, n_assets):
            return x_t

        raise ValueError(
            f"{name} must be a scalar, a vector of length {n_assets}, "
            f"or a square matrix of shape ({n_assets}, {n_assets}); "
            f"got shape {x_t.shape}"
        )

    def _split_y(self, y: np.ndarray):
        """Split the BVP state y into q, S, p_q, p_S.

        solve_bvp represents the solution as a 2D array:
            y.shape = (state_dimension, n_time_nodes)
        The layout is:
            y[0:n]       = q
            y[n:2n]      = S
            y[2n:3n]     = p_q
            y[3n:4n]     = p_S
            
        Each returned block has shape (n_assets, n_time_nodes)
        """
        n = self.n_assets

        q = y[0:n]
        S = y[n:2*n]
        p_q = y[2*n:3*n]
        p_S = y[3*n:4*n]

        return q, S, p_q, p_S

    # ------------------------------------------------------------------ #
    # Internal ODE/BC helpers                                            #
    # ------------------------------------------------------------------ #

    def _u_star(
        self,
        q: np.ndarray,
        S: np.ndarray,
        p_q: np.ndarray,
        p_S: np.ndarray,
    ) -> np.ndarray:
        """Compute the exact optimal control u* for gamma = 2.

        The reduced-state Hamiltonian is:

            H = 1/2 q^T Sigma q
                - S^T u
                + sum_i eta_i u_i^2
                + p_q^T (-u)
                + p_S^T (-diag(kappa) u)

        The first-order condition is:

            ∂H/∂u_i = 0

        which gives:

            2 eta_i u_i - S_i - p_q_i - kappa_i p_S_i = 0

        Therefore:

            u_i* = (S_i + p_q_i + kappa_i p_S_i) / (2 eta_i)

        In vectorized form, since kappa and eta are diagonal:

            u* = (S + p_q + kappa * p_S) / (2 eta)

        where all operations are componentwise.

        Shapes:
            S, p_q, p_S: (n_assets, n_time_nodes)
            kappa[:, None]: (n_assets, 1)
            eta[:, None]: (n_assets, 1)
        """
        cross_impact_costate = self.kappa.T @ p_S
        return (S + p_q + cross_impact_costate) / (2.0 * self.eta[:, None])

    def _odes(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ODE system for the PMP boundary value problem.

        The internal BVP system is written in terms of:

            q(t), S(t), p_q(t), p_S(t)

        The dynamics are:

            q_dot   = -u
            S_dot   = -kappa * u
            p_q_dot = -sigma^2 * q
            p_S_dot = u
        Returns
        -------
        np.ndarray (4*n_assets, n_time_nodes)
        """
        q, S, p_q, p_S = self._split_y(y)

        # Compute the optimal control at all BVP nodes.
        u = self._u_star(q, S, p_q, p_S)

        # State dynamics.
        dq = -u
        dS = -self.kappa @ u

        # Costate dynamics.
        dp_q = -self.sigma @ q
        dp_S = u

        # Stack the blocks back in the same order as y:
        return np.vstack([dq, dS, dp_q, dp_S])

    def _bc(
        self,
        ya: np.ndarray,
        yb: np.ndarray,
        q0: np.ndarray,
        S0: np.ndarray,
    ) -> np.ndarray:
        """Boundary conditions for the BVP.
        We impose two initial conditions and two terminal costate conditions.

        Initial conditions:
            q(0) = q0
            S(0) = S0

        Terminal conditions:
            p_q(T) = 2 alpha q(T)
            p_S(T) = 0

        The returned vector must have length 4*n_assets, exactly matching the
        BVP state dimension.
        """
        n = self.n_assets

        # Values at t = 0.
        q_a = ya[0:n]
        S_a = ya[n:2*n]

        # Values at t = T.
        q_b = yb[0:n]
        p_q_b = yb[2*n:3*n]
        p_S_b = yb[3*n:4*n]

        return np.concatenate([
            q_a - q0,                         # q(0) - q0 = 0
            S_a - S0,                         # S(0) - S0 = 0
            p_q_b - 2.0 * self.alpha * q_b,   # p_q(T) - 2 alpha q(T) = 0
            p_S_b,                            # p_S(T) = 0
        ])

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def solve(self, z0: ArrayLike, **kwargs: Any) -> Trajectory:
        """Solve the two-point BVP from one initial condition.

        Returns
        -------
        Trajectory
            t: time grid, shape (N,)
            z: state trajectory, shape (N, 2*n_assets + 1)
               with layout [q, S, X]
            u: control trajectory, shape (N-1, n_assets)
        """
        z0_np = _to_numpy(z0).reshape(-1)
        n = self.n_assets

        # Case 1: reduced state [q, S].
        if z0_np.shape == (2 * n,):
            q0 = z0_np[0:n]
            S0 = z0_np[n:2*n]
            # If the problem object has X0, use it; otherwise default to 0.
            X0 = float(getattr(self.prob, "X0", 0.0))

        # # Case 2: full observer state [q, S, X].
        # elif z0_np.shape == (2 * n + 1,):
        #     q0 = z0_np[0:n]
        #     S0 = z0_np[n:2*n]
        #     X0 = float(z0_np[2*n])

        # else:
        #     raise ValueError(
        #         f"z0 must have shape ({2*n},) or ({2*n+1},), got {z0_np.shape}"
        #     )

        # Time grid for the collocation BVP solver.
        t_nodes = np.linspace(self.t0, self.T, self.n_bvp_nodes)

        # Initial guess for the BVP solution.
        y_init = np.zeros((4 * n, len(t_nodes)))

        # Guess for q(t): linear liquidation from q0 to 0.
        for i in range(n):
            y_init[i] = np.linspace(q0[i], 0.0, len(t_nodes))

        # Guess for S(t): constant price at S0.
        for i in range(n):
            y_init[n + i] = S0[i]

        # Guess for p_q(t).
        y_init[2*n:3*n] = self.sigma @ y_init[0:n]

        # Guess for p_S(t): zero everywhere.
        y_init[3*n:4*n] = 0.0

        # Boundary condition closure.
        bc = lambda ya, yb: self._bc(ya, yb, q0, S0)

        # Solve the two-point boundary value problem.
        sol = solve_bvp(
            self._odes,
            bc,
            t_nodes,
            y_init,
            tol=self.bvp_tol,
            max_nodes=10000,
        )

        if not sol.success:
            warnings.warn(f"solve_bvp did not fully converge: {sol.message}")

        # Extract solution blocks.
        t_arr = sol.x
        q_sol, S_sol, p_q_sol, p_S_sol = self._split_y(sol.y)

        # Compute the optimal control along the BVP solution.
        u_sol_nodes = self._u_star(q_sol, S_sol, p_q_sol, p_S_sol)

        # Reconstruct cash X(t).
        # We use a trapezoidal rule on the BVP time grid.
        dt = np.diff(t_arr)
        X_sol = np.zeros_like(t_arr)
        X_sol[0] = X0

        for i, dt_i in enumerate(dt):
            # Midpoint approximations for S and u.
            u_mid = 0.5 * (u_sol_nodes[:, i] + u_sol_nodes[:, i + 1])
            S_mid = 0.5 * (S_sol[:, i] + S_sol[:, i + 1])

            # Cash flow:
            dX = (
                np.dot(S_mid, u_mid)
                - np.sum(self.eta * (u_mid ** 2 + self.epsilon))
            )

            X_sol[i + 1] = X_sol[i] + dt_i * dX

        # Pack the state trajectory.
        z_traj = np.concatenate(
            [q_sol.T, S_sol.T, X_sol[:, None]],
            axis=1,
        )

        # Control is defined on intervals, so we use left endpoints.
        u_traj = u_sol_nodes[:, :-1].T

        # Compute realised cost for reporting.
        running_integrand = 0.5 * np.einsum(
            "in,ij,jn->n",
            q_sol,
            self.sigma,
            q_sol,
        )

        running_cost = _cumtrapz_with_leading_zero(
            running_integrand,
            t_arr,
        )[-1]

        # Terminal cost:
        #   alpha ||q(T)||^2 - X(T)
        terminal_cost = self.alpha * np.dot(q_sol[:, -1], q_sol[:, -1]) - X_sol[-1]

        cost = float(running_cost + terminal_cost)

        # Metadata for debugging / plotting.
        meta = {
            "solver": "AlmgrenChrissBVPSolver",
            "n_assets": n,
            "q0": q0,
            "S0": S0,
            "X0_requested": X0,
            "gamma": self.gamma,
            "n_bvp_nodes": self.n_bvp_nodes,
            "bvp_tol": self.bvp_tol,
            "converged": bool(sol.success),
            "matrix_kappa": True,
            "matrix_sigma": True,
        }

        return Trajectory(
            t=t_arr,
            z=z_traj,
            u=u_traj,
            cost=cost,
            label="Exact BVP",
            style=dict(_DEFAULT_EXACT_STYLE),
            meta=meta,
        )




# =============================================================================
# Almgren-Chriss single-asset CLOSED-FORM solver (γ=2)
# =============================================================================

_DEFAULT_CF_STYLE = {"color": "#2166ac", "ls": "--", "lw": 2.0}


def _cumtrapz_with_leading_zero(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid that returns same length as ``x``, starting at 0.

    Used to integrate ``dX/dt = S u - eta (u^2 + epsilon)`` along the
    closed-form ``(q, S, u)`` trajectory so the returned ``X`` matches the
    shape contract of :class:`Trajectory` and stays consistent with the
    rest of the analytical solution.
    """
    dx = np.diff(x)
    inc = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty_like(y)
    out[0] = 0.0
    out[1:] = np.cumsum(inc)
    return out

class AlmgrenChrissClosedForm(ReferenceSolver):
    r"""Closed-form analytical reference for single-asset Almgren-Chriss
    liquidation with terminal inventory penalty (``γ = 2``).

    Equivalent to :class:`AlmgrenChrissBVPSolver` for the same problem, but
    evaluated analytically rather than via :func:`scipy.integrate.solve_bvp`,
    so the result is faster and free of BVP-collocation residuals.  Returns
    a :class:`Trajectory` with the same shape contract, hence is a drop-in
    replacement wherever the BVP solver is consumed.

    Parameters
    ----------
    prob
        Problem object exposing ``sigma, kappa, eta, gamma, epsilon,
        alpha, t_initial, t_final`` (and optionally ``nt`` for the default
        time grid). Mirrors :class:`AlmgrenChrissBVPSolver`.
    n_grid
        Number of nodes in ``t = linspace(t0, T, n_grid)``.  Defaults to
        ``prob.nt + 1`` so the closed-form grid coincides with the JFB
        rollout grid; for problems that don't expose ``nt`` falls back to
        ``500``.
    sigma_zero_atol
        Numerical threshold below which the analytical formula degenerates
        (``D = (finite) / (lambda * finite)`` with ``lambda → 0``); we
        switch to the algebraic ``λ → 0`` limit instead, which is the
        constant-rate TWAP-with-penalty solution.

    Math
    ----
    With Hamiltonian ``H = ½σ²q² + p_q(-u) + p_S(-κu) + p_X(Su - η(u²+ε))``,
    PMP gives ``p_X = -1`` (constant), ``p_S(T) = 0``, ``p_q(T) = 2 α q(T)``
    and after eliminating costates ``q'' = (σ² / (2η)) q``.  Setting
    ``λ = √(σ²/(2η))`` then yields

        q(t) = Q0 cosh(λt) + D sinh(λt)
        u(t) = -λ [Q0 sinh(λt) + D cosh(λt)]
        S(t) = S0 + κ (q(t) - Q0)

    where the closed-form constant ``D`` enforces the right-endpoint
    stationarity ``2η u(T) = S(T) + 2 α q(T)``:

        D = -[ 2 η λ Q0 sinh(λT) + S0 + κ Q0 (cosh(λT)-1)
              + 2 α Q0 cosh(λT) ]
            / [ 2 η λ cosh(λT) + (κ + 2 α) sinh(λT) ].

    ``X(t)`` is reconstructed by trapezoidal integration of
    ``dX/dt = S u - η (u² + ε)`` to match :meth:`compute_f` of
    :class:`LiquidationPortfolioOC` exactly (epsilon-aware).

    Sigma → 0 fallback
    ------------------
    When ``λ → 0`` the ODE degenerates to ``q'' = 0`` and the optimal
    control is constant: ``u* = (S0 + 2 α Q0) / (2 η + (κ + 2 α) T)``,
    giving ``q(t) = Q0 - u* t``, ``S(t) = S0 - κ u* t``.  Used whenever
    ``sigma <= sigma_zero_atol``.
    """

    def __init__(
        self,
        prob: Any,
        n_grid: Optional[int] = None,
        sigma_zero_atol: float = 1e-12,
    ):
        self.prob = prob

        # Mirror the scalar coercion in AlmgrenChrissBVPSolver so callers
        # can pass either the legacy scalar-attribute prob or the new
        # n_assets-vector LiquidationPortfolioOC.
        def _scalar(name: str) -> float:
            v = getattr(prob, name)
            if hasattr(v, "numel") or hasattr(v, "__len__"):
                arr = np.asarray(v if not hasattr(v, "detach") else v.detach().cpu())
                if arr.size != 1:
                    raise ValueError(
                        f"AlmgrenChrissClosedForm is single-asset only; "
                        f"prob.{name} has size {arr.size}."
                    )
                return float(arr.reshape(-1)[0])
            return float(v)

        self.sigma = _scalar("sigma")
        self.kappa = _scalar("kappa")
        self.eta = _scalar("eta")
        self.gamma = _scalar("gamma")
        self.epsilon = _scalar("epsilon")
        self.alpha = _scalar("alpha")
        self.t0 = float(prob.t_initial)
        self.T = float(prob.t_final)
        self.sigma_zero_atol = float(sigma_zero_atol)

        if abs(self.gamma - 2.0) >= 1e-6:
            raise ValueError(
                f"AlmgrenChrissClosedForm requires γ=2 (±1e-6); "
                f"got γ={self.gamma:.6f}."
            )
        if self.eta <= 0.0:
            raise ValueError(f"eta must be positive (got eta={self.eta}).")
        if self.T <= self.t0:
            raise ValueError(
                f"t_final must be > t_initial (got t0={self.t0}, T={self.T})."
            )
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be nonnegative (got sigma={self.sigma}).")

        if n_grid is None:
            n_grid = int(getattr(prob, "nt", 499)) + 1
        if n_grid < 2:
            raise ValueError(f"n_grid must be >= 2 (got {n_grid}).")
        self.n_grid = int(n_grid)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def solve(self, z0: ArrayLike, **kwargs: Any) -> Trajectory:
        """Closed-form trajectory starting at ``z0 = [q0, S0, X0]``.

        Same 3-component ``z0`` contract as :class:`AlmgrenChrissBVPSolver`.
        Callers using the post-reduction :class:`LiquidationPortfolioOC`
        (state_dim = 2 * n_assets) must append ``X0 = 0.0`` to their
        ``[q0, S0]`` before calling here — see the note in
        :meth:`AlmgrenChrissBVPSolver.solve`.

        Returns
        -------
        Trajectory
            Deterministic trajectory with state ``[q, S, X]`` on
            ``linspace(t0, T, n_grid)``, control ``u*`` on the left
            endpoints, label ``"Exact CF"`` and a dashed-blue style
            matching the BVP reference.  ``meta`` carries ``lam``, ``D``,
            ``gamma``, ``epsilon`` and ``solver``.
        """
        z0_np = _to_numpy(z0).reshape(-1)
        if z0_np.shape != (3,):
            raise ValueError(f"z0 must have shape (3,), got {z0_np.shape}")
        Q0, S0, X0 = float(z0_np[0]), float(z0_np[1]), float(z0_np[2])

        T = self.T - self.t0
        # Solve on a t-shifted grid so cosh/sinh start at 0 and we don't
        # need to subtract t0 inside every evaluation.  We then add t0
        # back to the returned time array.
        tau = np.linspace(0.0, T, self.n_grid)

        if self.sigma <= self.sigma_zero_atol:
            # λ → 0 limit: constant u, q affine, S affine.
            denom = 2.0 * self.eta + (self.kappa + 2.0 * self.alpha) * T
            u_const = (S0 + 2.0 * self.alpha * Q0) / denom
            u = np.full_like(tau, u_const)
            q = Q0 - u_const * tau
            S = S0 - self.kappa * u_const * tau
            D = float("nan")
            lam = 0.0
        else:
            lam = float(np.sqrt(self.sigma * self.sigma / (2.0 * self.eta)))
            cT = float(np.cosh(lam * T))
            sT = float(np.sinh(lam * T))

            num = (
                2.0 * self.eta * lam * Q0 * sT
                + S0
                + self.kappa * Q0 * (cT - 1.0)
                + 2.0 * self.alpha * Q0 * cT
            )
            den = 2.0 * self.eta * lam * cT + (self.kappa + 2.0 * self.alpha) * sT
            D = -num / den

            ct = np.cosh(lam * tau)
            st = np.sinh(lam * tau)

            q = Q0 * ct + D * st
            u = -lam * (Q0 * st + D * ct)
            # ∫_0^t u ds = Q0 - q(t), so S(t) = S0 - κ ∫ u = S0 + κ (q - Q0).
            S = S0 + self.kappa * (q - Q0)

        # X(t) by trapezoidal integration of the model's actual dX/dt so
        # the closed-form X stays numerically consistent with the
        # epsilon-smoothed compute_f used in JFB rollouts.
        rhs = S * u - self.eta * (u * u + self.epsilon)
        X = X0 + _cumtrapz_with_leading_zero(rhs, tau)

        # Assemble Trajectory.  Time grid is shifted back to absolute time.
        t_arr = self.t0 + tau
        z_traj = np.stack([q, S, X], axis=1)        # (N, 3)
        u_traj = u[:-1].reshape(-1, 1)              # (N-1, 1) on left endpoints

        # Realised cost J = ∫ ½σ²q² dt + α q(T)² - X(T) (matches
        # LiquidationPortfolioOC.compute_lagrangian + compute_G).
        running = _cumtrapz_with_leading_zero(0.5 * self.sigma**2 * q**2, tau)[-1]
        terminal = -float(X[-1]) + self.alpha * float(q[-1]) ** 2
        cost = float(running) + float(terminal)

        meta = {
            "solver": "AlmgrenChrissClosedForm",
            "Q0": Q0, "S0": S0, "X0": X0,
            "gamma": self.gamma, "epsilon": self.epsilon,
            "alpha": self.alpha, "sigma": self.sigma,
            "kappa": self.kappa, "eta": self.eta,
            "lam": lam, "D": D, "n_grid": self.n_grid,
            "t0": self.t0, "T": self.T,
        }
        return Trajectory(
            t=t_arr,
            z=z_traj,
            u=u_traj,
            cost=cost,
            label="Exact CF",
            style=dict(_DEFAULT_CF_STYLE),
            meta=meta,
        )


# =============================================================================
# JFB policy rollout
# =============================================================================

_DEFAULT_JFB_STYLE = {"color": "#d6604d", "lw": 2.0}


class JFBPolicyRollout(ReferenceSolver):
    """Explicit-Euler rollout of a trained JFB policy.

    Parameters
    ----------
    prob : ImplicitOC
        Provides dynamics via :meth:`ImplicitOC.compute_f`, plus the time
        grid (``t_initial``, ``t_final``, ``nt``), dimensions and device.
    policy : callable
        Any ``(z, t) -> u`` callable.  Typical choice is a trained
        :class:`ImplicitNets.ImplicitNetOC` instance.

    Notes
    -----
    Unlike the legacy ``LiquidationBenchmark._rollout_jfb`` which rolled
    out a full batch, this solver operates on a single initial condition
    at a time and returns a single-path :class:`Trajectory`.  Batch-style
    usage in the shim :mod:`liquidation_benchmark` is implemented by
    looping over this solver.
    """

    def __init__(self, prob: Any, policy: Any):
        self.prob = prob
        self.policy = policy

    def solve(self, z0: ArrayLike, **kwargs: Any) -> Trajectory:
        """Roll out the policy from a single ``z0`` and return a trajectory.

        Cash observer (post-reduction)
        ------------------------------
        For OC problems that expose :meth:`compute_cash_flow` (currently
        :class:`LiquidationPortfolioOC` after the ``X``-out-of-state
        refactor), this method integrates ``X(t)`` in parallel using
        the same explicit-Euler grid and starts from ``prob.X0``.
        ``X`` is **never** fed back into the policy or into
        ``compute_f`` — it is a pure observer. The returned
        :class:`Trajectory` packs the cash column at the final
        position so its ``z`` has shape ``(nt+1, state_dim + 1)``,
        matching the legacy ``[q, S, X]`` layout that
        :func:`benchmarking.plotter.liquidation_panels` reads from.
        For problems without ``compute_cash_flow`` the rollout
        is unchanged: ``z`` keeps shape ``(nt+1, state_dim)``.
        """
        prob = self.prob
        device = getattr(prob, "device", "cpu")
        nt = int(prob.nt)
        t0 = float(prob.t_initial)
        T = float(prob.t_final)
        dt = (T - t0) / nt

        z0_np = _to_numpy(z0).reshape(-1)
        if z0_np.shape[0] != prob.state_dim:
            raise ValueError(
                f"z0 has {z0_np.shape[0]} components but prob.state_dim={prob.state_dim}"
            )

        # Number of Monte-Carlo paths. >1 only makes sense for a stochastic
        # problem; for deterministic dynamics every path is identical so we
        # keep the legacy single-path behaviour.
        n_paths = int(kwargs.get("n_paths", 1))
        has_diff = bool(getattr(prob, "has_diffusion", lambda: False)())
        if not has_diff:
            n_paths = 1
        n_paths = max(n_paths, 1)

        # Optional seedable generator for reproducible Brownian increments.
        seed = kwargs.get("noise_seed", None)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        else:
            gen = getattr(prob, "noise_generator", None)

        # Batch the paths along dim 0: z has shape (P, state_dim).
        P = n_paths
        z0_t = torch.as_tensor(z0_np, dtype=torch.float32, device=device).unsqueeze(0)
        z0_t = z0_t.expand(P, -1).contiguous()

        z_traj = torch.zeros(P, prob.state_dim, nt + 1, device=device)
        u_traj = torch.zeros(P, prob.control_dim, nt, device=device)
        z_traj[:, :, 0] = z0_t

        # Detect a cash observer (post-reduction LiquidationPortfolioOC) and
        # set it up alongside the OC state. None for problems that don't
        # expose `compute_cash_flow` (e.g. multi-bicycle).
        cash_fn = getattr(prob, "compute_cash_flow", None)
        if callable(cash_fn):
            x_traj = torch.zeros(P, 1, nt + 1, device=device)
            x_traj[:, :, 0] = float(getattr(prob, "X0", 0.0))
        else:
            x_traj = None

        diff_incr = getattr(prob, "diffusion_increment", None)

        z = z0_t.clone()
        ti = t0
        with torch.no_grad():
            for i in range(nt):
                u = self.policy(z, float(ti)).view(P, prob.control_dim)
                u_traj[:, :, i] = u
                if x_traj is not None:
                    # Read S from the *current* (pre-update) state, matching
                    # the left-endpoint Euler convention used for compute_f.
                    x_traj[:, :, i + 1] = (
                        x_traj[:, :, i] + dt * cash_fn(float(ti), z, u)
                    )
                z = z + dt * prob.compute_f(float(ti), z, u)
                # Euler-Maruyama diffusion term (zero for deterministic probs).
                if has_diff and callable(diff_incr):
                    z = z + diff_incr(float(ti), z, u, dt, generator=gen)
                z_traj[:, :, i + 1] = z
                ti += dt

        t_jfb = np.linspace(t0, T, nt + 1)

        if P == 1:
            # Legacy deterministic 2D Trajectory contract.
            z_np = z_traj[0].cpu().numpy().T       # (nt+1, state_dim)
            u_np = u_traj[0].cpu().numpy().T       # (nt,   control_dim)
            if x_traj is not None:
                x_np = x_traj[0, 0].cpu().numpy()  # (nt+1,)
                z_np = np.concatenate([z_np, x_np[:, None]], axis=1)
        else:
            # Stochastic 3D Trajectory: (n_paths, nt+1, state_dim[+1]).
            z_np = z_traj.permute(0, 2, 1).cpu().numpy()   # (P, nt+1, state_dim)
            u_np = u_traj.permute(0, 2, 1).cpu().numpy()   # (P, nt, control_dim)
            if x_traj is not None:
                x_np = x_traj.permute(0, 2, 1).cpu().numpy()   # (P, nt+1, 1)
                z_np = np.concatenate([z_np, x_np], axis=2)

        meta = {
            "solver": "JFBPolicyRollout",
            "z0": z0_np.copy(), "nt": nt,
            "n_paths": P,
        }
        if x_traj is not None:
            meta["has_cash_observer"] = True
            meta["X0"] = float(getattr(prob, "X0", 0.0))

        return Trajectory(
            t=t_jfb,
            z=z_np,
            u=u_np,
            label="JFB",
            style=dict(_DEFAULT_JFB_STYLE),
            meta=meta,
        )


# =============================================================================
# Monte-Carlo reference band (stochastic baseline)
# =============================================================================

_DEFAULT_MC_STYLE = {"color": "#4393c3", "lw": 2.0, "alpha": 0.9}


def monte_carlo_policy_band(
    prob: Any,
    policy: Any,
    z0: ArrayLike,
    n_paths: int = 256,
    seed: "int | None" = 0,
    label: str = "MC band",
) -> Trajectory:
    """Roll a feedback ``policy`` over ``n_paths`` Euler-Maruyama paths.

    Returns a *stochastic* (3D) :class:`Trajectory` (shape
    ``(n_paths, nt+1, state_dim[+1])``) suitable for the ``plot_type="band"``
    extractors in :mod:`benchmarking.plotter` (mean +/- 1 std across paths).

    This is the stochastic analogue of a deterministic reference: instead of
    a single "truth" path it provides the Monte-Carlo distribution of the
    closed-loop dynamics under ``policy``. Use the trained policy to inspect
    its own dispersion, or pass an oracle / closed-form control wrapped as a
    ``(z, t) -> u`` callable for a certainty-equivalent baseline.

    For a deterministic problem (``prob.has_diffusion() is False``) every path
    is identical, so this collapses to a single-path rollout.
    """
    roller = JFBPolicyRollout(prob, policy)
    traj = roller.solve(z0, n_paths=int(n_paths), noise_seed=seed)
    # Re-label / restyle for the MC baseline without mutating the frozen
    # dataclass in place.
    return Trajectory(
        t=traj.t,
        z=traj.z,
        u=traj.u,
        label=label,
        style=dict(_DEFAULT_MC_STYLE),
        meta={**traj.meta, "solver": "monte_carlo_policy_band"},
    )
