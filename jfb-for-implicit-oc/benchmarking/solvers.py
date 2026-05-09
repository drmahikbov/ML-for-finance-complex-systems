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
# Almgren-Chriss single-asset BVP solver (γ=2)
# =============================================================================

_DEFAULT_EXACT_STYLE = {"color": "#2166ac", "ls": "--", "lw": 2.0}


class AlmgrenChrissBVPSolver(ReferenceSolver):
    """Closed-form reference for single-asset Almgren-Chriss liquidation.

    The problem must use ``γ = 2`` (within ``1e-6``) -- the stationarity
    condition of the Hamiltonian is linear in ``u`` in that case and the
    resulting 4-state two-point BVP can be solved to machine precision by
    :func:`scipy.integrate.solve_bvp`.  See the module docstring of
    :mod:`liquidation_benchmark` for the full derivation.

    Parameters
    ----------
    prob : LiquidationPortfolioOC or object with scalar attributes
        Used to read ``sigma``, ``kappa``, ``eta``, ``gamma``, ``epsilon``,
        ``alpha``, ``t_initial``, ``t_final``.
    n_bvp_nodes : int
        Number of collocation nodes used by :func:`scipy.integrate.solve_bvp`.
    bvp_tol : float
        Residual tolerance forwarded to :func:`scipy.integrate.solve_bvp`.

    Raises
    ------
    ValueError
        When ``abs(prob.gamma - 2.0) >= 1e-6`` at construction time.

    Notes
    -----
    The accumulated cash component ``X(t)`` is reconstructed **after** the
    BVP is solved by trapezoidal integration of ``dX/dt = Su - η(u²+ε)``.
    This is slightly inconsistent with the explicit-Euler rollout used for
    the JFB policy (see :class:`JFBPolicyRollout`), which accumulates
    ``X`` with the left-endpoint rule.  The discrepancy is small for the
    parameter regimes currently in use and is left as-is in this refactor.
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

        self.sigma = float(prob.sigma)
        self.kappa = float(prob.kappa)
        self.eta = float(prob.eta)
        self.gamma = float(prob.gamma)
        self.epsilon = float(prob.epsilon)
        self.alpha = float(prob.alpha)
        self.t0 = float(prob.t_initial)
        self.T = float(prob.t_final)

        if abs(self.gamma - 2.0) >= 1e-6:
            raise ValueError(
                f"AlmgrenChrissBVPSolver requires γ=2 (±1e-6); got γ={self.gamma:.6f}."
            )

    # ------------------------------------------------------------------ #
    # Internal ODE/BC helpers                                            #
    # ------------------------------------------------------------------ #

    def _u_star(self, q: np.ndarray, S: np.ndarray,
                p_q: np.ndarray, p_S: np.ndarray) -> np.ndarray:
        """Optimal control from the linear stationarity condition (γ=2, p_X=-1).

        With Hamiltonian ``H = L + p^T f`` (the convention used everywhere in
        this codebase, in particular in :meth:`ImplicitOC.compute_grad_H_u`),
        ``∂H/∂u = -p_q - κ p_S + p_X S - 2 η p_X u``.  Setting this to zero
        and using ``p_X = -1`` (since ``∂G/∂X = -1`` and ``∂H/∂X = 0``) gives

            u* = (p_q + κ p_S + S) / (2 η).

        The previous implementation returned the negative of this value,
        which forced the BVP into a sign-flipped costate solution that
        disagreed with the JFB Hamiltonian convention.
        """
        return (p_q + self.kappa * p_S + S) / (2.0 * self.eta)

    def _odes(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        q, S, p_q, p_S = y
        u = self._u_star(q, S, p_q, p_S)
        dq = -u
        dS = -self.kappa * u
        dp_q = -(self.sigma ** 2) * q
        dp_S = u  # ṗ_S = -∂H/∂S = -p_X·u = +u  (since p_X = -1)
        return np.array([dq, dS, dp_q, dp_S])

    def _bc(self, ya: np.ndarray, yb: np.ndarray,
            q0: float, S0: float) -> np.ndarray:
        return np.array([
            ya[0] - q0,
            ya[1] - S0,
            yb[2] - 2.0 * self.alpha * yb[0],
            yb[3],
        ])

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def solve(self, z0: ArrayLike, **kwargs: Any) -> Trajectory:
        """Solve the TPBVP starting at ``z0 = [q0, S0, X0]``.

        Parameters
        ----------
        z0 : array-like, shape (3,)
            Initial state ``[q0, S0, X0]``.  ``X0`` is forwarded to
            ``Trajectory.meta``; the reconstructed ``X(t)`` always starts
            from ``0`` (matching the legacy :mod:`liquidation_benchmark`
            convention).

            **Note (post-reduction):** the live :class:`LiquidationPortfolioOC`
            model now has ``state_dim = 2 * n_assets`` and its
            ``sample_initial_condition`` returns a 2-component ``[q0, S0]``.
            Callers that drive this BVP solver from the new ``z0`` shape
            must explicitly append ``X0 = 0.0`` (or whatever observer
            initial cash they want) before calling :meth:`solve`. The
            BVP itself is independent of the model's state layout —
            this solver still works on the 3-state ``[q, S, X]`` BVP
            because that is the analytical reference, not because it
            mirrors the OC state.

        Returns
        -------
        Trajectory
            A deterministic Trajectory with state ``[q, S, X]``, control
            ``u*``, label ``"Exact BVP"`` and the default dashed blue
            style.
        """
        z0_np = _to_numpy(z0).reshape(-1)
        if z0_np.shape != (3,):
            raise ValueError(f"z0 must have shape (3,), got {z0_np.shape}")
        q0, S0, X0 = float(z0_np[0]), float(z0_np[1]), float(z0_np[2])

        t_nodes = np.linspace(self.t0, self.T, self.n_bvp_nodes)
        y_init = np.zeros((4, len(t_nodes)))
        y_init[0] = np.linspace(q0, 0.0, len(t_nodes))
        y_init[1] = S0
        # Initial guess for p_q(t).  Adjoint ODE ``dp_q/dt = -σ² q`` with
        # ``p_q(T) = 2 α q(T)`` and a linear-liquidation guess for q(t)
        # integrates backwards as p_q(t) ≥ p_q(T) ≥ 0, so the sign here is
        # POSITIVE.  (The previous version had a minus sign here, consistent
        # only with the buggy sign-flipped ``_u_star``.)
        y_init[2] = (self.sigma ** 2) * q0 * (self.T - t_nodes)
        y_init[3] = 0.0

        bc = lambda ya, yb: self._bc(ya, yb, q0, S0)
        sol = solve_bvp(
            self._odes, bc, t_nodes, y_init,
            tol=self.bvp_tol, max_nodes=10000,
        )
        if not sol.success:
            warnings.warn(f"solve_bvp did not fully converge: {sol.message}")

        t_arr = sol.x
        q_sol, S_sol, p_q_sol, p_S_sol = sol.y
        u_sol_nodes = self._u_star(q_sol, S_sol, p_q_sol, p_S_sol)

        # Reconstruct X(t) by trapezoidal (mid-point) integration of
        # dX/dt = S u - eta (u^2 + epsilon)^(gamma/2).
        # Note: inconsistent with Euler rollout used for JFB; preserved
        # as-is for backwards-compatibility with the original benchmark.
        dt = np.diff(t_arr)
        X_sol = np.zeros_like(t_arr)
        X_sol[0] = 0.0
        for i, dt_i in enumerate(dt):
            u_mid = 0.5 * (u_sol_nodes[i] + u_sol_nodes[i + 1])
            S_mid = 0.5 * (S_sol[i] + S_sol[i + 1])
            dX = (
                S_mid * u_mid
                - self.eta * (u_mid ** 2 + self.epsilon) ** (self.gamma / 2.0)
            )
            X_sol[i + 1] = X_sol[i] + dt_i * dX

        # Pack the state trajectory: (N, 3)
        z_traj = np.stack([q_sol, S_sol, X_sol], axis=1)
        # Control on left endpoints: drop the right endpoint.
        u_traj = u_sol_nodes[:-1].reshape(-1, 1)

        # Terminal cost G = -X(T) + alpha * q(T)^2 (see problem class).
        G = float(-X_sol[-1] + self.alpha * q_sol[-1] ** 2)

        meta = {
            "solver": "AlmgrenChrissBVPSolver",
            "q0": q0, "S0": S0, "X0_requested": X0,
            "gamma": self.gamma, "n_bvp_nodes": self.n_bvp_nodes,
            "bvp_tol": self.bvp_tol, "converged": bool(sol.success),
        }
        return Trajectory(
            t=t_arr,
            z=z_traj,
            u=u_traj,
            cost=G,
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

        z0_t = torch.as_tensor(z0_np, dtype=torch.float32, device=device).unsqueeze(0)

        z_traj = torch.zeros(1, prob.state_dim, nt + 1, device=device)
        u_traj = torch.zeros(1, prob.control_dim, nt, device=device)
        z_traj[:, :, 0] = z0_t

        # Detect a cash observer (post-reduction LiquidationPortfolioOC) and
        # set it up alongside the OC state. None for problems that don't
        # expose `compute_cash_flow` (e.g. multi-bicycle).
        cash_fn = getattr(prob, "compute_cash_flow", None)
        if callable(cash_fn):
            x_traj = torch.zeros(1, 1, nt + 1, device=device)
            x_traj[:, :, 0] = float(getattr(prob, "X0", 0.0))
        else:
            x_traj = None

        z = z0_t.clone()
        ti = t0
        with torch.no_grad():
            for i in range(nt):
                u = self.policy(z, float(ti)).view(1, prob.control_dim)
                u_traj[:, :, i] = u
                if x_traj is not None:
                    # Read S from the *current* (pre-update) state, matching
                    # the left-endpoint Euler convention used for compute_f.
                    x_traj[:, :, i + 1] = (
                        x_traj[:, :, i] + dt * cash_fn(float(ti), z, u)
                    )
                z = z + dt * prob.compute_f(float(ti), z, u)
                z_traj[:, :, i + 1] = z
                ti += dt

        t_jfb = np.linspace(t0, T, nt + 1)
        z_np = z_traj[0].cpu().numpy().T       # (nt+1, state_dim)
        u_np = u_traj[0].cpu().numpy().T       # (nt,   control_dim)

        if x_traj is not None:
            x_np = x_traj[0, 0].cpu().numpy()  # (nt+1,)
            # Pack the observer cash at column `state_dim` so the plotter
            # extractor at index 2*n_assets keeps working unchanged.
            z_np = np.concatenate([z_np, x_np[:, None]], axis=1)

        meta = {
            "solver": "JFBPolicyRollout",
            "z0": z0_np.copy(), "nt": nt,
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
