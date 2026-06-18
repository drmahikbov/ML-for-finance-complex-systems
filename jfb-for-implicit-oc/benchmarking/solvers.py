"""
benchmarking.solvers
--------------------
Reference trajectory solvers.

Abstract `ReferenceSolver` plus two concrete implementations:
- `AlmgrenChrissBVPSolver`: closed-form γ=2 BVP solution for
  `LiquidationPortfolioOC`.
- `JFBPolicyRollout`: explicit-Euler rollout of a trained JFB policy.

Both produce single-path (deterministic) `Trajectory` objects.
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
        """Optimal control from the linear stationarity condition (γ=2, p_X=-1)."""
        return (-p_q - self.kappa * p_S - S) / (2.0 * self.eta)

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
        y_init[2] = -(self.sigma ** 2) * q0 * (self.T - t_nodes)
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
        """Roll out the policy from a single ``z0`` and return a trajectory."""
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

        z = z0_t.clone()
        ti = t0
        with torch.no_grad():
            for i in range(nt):
                u = self.policy(z, float(ti)).view(1, prob.control_dim)
                u_traj[:, :, i] = u
                z = z + dt * prob.compute_f(float(ti), z, u)
                z_traj[:, :, i + 1] = z
                ti += dt

        t_jfb = np.linspace(t0, T, nt + 1)
        z_np = z_traj[0].cpu().numpy().T       # (nt+1, state_dim)
        u_np = u_traj[0].cpu().numpy().T       # (nt,   control_dim)

        meta = {
            "solver": "JFBPolicyRollout",
            "z0": z0_np.copy(), "nt": nt,
        }
        return Trajectory(
            t=t_jfb,
            z=z_np,
            u=u_np,
            label="JFB",
            style=dict(_DEFAULT_JFB_STYLE),
            meta=meta,
        )
