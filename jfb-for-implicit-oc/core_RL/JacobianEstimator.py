"""
core_RL.JacobianEstimator
-------------------------
Per-time-step local linearisation of unknown dynamics f.

Provides `a_k = ∂f/∂z` (B, n, n) and `b_k = (∂f/∂u)ᵀ` (B, m, n) at each
trajectory step, used by the implicit policy and the backward adjoint pass.
Estimators store continuous-time Jacobians; the RLS update works in
discrete-time internally and strips the I/Δt before storing.

Two concrete implementations:
- `RLSJacobianEstimator`: block RLS with forgetting factor, one shared
  estimate per time step across the full batch.
- `OracleJacobianEstimator`: queries true analytical Jacobians; only for
  sanity-checking the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch


class JacobianEstimator(ABC):
    """Abstract per-time-step Jacobian estimator.

    A single instance owns the estimates for all ``N`` time steps of one
    rollout horizon. ``update`` is called sequentially during the forward
    rollout; ``AB`` is called both during the rollout (to fetch ``b_k`` for
    the implicit policy) and during the backward adjoint pass.

    Parameters
    ----------
    nt
        Number of discretization steps in the trajectory. We hold one
        estimate per step.
    state_dim, control_dim
        Match the problem.
    dt
        Discretization time step. Needed to convert between discrete-map
        and continuous-time Jacobians.
    device
        Torch device.
    """

    def __init__(
        self,
        nt: int,
        state_dim: int,
        control_dim: int,
        dt: float,
        device: str = "cpu",
    ):
        self.nt = nt
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.device = device

    @abstractmethod
    def update(
        self,
        k: int,
        z_k: torch.Tensor,
        u_k: torch.Tensor,
        z_kp1: torch.Tensor,
    ) -> None:
        """Incorporate the transition ``(z_k, u_k) -> z_{k+1}`` into the
        estimate at step ``k``. ``z_k, u_k, z_kp1`` are batched
        ``(B, state_dim/control_dim/state_dim)``.
        """
        ...

    @abstractmethod
    def AB(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a_k, b_k)`` for the trajectory at step ``k``.

        Shapes ``(B, n, n)`` and ``(B, m, n)`` respectively. The batch
        dimension may be 1 if the estimator shares one estimate across the
        batch (this is how :class:`RLSJacobianEstimator` works).
        """
        ...

    def reset(self) -> None:
        """Optional: reset all estimator state. Default no-op."""

    # Diagnostics — used by the trainer to log estimation health
    def linear_model_residual(
        self, k: int, z_k: torch.Tensor, u_k: torch.Tensor, z_kp1: torch.Tensor
    ) -> torch.Tensor:
        """Mean ``||z_{k+1} - F̂_k(z_k, u_k)||`` under the *current* estimate
        at step ``k``. Cheap, runs after ``update``. Useful trainer
        diagnostic: this should decrease and stabilise across epochs.
        """
        a_k, b_k = self.AB(k)
        if a_k.dim() == 3 and a_k.shape[0] == 1:
            a_k = a_k.expand(z_k.shape[0], -1, -1)
            b_k = b_k.expand(z_k.shape[0], -1, -1)
        # Continuous-time linearisation: f_k(z, u) ≈ f_k(z̄, ū) + a_k (z-z̄) + b_kᵀ (u-ū).
        # We don't track f_k(z̄, ū) explicitly here, so just check first-order
        # consistency of the increment relative to the batch mean.
        z_bar = z_k.mean(dim=0, keepdim=True)
        u_bar = u_k.mean(dim=0, keepdim=True)
        dz = z_k - z_bar  # (B, n)
        du = u_k - u_bar  # (B, m)
        # a_k @ dz: (B, n, n) @ (B, n, 1) -> (B, n, 1) -> (B, n)
        a_term = torch.bmm(a_k, dz.unsqueeze(-1)).squeeze(-1)
        # b_kᵀ @ du: in our convention b_k has shape (B, m, n), so b_kᵀ @ du
        # means (b_k.transpose(1,2)) @ du.unsqueeze(-1).
        b_term = torch.bmm(b_k.transpose(1, 2), du.unsqueeze(-1)).squeeze(-1)
        df_pred = a_term + b_term  # (B, n)
        df_actual = (z_kp1 - z_k) / self.dt - (z_kp1 - z_k).mean(dim=0, keepdim=True) / self.dt
        return (df_pred - df_actual).norm(dim=1).mean()



# Concrete: recursive least squares
class RLSJacobianEstimator(JacobianEstimator):
    r"""Block-RLS estimator with a forgetting factor.

    Maintains, for each time step ``t = 0, ..., nt-1``, an affine model

        z_{t+1} ≈ F_t · [z_t; u_t; 1]

    where ``F_t ∈ R^{n × (n+m+1)}`` is fitted by recursive least squares with
    forgetting factor ``alpha_rls ∈ (0, 1]``. The structure follows
    Eberhard, Vernade & Muehlebach (2024) eqns. (15)–(16), generalised to
    block updates so that the whole batch contributes one step at a time.

    The estimate is **shared across the batch** at each time step (the
    function ``f`` is the same for every initial condition; batching just
    gives us more data per update). ``AB(k)`` therefore returns a
    ``(1, n, n)`` / ``(1, m, n)`` tuple that consumers can broadcast.

    Hyperparameters
    ---------------
    alpha_rls : forgetting factor. ``→ 1`` ⇒ infinite memory (best when
        dynamics are stationary); ``≈ 0.9`` ⇒ tracks slow non-stationarities;
        ``< 0.7`` ⇒ aggressive tracking, noisy estimates.
    q0 : initial precision regularisation. The prior precision matrix is
        ``q0 · I``; large ``q0`` makes the estimator skeptical of early
        data, which is helpful during the warm-up phase when the policy is
        random.
    """

    def __init__(
        self,
        nt: int,
        state_dim: int,
        control_dim: int,
        dt: float,
        alpha_rls: float = 0.9,
        q0: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(nt, state_dim, control_dim, dt, device)
        self.alpha_rls = alpha_rls
        self.q0 = q0

        d = state_dim + control_dim + 1  # regression-vector length
        # F_t : (n, d), one matrix per time step.
        # Initialise A_disc = I (identity block) so that before any data the
        # continuous-time estimate is a_k = (I − I)/dt = 0 rather than
        # (0 − I)/dt = −I/dt, which would collapse the adjoint to zero
        # immediately and kill gradient signal during the RLS warm-up.
        F0 = torch.zeros(state_dim, d, device=device)
        F0[:state_dim, :state_dim] = torch.eye(state_dim, device=device)
        self.F = [F0.clone() for _ in range(nt)]
        # Q_t : (d, d), inverse-covariance accumulator
        self.Q = [q0 * torch.eye(d, device=device) for _ in range(nt)]
        self._d = d

    def reset(self) -> None:
        d = self._d
        n = self.state_dim
        for t in range(self.nt):
            self.F[t].zero_()
            self.F[t][:n, :n] = torch.eye(n, device=self.device)
            self.Q[t] = self.q0 * torch.eye(d, device=self.device)

    def update(
        self,
        k: int,
        z_k: torch.Tensor,
        u_k: torch.Tensor,
        z_kp1: torch.Tensor,
    ) -> None:
        # Build regression matrix X ∈ R^{B × d} and target Y ∈ R^{B × n}.
        B = z_k.shape[0]
        ones = torch.ones(B, 1, device=z_k.device, dtype=z_k.dtype)
        X = torch.cat([z_k, u_k, ones], dim=1)              # (B, d)
        Y = z_kp1                                            # (B, n)

        # Block RLS update.
        # Q_new = α Q_old + (1-α) q0 I + Xᵀ X
        Q_old = self.Q[k]
        d = X.shape[1]
        Q_new = (
            self.alpha_rls * Q_old
            + (1.0 - self.alpha_rls) * self.q0 * torch.eye(d, device=self.device, dtype=X.dtype)
            + X.t() @ X
        )

        # F_new = F_old + (Y - F_old @ Xᵀ.T)ᵀ ... — the cleanest form in the
        # block setting is the normal equation
        #     F_new = F_old + (Y - X F_oldᵀ)ᵀ (Q_new)^{-1} X
        # which follows from minimising the forgetting-factor weighted
        # residual sum of squares. See Eberhard et al., Algorithm 1.
        F_old = self.F[k]                                    # (n, d)
        innovation = Y - X @ F_old.t()                       # (B, n)
        # innovation.T @ X has shape (n, d), then we right-solve with Q_new:
        update = torch.linalg.solve(Q_new, (innovation.t() @ X).t()).t()  # (n, d)
        self.F[k] = F_old + update
        self.Q[k] = Q_new

    def AB(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # F_k = [A_disc | B_disc | c_disc] where:
        #   A_disc has shape (n, n)        rows=output state, cols=input state
        #   B_disc has shape (n, m)        rows=output state, cols=input ctrl
        #   c_disc has shape (n, 1)        intercept (we discard it)
        F_k = self.F[k]
        n, m = self.state_dim, self.control_dim
        A_disc = F_k[:, :n]                # (n, n)
        B_disc = F_k[:, n:n + m]           # (n, m)

        # Continuous-time conversion.
        # a_k (standard Jacobian, rows=output) = (A_disc - I) / dt
        # b_k (transpose Jacobian, rows=control input) = B_discᵀ / dt
        I = torch.eye(n, device=self.device, dtype=F_k.dtype)
        a_k = (A_disc - I) / self.dt          # (n, n)
        b_k = B_disc.t() / self.dt            # (m, n)
        # Add a leading batch dim of 1 so consumers can broadcast against a
        # full batch.
        return a_k.unsqueeze(0), b_k.unsqueeze(0)


# Concrete: oracle (uses true f's analytical Jacobians) — sanity-check only
class OracleJacobianEstimator(JacobianEstimator):
    """Estimator that *cheats* by querying the true dynamics' analytical
    Jacobians. Used only to validate that the rest of the RL pipeline (env
    rollout + manual JFB surrogate) gives the same answer as the
    known-dynamics JFB pipeline. If the new code produces wildly different
    results when this estimator is plugged in, the bug is in the
    surrogate-construction code, not in the Jacobian estimation.

    Parameters
    ----------
    grad_f_z, grad_f_u
        Callables matching the signatures of
        ``ImplicitOC.compute_grad_f_z(t, z, u)`` and
        ``ImplicitOC.compute_grad_f_u(t, z, u)``: take ``(t, z, u)``,
        return ``(B, n, n)`` and ``(B, m, n)`` respectively.
    schedule_t
        Callable ``schedule_t(k) -> t`` returning the wall time at step
        ``k``. The trainer can pass ``lambda k: t_initial + k * dt`` here.
    """

    def __init__(
        self,
        nt: int,
        state_dim: int,
        control_dim: int,
        dt: float,
        grad_f_z: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
        grad_f_u: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
        schedule_t: Callable[[int], float],
        device: str = "cpu",
    ):
        super().__init__(nt, state_dim, control_dim, dt, device)
        self._grad_f_z = grad_f_z
        self._grad_f_u = grad_f_u
        self._t_of = schedule_t

        # Cache the latest (z_k, u_k) seen by ``update`` so that ``AB``
        # returns Jacobians evaluated at the current trajectory point.
        self._cache_z = [None] * nt
        self._cache_u = [None] * nt

    @torch.no_grad()
    def update(
        self,
        k: int,
        z_k: torch.Tensor,
        u_k: torch.Tensor,
        z_kp1: torch.Tensor,
    ) -> None:
        # We only need (z_k, u_k); the next-state z_kp1 is irrelevant for an
        # oracle that knows f exactly.
        self._cache_z[k] = z_k.detach().clone()
        self._cache_u[k] = u_k.detach().clone()

    @torch.no_grad()
    def AB(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._cache_z[k]
        u = self._cache_u[k]
        if z is None:
            raise RuntimeError(
                f"OracleJacobianEstimator.AB({k}) called before update({k}). "
                "Run a forward rollout first."
            )
        t = self._t_of(k)
        a_k = self._grad_f_z(t, z, u)        # (B, n, n)
        b_k = self._grad_f_u(t, z, u)        # (B, m, n)
        return a_k, b_k