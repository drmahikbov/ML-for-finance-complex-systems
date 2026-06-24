"""
core_RL.ImplicitOC_RL
---------------------
Abstract base for optimal-control problems with unknown dynamics.

Like `core/ImplicitOC` but without `compute_f`, `compute_grad_f_u`, or
`compute_grad_f_z` — the agent never sees the dynamics. Key additions:
- `compute_grad_H_u_estimated`: Hamiltonian gradient using estimated `b_k`
  in place of the true `∂f/∂u`.
- `compute_loss_RL`: full training step — optional exploration rollout, clean
  rollout, data-driven backward adjoint, and JFB surrogate scalar.

Sign convention: H = L + ⟨p, f⟩; b_k shape (B, m, n) with b_k[:, i, j] = ∂f_j/∂u_i.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from core_RL.Environment import Environment
from core_RL.JacobianEstimator import JacobianEstimator


TimeLike = float | torch.Tensor


class ImplicitOC_RL(ABC):
    """Abstract optimal-control problem with **unknown** dynamics.

    Concrete subclasses must implement only the *designer-side* pieces:
    running cost, terminal cost, and an initial-condition sampler. The
    dynamics ``f`` belong to the :class:`Environment`, not to this class.

    Parameters
    ----------
    state_dim, control_dim
    batch_size            initial-condition batch size (forwarded to
                          ``sample_initial_condition``).
    t_initial, t_final, nt
    alphaL, alphaG        weights on running and terminal cost (kept from
                          the original ``ImplicitOC`` so the loss stays in
                          comparable units).
    device
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        batch_size: int,
        t_initial: float,
        t_final: float,
        nt: int,
        alphaL: float = 1.0,
        alphaG: float = 1.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.batch_size = batch_size
        self.t_initial = t_initial
        self.t_final = t_final
        self.nt = nt
        self.h = (t_final - t_initial) / nt
        self.alphaL = alphaL
        self.alphaG = alphaG
        self.device = device

        self.oc_problem_name = "Generic ImplicitOC_RL"

    # Abstract: designer-side knowns
    @abstractmethod
    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Running cost ``L(t, z, u)``. Shape ``(B,)``."""

    @abstractmethod
    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∇_u L``. Shape ``(B, control_dim)``."""

    def compute_grad_lagrangian_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∇_z L``, used in the backward adjoint pass.

        **Default**: autograd-based. Concrete problems whose ``L`` is
        independent of ``z`` (e.g. our Merton example, where the running
        cost depends only on the portfolio weight ``π``) can leave this
        unchanged — autograd will return zeros automatically. Problems
        with a non-trivial ``z``-dependence (e.g. consumption-savings
        with an inventory-risk term ``½ σ² q²``) should override this with
        the analytical formula for speed and numerical stability.
        """
        z_req = z.detach().requires_grad_(True)
        L = self.compute_lagrangian(t, z_req, u).sum()
        (grad,) = torch.autograd.grad(L, z_req, create_graph=False)
        return grad.detach()

    @abstractmethod
    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """Terminal cost ``G(z(T))``. Shape ``(B,)``."""

    @abstractmethod
    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """``∇G``. Shape ``(B, state_dim)``."""

    @abstractmethod
    def sample_initial_condition(self) -> torch.Tensor:
        """Draw a batch of initial states ``z_0 ∈ R^{B × state_dim}``."""

    # The Hamiltonian gradient — using estimated b_k
    def compute_grad_H_u_estimated(
        self,
        t: TimeLike,
        z: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        b_k: torch.Tensor,
    ) -> torch.Tensor:
        """``∇_u H`` using estimated control Jacobian ``b_k`` in place of
        the true ``∂f/∂u``.

        Parameters
        ----------
        t : scalar or 0-d tensor
        z : ``(B, state_dim)``
        u : ``(B, control_dim)``
        p : ``(B, state_dim)``  — costate (typically ``∇_z φ_θ(t, z)``)
        b_k : ``(B, m, n)`` *or* ``(1, m, n)`` (broadcastable from a
              shared-across-batch estimator like RLS).

        Returns
        -------
        ``(B, control_dim)``.

        This is the RL counterpart of
        ``ImplicitOC.compute_grad_H_u`` — same shape, same sign — but with
        ``b_k`` injected from the outside instead of ``self.compute_grad_f_u(...)``.
        """
        B = z.shape[0]
        # αL · ∇_u L
        grad_L = self.alphaL * self.compute_grad_lagrangian(t, z, u)   # (B, m)
        # b_k @ p — broadcast b_k if it's (1, m, n)
        if b_k.shape[0] == 1 and B > 1:
            b_k = b_k.expand(B, -1, -1)
        grad_pf = torch.bmm(b_k, p.unsqueeze(-1)).view(B, self.control_dim)  # (B, m)
        return grad_L + grad_pf

    # The RL training-loss routine
    def compute_loss_RL(
        self,
        policy,                       # ImplicitNetOC_RL
        env: Environment,
        jac_est: JacobianEstimator,
        z0: torch.Tensor,
        exploration_std: float = 0.0,
    ) -> dict:
        """End-to-end RL JFB loss computation — two-rollout variant.

        When ``exploration_std > 0``, two rollouts are executed per epoch:

        1a. **Exploration rollout** (no grad): noisy controls are sent to
            ``env.step`` and ``jac_est.update`` so the RLS estimator sees
            sufficient control variation to identify ``b_k = ∂f/∂u``.
            This rollout is discarded after updating the estimator.

        1b. **Clean rollout** (no grad): the policy is queried with its
            own (noise-free) controls using the freshly updated ``b_k``
            estimates. The resulting trajectory is internally consistent —
            the same states and controls feed the cost reporting, the
            backward adjoint pass, and the JFB surrogate. The logged
            ``terminal_cost`` therefore reflects the policy's actual
            performance rather than a noise-corrupted trajectory.

        When ``exploration_std == 0`` a single rollout is run (original
        behaviour; ``jac_est.update`` is called inside that rollout).

        2. **Backward adjoint pass** (no grad):
               p_N = ∇G(z_N)
               p_k = p_{k+1} + Δt (a_kᵀ p_{k+1} + ∇_z L_k)

        3. **JFB surrogate construction**: a scalar ``S(θ)`` such that
           ``S(θ).backward()`` populates ``param.grad`` with the
           JFB-with-estimates gradient:

               S(θ) = Σ_k ⟨ T̂_k(ū_k; z̄_k),  Δt · (∇_u L + b_k @ p̂_{k+1}) ⟩

        4. **Loss reporting**: cost on the clean trajectory.

        Returns
        -------
        dict with keys
          ``surrogate``       — scalar tensor with autograd graph.
          ``total_cost``      — float, J on the **clean** trajectory.
          ``running_cost``    — float, Σ L Δt on the clean trajectory.
          ``terminal_cost``   — float, G(z_N) on the clean trajectory.
          ``z_traj``          — ``(B, n, nt+1)``, detached, clean trajectory.
          ``u_traj``          — ``(B, m, nt)``, detached, clean controls.
          ``p_traj``          — ``(B, n, nt+1)``, detached costate sequence.
          ``lin_residual``    — float, mean linear-model residual from the
                                exploration rollout (or clean rollout when
                                ``exploration_std == 0``).
        """
        device = z0.device
        B = z0.shape[0]
        n, m = self.state_dim, self.control_dim
        N = self.nt
        dt = self.h

        lin_residual_acc = 0.0

        # -------- 1a. Exploration rollout (only when noise is active) --- #
        # Noisy controls excite the env so the RLS estimator can identify
        # b_k = ∂f/∂u. The trajectory produced here is discarded.
        if exploration_std > 0.0:
            z_exp = z0.detach()
            t = self.t_initial
            with torch.no_grad():
                for k in range(N):
                    _, b_k = jac_est.AB(k)
                    policy.set_step_jacobian(b_k)
                    u_k_policy = policy(z_exp, t).view(B, m)
                    u_k_explore = u_k_policy + exploration_std * torch.randn_like(u_k_policy)
                    if getattr(policy, "use_control_limits", False):
                        u_k_explore = u_k_explore.clamp(policy.u_min, policy.u_max)
                    z_next = env.step(z_exp, u_k_explore, t)
                    jac_est.update(k, z_exp, u_k_explore, z_next)
                    lin_residual_acc += jac_est.linear_model_residual(
                        k, z_exp, u_k_explore, z_next
                    ).item()
                    z_exp = z_next
                    t = t + dt

        # -------- 1b. Clean rollout (no grad) -------------------------- #
        # Policy controls only — uses the freshly updated b_k estimates.
        # All downstream quantities (adjoint, surrogate, costs) are built
        # from this trajectory, so they reflect the policy's true behaviour.
        z_traj = torch.zeros(B, n, N + 1, device=device, dtype=z0.dtype)
        u_traj = torch.zeros(B, m, N, device=device, dtype=z0.dtype)
        z_traj[:, :, 0] = z0
        z = z0.detach()
        t = self.t_initial
        running_cost_acc = torch.zeros(B, device=device, dtype=z0.dtype)

        with torch.no_grad():
            for k in range(N):
                _, b_k = jac_est.AB(k)
                policy.set_step_jacobian(b_k)
                u_k = policy(z, t).view(B, m)
                z_next = env.step(z, u_k, t)

                # When not exploring, update the estimator from the clean
                # rollout (preserves the original single-rollout behaviour).
                if exploration_std == 0.0:
                    jac_est.update(k, z, u_k, z_next)
                    lin_residual_acc += jac_est.linear_model_residual(
                        k, z, u_k, z_next
                    ).item()

                running_cost_acc = running_cost_acc + dt * self.compute_lagrangian(t, z, u_k)
                u_traj[:, :, k] = u_k
                z_traj[:, :, k + 1] = z_next
                z = z_next
                t = t + dt

            terminal_cost_per_sample = self.compute_G(z)  # (B,)

        # -------- 2. Backward adjoint pass (no grad) ------------------- #
        p_traj = torch.zeros(B, n, N + 1, device=device, dtype=z0.dtype)
        with torch.no_grad():
            # p_N = αG · ∇G — scale by alphaG so the terminal cost carries
            # the correct weight in both the adjoint and the bracket below.
            p_kp1 = self.alphaG * self.compute_grad_G_z(z_traj[:, :, N])  # (B, n)
            p_traj[:, :, N] = p_kp1
            t_back = self.t_initial + (N - 1) * dt
            for k in range(N - 1, -1, -1):
                z_k = z_traj[:, :, k]
                u_k = u_traj[:, :, k]
                a_k, _ = jac_est.AB(k)
                if a_k.shape[0] == 1 and B > 1:
                    a_k = a_k.expand(B, -1, -1)
                # a_kᵀ @ p_{k+1}: (B, n, n)ᵀ @ (B, n, 1) -> (B, n, 1) -> (B, n)
                aT_p = torch.bmm(a_k.transpose(1, 2), p_kp1.unsqueeze(-1)).squeeze(-1)
                # αL · ∇_z L
                grad_z_L = self.alphaL * self.compute_grad_lagrangian_z(t_back, z_k, u_k)
                p_k = p_kp1 + dt * (aT_p + grad_z_L)
                p_traj[:, :, k] = p_k
                p_kp1 = p_k
                t_back = t_back - dt

        # -------- 3. JFB surrogate (autograd through phi only) --------- #
        surrogate = torch.zeros((), device=device, dtype=z0.dtype)
        for k in range(N):
            t_k = self.t_initial + k * dt
            z_k = z_traj[:, :, k].detach()
            u_k = u_traj[:, :, k].detach()
            _, b_k = jac_est.AB(k)
            b_k_det = b_k.detach()
            p_kp1_det = p_traj[:, :, k + 1].detach()

            # αL · ∇_u L — consistent with the αL scaling in compute_grad_H_u_estimated
            # and in the adjoint step above.
            grad_uL = (self.alphaL * self.compute_grad_lagrangian(t_k, z_k, u_k)).detach()

            # θ-dependent piece: ∇_z φ_θ(t, z̄_k). Gradient flows through
            # θ even though z_k is detached.
            p_phi = policy.p_net(t_k, z_k)                # (B, n)

            if b_k_det.shape[0] == 1 and B > 1:
                b_k_b = b_k_det.expand(B, -1, -1)
            else:
                b_k_b = b_k_det
            grad_uH = grad_uL + torch.bmm(b_k_b, p_phi.unsqueeze(-1)).view(B, m)
            T_k = policy.apply_control_limits(u_k - policy.alpha * grad_uH)  # (B, m)

            bracket = dt * (grad_uL + torch.bmm(b_k_b.detach(), p_kp1_det.unsqueeze(-1)).view(B, m))
            bracket = bracket.detach()

            surrogate = surrogate + (T_k * bracket).sum(dim=1).mean()

        running_cost_mean = running_cost_acc.mean().item()
        terminal_cost_mean = terminal_cost_per_sample.mean().item()
        total_cost = self.alphaL * running_cost_mean + self.alphaG * terminal_cost_mean

        return {
            "surrogate": surrogate,
            "total_cost": total_cost,
            "running_cost": running_cost_mean,
            "terminal_cost": terminal_cost_mean,
            "z_traj": z_traj.detach(),
            "u_traj": u_traj.detach(),
            "p_traj": p_traj.detach(),
            "lin_residual": lin_residual_acc / N,
        }

    # Convenience: deterministic rollout for plotting
    def generate_trajectory(
        self,
        policy,
        env: Environment,
        z0: torch.Tensor,
        return_full_trajectory: bool = True,
    ) -> torch.Tensor:
        """Stand-in for the original ``generate_trajectory`` that uses
        ``env.step`` instead of ``compute_f``. The trainer's plotting
        dispatch calls this instead of going through the model class's
        analytical rollout.
        """
        z_traj, _ = env.rollout(policy, z0, return_full_trajectory=return_full_trajectory)
        return z_traj