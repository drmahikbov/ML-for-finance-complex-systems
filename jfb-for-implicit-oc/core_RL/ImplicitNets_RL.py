"""
core_RL.ImplicitNets_RL
-----------------------
Drop-in subclass of `ImplicitNetOC` for the RL setting.

Overrides only `T`: uses `compute_grad_H_u_estimated` with an externally
supplied `b_k` (estimated `∂f/∂u`) instead of the true analytical Jacobian.
Everything else — FP loop, Anderson acceleration, control clamping, convergence
tracking — is inherited unchanged.

The trainer must call `policy.set_step_jacobian(b_k)` before each
`policy(z, t)` invocation; calling `T` without it raises at runtime.
"""

from __future__ import annotations

import torch

# core/ is on sys.path; flat import works.
from core.ImplicitNets import ImplicitNetOC


class ImplicitNetOC_RL(ImplicitNetOC):
    """Implicit policy that uses an externally supplied ``b_k`` inside its
    fixed-point operator. Drop-in replacement for :class:`ImplicitNetOC`
    in the RL training pipeline.

    Parameters
    ----------
    Same as :class:`ImplicitNetOC`, with the added expectation that
    ``oc_problem`` is a :class:`ImplicitOC_RL` instance (i.e. it provides
    ``compute_grad_H_u_estimated`` rather than ``compute_grad_H_u``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sentinel value: ``None`` until the trainer pushes a real estimate.
        # We reach for a deliberately unhelpful error if T is called before
        # ``set_step_jacobian`` so misuse fails fast.
        self._current_b_k: torch.Tensor | None = None

    # Setter the trainer / loss-routine calls before each step
    def set_step_jacobian(self, b_k: torch.Tensor) -> None:
        """Set the local control Jacobian to be used by the next FP run.

        Expected shape: ``(B, control_dim, state_dim)`` *or*
        ``(1, control_dim, state_dim)`` (broadcastable from a
        shared-across-batch estimator).
        """
        self._current_b_k = b_k

    def clear_step_jacobian(self) -> None:
        """Reset to ``None`` — useful between epochs to surface bugs early."""
        self._current_b_k = None

    # Override only T
    def T(self, u: torch.Tensor, x: torch.Tensor, t) -> torch.Tensor:
        """One gradient-ascent step on the **estimated** Hamiltonian.

        ``T̂_k(u; z) = u - α · ∇_u Ĥ(t, z, u, ∇_z φ_θ(t, z), b_k)``

        Same sign convention as the parent class.
        """
        if self._current_b_k is None:
            raise RuntimeError(
                "ImplicitNetOC_RL.T() called without a current b_k. "
                "The training loop must call policy.set_step_jacobian(b_k) "
                "before each policy(z, t) invocation."
            )

        batch_size = x.shape[0]
        t_scalar = torch.ones(1, device=x.device, dtype=x.dtype) * t
        assert x.shape == (batch_size, self.state_dim)

        # Costate: ∇_z φ_θ(t, z). The parent's ``Phi`` returns this
        # directly when called as ``Phi(t, z)`` (vs ``Phi.getPhi(t, z)``
        # which returns the scalar value).
        p = self.p_net(t, x)

        grad_H_u = self.oc_problem.compute_grad_H_u_estimated(
            t_scalar, x, u, p, self._current_b_k
        )
        assert grad_H_u.shape == u.shape, (
            f"compute_grad_H_u_estimated returned shape {tuple(grad_H_u.shape)}, "
            f"expected {tuple(u.shape)}"
        )

        # Sign convention identical to the parent: T(u) = u - α ∇_u H.
        # Clamp inside the FP loop so iterates stay in [u_min, u_max]; without
        # this, a large random p_net × non-zero b_k pushes the FP target
        # outside the contractive region and the iteration diverges to NaN.
        return self.apply_control_limits(u - self.alpha * grad_H_u)