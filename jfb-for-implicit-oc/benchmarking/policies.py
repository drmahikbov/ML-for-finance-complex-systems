"""
benchmarking.policies
---------------------
Diagnostic-only policy adapters used by the benchmarking harness.

These callables satisfy the same ``policy(z, t) -> u`` contract as
:class:`ImplicitNets.ImplicitNetOC` so they plug straight into
:class:`benchmarking.solvers.JFBPolicyRollout` and the
:class:`Mapping[label, policy]` overlay branch of
:class:`liquidation_benchmark.LiquidationBenchmark.plot_comparison`.

They are **never** intended for training.  In particular,
:class:`LearnedCostatePolicy` carries ``is_direct_control = True`` so that
:meth:`ImplicitOC.compute_loss` skips its HJB / adjoint diagnostics if
the adapter is ever accidentally handed to the trainer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LearnedCostatePolicy(nn.Module):
    """Closed-form ``u*(t, z)`` evaluated on a learned costate
    ``p_θ = ∇_z φ`` (no inner fixed-point iteration).

    Used to **decompose** the gap ``JFB ↔ Exact BVP`` into

      * ``JFB ↔ JFB u*(p_θ)``  -- inner-FP convergence error,
      * ``JFB u*(p_θ) ↔ Exact BVP``  -- costate-learning error.

    Construction requires ``prob.has_closed_form_u_star() is True``.
    Overlaying this policy on the comparison figure therefore happens
    only for problems where a closed-form ``argmin_u H`` is meaningful
    (e.g. the post-reduction :class:`LiquidationPortfolioOC` at γ=2,
    where ``u* = (S + p_q + κ p_S) / (2η)`` — no ``p_X`` involved).

    Parameters
    ----------
    p_net
        Learned costate network exposed by :class:`ImplicitNets.ImplicitNetOC`
        (typically ``inn.p_net``).  Must implement ``p_net(t, z) -> p`` of
        shape ``(batch, state_dim)`` -- exactly what :class:`Phi` returns
        with ``full_grad=False``.
    prob
        Concrete :class:`ImplicitOC` problem providing the closed-form
        formula via :meth:`ImplicitOC.optimal_u_from_costate`.

    Notes
    -----
    The legacy ``pin_p_X`` / ``p_x_floor`` knobs are gone: in the
    reduced :class:`LiquidationPortfolioOC` formulation there is no
    ``p_X`` to pin (cash ``X`` is no longer an OC state) and no
    division by ``p_X`` in :meth:`optimal_u_from_costate`, so the
    failure mode they guarded against cannot occur.
    """

    # The trainer's compute_loss switches off HJB/adjoint diagnostics
    # when this attribute is True; setting it here means a stray use in
    # training cannot pollute the optimisation with bogus residuals.
    is_direct_control = True

    def __init__(self, p_net: nn.Module, prob: Any):
        super().__init__()
        if not prob.has_closed_form_u_star():
            raise ValueError(
                f"{type(prob).__name__} does not implement a closed-form u*; "
                "LearnedCostatePolicy cannot be used."
            )
        # Register p_net as a child module so .to(device) propagates.
        self.p_net = p_net
        # ``prob`` carries non-Module parameters (eta, kappa, ...); store
        # as a plain attribute, not a submodule.
        self._prob = prob

    @property
    def prob(self) -> Any:
        return self._prob

    def forward(self, z: torch.Tensor, t: float, **_: Any) -> torch.Tensor:
        """Return ``u*(t, z)`` evaluated through the closed-form formula.

        ``**_`` swallows extra kwargs (``track_all_fp_iters``, ``record_trace``,
        etc.) that :class:`ImplicitNetOC.forward` accepts, so this adapter
        is a drop-in replacement at any policy call site.
        """
        with torch.no_grad():
            p = self.p_net(float(t), z)                       # (B, state_dim)
            u = self._prob.optimal_u_from_costate(float(t), z, p)
        return u


__all__ = ["LearnedCostatePolicy"]
