"""
benchmarking.diagnostics
------------------------
Inner-fixed-point and policy diagnostic plots for trained JFB policies.

What this module gives you
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`diagnostic_rollout` -- run one explicit-Euler rollout of the
  given policy on a single ``z0`` while harvesting, **at every
  timestep**, the inner-fixed-point depth, residual, the Hamiltonian
  stationarity residual and the policy-network costate ``p_θ``. Returns
  a :class:`benchmarking.Trajectory` whose ``meta`` carries those
  arrays. The first timestep additionally records the per-iteration
  residual trace of the inner FP solver.

* :func:`diagnostic_panels` -- a 6-panel layout consuming the diagnostic
  trajectory above. Pair with :class:`benchmarking.BenchmarkPlotter`.

* :func:`liquidation_costate_vs_bvp_panels` -- liquidation-specific
  overlay panels comparing the learned ``p_q(t), p_S(t)`` against the
  exact-BVP solution. Adds two panels on top of the generic ones.

Why a separate module?
~~~~~~~~~~~~~~~~~~~~~~

The standard :class:`benchmarking.BenchmarkPlotter` workflow draws
state/control panels for *one or more* trajectories. The diagnostic
plots are policy-specific (they need access to the policy and the
problem to recompute the Hamiltonian residual / costate), and they
*augment* the standard plots rather than replacing them, so they live
in their own module to avoid bloating the core plotter.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from .plotter import Panel
from .trajectory import Trajectory
from .solvers import AlmgrenChrissBVPSolver


# =============================================================================
# Diagnostic rollout
# =============================================================================

_DIAG_STYLE = {"color": "#d6604d", "lw": 1.6}


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def diagnostic_rollout(
    prob: Any,
    policy: Any,
    z0: torch.Tensor,
    label: str = "JFB diag",
    record_trace_at_t0: bool = True,
) -> Trajectory:
    """Roll out the policy and harvest inner-solver diagnostics.

    Parameters
    ----------
    prob
        Concrete :class:`ImplicitOC` problem providing ``compute_f``,
        ``compute_grad_H_u`` and the time grid (``t_initial``, ``t_final``,
        ``nt``).
    policy
        Trained :class:`ImplicitNetOC`. Must expose
        ``get_convergence_stats()`` (which it does by default) and accept
        the keyword ``record_trace`` on its ``forward`` call.
    z0
        Initial state, shape ``(state_dim,)`` or ``(1, state_dim)``.
    label
        Display label propagated to the resulting :class:`Trajectory`.
    record_trace_at_t0
        If ``True`` the very first inner-FP call records its per-iteration
        residual trace, exposed via ``meta["fp_trace_t0"]``.

    Returns
    -------
    Trajectory
        Standard ``(t, z, u)`` trajectory. ``meta`` carries:

        * ``fp_depth``         -- ``(nt,)`` ints, per-timestep depth.
        * ``fp_res_norm``      -- ``(nt,)`` floats, per-timestep residual.
        * ``grad_H_u_norm``    -- ``(nt,)`` floats,
          ``||∇_u H(t, z, u_returned, p_θ(t, z))||₂``.
        * ``p_theta``          -- ``(nt+1, state_dim)`` learned costate.
        * ``fp_trace_t0``      -- ``(K,)`` per-inner-iter residuals at ``t=t₀``
          (only when ``record_trace_at_t0=True``).
    """
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

    z = torch.as_tensor(z0_np, dtype=torch.float32, device=device).unsqueeze(0)

    z_traj = np.zeros((nt + 1, prob.state_dim), dtype=np.float64)
    u_traj = np.zeros((nt, prob.control_dim), dtype=np.float64)
    p_traj = np.zeros((nt + 1, prob.state_dim), dtype=np.float64)

    fp_depth = np.zeros(nt, dtype=np.int64)
    fp_res = np.zeros(nt, dtype=np.float64)
    grad_H_u_norm = np.zeros(nt, dtype=np.float64)
    fp_trace_t0: List[float] = []

    z_traj[0] = z.detach().cpu().numpy().reshape(-1)

    # Pre-compute p_θ(t₀, z₀) so the costate panel has the leading time-node.
    with torch.no_grad():
        p_traj[0] = policy.p_net(t0, z).detach().cpu().numpy().reshape(-1)

    was_eval = not policy.training
    policy.eval()
    try:
        ti = t0
        for i in range(nt):
            do_trace = record_trace_at_t0 and i == 0
            with torch.no_grad():
                u = policy(z, float(ti), record_trace=do_trace).view(1, prob.control_dim)

            stats = policy.get_convergence_stats()
            fp_depth[i] = int(stats.get("fp_depth", 0))
            fp_res[i] = float(stats.get("max_res_norm", float("nan")))
            if do_trace:
                fp_trace_t0 = list(stats.get("residual_trace", []))

            # Hamiltonian residual at the returned u and p_θ.
            with torch.no_grad():
                p_theta = policy.p_net(float(ti), z)
                t_scalar = torch.ones(1, device=z.device) * float(ti)
                grad_H_u_val = prob.compute_grad_H_u(t_scalar, z, u, p_theta)
            grad_H_u_norm[i] = float(torch.linalg.vector_norm(grad_H_u_val, dim=1).max().item())

            u_traj[i] = u.detach().cpu().numpy().reshape(-1)
            with torch.no_grad():
                z = z + dt * prob.compute_f(float(ti), z, u)
            z_traj[i + 1] = z.detach().cpu().numpy().reshape(-1)

            with torch.no_grad():
                p_traj[i + 1] = policy.p_net(float(ti) + dt, z).detach().cpu().numpy().reshape(-1)

            ti += dt
    finally:
        if not was_eval:
            policy.train()

    t_arr = np.linspace(t0, T, nt + 1)

    meta = {
        "fp_depth": fp_depth,
        "fp_res_norm": fp_res,
        "grad_H_u_norm": grad_H_u_norm,
        "p_theta": p_traj,
        "fp_trace_t0": np.asarray(fp_trace_t0, dtype=np.float64),
        "fp_max_iters": int(getattr(policy, "max_iters", 0)),
        "fp_tol": float(getattr(policy, "tol", 0.0)),
        "fp_alpha": float(getattr(policy, "alpha", 0.0)),
        "use_anderson": bool(getattr(policy, "use_anderson", False)),
    }

    return Trajectory(
        t=t_arr,
        z=z_traj,
        u=u_traj,
        label=label,
        style=dict(_DIAG_STYLE),
        meta=meta,
    )


# =============================================================================
# Diagnostic panels
# =============================================================================

def _meta_extractor(key: str, t_kind: str = "interior"):
    """Build an extractor that pulls ``traj.meta[key]`` against ``t``.

    ``t_kind``:
      * ``"interior"`` -- use ``traj.t[:-1]`` (one value per Euler step,
        i.e. matches per-timestep arrays of length ``nt``).
      * ``"all"``      -- use ``traj.t`` (matches arrays of length ``nt+1``).
    """
    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(traj.meta.get(key, np.array([])))
        if t_kind == "interior":
            x = traj.t[:-1]
        else:
            x = traj.t
        return x, y
    return _extract


def _costate_extractor(component: int):
    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        p = traj.meta.get("p_theta")
        if p is None:
            return traj.t, np.zeros_like(traj.t)
        return traj.t, np.asarray(p)[:, component]
    return _extract


def _trace_extractor():
    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        trace = np.asarray(traj.meta.get("fp_trace_t0", np.array([])))
        x = np.arange(1, trace.shape[0] + 1) if trace.size else np.empty(0)
        return x, trace
    return _extract


def diagnostic_panels(state_components: Tuple[int, int] = (0, 1)) -> List[Panel]:
    """6-panel diagnostic layout for inner-FP and policy quality.

    Panels:

    1. ``fp_depth(t)``         -- inner FP depth per Euler step.
    2. ``res_norm(t)``         -- inner FP residual per Euler step (log y).
    3. ``||∇_u H||(t)``        -- Hamiltonian stationarity residual (log y).
    4. ``p_θ[component_a](t)`` -- learned costate (default ``p_q``).
    5. ``p_θ[component_b](t)`` -- learned costate (default ``p_S``).
    6. inner-FP residual trace at ``t=t₀`` (log y).

    Parameters
    ----------
    state_components
        Pair ``(a, b)`` indicating which components of ``p_θ`` to plot in
        panels 4 and 5. Default ``(0, 1)`` matches the liquidation
        layout ``[q, S, X]`` (so the panels show ``p_q`` and ``p_S``).
    """
    a, b = state_components
    return [
        Panel(
            "Inner-FP depth  k(t)",
            _meta_extractor("fp_depth", t_kind="interior"),
            ylabel="iterations",
            xlabel="t",
            yscale="linear",
        ),
        Panel(
            "Inner-FP residual  ||u_{k+1}−u_k||/α (t)",
            _meta_extractor("fp_res_norm", t_kind="interior"),
            ylabel="residual",
            xlabel="t",
            yscale="log",
        ),
        Panel(
            "Hamiltonian residual  ||∇_u H||(t)",
            _meta_extractor("grad_H_u_norm", t_kind="interior"),
            ylabel="||∇_u H||",
            xlabel="t",
            yscale="log",
        ),
        Panel(
            f"Learned costate  p_θ[{a}](t)",
            _costate_extractor(a),
            ylabel=f"p_θ[{a}]",
            xlabel="t",
        ),
        Panel(
            f"Learned costate  p_θ[{b}](t)",
            _costate_extractor(b),
            ylabel=f"p_θ[{b}]",
            xlabel="t",
        ),
        Panel(
            "Inner-FP convergence trace at t = t₀",
            _trace_extractor(),
            ylabel="residual",
            xlabel="inner iteration k",
            yscale="log",
        ),
    ]


# =============================================================================
# Liquidation-specific: costate-vs-BVP overlay
# =============================================================================

def attach_bvp_costate_to_meta(traj: Trajectory, prob: Any, z0: np.ndarray) -> Trajectory:
    """Add the exact BVP costate ``(p_q, p_S)`` to a diagnostic trajectory.

    Returns a *new* :class:`Trajectory` whose ``meta`` gains
    ``p_bvp`` (shape ``(N, 2)``) and ``t_bvp`` (shape ``(N,)``). Use the
    factory :func:`liquidation_costate_vs_bvp_panels` together with this
    augmentation.

    Accepts ``z0`` in either the legacy 3-component layout
    ``[q0, S0, X0]`` or the post-reduction 2-component layout
    ``[q0, S0]`` — single-asset only either way. The 2-component case
    is auto-padded with ``X0 = 0.0`` before being handed to the
    reference BVP solver, which still requires 3 components.
    """
    if abs(float(prob.gamma) - 2.0) >= 1e-6:
        raise ValueError(
            f"BVP costate available only for γ=2; prob.gamma={prob.gamma}"
        )

    z0_arr = np.asarray(z0).reshape(-1)
    if z0_arr.shape == (2,):
        z0_arr = np.concatenate([z0_arr, np.array([0.0])])

    solver = AlmgrenChrissBVPSolver(prob)
    bvp_traj = solver.solve(z0_arr)
    # Reconstruct (p_q, p_S) on the BVP grid by re-running the BVP at the
    # same nodes.  AlmgrenChrissBVPSolver doesn't expose them directly,
    # so we redo the solve and read off the costates.
    from scipy.integrate import solve_bvp
    q0, S0 = float(z0_arr[0]), float(z0_arr[1])
    t_nodes = np.linspace(prob.t_initial, prob.t_final, solver.n_bvp_nodes)
    y_init = np.zeros((4, len(t_nodes)))
    y_init[0] = np.linspace(q0, 0.0, len(t_nodes))
    y_init[1] = S0
    y_init[2] = -(solver.sigma ** 2) * q0 * (prob.t_final - t_nodes)
    y_init[3] = 0.0
    sol = solve_bvp(
        solver._odes,
        lambda ya, yb: solver._bc(ya, yb, q0, S0),
        t_nodes, y_init, tol=solver.bvp_tol, max_nodes=10000,
    )
    p_bvp = np.stack([sol.y[2], sol.y[3]], axis=1)  # (N, 2)

    new_meta = dict(traj.meta)
    new_meta["p_bvp"] = p_bvp
    new_meta["t_bvp"] = sol.x

    from dataclasses import replace
    return replace(traj, meta=new_meta)


def _bvp_costate_extractor(component: int):
    """Extract (t_bvp, p_bvp[:, component]) from a trajectory enriched by
    :func:`attach_bvp_costate_to_meta`. Returns empty arrays when the
    trajectory was not enriched (so the BenchmarkPlotter quietly skips it).
    """
    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        p_bvp = traj.meta.get("p_bvp")
        t_bvp = traj.meta.get("t_bvp")
        if p_bvp is None or t_bvp is None:
            return np.empty(0), np.empty(0)
        return np.asarray(t_bvp), np.asarray(p_bvp)[:, component]
    return _extract


def liquidation_costate_vs_bvp_panels() -> List[Panel]:
    """Two extra panels overlaying ``p_θ`` against the exact BVP costates.

    These are *additive* on top of the panels produced by
    :func:`diagnostic_panels`. Render them by passing two trajectories:
    the diagnostic one (carrying ``p_theta``) **and** an enriched copy
    carrying ``p_bvp`` / ``t_bvp`` (see :func:`attach_bvp_costate_to_meta`).

    Order:
      1. learned ``p_θ[0]`` vs exact ``p_q(t)``
      2. learned ``p_θ[1]`` vs exact ``p_S(t)``
    """
    return [
        Panel(
            "Costate p_q(t):  learned vs exact BVP",
            _bvp_costate_extractor(0),
            ylabel="p_q",
            xlabel="t",
        ),
        Panel(
            "Costate p_S(t):  learned vs exact BVP",
            _bvp_costate_extractor(1),
            ylabel="p_S",
            xlabel="t",
        ),
    ]


def liquidation_u_decomposition_panel(prob: Any, asset: int = 0) -> Panel:
    """One additive panel: ``u*(t) = prob.optimal_u_from_costate(t, z, p_θ)``
    sampled along the JFB rollout.

    Renders the closed-form PMP optimum evaluated on the **learned**
    costate ``p_θ`` from ``meta["p_theta"]`` of a trajectory produced by
    :func:`diagnostic_rollout`.  Because ``z(t)`` and ``p_θ(t)`` come
    from the JFB rollout, the gap between this curve and the trajectory's
    own ``u(t)`` isolates the **inner-FP convergence error** (no
    re-rollout needed).

    Parameters
    ----------
    prob
        Problem exposing :meth:`ImplicitOC.optimal_u_from_costate`.
        Must satisfy ``prob.has_closed_form_u_star() is True``.
    asset
        Which control component to plot (default 0).  For single-asset
        problems this is the only choice.

    Notes
    -----
    The legacy ``pin_p_X`` knob has been removed: in the reduced
    :class:`LiquidationPortfolioOC` formulation there is no ``p_X`` in
    the costate (cash ``X`` is no longer an OC state) and
    ``optimal_u_from_costate`` does not divide by ``p_X``, so the
    blow-up the knob used to mask cannot occur.

    Trajectories produced by :class:`benchmarking.JFBPolicyRollout` carry
    a cash observer column (``[q, S, X_obs]``, shape ``(N, 2n+1)``); we
    slice off the OC state (the first ``state_dim`` columns) before
    feeding it into ``optimal_u_from_costate``.
    """
    if not prob.has_closed_form_u_star():
        raise ValueError(
            f"{type(prob).__name__} does not implement a closed-form u*; "
            "liquidation_u_decomposition_panel requires it."
        )

    state_dim = int(getattr(prob, "state_dim"))

    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        p_arr = traj.meta.get("p_theta")
        if p_arr is None:
            return np.empty(0), np.empty(0)
        # Trajectory.z may include a trailing cash observer column
        # (rollouts) or be exactly state_dim wide (diagnostic rollouts);
        # always feed the closed-form formula the OC state only.
        z_full = np.asarray(traj.z)
        z_oc = z_full[:, :state_dim]
        z_t = torch.as_tensor(z_oc, dtype=torch.float32)          # (N, state_dim)
        p_t = torch.as_tensor(np.asarray(p_arr), dtype=torch.float32)
        with torch.no_grad():
            u = prob.optimal_u_from_costate(0.0, z_t, p_t)        # (N, control_dim)
        return traj.t, u.cpu().numpy()[:, asset]

    return Panel(
        f"u*(p_θ) along JFB rollout  [asset {asset}]",
        _extract,
        ylabel="u*",
        xlabel="t",
    )


__all__ = [
    "diagnostic_rollout",
    "diagnostic_panels",
    "attach_bvp_costate_to_meta",
    "liquidation_costate_vs_bvp_panels",
    "liquidation_u_decomposition_panel",
]
