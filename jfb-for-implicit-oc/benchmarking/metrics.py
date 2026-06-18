"""
benchmarking.metrics
--------------------
Problem-agnostic error metrics (MAE, RMSE, L∞, terminal error) between two
`Trajectory` objects. Interpolates the reference onto the test time grid when
the grids differ.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np

from .trajectory import Trajectory


def _collapse_paths(a: np.ndarray) -> np.ndarray:
    """Average over the leading path axis if present."""
    if a.ndim > 1 and a.shape[0] > 1 and a.ndim == 2:
        # Treat 2D as (paths, N); caller responsibility not to pass (N, dim) here.
        return a.mean(axis=0)
    if a.ndim == 2 and a.shape[0] == 1:
        return a[0]
    return a


def _extract_component(
    traj: Trajectory, component: int, kind: Literal["state", "control"]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_series, series)`` with paths collapsed to mean."""
    if kind == "state":
        if traj.is_stochastic:
            series = traj.z[..., component].mean(axis=0)
        else:
            series = traj.z[..., component]
        return traj.t, series
    elif kind == "control":
        if traj.u is None:
            raise ValueError("Trajectory has no control data.")
        if traj.is_stochastic:
            series = traj.u[..., component].mean(axis=0)
        else:
            series = traj.u[..., component]
        return traj.t[:-1], series
    else:
        raise ValueError(f"kind must be 'state' or 'control', got {kind!r}")


def trajectory_error(
    traj_test: Trajectory,
    traj_ref: Trajectory,
    component: int,
    kind: Literal["state", "control"],
) -> Dict[str, float]:
    """Compute MAE / RMSE / L∞ / terminal error for one scalar component.

    Parameters
    ----------
    traj_test : Trajectory
        Trajectory under evaluation.
    traj_ref : Trajectory
        Reference / ground-truth trajectory.  Its time grid is mapped onto
        ``traj_test.t`` (or ``traj_test.t[:-1]`` for controls) via
        :func:`numpy.interp`.
    component : int
        Component index into the state (if ``kind='state'``) or control
        (if ``kind='control'``) vector.
    kind : {"state", "control"}

    Returns
    -------
    dict
        Keys ``"mae"``, ``"rmse"``, ``"l_inf"``, ``"terminal_abs"``.
    """
    t_test, y_test = _extract_component(traj_test, component, kind)
    t_ref, y_ref = _extract_component(traj_ref, component, kind)
    y_ref_on_test = np.interp(t_test, t_ref, y_ref)
    diff = y_test - y_ref_on_test
    return {
        "mae":          float(np.mean(np.abs(diff))),
        "rmse":         float(np.sqrt(np.mean(diff ** 2))),
        "l_inf":        float(np.max(np.abs(diff))),
        "terminal_abs": float(abs(diff[-1])),
    }


def cost_error(
    traj_test: Trajectory, traj_ref: Trajectory
) -> Optional[float]:
    """Return ``|J_test - J_ref|`` if both costs are defined, else ``None``."""
    if traj_test.cost is None or traj_ref.cost is None:
        return None
    return float(abs(traj_test.cost - traj_ref.cost))


def format_error_table(
    name: str, errors: Dict[str, Dict[str, float]]
) -> str:
    """Pretty-print a nested error report.

    Parameters
    ----------
    name : str
        Header label (e.g. problem name).
    errors : dict
        Mapping ``variable_name -> {metric_name -> value}``.

    Returns
    -------
    str
        Multi-line string suitable for ``print`` / logging.
    """
    width = 52
    lines = ["", "=" * width, f"  {name}".center(width), "=" * width]
    for var, metrics in errors.items():
        lines.append(f"  [{var}]")
        for metric, value in metrics.items():
            if value is None:
                continue
            lines.append(f"    {metric:<14} : {value:.6f}")
    lines.append("=" * width)
    return "\n".join(lines) + "\n"
