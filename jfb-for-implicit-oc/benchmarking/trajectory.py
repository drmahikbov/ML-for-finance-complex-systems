"""
benchmarking.trajectory
-----------------------
Dataclass container for a state/control trajectory.

Deterministic: z (N, state_dim), u (N-1, control_dim). Stochastic bundle:
z (n_paths, N, state_dim), u (n_paths, N-1, control_dim). Control u sits on
the left endpoints of the Euler intervals. Payloads are numpy arrays only —
convert torch tensors to numpy before constructing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """A state/control path with presentation metadata.

    Parameters
    ----------
    t : np.ndarray
        Shape ``(N,)``.  Time grid.
    z : np.ndarray
        Shape ``(N, state_dim)`` for deterministic data or
        ``(n_paths, N, state_dim)`` for stochastic data.
    u : np.ndarray, optional
        Shape ``(N-1, control_dim)`` or ``(n_paths, N-1, control_dim)``.
        ``None`` when control is not available (e.g. the exact-solution
        plotter is passed a state-only reference).
    cost : float, optional
        Realised total cost ``J`` along the trajectory.  Mean over paths
        for stochastic shapes.
    label : str
        Display label used by :class:`BenchmarkPlotter` for legends.
    style : dict
        Matplotlib keyword arguments (``color``, ``ls``, ``lw``, ``alpha``,
        ``marker``, ...).  Forwarded verbatim to the plotting call.
    meta : dict
        Arbitrary user metadata (initial conditions, seed, solver
        parameters).  Never read by the plotter.
    """

    t: np.ndarray
    z: np.ndarray
    u: Optional[np.ndarray] = None
    cost: Optional[float] = None
    label: str = ""
    style: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.t)
        z = np.asarray(self.z)
        if t.ndim != 1:
            raise ValueError(f"t must be 1D, got shape {t.shape}")
        if z.ndim not in (2, 3):
            raise ValueError(
                f"z must be 2D (deterministic) or 3D (stochastic), got shape {z.shape}"
            )
        n_steps = t.shape[0]
        n_from_z = z.shape[-2]
        if n_from_z != n_steps:
            raise ValueError(
                f"t has length {n_steps} but z has {n_from_z} time nodes "
                f"(z.shape={z.shape})"
            )

        if self.u is not None:
            u = np.asarray(self.u)
            if z.ndim == 2:
                if u.ndim != 2:
                    raise ValueError(
                        f"Deterministic trajectory expects 2D u, got shape {u.shape}"
                    )
                if u.shape[0] != n_steps - 1:
                    raise ValueError(
                        f"u has {u.shape[0]} time steps but expected {n_steps - 1}"
                    )
            else:  # z.ndim == 3 (stochastic)
                if u.ndim != 3:
                    raise ValueError(
                        f"Stochastic trajectory expects 3D u, got shape {u.shape}"
                    )
                if u.shape[0] != z.shape[0]:
                    raise ValueError(
                        f"u has {u.shape[0]} paths but z has {z.shape[0]}"
                    )
                if u.shape[1] != n_steps - 1:
                    raise ValueError(
                        f"u has {u.shape[1]} time steps but expected {n_steps - 1}"
                    )
            object.__setattr__(self, "u", u)

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "z", z)

    # ------------------------------------------------------------------ #
    # Shape introspection                                                #
    # ------------------------------------------------------------------ #

    @property
    def is_stochastic(self) -> bool:
        """``True`` iff this trajectory carries a leading ``n_paths`` axis."""
        return self.z.ndim == 3

    @property
    def n_paths(self) -> int:
        return self.z.shape[0] if self.is_stochastic else 1

    @property
    def n_steps(self) -> int:
        return self.t.shape[0]

    @property
    def state_dim(self) -> int:
        return self.z.shape[-1]

    @property
    def control_dim(self) -> int:
        if self.u is None:
            return 0
        return self.u.shape[-1]

    # ------------------------------------------------------------------ #
    # Derived views                                                      #
    # ------------------------------------------------------------------ #

    def terminal_state(self) -> np.ndarray:
        """Return ``z[..., -1, :]``.

        Shape is ``(state_dim,)`` for deterministic trajectories and
        ``(n_paths, state_dim)`` for stochastic ones.
        """
        return self.z[..., -1, :]

    def mean_path(self) -> "Trajectory":
        """Average across the path axis.

        For a deterministic trajectory this is a no-op.  For a stochastic
        trajectory it returns a fresh deterministic :class:`Trajectory`
        with ``z`` and ``u`` reduced by :func:`numpy.mean` over the path
        axis.  The ``cost`` and ``label`` fields are preserved; ``meta``
        gains a ``reduced_from_n_paths`` entry for traceability.
        """
        if not self.is_stochastic:
            return self
        z_mean = self.z.mean(axis=0)
        u_mean = None if self.u is None else self.u.mean(axis=0)
        new_meta: dict[str, Any] = dict(self.meta)
        new_meta["reduced_from_n_paths"] = self.n_paths
        return replace(self, z=z_mean, u=u_mean, meta=new_meta)
