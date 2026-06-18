"""
benchmarking.plotter
--------------------
Generic multi-panel comparison plotter.

`BenchmarkPlotter` overlays a list of `Trajectory` objects on each `Panel`.
Panels are data-agnostic — an `extract` callable maps a trajectory to (x, y).
Stochastic trajectories get band plots; single-path trajectories get lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .trajectory import Trajectory


# -------------------------------------------------------------------------
# Shared styling constants (mirroring the legacy liquidation_benchmark)
# -------------------------------------------------------------------------

_COLORS = {
    "exact": "#2166ac",   # blue  - BVP / closed-form
    "jfb":   "#d6604d",   # red   - JFB learned policy
    "band":  "#92c5de",   # light blue - ±1 std band around exact
}
_LW = 2.0


# -------------------------------------------------------------------------
# Panel spec
# -------------------------------------------------------------------------

ExtractFn = Callable[[Trajectory], Tuple[np.ndarray, np.ndarray]]


@dataclass
class Panel:
    """Declarative description of one subplot.

    Parameters
    ----------
    title : str
        Title rendered above the subplot.
    extract : callable
        ``trajectory -> (x, y)``.  For ``plot_type="band"`` on stochastic
        data the extractor should return ``y`` as a 2D array of shape
        ``(n_paths, N)``; the plotter computes the mean and ±1 std
        itself.  For every other case ``y`` should be a 1D array.
    ylabel : str
    xlabel : str
    plot_type : {"line", "bar", "scatter", "band"}
    yscale : {"linear", "log"}
    """

    title: str
    extract: ExtractFn
    ylabel: str
    xlabel: str = "t"
    plot_type: Literal["line", "bar", "scatter", "band"] = "line"
    yscale: Literal["linear", "log"] = "linear"
    extra: dict = field(default_factory=dict)


# -------------------------------------------------------------------------
# Plotter
# -------------------------------------------------------------------------

class BenchmarkPlotter:
    """Render a grid of :class:`Panel` objects for a list of trajectories.

    Parameters
    ----------
    panels : list of Panel
    ncols : int
        Number of columns in the subplot grid.  Rows are derived from
        ``ceil(len(panels) / ncols)``.
    figsize_per_panel : (width, height) in inches
        Per-panel cell size; total figure size is ``(ncols * w, nrows * h)``.
    hspace, wspace : float
        ``matplotlib.gridspec.GridSpec`` spacing parameters.
    """

    def __init__(
        self,
        panels: List[Panel],
        ncols: int = 2,
        figsize_per_panel: Tuple[float, float] = (5.5, 4.0),
        hspace: float = 0.42,
        wspace: float = 0.32,
    ):
        if not panels:
            raise ValueError("BenchmarkPlotter needs at least one panel.")
        self.panels = list(panels)
        self.ncols = int(ncols)
        self.nrows = int(np.ceil(len(panels) / self.ncols))
        self.figsize_per_panel = figsize_per_panel
        self.hspace = hspace
        self.wspace = wspace

    # ------------------------------------------------------------------
    def plot(
        self,
        trajectories: List[Trajectory],
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> "matplotlib.figure.Figure":
        """Render the grid of panels.

        Parameters
        ----------
        trajectories : list of Trajectory
            Each trajectory is drawn on every panel.  The legend displays
            each unique ``trajectory.label`` exactly once across the whole
            figure (computed per-axis but the label itself appears only
            once per axis).
        save_path : str, optional
            If given, ``fig.savefig(save_path, ...)`` is called and the
            figure is closed.  Otherwise ``plt.show()`` is called.
        title : str, optional
            Figure-level suptitle.
        """
        w, h = self.figsize_per_panel
        fig = plt.figure(figsize=(self.ncols * w, self.nrows * h))
        gs = gridspec.GridSpec(
            self.nrows, self.ncols, figure=fig,
            hspace=self.hspace, wspace=self.wspace,
        )
        axes = []
        for idx, panel in enumerate(self.panels):
            r, c = divmod(idx, self.ncols)
            ax = fig.add_subplot(gs[r, c])
            axes.append(ax)
            self._draw_panel(ax, panel, trajectories)

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    def _draw_panel(
        self,
        ax: "matplotlib.axes.Axes",
        panel: Panel,
        trajectories: List[Trajectory],
    ) -> None:
        seen_labels: set[str] = set()
        bar_offset = 0.0
        for traj in trajectories:
            x, y = panel.extract(traj)
            style = dict(traj.style)
            label = traj.label if traj.label and traj.label not in seen_labels else None
            if label:
                seen_labels.add(traj.label)

            if panel.plot_type == "line":
                y_line = _reduce_to_line(y)
                ax.plot(x, y_line, label=label, **style)

            elif panel.plot_type == "band":
                _plot_band(ax, x, y, style, label)

            elif panel.plot_type == "scatter":
                scatter_style = _scatter_style(style)
                ax.scatter(x, y, label=label, **scatter_style)

            elif panel.plot_type == "bar":
                bar_style = _bar_style(style)
                ax.bar(x + bar_offset, y, width=0.4, label=label, **bar_style)
                bar_offset += 0.4

            else:
                raise ValueError(f"Unknown plot_type: {panel.plot_type!r}")

        _label_ax(ax, panel.title, panel.xlabel, panel.ylabel)
        ax.set_yscale(panel.yscale)
        _add_legend(ax)


# -------------------------------------------------------------------------
# Drawing helpers
# -------------------------------------------------------------------------

def _reduce_to_line(y: np.ndarray) -> np.ndarray:
    """Collapse an extractor output to a 1D series for line plots."""
    y = np.asarray(y)
    if y.ndim == 2:
        return y.mean(axis=0)
    return y


def _plot_band(ax, x, y, style: dict, label: Optional[str]) -> None:
    """Mean line plus ±1 std band for stochastic extractor output.

    Falls back to a single line when only one path is present.
    """
    y = np.asarray(y)
    if y.ndim == 1 or (y.ndim == 2 and y.shape[0] == 1):
        ax.plot(x, np.squeeze(y), label=label, **style)
        return
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    color = style.get("color", "#1f77b4")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0)
    line_style = dict(style)
    ax.plot(x, mean, label=label, **line_style)


def _scatter_style(style: dict) -> dict:
    """Translate line-style kwargs into scatter kwargs."""
    out = {}
    if "color" in style:
        out["color"] = style["color"]
    if "alpha" in style:
        out["alpha"] = style["alpha"]
    if "marker" in style:
        out["marker"] = style["marker"]
    out.setdefault("s", 60)
    out.setdefault("zorder", 3)
    return out


def _bar_style(style: dict) -> dict:
    out = {}
    if "color" in style:
        out["color"] = style["color"]
    out["alpha"] = style.get("alpha", 0.7)
    return out


def _label_ax(ax, title: str, xlabel: str, ylabel: str, fontsize: int = 10) -> None:
    ax.set_title(title, fontsize=fontsize, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, ls="--", alpha=0.35)
    ax.tick_params(labelsize=8)


def _add_legend(ax, fontsize: int = 8) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=fontsize, framealpha=0.7)


# -------------------------------------------------------------------------
# Pre-built panel sets
# -------------------------------------------------------------------------

def _state_extractor(component: int) -> ExtractFn:
    """Returns ``(t, z[...,component])`` reduced across paths if any."""
    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        if traj.is_stochastic:
            y = traj.z[..., component]          # (n_paths, N)
        else:
            y = traj.z[..., component]          # (N,)
        return traj.t, y
    return _extract


def _control_extractor(component: int) -> ExtractFn:
    """Returns ``(t[:-1], u[...,component])``.  Empty series when u is None."""
    def _extract(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        if traj.u is None:
            return np.empty(0), np.empty(0)
        t_u = traj.t[:-1]
        if traj.is_stochastic:
            y = traj.u[..., component]          # (n_paths, N-1)
        else:
            y = traj.u[..., component]          # (N-1,)
        return t_u, y
    return _extract


def almgren_chriss_panels() -> List[Panel]:
    """Standard 4-panel layout for single-asset Almgren-Chriss liquidation.

    Panels: inventory ``q(t)``, trading rate ``u*(t)``, impacted price
    ``S(t)``, accumulated cash ``X(t)``.  State layout assumed to be
    ``[q, S, X]`` and control ``u`` scalar (component 0).
    """
    return [
        Panel("Inventory  q(t)",      _state_extractor(0),   "q(t)"),
        Panel("Trading Rate  u*(t)",  _control_extractor(0), "u*(t)"),
        Panel("Impacted Price  S(t)", _state_extractor(1),   "S(t)"),
        Panel("Accumulated Cash  X(t)", _state_extractor(2), "X(t)"),
    ]
