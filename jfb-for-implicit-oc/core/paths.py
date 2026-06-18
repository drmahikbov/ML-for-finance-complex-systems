"""
core.paths
----------
Single source of truth for filesystem paths under results/.

`results_dir(problem_cls_name, subfolder)` returns the canonical path and
creates the directory. Subfolders: training/, rollouts/, reference/,
benchmark/. Anchored to this file's location so scripts work from any cwd.
"""

from __future__ import annotations

import os

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_ROOT = os.path.join(_PKG_ROOT, "results")


def results_dir(*parts: str) -> str:
    """Return ``<package>/results/<parts...>``, creating it on demand.

    Parameters
    ----------
    *parts : str
        Path components below ``results/``. Typical usage::

            results_dir("LiquidationPortfolioOC", "training")
            results_dir("LiquidationPortfolioOC", "training", "training-plots")
            results_dir("LiquidationPortfolioOC", "benchmark")

    Returns
    -------
    str
        Absolute path to the directory; always exists when this returns.
    """
    p = os.path.join(_RESULTS_ROOT, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def results_root() -> str:
    """Return the absolute path to ``<package>/results/`` itself."""
    os.makedirs(_RESULTS_ROOT, exist_ok=True)
    return _RESULTS_ROOT
