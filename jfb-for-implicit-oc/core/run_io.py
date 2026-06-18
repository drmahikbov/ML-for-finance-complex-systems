"""
core.run_io
-----------
Binds run identity (problem class, tag, timestamp) to artifact paths.

`RunIO` owns all filename decisions — the trainer and runners never build
paths themselves. Stem = f"{tag}_{run_id}". Artifacts land in:
  results/<ProblemClassName>/training/   — checkpoint, history, loss curve, plots
  results/<ProblemClassName>/rollouts/   — final rollout figure and trajectory tensor
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from core.paths import results_dir


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class RunIO:
    """Path/filename policy for a single training run."""

    problem_cls_name: str
    tag: str = "JFB"
    run_id: str = field(default_factory=_default_run_id)

    @property
    def stem(self) -> str:
        """The shared filename prefix, e.g. ``"JFB_20260425_154755"``."""
        return f"{self.tag}_{self.run_id}"

    # ------------------------------------------------------------------
    # Directories (each created on demand by ``results_dir``).
    # ------------------------------------------------------------------
    @property
    def train_dir(self) -> str:
        return results_dir(self.problem_cls_name, "training")

    @property
    def plots_dir(self) -> str:
        return results_dir(self.problem_cls_name, "training", "training-plots")

    @property
    def rollout_dir(self) -> str:
        return results_dir(self.problem_cls_name, "rollouts")

    @property
    def benchmark_dir(self) -> str:
        return results_dir(self.problem_cls_name, "benchmark")

    @property
    def reference_dir(self) -> str:
        return results_dir(self.problem_cls_name, "reference")

    # ------------------------------------------------------------------
    # Per-artifact filenames.
    # ------------------------------------------------------------------
    def policy_path(self) -> str:
        return os.path.join(self.train_dir, f"best_policy_{self.stem}.pth")

    def history_path(self) -> str:
        return os.path.join(self.train_dir, f"history_{self.stem}.csv")

    def loss_curve_path(self) -> str:
        return os.path.join(self.train_dir, f"loss_curve_{self.stem}.png")

    def training_plot_path(self, epoch: int) -> str:
        return os.path.join(self.plots_dir, f"rollout_{self.stem}_{epoch:04d}.png")

    def rollout_path(self) -> str:
        return os.path.join(self.rollout_dir, f"policy_rollout_{self.stem}.png")

    def trajectory_path(self) -> str:
        return os.path.join(self.rollout_dir, f"trajectory_{self.stem}.pth")
