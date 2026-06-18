"""
examples.explicit_ustar.liquidation_benchmark
----------------------------------------------
Backwards-compatibility shim exposing the legacy `LiquidationBenchmark` API
(plot_comparison, error_report, etc.) over the `benchmarking` package.

New code should import directly from `benchmarking`. The γ=2 closed-form
optimum u*(t) = (−p_q − κ p_S − S) / (2η) is provided by
`AlmgrenChrissBVPSolver`.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # remove if running interactively
import matplotlib.pyplot as plt

from benchmarking import (
    Trajectory,
    AlmgrenChrissBVPSolver,
    JFBPolicyRollout,
    BenchmarkPlotter,
    Panel,
    almgren_chriss_panels,
)
from benchmarking import paths as _paths
from benchmarking import gradient_checks as _gradient_checks
from benchmarking.plotter import (
    _label_ax as _label_ax,
    _add_legend as _add_legend,
)
from benchmarking.metrics import trajectory_error, cost_error


# ──────────────────────────────────────────────────────────────────────────────
# Legacy output directory defaults
# ──────────────────────────────────────────────────────────────────────────────

from core.paths import results_dir as _results_dir

# Default benchmark directory: <pkg>/results/LiquidationPortfolioOC/benchmark/
_LEGACY_DEFAULT_DIR = _results_dir("LiquidationPortfolioOC", "benchmark")


def benchmark_png_dir() -> str:
    """Directory for liquidation benchmark PNGs.

    Preserves the legacy default ``results_liquidation_benchmark/`` next
    to this file.  ``LIQUIDATION_BENCHMARK_PNG_DIR`` still takes priority,
    followed by the generic ``BENCHMARK_PNG_DIR``.
    """
    return _paths.benchmark_png_dir(default_dir=_LEGACY_DEFAULT_DIR)


def benchmark_png_path(filename: str) -> str:
    """Full path for ``filename`` under :func:`benchmark_png_dir`, mkdir on demand."""
    d = benchmark_png_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


# ──────────────────────────────────────────────────────────────────────────────
# LiquidationBenchmark -- backwards-compatibility wrapper
# ──────────────────────────────────────────────────────────────────────────────


class LiquidationBenchmark:
    """Legacy facade for the liquidation portfolio benchmark.

    Delegates to :mod:`benchmarking` but keeps the legacy method
    signatures so existing scripts that rely on this class continue to
    work.  See the module docstring for the underlying mathematics.
    """

    _COLORS = {
        "exact": "#2166ac",
        "jfb":   "#d6604d",
        "band":  "#92c5de",
    }
    _LW = 2.0

    def __init__(self, prob: Any, n_bvp_nodes: int = 500, bvp_tol: float = 1e-9):
        self.prob = prob
        self.n_bvp_nodes = n_bvp_nodes
        self.bvp_tol = bvp_tol

        self.sigma = prob.sigma
        self.kappa = prob.kappa
        self.eta = prob.eta
        self.gamma = prob.gamma
        self.epsilon = prob.epsilon
        self.alpha = prob.alpha
        self.T = prob.t_final
        self.t0 = prob.t_initial

        self._gamma2_available = abs(self.gamma - 2.0) < 1e-6
        self._bvp_solver: Optional[AlmgrenChrissBVPSolver] = None
        if self._gamma2_available:
            self._bvp_solver = AlmgrenChrissBVPSolver(
                prob, n_bvp_nodes=n_bvp_nodes, bvp_tol=bvp_tol,
            )

    # ------------------------------------------------------------------
    # solve_exact -- legacy tuple return
    # ------------------------------------------------------------------

    def solve_exact(
        self, q0: float = 1.0, S0: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact BVP solution for a single ``(q0, S0)`` (γ=2 only).

        Returns
        -------
        t_arr : (N,) ndarray
        traj  : (3, N) ndarray -- rows ``[q(t), S(t), X(t)]``
        u_arr : (N,) ndarray of ``u*(t)`` on the BVP time grid
        """
        if not self._gamma2_available:
            raise ValueError(
                f"Exact solver only available for γ=2; this problem has γ={self.gamma:.3f}."
            )
        traj = self._bvp_solver.solve(np.array([q0, S0, 0.0]))
        t_arr = traj.t
        traj_mat = traj.z.T  # (3, N)
        # Legacy u_arr is the BVP-grid u* evaluated on *every* node (N values).
        # Reconstruct the right endpoint by extending with the final optimal control.
        u_left = traj.u[:, 0]
        u_right = u_left[-1:] if u_left.size else u_left
        # We recompute u*(T) from the raw BVP output for parity with the
        # legacy implementation which returned N values (not N-1).
        u_final = self._bvp_solver._u_star(
            traj.z[-1, 0], traj.z[-1, 1],
            # Reconstruct p_q(T) and p_S(T) from boundary conditions:
            2.0 * self.alpha * traj.z[-1, 0],
            0.0,
        )
        u_arr = np.concatenate([u_left, np.array([u_final])])
        return t_arr, traj_mat, u_arr

    # ------------------------------------------------------------------
    # plot_comparison -- 6-panel legacy figure
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        policy: Any,
        z0_batch: torch.Tensor,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        n_show: int = 5,
    ) -> plt.Figure:
        """Six-panel comparison figure preserving the legacy layout."""
        import matplotlib.gridspec as gridspec

        prob = self.prob
        batch = min(z0_batch.shape[0], n_show)
        z0 = z0_batch[:batch].to(prob.device)

        jfb_solver = JFBPolicyRollout(prob, policy)
        jfb_trajs = [jfb_solver.solve(z0[b]) for b in range(batch)]
        t_jfb = jfb_trajs[0].t
        t_u = t_jfb[:-1]

        exact_trajs: list[Trajectory] = []
        if self._gamma2_available:
            for b in range(batch):
                q0_b = float(z0[b, 0].item())
                S0_b = float(z0[b, 1].item())
                exact_trajs.append(
                    self._bvp_solver.solve(np.array([q0_b, S0_b, 0.0]))
                )

        fig = plt.figure(figsize=(14, 13))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)
        axs = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

        c_ex = self._COLORS["exact"]
        c_jfb = self._COLORS["jfb"]
        lw = self._LW
        alpha_line = 0.75

        for b in range(batch):
            kw_jfb = dict(color=c_jfb, lw=lw, alpha=alpha_line,
                          label="JFB" if b == 0 else None)
            kw_ex = dict(color=c_ex, lw=lw, alpha=alpha_line, ls="--",
                         label="Exact (BVP)" if b == 0 else None)

            jfb = jfb_trajs[b]
            q_jfb = jfb.z[:, 0]
            S_jfb = jfb.z[:, 1]
            X_jfb = jfb.z[:, 2]
            u_vals = jfb.u[:, 0]

            axs[0][0].plot(t_jfb, q_jfb, **kw_jfb)
            axs[0][1].plot(t_u,   u_vals, **kw_jfb)
            axs[1][0].plot(t_jfb, S_jfb, **kw_jfb)
            axs[1][1].plot(t_jfb, X_jfb, **kw_jfb)

            if self._gamma2_available:
                ex = exact_trajs[b]
                axs[0][0].plot(ex.t, ex.z[:, 0], **kw_ex)
                axs[0][1].plot(ex.t[:-1], ex.u[:, 0], **kw_ex)
                axs[1][0].plot(ex.t, ex.z[:, 1], **kw_ex)
                axs[1][1].plot(ex.t, ex.z[:, 2], **kw_ex)

        _label_ax(axs[0][0], "Inventory  q(t)",         "Time", "q(t)")
        _label_ax(axs[0][1], "Trading Rate  u*(t)",     "Time", "u*(t)")
        _label_ax(axs[1][0], "Impacted Price  S(t)",    "Time", "S(t)")
        _label_ax(axs[1][1], "Accumulated Cash  X(t)",  "Time", "X(t)")
        for row in range(2):
            for col in range(2):
                _add_legend(axs[row][col])

        # Panel [2,0]: terminal inventory bar chart
        q_T_jfb = np.array([jfb.z[-1, 0] for jfb in jfb_trajs])
        xs = np.arange(batch)
        axs[2][0].bar(xs, q_T_jfb, color=c_jfb, alpha=0.7, label="JFB  q(T)")
        if self._gamma2_available:
            q_T_exact = np.array([ex.z[-1, 0] for ex in exact_trajs])
            axs[2][0].bar(xs + 0.4, q_T_exact, 0.4, color=c_ex, alpha=0.7,
                          label="Exact q(T)")
        axs[2][0].axhline(0, color="k", lw=0.8, ls="--")
        _label_ax(axs[2][0], "Terminal Inventory  q(T)", "Trajectory index", "q(T)")
        _add_legend(axs[2][0])

        # Panel [2,1]: u*(0) vs q0 linearity check
        q0_vals = z0[:, 0].cpu().numpy()
        u0_jfb = np.array([jfb.u[0, 0] for jfb in jfb_trajs])
        axs[2][1].scatter(q0_vals, u0_jfb, color=c_jfb, s=60, zorder=3,
                          label="JFB  u*(0)")
        if self._gamma2_available:
            u0_exact = np.array([ex.u[0, 0] for ex in exact_trajs])
            axs[2][1].scatter(q0_vals, u0_exact, color=c_ex, s=60, marker="^",
                              zorder=3, label="Exact u*(0)")
            if batch > 1:
                m, c_fit = np.polyfit(q0_vals, u0_exact, 1)
                x_line = np.linspace(q0_vals.min(), q0_vals.max(), 50)
                axs[2][1].plot(x_line, m * x_line + c_fit, color=c_ex, lw=1.2,
                               ls=":", alpha=0.6)
        _label_ax(axs[2][1], "Initial Rate u*(0) vs  q₀",
                  "Initial inventory  q₀", "u*(0)")
        _add_legend(axs[2][1])

        _title = title or (
            f"LiquidationPortfolio — JFB vs Exact BVP  "
            f"(γ={self.gamma:.1f}, η={self.eta}, κ={self.kappa:.0e})"
        )
        fig.suptitle(_title, fontsize=13, fontweight="bold", y=0.995)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # error_report
    # ------------------------------------------------------------------

    def error_report(
        self,
        policy: Any,
        z0_batch: torch.Tensor,
        verbose: bool = True,
    ) -> dict:
        """Per-batch JFB-vs-Exact error metrics (γ=2 only)."""
        if not self._gamma2_available:
            raise ValueError("error_report only available for γ=2.")

        prob = self.prob
        batch = z0_batch.shape[0]
        z0 = z0_batch.to(prob.device)

        jfb_solver = JFBPolicyRollout(prob, policy)

        u_maes, u_rmses, q_T_maes, X_T_maes, G_maes = [], [], [], [], []

        for b in range(batch):
            jfb_traj = jfb_solver.solve(z0[b])
            q0_b = float(z0[b, 0].item())
            S0_b = float(z0[b, 1].item())
            exact_traj = self._bvp_solver.solve(np.array([q0_b, S0_b, 0.0]))

            u_err = trajectory_error(jfb_traj, exact_traj, 0, "control")
            q_err = trajectory_error(jfb_traj, exact_traj, 0, "state")
            X_err = trajectory_error(jfb_traj, exact_traj, 2, "state")

            u_maes.append(u_err["mae"])
            u_rmses.append(u_err["rmse"])
            q_T_maes.append(q_err["terminal_abs"])
            X_T_maes.append(X_err["terminal_abs"])

            z_T_jfb = jfb_traj.z[-1]
            G_jfb = float(-z_T_jfb[2] + self.alpha * z_T_jfb[0] ** 2)
            # exact_traj.cost already holds G for the exact path.
            G_exact = exact_traj.cost if exact_traj.cost is not None else float(
                -exact_traj.z[-1, 2] + self.alpha * exact_traj.z[-1, 0] ** 2
            )
            G_maes.append(abs(G_jfb - G_exact))

        results = {
            "u_mae":   float(np.mean(u_maes)),
            "u_rmse":  float(np.mean(u_rmses)),
            "q_T_mae": float(np.mean(q_T_maes)),
            "X_T_mae": float(np.mean(X_T_maes)),
            "G_mae":   float(np.mean(G_maes)),
        }

        if verbose:
            print("\n" + "=" * 52)
            print("  LiquidationPortfolio — JFB vs Exact BVP  ")
            print(f"  γ={self.gamma:.2f}  batch={batch}  T={self.T}")
            print("=" * 52)
            print(f"  Trading rate  MAE  : {results['u_mae']:.6f}")
            print(f"  Trading rate  RMSE : {results['u_rmse']:.6f}")
            print(f"  Terminal q(T) MAE  : {results['q_T_mae']:.6f}")
            print(f"  Terminal X(T) MAE  : {results['X_T_mae']:.6f}")
            print(f"  Terminal cost MAE  : {results['G_mae']:.6f}")
            print("=" * 52 + "\n")

        return results

    # ------------------------------------------------------------------
    # gradient_check -- delegates to benchmarking.gradient_checks
    # ------------------------------------------------------------------

    def gradient_check(
        self,
        z: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
    ) -> dict:
        """Run all analytical-gradient tests + a Taylor-convergence check."""
        return _gradient_checks.gradient_check(
            self.prob, z=z, u=u, save_path=save_path,
        )

    # ------------------------------------------------------------------
    # plot_training_history  (unchanged behaviour from legacy version)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_training_history(
        history_csv: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training-history curves from the trainer's CSV file."""
        import pandas as pd
        df = pd.read_csv(history_csv)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        if "loss" in df.columns:
            axes[0].semilogy(df.index, df["loss"], lw=1.8, color="#1a1a2e")
        axes[0].set_title("Total loss"); axes[0].set_xlabel("Epoch")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)

        for col, color, label in [
            ("running_cost",  "#2166ac", "Running cost L"),
            ("terminal_cost", "#d6604d", "Terminal cost G"),
        ]:
            if col in df.columns:
                axes[1].semilogy(df.index, df[col].abs(), lw=1.8,
                                 color=color, label=label)
        axes[1].set_title("Cost decomposition"); axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=9); axes[1].grid(True, which="both", ls="--", alpha=0.4)

        for col, color, label in [
            ("chjb",    "#4dac26", "cHJB"),
            ("chjbfin", "#b8e186", "cHJBfin"),
        ]:
            if col in df.columns:
                axes[2].semilogy(df.index, df[col].abs() + 1e-12, lw=1.8,
                                 color=color, label=label)
        axes[2].set_title("HJB residuals"); axes[2].set_xlabel("Epoch")
        axes[2].legend(fontsize=9); axes[2].grid(True, which="both", ls="--", alpha=0.4)

        fig.suptitle("Training history — LiquidationPortfolio",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # plot_exact_trajectories
    # ------------------------------------------------------------------

    def plot_exact_trajectories(
        self,
        q0_values: Optional[list[float]] = None,
        S0: float = 1.0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot exact BVP trajectories for several initial inventories."""
        if not self._gamma2_available:
            raise ValueError("Exact plot only available for γ=2.")
        if q0_values is None:
            q0_values = [0.5, 1.0, 1.5]

        cmap = plt.cm.Blues(np.linspace(0.45, 0.9, len(q0_values)))
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        for i, q0 in enumerate(q0_values):
            traj = self._bvp_solver.solve(np.array([q0, S0, 0.0]))
            kw = dict(color=cmap[i], lw=self._LW, label=f"q₀={q0:.1f}")
            axs[0].plot(traj.t,       traj.z[:, 0], **kw)
            axs[1].plot(traj.t[:-1],  traj.u[:, 0], **kw)
            axs[2].plot(traj.t,       traj.z[:, 1], **kw)
            axs[3].plot(traj.t,       traj.z[:, 2], **kw)

        _label_ax(axs[0], "Inventory q(t)",       "t", "q")
        _label_ax(axs[1], "Trading rate u*(t)",   "t", "u*")
        _label_ax(axs[2], "Impacted price S(t)",  "t", "S")
        _label_ax(axs[3], "Accumulated cash X(t)","t", "X")
        for ax in axs:
            _add_legend(ax)
        fig.suptitle(f"Exact BVP solution — γ=2, S₀={S0}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test / standalone demo (no policy required)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from LiquidationPortfolio import LiquidationPortfolioOC

    print("── LiquidationBenchmark smoke test ──")
    prob = LiquidationPortfolioOC(
        batch_size=10, t_initial=0.0, t_final=2.0, nt=100,
        sigma=0.02, kappa=1e-4, eta=0.1, gamma=2.0,
        epsilon=1e-2, alpha=30, q0_min=0.5, q0_max=1.5,
    )
    bench = LiquidationBenchmark(prob)

    out_dir = benchmark_png_dir()
    print(f"   PNG output directory: {out_dir}")

    print("\n1. Plotting exact BVP trajectories …")
    path_exact = os.path.join(
        _results_dir("LiquidationPortfolioOC", "reference"),
        "exactbvp_reference.png",
    )
    bench.plot_exact_trajectories(
        q0_values=[0.5, 1.0, 1.5],
        save_path=path_exact,
    )
    print(f"   Saved: {path_exact}")

    print("\n2. Running gradient checks …")
    path_taylor = benchmark_png_path("taylor_vs_analytic_benchmark.png")
    bench.gradient_check(save_path=path_taylor)
    print(f"   Saved: {path_taylor}")

    t_arr, traj, u_arr = bench.solve_exact(q0=1.0, S0=1.0)
    print(f"\n3. BVP solution summary (q₀=1, S₀=1):")
    print(f"   q(T)   = {traj[0,-1]:.4f}  (target ≈ 0)")
    print(f"   X(T)   = {traj[2,-1]:.4f}")
    G_val = -traj[2, -1] + 30.0 * traj[0, -1] ** 2
    print(f"   G(T)   = {G_val:.4f}")
    print(f"   u mean = {u_arr.mean():.4f},  u std = {u_arr.std():.4f}")
    print("   (near-constant u confirms risk-neutral VWAP-like execution)")
