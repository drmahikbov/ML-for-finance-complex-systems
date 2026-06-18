"""
core_RL.OptimalControlTrainer_RL
---------------------------------
Subclass of `OptimalControlTrainer` for the RL pipeline.

Owns an `Environment` and a `JacobianEstimator` in addition to the policy and
optimizer. Each epoch calls `compute_loss_RL`, backpropagates the surrogate,
and decays the exploration noise. Plotting routes through `env.rollout` rather
than the analytical `compute_f`. The post-training artifact bundle (RunIO, loss
curves, checkpoint, trajectory) is fully inherited.
"""

from __future__ import annotations

import os
import time

import torch
import pandas as pd
import psutil

from core.OptimalControlTrainer import OptimalControlTrainer
from core.log_format import EpochColourizer
from benchmarking import BenchmarkPlotter

from core_RL.Environment import Environment
from core_RL.JacobianEstimator import JacobianEstimator


class OptimalControlTrainer_RL(OptimalControlTrainer):
    """RL-flavoured trainer.

    Parameters
    ----------
    policy_net : :class:`ImplicitNetOC_RL`
    oc_problem : :class:`ImplicitOC_RL`
    env        : :class:`Environment`
    jac_est    : :class:`JacobianEstimator`
    optimizer  : torch optimizer
    scheduler  : LR scheduler with ``get_last_lr()`` (e.g.
                 ``torch.optim.lr_scheduler.ReduceLROnPlateau``).
    Other args same as the parent: ``device``, ``tag``, ``run_io``.
    """

    def __init__(
        self,
        policy_net,
        oc_problem,
        env: Environment,
        jac_est: JacobianEstimator,
        optimizer,
        scheduler=None,
        ver: bool = False,
        device: str = "cpu",
        tag: str = "JFB-RL",
        run_io=None,
        exploration_std: float = 0.5,
        exploration_decay: float = 0.99,
    ):
        super().__init__(
            policy_net=policy_net,
            oc_problem=oc_problem,
            optimizer=optimizer,
            scheduler=scheduler,
            ver=ver,
            device=device,
            tag=tag,
            run_io=run_io,
        )
        self.env = env
        self.jac_est = jac_est
        self.mode = "rl"

        # Exploration schedule: start with wide Gaussian noise on u_k during
        # rollout so the RLS estimator sees control variation and learns b_k.
        # Decayed multiplicatively each epoch so that late training is clean.
        self.exploration_std = exploration_std
        self.exploration_decay = exploration_decay

        # Extend the history schema with RL-specific diagnostics.
        for k in ("lin_residual", "exploration_std"):
            if k not in self.history:
                self.history[k] = []

    # The single training step                                           #
    def rl_step(self, z0: torch.Tensor) -> dict:
        """One epoch's worth of forward rollout + backward adjoint +
        JFB-surrogate optimisation.
        """
        self.policy.train()
        self.optimizer.zero_grad()

        out = self.oc_problem.compute_loss_RL(
            policy=self.policy,
            env=self.env,
            jac_est=self.jac_est,
            z0=z0,
            exploration_std=self.exploration_std,
        )
        surrogate = out["surrogate"]

        # The surrogate's gradient w.r.t. θ equals the JFB-with-estimates
        # gradient of J. ``backward`` populates param.grad.
        surrogate.backward()

        if self.enable_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.grad_clip_value
            )
        self.optimizer.step()

        # Ask the policy for FP-convergence stats from the most recent
        # forward pass (for diagnostics — same field the parent populates).
        conv = self.policy.get_convergence_stats()

        return {
            "loss": out["total_cost"],
            "running_cost": out["running_cost"],
            "terminal_cost": out["terminal_cost"],
            # The HJB / adjoint consistency penalties don't apply here —
            # they require analytical f. Keep the keys for CSV alignment.
            "cHJB": 0.0,
            "cHJBfin": 0.0,
            "cadj": 0.0,
            "cadjfin": 0.0,
            "max_fp_itrs": conv["fp_depth"],
            "max_fp_res_norm": conv["max_res_norm"],
            "lr": self.scheduler.get_last_lr()[0] if self.scheduler else float("nan"),
            # Hamiltonian-gradient diagnostics that require f are zeroed.
            # We keep these keys so the existing CSV / colourizer keep working.
            "max_grad_H": 0.0,
            "avg_grad_H": 0.0,
            # New RL diagnostic.
            "lin_residual": out["lin_residual"],
        }

    # The parent dispatches via ``train_epoch``. We override it to point
    # at our own step.
    def train_epoch(self, z0: torch.Tensor) -> dict:
        return self.rl_step(z0)

    # Plotting dispatch — go through env, not compute_f                  #
    def _plot_rollout(self, z_traj: torch.Tensor, save_path: str) -> None:
        """Same panels API as the parent, but the trajectory was produced
        by ``env.rollout`` instead of an analytical Euler march.
        """
        if self._has_benchmark_plotter_api():
            traj = self.oc_problem.to_trajectory(z_traj.detach(), self.policy)
            BenchmarkPlotter(self.oc_problem.panels()).plot([traj], save_path=save_path)
        else:
            # Legacy fallback: the model's bespoke plot_position_trajectories.
            self.oc_problem.plot_position_trajectories(
                z_traj.detach(), save_path=save_path
            )

    # Override the part of ``train`` that calls ``generate_trajectory``  #
    # The parent's ``train`` and ``_finalize`` methods invoke
    # ``self.oc_problem.generate_trajectory(self.policy, z0, self.oc_problem.nt,
    # return_full_trajectory=True)`` — which in core/ uses compute_f. Our
    # ImplicitOC_RL.generate_trajectory takes an extra ``env`` arg, so we
    # adapt by replacing those two call sites. We re-implement ``train``
    # only minimally; everything else is inherited.
    def train(self, z0, num_epochs, verbose=True, plot_frequency=25):
        save_path = self.run_io.policy_path()
        history_path = self.run_io.history_path()
        print(f"Starting RL training in '{self.mode}' mode for {num_epochs} epochs.")
        print(f"  run_id      : {self.run_io.run_id}")
        print(f"  output root : {self.run_io.train_dir}")
        if verbose:
            print("-" * 60)
            print(EpochColourizer.legend(
                fp_max_iters=getattr(self.policy, "max_iters", 0),
                fp_tol=getattr(self.policy, "tol", 1e-4),
                fp_alpha=getattr(self.policy, "alpha", 1e-3),
            ))
        print("-" * 60)
        best_loss = float("inf")

        process = psutil.Process(os.getpid())
        colour = EpochColourizer(history_window=10)
        for epoch in range(1, num_epochs + 1):
            gpu_memory_MB = 0.0
            gpu_max_memory_MB = 0.0
            max_memory_MB = 0.0
            epoch_start_time = time.time()
            step_info = self.train_epoch(z0)
            step_info["exploration_std"] = self.exploration_std
            self.exploration_std *= self.exploration_decay
            if self.scheduler is not None:
                self.scheduler.step(step_info["loss"])

            memory_MB = process.memory_info().rss / 1024 / 1024
            if memory_MB > max_memory_MB:
                max_memory_MB = memory_MB
            if torch.cuda.is_available():
                gpu_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
            time_per_epoch = time.time() - epoch_start_time

            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in self.policy.parameters()
                if p.grad is not None
            ) ** 0.5

            work_units = self.oc_problem.batch_size * self.policy.tracked_iters
            if (
                step_info["max_fp_itrs"] < self.policy.tracked_iters
                or getattr(self.oc_problem, "track_all_fp_iters", False)
            ):
                work_units = self.oc_problem.batch_size * step_info["max_fp_itrs"]

            for key in self.history:
                if key == "memory_MB":
                    self.history[key].append(memory_MB)
                elif key == "max_memory_MB":
                    self.history[key].append(max_memory_MB)
                elif key == "gpu_memory_MB":
                    self.history[key].append(gpu_memory_MB)
                elif key == "gpu_max_memory_MB":
                    self.history[key].append(gpu_max_memory_MB)
                elif key == "work_units":
                    self.history[key].append(work_units)
                else:
                    self.history[key].append(
                        locals().get(key, step_info.get(key, 0))
                    )

            if verbose:
                fp_cap = getattr(self.policy, "max_iters", 0)
                fp_tol = getattr(self.policy, "tol", 1e-4)
                fp_alpha = getattr(self.policy, "alpha", 1e-3)
                line = (
                    f"{colour.epoch(epoch)} | "
                    f"Loss: {colour.loss(step_info['loss'])} | "
                    f"L: {step_info['running_cost']:.3e} | "
                    f"G: {step_info['terminal_cost']:.3e} | "
                    f"linRes: {step_info['lin_residual']:.3e} | "
                    f"Grad: {colour.grad_norm(grad_norm)} | "
                    f"Time: {colour.time(time_per_epoch)}s | "
                    f"CPU Mem: {memory_MB:.1f}MB | "
                    f"GPU Mem: {gpu_memory_MB:.1f}MB | "
                    f"lr: {colour.lr(step_info['lr'])} | "
                    f"max_fp_itrs: {colour.fp_itrs(step_info['max_fp_itrs'], fp_cap)} | "
                    f"res_norm: {colour.res_norm(step_info['max_fp_res_norm'], fp_tol)}\n"
                )
                print(line)
            colour.update_history(
                loss=step_info["loss"],
                grad_norm=grad_norm,
                time_per_epoch=time_per_epoch,
                lr=step_info["lr"],
                cadj=None,
                cHJB=None,
            )

            if step_info["loss"] < best_loss:
                best_loss = step_info["loss"]
                torch.save(self.policy.state_dict(), save_path)
                if verbose:
                    print(
                        f"    -> New best model saved to {save_path} "
                        f"with loss {best_loss:.4e}"
                    )

            if plot_frequency and epoch % plot_frequency == 0:
                # IMPORTANT: rollout via env, not analytical compute_f.
                with torch.no_grad():
                    # Make sure the policy is using the latest b_k along the
                    # rollout — wire up a small setter callback.
                    def _setter(k):
                        _, b_k = self.jac_est.AB(k)
                        self.policy.set_step_jacobian(b_k)

                    z_traj, _ = self.env.rollout(
                        self.policy, z0, jac_setter=_setter, return_full_trajectory=True
                    )
                self._plot_rollout(
                    z_traj, save_path=self.run_io.training_plot_path(epoch)
                )

            pd.DataFrame(self.history).to_csv(history_path, index=False)

        self._finalize_rl(best_loss=best_loss, verbose=verbose, z0_for_rollout=z0)
        return self.history

    # Override _finalize to use env rollout instead of compute_f         #
    def _finalize_rl(
        self, best_loss: float, verbose: bool, z0_for_rollout: torch.Tensor
    ) -> None:
        if verbose:
            print("-" * 60)
            print("Finalising RL run: writing loss curve, final rollout figure, "
                  "and saving the rolled-out trajectory.")

        self.plot_loss_curve(self.run_io.loss_curve_path())

        ckpt_path = self.run_io.policy_path()
        if os.path.isfile(ckpt_path):
            self.policy.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True),
            )
            if verbose:
                print(
                    f"  reloaded best checkpoint (loss={best_loss:.4e}) "
                    f"from {ckpt_path}"
                )
        self.policy.eval()

        # Test rollout on a fresh IC.
        with torch.no_grad():
            z0_test = self.oc_problem.sample_initial_condition()

            def _setter(k):
                _, b_k = self.jac_est.AB(k)
                self.policy.set_step_jacobian(b_k)

            z_traj, _ = self.env.rollout(
                self.policy, z0_test, jac_setter=_setter, return_full_trajectory=True
            )

        self._plot_rollout(z_traj, save_path=self.run_io.rollout_path())
        if verbose:
            print(f"  wrote rollout figure -> {self.run_io.rollout_path()}")

        torch.save(
            {
                "z_traj": z_traj.detach().cpu(),
                "z0": z0_test.detach().cpu(),
                "run_id": self.run_io.run_id,
                "tag": self.run_io.tag,
                "problem_cls_name": self.run_io.problem_cls_name,
            },
            self.run_io.trajectory_path(),
        )
        if verbose:
            print(f"  wrote trajectory tensor -> {self.run_io.trajectory_path()}")