"""
core.OptimalControlTrainer
--------------------------
JFB training loop for `ImplicitNetOC` policies.

Each epoch: rollout, `.backward()` on the JFB surrogate, optimizer step.
Artifacts (checkpoint, loss curve, rollout figures, trajectory tensor) are
managed by a `RunIO` instance. Plotting delegates to `BenchmarkPlotter` for
models with `panels()` / `to_trajectory()`, with a legacy fallback for older
models.
"""

from __future__ import annotations

import os
import time
import copy

import torch
import torch.nn as nn
import pandas as pd
import psutil
from matplotlib import pyplot as plt

from Quadcopter import QuadcopterOC
from CVXPolicy import CVXPolicy_MC, CVXPolicy_Quadcopter
from ImplicitNets import ImplicitNetOC
## for optimal consumption example, use --
# from ImplicitNets import ImplicitNetOC_pos as ImplicitNetOC
from core.paths import results_dir
from core.run_io import RunIO
from core.log_format import EpochColourizer
from benchmarking import BenchmarkPlotter


class LRScheduler:
    """
    Custom LR scheduler class that is similar to PyTorch's
    ReduceLROnPlateau LR scheduler, but reduces the learning rate
    by some factor after a fixed number of consecutive epochs during 
    which there is no decrease in the loss function. ReduceLROnPlateau 
    reduces the learing rate only after a fixed number of epochs in which 
    the loss function is less than the best loss achieved up to that point
    """

    def __init__(self, optim, init_lr, min_lr, fact, pat):
        self.optimizer = optim
        self.initial_lr = init_lr
        self.min_lr = min_lr
        self.factor = fact
        self.patience = pat
        self.current_lr = init_lr
        self.num_no_decr = 0
        self.prev_loss = float('inf')
        self.current_epoch = 0

    def get_initial_lr(self):
        return self.initial_lr

    def get_current_lr(self):
        return self.current_lr

    def get_current_epoch(self):
        return self.current_epoch

    def get_prev_loss(self):
        return self.get_prev_loss

    def step(self, new_loss):
        self.current_epoch += 1

        if new_loss < self.prev_loss:
            self.num_no_decr = 0
        else:
            self.num_no_decr += 1
        self.prev_loss = new_loss

        if (self.num_no_decr > self.patience) and (self.current_epoch != 1):
            self.current_lr *= self.factor

            if self.current_lr < self.min_lr:
                self.current_lr = self.min_lr

            for g in self.optimizer.param_groups:
                g['lr'] = self.current_lr

            self.num_no_decr = 0


class OptimalControlTrainer:
    def __init__(
        self,
        policy_net,
        oc_problem,
        optimizer,
        scheduler=None,
        ver=False,
        device='cpu',
        tag: str = "JFB",
        run_io: RunIO | None = None,
    ):
        self.policy = policy_net
        self.oc_problem = oc_problem
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mode = 'standard' # Default mode
        self.verify = ver # Whether or not to do additional computations to verify theoretical assumptions, default is False
        # Gradient clipping for direct control methods
        self.enable_grad_clip = False
        self.grad_clip_value = 1.0

        self.run_io = run_io or RunIO(
            problem_cls_name=type(oc_problem).__name__,
            tag=tag,
        )

        self.history = {k: [] for k in [
            'epoch', 'loss', 'running_cost', 'terminal_cost', 
            'cHJB', 'cHJBfin', 'cadj', 'cadjfin',
            'time_per_epoch', 'grad_norm', 'lr', 'max_fp_itrs',
            'max_fp_res_norm', 'memory_MB', 'max_memory_MB',
            'gpu_memory_MB', 'gpu_max_memory_MB', 'work_units',
            'max_grad_H', 'avg_grad_H', 'smallest_M_sdval', 
            'largest_M_sdval', 'smallest_lambda_min', 'largest_lambda_max',
            'max_grad_T_u', 'avg_grad_T_u', 'sd_grad_T_u', 'angle'
        ]}

    def set_mode(self, mode='standard'):
        if mode not in ['standard', 'cvx']:
            raise ValueError("Mode must be 'standard' or 'cvx'")
        self.mode = mode
        print(f"Trainer mode set to '{self.mode}'")

    def standard_step(self, z0):
        self.policy.train()
        max_fp_itrs = 0.0
        max_fp_res_norm = 0.0

        if self.mode == 'standard':
            convergence_stats = self.policy.get_convergence_stats()
            max_fp_itrs = convergence_stats['fp_depth']
            max_fp_res_norm = convergence_stats['max_res_norm']

        self.optimizer.zero_grad()
        # === CHANGE: Unpack all 7 values from compute_loss ===
        total_cost, run_cost, term_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = self.oc_problem.compute_loss(self.policy, z0)
        total_cost.backward()
        if self.enable_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_value)
        self.optimizer.step()
        # === CHANGE: Return all cost components ===
        return {
            'loss': total_cost.item(), 
            'running_cost': run_cost.item(), 
            'terminal_cost': term_cost.item(),
            'cHJB': cHJB.item(),
            'cHJBfin': cHJBfin.item(),
            'cadj': cadj.item(),
            'cadjfin': cadjfin.item(),
            'max_fp_itrs': max_fp_itrs,
            'max_fp_res_norm': max_fp_res_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'max_grad_H': max_grad_H,
            'avg_grad_H': avg_grad_H
        }
    
    def standard_step_verify(self, z0):
        self.policy.train()
        max_fp_itrs = 0.0
        max_fp_res_norm = 0.0
        angle = 0.0
        
        if self.mode == 'standard':
            convergence_stats = self.policy.get_convergence_stats()
            max_fp_itrs = convergence_stats['fp_depth']
            max_fp_res_norm = convergence_stats['max_res_norm']

        self.optimizer.zero_grad()

        # Copy weights to compute gradient using full AD as well as JFB for angle computation
        #policy_AD = copy.deepcopy(self.policy) 
        # === CHANGE: Unpack all values from compute_loss ===
        total_cost, run_cost, term_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H, smallest_M_sdval, largest_M_sdval, smallest_lambda_min, largest_lambda_max, max_grad_T_u, avg_grad_T_u, sd_grad_T_u = self.oc_problem.compute_loss_verify(self.policy, z0)
        #og_full_AD = self.oc_problem.track_all_fp_iters
        #self.oc_problem.track_all_fp_iters = True
        #total_cost_ad, run_cost_ad, term_cost_ad, cHJB_ad, cHJBfin_ad, cadj_ad, cadjfin_ad, max_grad_H_ad, avg_grad_H_ad = self.oc_problem.compute_loss(policy_AD, z0)
        #self.oc_problem.track_all_fp_iters = og_full_AD
        total_cost.backward()
        if self.enable_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_value)
        #total_cost_ad.backward()
        """
        nabla_theta_J_ = []
        for p_ad in policy_AD.parameters():
            if p_ad.grad is not None:
                nabla_theta_J_.append(p_ad.grad.view(-1))
        nabla_theta_J = torch.cat(nabla_theta_J_)
        d_JFB_ = []
        for p in self.policy.parameters():
            if p.grad is not None:
                d_JFB_.append(p.grad.view(-1))
        d_JFB = torch.cat(d_JFB_)
        angle = torch.acos(torch.dot(nabla_theta_J, d_JFB)/(torch.linalg.norm(nabla_theta_J, ord=2)*torch.linalg.norm(d_JFB, ord=2))).item()
        print(f"angle between true gradient and JFB approximation: {angle:.4e}")
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        """
        self.optimizer.step()
        # === CHANGE: Return all cost components ===
        return {
            'loss': total_cost.item(),
            'running_cost': run_cost.item(),
            'terminal_cost': term_cost.item(),
            'cHJB': cHJB.item(),
            'cHJBfin': cHJBfin.item(),
            'cadj': cadj.item(),
            'cadjfin': cadjfin.item(),
            'max_fp_itrs': max_fp_itrs,
            'max_fp_res_norm': max_fp_res_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'max_grad_H': max_grad_H,
            'avg_grad_H': avg_grad_H,
            'smallest_M_sdval': smallest_M_sdval,
            'largest_M_sdval': largest_M_sdval,
            'smallest_lambda_min': smallest_lambda_min,
            'largest_lambda_max': largest_lambda_max,
            'max_grad_T_u': max_grad_T_u,
            'avg_grad_T_u': avg_grad_T_u,
            'sd_grad_T_u': sd_grad_T_u,
            'angle': angle
        }

    def cvx_step(self, z0):
        # The logic is identical for CVX mode, but we keep it separate for clarity
        return self.standard_step(z0)

    def train_epoch(self, z0):
        if self.mode == 'cvx':
            return self.cvx_step(z0)
        elif self.mode == 'standard' and self.verify: # verify assumptions
            return self.standard_step_verify(z0)
        else: # standard mode
            return self.standard_step(z0)

    # ------------------------------------------------------------------
    # Plotting dispatch
    # ------------------------------------------------------------------
    def _has_benchmark_plotter_api(self) -> bool:
        return hasattr(self.oc_problem, "panels") and hasattr(self.oc_problem, "to_trajectory")

    def _plot_rollout(self, z_traj: torch.Tensor, save_path: str) -> None:
        """Dispatch a rollout figure to whichever plotting API the model exposes."""
        if self._has_benchmark_plotter_api():
            traj = self.oc_problem.to_trajectory(z_traj.detach(), self.policy)
            BenchmarkPlotter(self.oc_problem.panels()).plot([traj], save_path=save_path)
        else:
            # Legacy path for models that have not yet been migrated
            # (Quadcopter, MultiBicycle): their plot_position_trajectories
            # accepts (z_traj, ..., save_path=...).
            self.oc_problem.plot_position_trajectories(z_traj.detach(), save_path=save_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, z0, num_epochs, verbose=True, plot_frequency=25):
        save_path = self.run_io.policy_path()
        history_path = self.run_io.history_path()
        print(f"Starting training in '{self.mode}' mode for {num_epochs} epochs.")
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
        best_loss = float('inf')

        process = psutil.Process(os.getpid())
        colour = EpochColourizer(history_window=10)
        for epoch in range(1, num_epochs + 1):
            gpu_memory_MB = 0.0
            gpu_max_memory_MB = 0.0
            max_memory_MB = 0.0
            epoch_start_time = time.time()
            step_info = self.train_epoch(z0)
            self.scheduler.step(step_info['loss'])

            memory_MB = process.memory_info().rss / 1024 / 1024
            if memory_MB > max_memory_MB:
                max_memory_MB = memory_MB
            if torch.cuda.is_available():
                gpu_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
            time_per_epoch = time.time() - epoch_start_time

            grad_norm = sum(p.grad.norm().item()**2 for p in self.policy.parameters() if p.grad is not None)**0.5

            work_units = self.oc_problem.batch_size * self.policy.tracked_iters
            if (step_info['max_fp_itrs'] < self.policy.tracked_iters) or self.oc_problem.track_all_fp_iters:
                work_units = self.oc_problem.batch_size * step_info['max_fp_itrs']

            for key in self.history:
                if key == 'memory_MB':
                    self.history[key].append(memory_MB)
                elif key == 'max_memory_MB':
                    self.history[key].append(max_memory_MB)
                elif key == 'gpu_memory_MB':
                    self.history[key].append(gpu_memory_MB)
                elif key == 'gpu_max_memory_MB':
                    self.history[key].append(gpu_max_memory_MB)
                elif key == 'work_units':
                    self.history[key].append(work_units)
                else:
                    self.history[key].append(locals().get(key, step_info.get(key, 0)))

            # === Colourised per-epoch log ===
            # Each numeric field is wrapped in an ANSI escape determined
            # by `EpochColourizer`. Rolling-history rules use the values
            # *before* the current epoch is appended -- update_history()
            # is called at the very end so the current epoch is judged
            # against the previous K epochs.
            if verbose:
                fp_cap = getattr(self.policy, "max_iters", 0)
                fp_tol = getattr(self.policy, "tol", 1e-4)
                fp_alpha = getattr(self.policy, "alpha", 1e-3)
                line = (
                    f"{colour.epoch(epoch)} | "
                    f"Loss: {colour.loss(step_info['loss'])} | "
                    f"L: {step_info['running_cost']:.3e} | "
                    f"G: {step_info['terminal_cost']:.3e} | "
                    f"HJB: {colour.cHJB(step_info.get('cHJB', 0.0))} | "
                    f"HJB fin: {step_info.get('cHJBfin', 0.0):.3e} | "
                    f"Adj: {colour.cadj(step_info.get('cadj', 0.0))} | "
                    f"Grad: {colour.grad_norm(grad_norm)} | "
                    f"Time: {colour.time(time_per_epoch)}s | "
                    f"CPU Mem: {memory_MB:.1f}MB | Max CPU: {max_memory_MB:.1f}MB | "
                    f"GPU Mem: {gpu_memory_MB:.1f}MB | Max GPU: {gpu_max_memory_MB:.1f}MB | "
                    f"lr: {colour.lr(step_info['lr'])} | "
                    f"max_fp_itrs: {colour.fp_itrs(step_info['max_fp_itrs'], fp_cap)} | "
                    f"res_norm: {colour.res_norm(step_info['max_fp_res_norm'], fp_tol)} | "
                    f"max_grad_H: {colour.max_grad_H(step_info['max_grad_H'], fp_alpha)} | "
                    f"avg_grad_H: {step_info['avg_grad_H']:.3e}\n"
                )
                print(line)
            colour.update_history(
                loss=step_info['loss'],
                grad_norm=grad_norm,
                time_per_epoch=time_per_epoch,
                lr=step_info['lr'],
                cadj=step_info.get('cadj', None),
                cHJB=step_info.get('cHJB', None),
            )

            if step_info['loss'] < best_loss:
                best_loss = step_info['loss']
                torch.save(self.policy.state_dict(), save_path)
                if verbose:
                    print(f"    -> New best model saved to {save_path} with loss {best_loss:.4e}")

            if plot_frequency and epoch % plot_frequency == 0:
                z_traj = self.oc_problem.generate_trajectory(
                    self.policy, z0, self.oc_problem.nt, return_full_trajectory=True,
                )
                self._plot_rollout(z_traj, save_path=self.run_io.training_plot_path(epoch))

            # === Save CSV after each epoch ===
            pd.DataFrame(self.history).to_csv(history_path, index=False)

        self._finalize(best_loss=best_loss, verbose=verbose)
        return self.history

    # ------------------------------------------------------------------
    # Post-training artifacts
    # ------------------------------------------------------------------
    def _finalize(self, best_loss: float, verbose: bool = True) -> None:
        """Reload the best policy, write the loss curve, the final rollout
        figure and the saved trajectory tensor."""
        if verbose:
            print("-" * 60)
            print("Finalizing run: writing loss curve, final rollout figure, "
                  "and saving the rolled-out trajectory.")

        # 1) Loss curve over the whole run.
        self.plot_loss_curve(self.run_io.loss_curve_path())

        # 2) Reload best checkpoint so the final artifacts reflect best loss.
        ckpt_path = self.run_io.policy_path()
        if os.path.isfile(ckpt_path):
            self.policy.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True),
            )
            if verbose:
                print(f"  reloaded best checkpoint (loss={best_loss:.4e}) from {ckpt_path}")
        self.policy.eval()

        # 3) Roll out the best policy on a fresh test IC.
        with torch.no_grad():
            z0_test = self.oc_problem.sample_initial_condition()
            z_traj = self.oc_problem.generate_trajectory(
                self.policy, z0_test, self.oc_problem.nt, return_full_trajectory=True,
            )

        # 4) Final rollout figure -> rollouts/.
        self._plot_rollout(z_traj, save_path=self.run_io.rollout_path())
        if verbose:
            print(f"  wrote rollout figure -> {self.run_io.rollout_path()}")

        # 5) Saved trajectory tensor for later replay / analysis -> rollouts/.
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

    def plot_loss_curve(self, save_path: str) -> None:
        """Render the loss curve for the current run to ``save_path``."""
        if not self.history.get("epoch"):
            return
        fig = plt.figure(figsize=(10, 6))
        plt.yscale('log')
        plt.grid(True)
        plt.plot(self.history['epoch'], self.history['loss'], label='Total Loss')
        plt.title(f'Training Loss (Mode: {self.mode}, run: {self.run_io.run_id})')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
