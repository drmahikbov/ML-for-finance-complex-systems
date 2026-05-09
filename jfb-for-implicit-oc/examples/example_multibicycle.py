"""
Multi-bicycle optimal control: train the JFB policy and let the trainer
write every artifact to ``results/MultiBicycleOC/``.

Mirrors the structure of ``example_liquidationportfolio.py``:
arv
1. instantiate :class:`MultiBicycleOC` with concrete parameters,
2. wire up :class:`ImplicitNetOC_MB` + optimizer + scheduler,
3. hand them to :class:`OptimalControlTrainer` with a ``tag``; the
   trainer's :class:`core.run_io.RunIO` decides every output filename.

All path / filename reasoning lives in :mod:`core.run_io` and
:func:`core.paths.results_dir`. The standard six-artifact bundle
(best_policy / history / loss_curve / training-plots / policy_rollout /
trajectory) ends up under ``results/MultiBicycleOC/`` automatically.
"""

import os
import sys
import time

import numpy as np
import torch

# Make the reorganised package importable when running this file directly:
# core/ and models/ use flat imports (e.g. `from ImplicitNets import ...`),
# so they need to be on sys.path themselves; the project root is needed for
# `core.paths`.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../jfb-for-implicit-oc/examples
_ROOT = os.path.dirname(_HERE)                               # .../jfb-for-implicit-oc
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MultiBicycle import MultiBicycleOC                       # models/
from ImplicitNets import ImplicitNetOC_MB, Phi                # core/
from OptimalControlTrainer import OptimalControlTrainer       # core/
from core.paths import results_dir
# torch.set_default_dtype(torch.float64)


EXPERIMENT_TAG_SUFFIX = "example-run"


def run_mb_jfb(
    config_oc,
    config_train,
    N,
    full_AD_mode=False,
    device="cpu",
    plot_frequency=None,
    load_prev_model=False,
    model_path=None,
    tag_suffix=EXPERIMENT_TAG_SUFFIX,
):
    """Solve the Multi-Bicycle optimal control problem with INN + JFB.

    All artifacts are routed through :class:`core.run_io.RunIO` -> the
    trainer infers the output directory from ``type(oc_problem).__name__``,
    i.e. ``results/MultiBicycleOC/``.
    """
    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############        MultiBicycle OC with INN        ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()
    print(f"Full AD Mode: {full_AD_mode}")
    print()

    mb_std = MultiBicycleOC(n_b=N, device=device, **config_oc)
    print(f"ic_var = {mb_std.ic_var}")
    mb_std.track_all_fp_iters = full_AD_mode

    Phi_Net = Phi(3, 100, mb_std.state_dim).to(device)
    total_params = sum(p.numel() for p in Phi_Net.parameters() if p.requires_grad)
    print(f"total params: {total_params}")

    # Steering-angle limits for the INN's projected fixed-point iteration.
    angle_min = (-0.5 * np.pi) + 0.25
    angle_max = (0.5 * np.pi) - 0.25
    tracked_iters = 1
    max_itr = 1000 - tracked_iters

    implicit_net = ImplicitNetOC_MB(
        mb_std.state_dim, mb_std.control_dim,
        alpha=5.0e-5, max_iters=max_itr, tol=0.1,
        tracked_iters=tracked_iters,
        oc_problem=mb_std,
        use_control_limits=True, u_min=angle_min, u_max=angle_max,
        p_net=Phi_Net, dev=device, use_aa=False,
    ).to(device)

    if load_prev_model:
        if not model_path:
            print("load_prev_model=True but no model_path provided; skipping load.")
        else:
            try:
                implicit_net.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )
                print(f"Model loaded successfully from {model_path}")
            except FileNotFoundError:
                print(f"Error: Model file not found at {model_path}.")
                return
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Ensure the model architecture matches the saved state dictionary.")
                return
            implicit_net.train()

    optimizer_std = torch.optim.Adam(implicit_net.parameters(), lr=config_train["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_std, mode="min", factor=0.5, patience=10, min_lr=1.0e-8,
    )

    tag_prefix = "FullAD" if full_AD_mode else "JFB"
    tag = (
        f"{tag_prefix}_N{N}_bs{config_oc['batch_size']}_nt{config_oc['nt']}"
        f"_aG{config_oc['alphaG']:.0e}_lr{config_train['lr']:.0e}"
        f"_{tag_suffix}"
    ).replace(".", "_")

    trainer_std = OptimalControlTrainer(
        implicit_net, mb_std, optimizer_std,
        scheduler=scheduler, device=device, tag=tag,ver=True,
    )
    trainer_std.set_mode("standard")

    z0_std = mb_std.sample_initial_condition()
    trainer_std.train(
        z0_std,
        num_epochs=config_train["epochs"],
        plot_frequency=plot_frequency,
    )
    return trainer_std


def run_mb_direct_transcription(N, config_oc, device):
    """Legacy direct-transcription baseline: optimise the control sequence
    directly. Output figure lands under ``results/MultiBicycleOC/rollouts/``."""
    mb_oc = MultiBicycleOC(n_b=N, device=device, **config_oc)
    z0 = mb_oc.sample_initial_condition()
    u_true = torch.randn(
        config_oc["batch_size"], mb_oc.control_dim, config_oc["nt"], device=device,
    )
    z_traj = mb_oc.generate_trajectory(u_true, z0, config_oc["nt"], return_full_trajectory=True)
    mb_oc.plot_position_trajectories(z_traj.detach())

    u_true.requires_grad = True
    optimizer = torch.optim.Adam([u_true], lr=1.0e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1.0e-9,
    )
    max_iters = int(3000)
    for i in range(1, max_iters + 1):
        optimizer.zero_grad()
        loss, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = (
            mb_oc.compute_loss(u_true, z0)
        )
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        s_dict = optimizer.state_dict()
        s_dict["lr"] = scheduler.get_last_lr()[0]
        optimizer.load_state_dict(s_dict)

        grad_norm = torch.norm(u_true.grad)
        loss, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = (
            mb_oc.compute_loss(u_true, z0)
        )
        print(
            "iter %d | loss %.3e | grad %.3e | L %.3e | G %.3e | HJB %.3e | "
            "HJBfin %.3e | Adj %.3e | Adjfin %.3e | lr %.3e"
            % (
                i, loss.item(), grad_norm.item(), running_cost.item(),
                terminal_cost.item(), cHJB.item(), cHJBfin.item(),
                cadj.item(), cadjfin.item(), s_dict["lr"],
            )
        )

    u_true = u_true.detach()
    z0 = z0.detach()

    z_traj = mb_oc.generate_trajectory(u_true, z0, config_oc["nt"], return_full_trajectory=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        results_dir("MultiBicycleOC", "rollouts"),
        f"mb_traj_direct_transcription_{timestamp}.png",
    )
    mb_oc.plot_position_trajectories(z_traj.detach(), save_path=save_path)


def main():
    seed = 8005
    torch.manual_seed(seed)
    np.random.seed(seed)

    # nt=40 => dt=0.1 for t_final=4; richer trajectories for training/plots.
    config_mb_oc = {
        "batch_size": 1,
        "nt": 40,
        "t_final": 4.0,
        "alpha_interaction": 5.0,
        "alphaG": 500.0,
        "alphaHJB": [1.0e-4, 1.0e-2],
        "pen_pos": True,
        "ic_var": 0.1,
    }
    config_mb_train = {"lr": 5.0e-4, "epochs": 150}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU")
    N = 1            # number of bicycles
    n_trials = 2

    # run_mb_direct_transcription(N, config_mb_oc, device)
    for trial in range(n_trials):
        run_mb_jfb(
            config_mb_oc,
            config_mb_train,
            N,
            full_AD_mode=False,
            device=device,
            plot_frequency=50,
            load_prev_model=False,
            model_path=None,
            tag_suffix=f"{EXPERIMENT_TAG_SUFFIX}_trial{trial}",
        )


if __name__ == "__main__":
    main()
