"""
examples.example_multibicycle
------------------------------
Train the JFB implicit policy on the multi-bicycle formation problem
(MultiBicycleOC). Set full_AD_mode=True for full autodiff comparison.
Results logged to results/MultiBicycleOC/training/.
"""
import torch
import numpy as np
from MultiBicycle import MultiBicycleOC
from ImplicitNets import ImplicitNetOC, ImplicitNetOC_MB, Phi
from OptimalControlTrainer import OptimalControlTrainer, LRScheduler
from CVXPolicy import CVXPolicy_MultiBicycle
from core.paths import results_dir
import time
import sys
import os
# torch.set_default_dtype(torch.float64)

class Logger(object):
    def __init__(self, filename="mb_run.log"):
        self.terminal = sys.stdout
        logdir = os.path.dirname(filename)
        if logdir and not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)  # Ensure directory exists
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(os.path.join(results_dir("MultiBicycleOC", "training"), "mb_run.log"))
def run_mb_jfb(config_oc, config_train, N, full_AD_mode=False, device='cpu', plot_frequency=None, load_prev_model=False, model_path=""):
    """
    Solves Quadcopter optimal control problem with INN + JFB
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
    print(mb_std.ic_var)
    mb_std.track_all_fp_iters = full_AD_mode
    Phi_Net = Phi(3, 100, mb_std.state_dim).to(device)
    total_params = sum(p.numel() for p in Phi_Net.parameters() if p.requires_grad)
    print(f"total params: {total_params}")
    angle_min = (-0.5*np.pi) + 0.25
    angle_max = (0.5*np.pi) - 0.25
    tracked_iters = 1
    max_itr = 1000-tracked_iters
    implicit_net = ImplicitNetOC_MB(mb_std.state_dim, mb_std.control_dim, alpha=5.0e-5, max_iters = max_itr, tol=0.1, 
            tracked_iters=tracked_iters, oc_problem=mb_std, use_control_limits=True, u_min=angle_min, u_max=angle_max, p_net=Phi_Net, dev=device, use_aa=False).to(device)

    if load_prev_model == True:
        # Load the trained model's state dictionary
        try:
            implicit_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Please check the path.")
            return
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Ensure the model architecture matches the saved state dictionary.")
            return
        implicit_net.train()# Set mode to train

    optimizer_std = torch.optim.Adam(implicit_net.parameters(), lr=config_train['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_std, mode='min', factor=0.5, patience=10, min_lr=1.0e-8)
    trainer_std = OptimalControlTrainer(implicit_net, mb_std, optimizer_std, scheduler=scheduler, device=device)
    trainer_std.set_mode('standard')
    save_model_name = f"best_policy_JFB_BatchSize_{config_oc['batch_size']}_nt_{config_oc['nt']}_alphaG_{config_oc['alphaG']:.3e}_initlr_{config_train['lr']:.3e}_trackeditrs_{tracked_iters}_{time.ctime().replace(' ','_').replace(':','_')}"
    save_model_name = save_model_name.replace(".","_")
    z0_std = mb_std.sample_initial_condition()
    trainer_std.train(z0_std, num_epochs=config_train['epochs'], plot_frequency=plot_frequency, save_model_name=save_model_name)

def run_mb_direct_transcription(N,config_oc, device):
    mb_oc = MultiBicycleOC(n_b=N, device=device, **config_oc)
    z0 = mb_oc.sample_initial_condition()
    u_true = torch.randn(config_oc['batch_size'], mb_oc.control_dim, config_oc['nt'], device=device)
    z_traj = mb_oc.generate_trajectory(u_true, z0, config_oc['nt'], return_full_trajectory=True)
    mb_oc.plot_position_trajectories(z_traj.detach(), u_true)
    u_true.requires_grad = True
    optimizer = torch.optim.Adam([u_true], lr=1.0e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=1.0e-9)
    max_iters = int(3000)
    for i in range(1, max_iters+1):
        optimizer.zero_grad()
        loss, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = mb_oc.compute_loss(u_true, z0)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        s_dict = optimizer.state_dict()
        s_dict['lr'] = scheduler.get_last_lr()[0]
        optimizer.load_state_dict(s_dict)

        grad_norm = torch.norm(u_true.grad)
        loss, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = mb_oc.compute_loss(u_true, z0)
        # print in %1.3e format
        print("iter %d | loss %.3e | grad %.3e | L %.3e | G %.3e | HJB %.3e | HJBfin %.3e | Adj %.3e | Adjfin %.3e | lr %.3e" % (i, loss.item(), grad_norm.item(), running_cost.item(), terminal_cost.item(), cHJB.item(), cHJBfin.item(), cadj.item(), cadjfin.item(), s_dict['lr']))

    # detach variables
    u_true = u_true.detach()
    z0 = z0.detach()

    # Plot trajectory
    z_traj = mb_oc.generate_trajectory(u_true, z0, config_oc['nt'], return_full_trajectory=True)
    save_path = os.path.join(
        results_dir("MultiBicycleOC", "rollouts"),
        f"mb_traj_{time.ctime().replace(' ','_').replace(':','_')}_direct_transcription.png",
    )
    mb_oc.plot_position_trajectories(z_traj.detach(), u_true, save_path=save_path)
    
def main():
    # Seed for reproducibility
    seed = 8005
    torch.manual_seed(seed)
    np.random.seed(seed)
    # nt=40 => dt=0.1 for t_final=4; richer trajectories for training/plots (nt=1 is only two points in time).
    config_mb_oc = {
        'batch_size': 1,
        'nt': 40,
        't_final': 4.0,
        'alpha_interaction': 5.0,
        'alphaG': 500.0,
        'alphaHJB': [1.0e-4, 1.0e-2],
        'pen_pos': True,
        'ic_var': 0.1,
    } #nt=40 alpha_interaction=25.0 alphaG=100.0
    config_mb_train = {'lr': 5.0e-4, 'epochs': 500}

    # Device selection with validation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU")
    N = 1 # Number of bicycles
    n_trials = 2

    #run_mb_direct_transcription(N, config_mb_oc, device)
    # JFB
    for n in np.arange(n_trials):
        run_mb_jfb(
            config_mb_oc,
            config_mb_train,
            N,
            full_AD_mode=False,
            device=device,
            plot_frequency=50,
            load_prev_model=False,
            model_path=None,
        )

if __name__ == "__main__":
    main()
