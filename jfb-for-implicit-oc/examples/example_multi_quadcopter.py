"""
examples.example_multi_quadcopter
-----------------------------------
Train the JFB implicit policy on the multi-quadcopter formation problem
(MultiQuadcopterOC). Set full_AD_mode=True for full autodiff comparison.
Results logged to results/MultiQuadcopterOC/training/.
"""
import torch
import numpy as np
import argparse
from Quadcopter import QuadcopterOC, MultiQuadcopterOC
from ImplicitNets import ImplicitNetOC, Phi
from OptimalControlTrainer import OptimalControlTrainer
from CVXPolicy import CVXPolicy_MultiQuadcopter
from core.paths import results_dir
import time
import sys, os
#torch.set_default_dtype(torch.float64)

class Logger(object):
    def __init__(self, filename="multi_quadcopter_run.log"):
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

sys.stdout = Logger(os.path.join(results_dir("MultiQuadcopterOC", "training"), "multi_quadcopter_run.log"))
def run_quadcopter_jfb(config_oc, config_train, full_AD_mode=False, device='cpu', plot_frequency=100):
    """
    Solves Quadcopter optimal control problem with INN + JFB
    """
    print()
    print("##################################################################")
    print("##############                                      ##############")
    print("##############        Quadcopter OC with INN        ##############")
    print("##############                                      ##############")
    print("##################################################################")
    print()
    print(f"Full AD Mode: {full_AD_mode}")
    print()
    copter_std = MultiQuadcopterOC(device=device, **config_oc)
    print(f"Running high-dimensional experiment with {copter_std.num_agents} quadcopters.")
    print(f"State Dimension: {copter_std.state_dim}, Control Dimension: {copter_std.control_dim}")
    tracked_iters=1
    copter_std.track_all_fp_iters = full_AD_mode
    Phi_Net = Phi(3, 96, copter_std.state_dim)
    implicit_net = ImplicitNetOC(copter_std.state_dim, copter_std.control_dim, max_iters = int(500-tracked_iters), alpha=1.0e-4, tol=1.0e-1, tracked_iters=tracked_iters, oc_problem=copter_std, p_net=Phi_Net, dev=device).to(device)
    optimizer_std = torch.optim.Adam(implicit_net.parameters(), lr=config_train['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_std, mode='min', factor=0.5, patience=10)
    trainer_std = OptimalControlTrainer(implicit_net, copter_std, optimizer_std, scheduler=scheduler, device=device)
    trainer_std.set_mode('standard')
    save_model_name = f"best_policy_JFB_BatchSize_{config_oc['batch_size']}_alphaG_{config_oc['alphaG']}_num_{config_oc['num_quadcopters']}_{time.ctime().replace(' ','_').replace(':','_')}"
    if full_AD_mode:
        save_model_name = f"best_policy_FullAD_BatchSize_{config_oc['batch_size']}_alphaG_{config_oc['alphaG']}_num_{config_oc['num_quadcopters']}_{time.ctime().replace(' ','_').replace(':','_')}"
    z0_std = copter_std.sample_initial_condition()
    trainer_std.train(z0_std, num_epochs=config_train['epochs'], plot_frequency=None, save_model_name=save_model_name)

def run_quadcopter_jbb(config_oc, config_train, device='cpu', plot_frequency=100):
    """
    Solves Quadcopter optimal control problem with INN and 
    Jacobian-based backpropagation (JBB)
    """
    print()
    print("##########################################################################")
    print("#################                                           ##############")
    print("#################     Quadcopter OC with INN + CVX (JBB)    ##############")
    print("#################                                           ##############")
    print("##########################################################################")
    print()
    copter_cvx = MultiQuadcopterOC(device=device, **config_oc)
    Phi_Net = Phi(3, 60, copter_cvx.state_dim)
    cvx_policy = CVXPolicy_MultiQuadcopter(config_oc['num_quadcopters'], p_net=Phi_Net, tol=2.0e-3, dev=device).to(device)
    optimizer_cvx = torch.optim.Adam(cvx_policy.parameters(), lr=config_train['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cvx, mode='min', factor=0.5, patience=10)
    trainer_cvx = OptimalControlTrainer(cvx_policy, copter_cvx, optimizer_cvx, scheduler=scheduler, device=device)
    trainer_cvx.set_mode('cvx')
    save_model_name = f"best_policy_JBB_BatchSize_{config_oc['batch_size']}_alphaG_{config_oc['alphaG']}_num_{config_oc['num_quadcopters']}_{time.ctime().replace(' ','_').replace(':','_')}"
    z0_cvx = copter_cvx.sample_initial_condition()
    trainer_cvx.train(z0_cvx, num_epochs=config_train['epochs'], plot_frequency=None, save_model_name=save_model_name)

def main():
    parser = argparse.ArgumentParser(description='Train Quadrotor Optimal Control with Implicit Networks')
    
    # Training method flags
    parser.add_argument('--train_jfb', action='store_true', default=True,
                        help='Train with Jacobian-Free Backpropagation (default: True)')
    parser.add_argument('--no_train_jfb', action='store_false', dest='train_jfb',
                        help='Disable JFB training')
    parser.add_argument('--train_full_ad', action='store_true', default=False,
                        help='Train with full automatic differentiation (default: False)')
    parser.add_argument('--train_jbb', action='store_true', default=False,
                        help='Train with Jacobian-Based Backpropagation/CVXPyLayers (default: False)')
    
    # Configuration parameters
    parser.add_argument('--num_quadcopters', type=int, default=100,
                        help='Number of quadrotors: 1, 6, or 100 (default: 100)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for training (default: 50)')
    parser.add_argument('--nt', type=int, default=160,
                        help='Number of time steps (default: 160)')
    parser.add_argument('--t_final', type=float, default=4.5,
                        help='Final time horizon (default: 4.5)')
    parser.add_argument('--alphaG', type=float, default=1.0e3,
                        help='Terminal cost weight (default: 1000.0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1.0e-2,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--n_trials', type=int, default=2,
                        help='Number of training trials (default: 2)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training: cpu, cuda, cuda:0, cuda:1, etc. (default: cpu)')
    parser.add_argument('--seed', type=int, default=1015,
                        help='Random seed for reproducibility (default: 1015)')
    
    args = parser.parse_args()
    
    # Validate and set device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            args.device = 'cpu'
        elif ':' in args.device:
            device_id = int(args.device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                print(f"Warning: CUDA device {device_id} not available (only {torch.cuda.device_count()} device(s)), using cuda:0")
                args.device = 'cuda:0'
    
    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configuration dictionaries
    config_oc = {
        'batch_size': args.batch_size,
        'nt': args.nt,
        't_final': args.t_final,
        'num_quadcopters': args.num_quadcopters,
        'alphaG': args.alphaG
    }
    config_train = {'lr': args.lr, 'epochs': args.epochs}

    for n in np.arange(args.n_trials):
        if args.train_jfb:
            run_quadcopter_jfb(config_oc, config_train, full_AD_mode=False, device=args.device)

        if args.train_full_ad:
            run_quadcopter_jfb(config_oc, config_train, full_AD_mode=True, device=args.device)
        
        if args.train_jbb:
            run_quadcopter_jbb(config_oc, config_train, args.device)       

if __name__ == "__main__":
    main()
