"""
examples.example_multiConsumption
----------------------------------
Train the JFB implicit policy on the multi-dimensional consumption-savings
problem with habit formation (ConsumptionSavingsOC). Set full_AD=True to
switch to full autodiff (BPTT) as a baseline.
"""
import torch, numpy as np, sys, time
from Consumption import ConsumptionSavingsOC
from ImplicitNets            import Phi
from ImplicitNets            import ImplicitNetOC_pos as ImplicitNetOC #this is needed in OptimalControlTrainer too. but this hardcoded way should be improved later
from OptimalControlTrainer   import OptimalControlTrainer


# class Logger:
#     def __init__(self, fname="consumption_run.log"):
#         self.terminal = sys.stdout
#         self.log = open(fname, "a")
#     def write(self, msg):
#         self.terminal.write(msg); self.log.write(msg)
#     def flush(self):  self.terminal.flush(); self.log.flush()

# sys.stdout = Logger("results_ConsumptionOC/consumption_run.log")



def run_consumption_jfb(config_oc: dict,
                        config_train: dict,
                        full_AD: bool = False,
                        device: str = "cpu",
                        plot_frequency=None):
    """
    Solves multi-dimensional optimal consumption problem with INN + JFB
    """

    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############        Consumption OC with INN        ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()


    m = 100
    A = torch.eye(m, device=device)
    B = torch.eye(m, device=device)

    cs = ConsumptionSavingsOC(
        m=m, A=A, B=B,
        eta=0.9, theta=0.9,
        batch_size=512,             
        t_initial=0.0, t_final=2.0,
        nt=100,
        r=3, delta=0.1,
        gamma=0.5, epsilon=0.1,
        device=device,
    )

    cs.track_all_fp_iters = full_AD     
    phi = Phi(3, 50, cs.state_dim, dev=device)
    inn = ImplicitNetOC(cs.state_dim, cs.control_dim,
                        alpha=1e-4, max_iters=200, tol=1e-4,
                        p_net=phi, oc_problem=cs,
                        use_control_limits=False,            
                        dev=device).to(device)

    opt       = torch.optim.Adam(inn.parameters(), lr=config_train["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=10)

    trainer = OptimalControlTrainer(inn, cs, opt, scheduler=scheduler,
                                    device=device)
    trainer.set_mode("standard")           # JFB = standard

    tag = "FullAD" if full_AD else "JFB"
    save_name = (f"best_policy_{tag}"
                 f"_Batch{config_oc['batch_size']}_"
                 f"{time.ctime().replace(' ','_').replace(':','_')}")

    z0 = cs.sample_initial_condition()
    trainer.train(z0,
                  num_epochs=config_train["epochs"],
                  plot_frequency=plot_frequency,
                  save_model_name=save_name)

def main():
    seed = 420
    torch.manual_seed(seed);  np.random.seed(seed)

    config_oc = dict(batch_size=1,
                     nt=2,
                     t_final=2.0)        
    config_train = dict(lr=1e-3, epochs=500)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    n_trials   = 3                    
    # JFB
    for n in np.arange(n_trials):
        run_consumption_jfb(config_oc, config_train,
                            full_AD=False, device=device)



if __name__ == "__main__":
    main()
