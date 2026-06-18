"""
models.MultiBicycle
-------------------
Formation control for a fleet of bicycles as an ImplicitOC problem.

State: 4D per bicycle (x, y, heading, speed). Control: 2D per bicycle (steering
angle, acceleration). Includes a pairwise interaction penalty to discourage
collisions.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Workaround for OMP: Error #15
from ImplicitNets import ImplicitNetOC
from ImplicitNets import Phi
from ImplicitOC import ImplicitOC, TimeLike
import torch
import matplotlib.pyplot as plt
import time
import matplotlib
#matplotlib.use('TkAgg')

class MultiBicycleOC(ImplicitOC):
    """
    OC problem involving multiple bicycles
    """
    def __init__(self, batch_size=50, t_initial=0.0, t_final=10.0, nt=250, 
                 n_b=3, alphaL=1.0, alphaG=5.0, alphaHJB=[0.0,0.0], device='cpu', 
                 alpha_interaction=0.1, pen_pos=False, ic_mean=0.0, ic_var=0.1):
        """
        Initialize the MultiBicycle OC problem.
        
        Args:
            batch_size (int): Batch size for trajectory optimization
            t_initial (float): Initial time
            t_final (float): Final time
            nt (int): Number of time steps
            n_b (int): Number of bicycles
            device (str): Device for computation
        """
        self.num_agents = n_b
        self.single_state_dim = 4
        self.single_control_dim = 2
        state_dim = self.single_state_dim * n_b
        control_dim = 2*n_b
        super().__init__(state_dim, control_dim, batch_size, t_initial, t_final, nt, alphaL, alphaG, device, alphaHJB, pen_pos=pen_pos)
        self.oc_problem_name = "Multi Bicycle"

        # Mean and variance for sampling initial condition
        self.ic_mean = ic_mean
        self.ic_var = ic_var
        
        # Define target at final time
        self.z_target = torch.zeros(batch_size, self.state_dim, device=self.device)#self.setup_targets()
        self.z_target[:,0:self.state_dim:self.single_state_dim] = 1.0
        self.z_target[:,1:self.state_dim:self.single_state_dim] = 1.0
        
        self.l = 0.05 # Distance between front and rear wheels on bicycle, in meters

        # --- Parameters for interaction cost ---
        self.alpha_interaction = alpha_interaction
        self.r = 1.0  # Characteristic distance of the potential field
        self.interaction_range = 2.0 * self.r # Distance at which penalty becomes active

    # Generate target positions ---
    def setup_targets(self):
        # Split targets into 3 5x5 grids
        Mx = 1
        My = 5
        # Group 1
        x_coords = torch.linspace(0.5, 0.5, Mx)
        y_coords = torch.linspace(0.5, 0.5, My)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        target_positions_1 = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        # Group 2
        x_coords = torch.linspace(0.5, 0.5, Mx)
        y_coords = torch.linspace(0.0, 0.0, My)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        target_positions_2 = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        # Group 3
        Mx = 5
        My = 1
        x_coords = torch.linspace(0.0, 0.0, Mx)
        y_coords = torch.linspace(0.5, 0.5, My)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        target_positions_3 = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # Embed positions into the full state vector
        target_positions = torch.cat((target_positions_1, target_positions_2, target_positions_3), dim=0)
        z_target_template = torch.zeros(self.num_agents, self.single_state_dim, device=self.device)
        z_target_template[:, :2] = target_positions
        """
        z_target_template = torch.zeros(self.num_agents, self.single_state_dim, device=self.device)
        z_target_template[0, :2] = torch.tensor([4.25, 4.5])
        z_target_template[1, :2] = torch.tensor([4.5, 4.25])
        z_target_template[2, :2] = torch.tensor([4.25, 4.0])
        z_target_template[3, :2] = torch.tensor([4.0, 4.25])
        z_target_template[4, :2] = torch.tensor([4.25, 4.25])
        """
        z_target_flat = z_target_template.view(1, self.state_dim)

        return z_target_flat.expand(self.batch_size, -1)

    # --- NEW METHOD ---
    def _compute_interaction_term(self, z):
        """
        Taken from Multiple Quadcopter
        """
        batch_size = z.shape[0]
        positions = z.view(batch_size, self.num_agents, self.single_state_dim)[:, :, :2]
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dists_sq = pos_diff.pow(2).sum(dim=-1)
        mask = (dists_sq < self.interaction_range**2) & (dists_sq > 1e-9)
        potential = torch.exp(-dists_sq / (2 * self.r**2))
        interaction_cost = torch.sum(potential * mask, dim=[1, 2]) / 2.0
        grad_potential = (-1 / (2 * self.r**2)) * potential
        term1 = (grad_potential * mask).unsqueeze(-1)
        grad_pos = 2.0 * torch.sum(term1 * pos_diff, dim=2)
        grad_z = torch.zeros_like(z)
        grad_z_reshaped = grad_z.view(batch_size, self.num_agents, self.single_state_dim)
        grad_z_reshaped[:, :, :2] = grad_pos
        return interaction_cost / (self.num_agents*2.0), (grad_z_reshaped/ (self.num_agents*2.0)).view(batch_size, self.state_dim)

    # --- MODIFIED METHOD ---
    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Lagrangian (running cost) of multi bicycle OC problem

        Args:
            t (torch.tensor or float): Current time
            z (torch.tensor): State vector of shape (batch_size,state_dim) 
            u (torch.tensor): Control vector shape (batch_size,control_dim)

        Return:
            torch.Tensor: Lagrangian values of shape (batch_size,)
        """
        u = u.view(self.batch_size, self.control_dim)

        # compute the loss along the feature dimension
        loss = 0.5*torch.norm(u, dim=1)**2 
        interaction_cost, _ = self._compute_interaction_term(z)
        z_temp = z.view(self.batch_size*self.num_agents, -1)
        z_target_temp = self.z_target.reshape(self.batch_size*self.num_agents, -1)
        diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(self.batch_size,-1)
        #Gt = 0.5*torch.norm(diff_p, dim=1)**2
        return loss + (self.alpha_interaction * interaction_cost)
    
    # --- NEW METHOD ---
    def compute_grad_lagrangian_z(self, t, z, u):
        """
        Computes the gradient of the Lagrangian with respect to the state z.
        """
        _, grad_interaction_z = self._compute_interaction_term(z)
        dGt_dz = z - self.z_target
        if self.pen_pos:
            dGt_dz[:,2:self.state_dim:self.single_state_dim] = 0.0
            dGt_dz[:,3:self.state_dim:self.single_state_dim] = 0.0
        return (self.alpha_interaction * grad_interaction_z) + dGt_dz

    # --- NEW METHOD (Overrides base (implicitOC) class) ---
    def compute_grad_H_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the Hamiltonian H = L(z,u) + p^T f(z,u) w.r.t. state z.
        This override is necessary because the Lagrangian L now depends on state z.
        """
        grad_L_z = self.compute_grad_lagrangian_z(t, z, u)
        grad_H_z_from_f = super().compute_grad_H_z(t, z, u, p)
        return grad_L_z + grad_H_z_from_f

    
    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate gradient with respect to u of Lagrangian (running cost) 

        Args:
            t (torch.tensor or float): Current time
            z (torch.tensor): State vector of shape (batch_size,state_dim) 
            u (torch.tensor): Control vector of shape (batch_size,control_dim)

        Return:
            torch.Tensor: Gradient of Lagrangian with respect to u of shape (batch_size,)
        """
        u = u.view(self.batch_size, self.control_dim)
        return u

    def compute_grad_lagrangian_(self, t, z, u):
        """
        Calculate non-batched gradient with respect to u of Lagrangian (running cost)

        Args:
            t (torch.tensor or float): Current time
            z (torch.tensor): State vector of shape (state_dim,)
            u (torch.tensor): Control vector of shape (control_dim,)

        Return:
            torch.Tensor: Gradient of Lagrangian with respect to u of shape (control_dim,)
        """
        return u
    
    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate terminal cost of multi bicycle OC problem

        Args:
            z (torch.tensor): State vector of shape (batch_size,state_dim) 

        Return:
            torch.Tensor: Terminal cost values of shape (batch_size,)
        """
        if self.pen_pos:
            batch_size = z.shape[0]
            """
            diff_p1 = torch.zeros(batch_size, int(2*self.num_agents), device=self.device)
            for i in range(self.num):
                x_idx = int(4*i)
                y_idx = x_idx + 1
                diff_p1[:,int(2*i)] = z[:,x_idx] - self.z_target[:,x_idx]
                diff_p1[:,int(2*i)+1] = z[:,y_idx] - self.z_target[:,y_idx]
            """
            z_temp = z.view(batch_size*self.num_agents, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num_agents, -1)
            diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(batch_size,-1)
            G = 0.5*torch.norm(diff_p, dim=1)**2
            #G_huber = torch.nn.functional.huber_loss(z_temp[:,:2], z_target_temp[:,:2], reduction='none', delta=1.0e-5)
            #G = torch.sum(G_huber, dim=1)
            return G
        else:
            diff = z - self.z_target
            # compute the fixed cost G
            G = 0.5*torch.norm(diff, dim=1)**2
            #G_huber = torch.nn.functional.huber_loss(z, self.z_target, reduction='none', delta=1.0e-5)
            #G = torch.sum(G_huber, dim=1)
            return G
    
    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gradient of terminal cost G with respect to z

        Args:
            z (torch.Tensor): State vector of shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Gradient of G with respect to z of shape (batch_size, state_dim)
        """
        if self.pen_pos:
            batch_size = z.shape[0]
            z_temp = z.view(batch_size*self.num_agents, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num_agents, -1)
            diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(batch_size,-1)
            grad_G_z = diff_p
            return grad_G_z 
        else:
            diff = z - self.z_target
            grad_G_z = diff
            return grad_G_z

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the time derivative of the state vector z 
        for the multi bicycle OC problem

        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control vector of shape (batch_size, control_dim) 

        Returns:
            torch.Tensor: Time derivative of z, i.e. dz/dt, shape (batch_size, state_dim)
        """
        # Ensure all variables are on the right device
        z = z.to(self.device)
        u = u.to(self.device)
        dz_dt = torch.zeros(self.batch_size, self.state_dim, device=self.device)

        # Compute state derivatives
        dz_dt[:,0:self.state_dim:4] = z[:,3:self.state_dim:4]*torch.cos(z[:,2:self.state_dim:4])
        dz_dt[:,1:self.state_dim:4] = z[:,3:self.state_dim:4]*torch.sin(z[:,2:self.state_dim:4])
        dz_dt[:,2:self.state_dim:4] = (z[:,3:self.state_dim:4]/self.l)*torch.tan(u[:,0:self.control_dim:2])
        dz_dt[:,3:self.state_dim:4] = u[:,1:self.control_dim:2]

        return dz_dt
    
    def compute_grad_f_u_(self, z, u, grad_f_u_):
        """
        Computes the gradient of the dynamics f with respect to the control u
        for one sample in a batch

        Args:
            z (torch.Tensor): State vector of shape (state_dim)
            u (torch.Tensor): Control vector of shape (control_dim) 
            grad_f_u_ (torch.Tensor): Output tensor of shape (control_dim, state_dim)

        Return:
            torch.Tensor: Gradient of f with respect to u for one sample of shape 
                          (control_dim,state_dim)
        """
        grad_f_u_[0:self.control_dim:2,2:self.state_dim:4] = torch.diag((z[3:self.state_dim:4]/self.l)*((1.0/torch.cos(u[0:self.control_dim:2]))**2))
        grad_f_u_[1:self.control_dim:2,3:self.state_dim:4] = torch.eye(self.num_agents, self.num_agents, device=self.device)
        return grad_f_u_
    
    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient of the dynamics f with respect to the control u.

        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control vector of shape (batch_size, control_dim) 

        Return:
            torch.Tensor: Gradient of f with respect to u of shape (batch_size, control_dim, state_dim)
        """
        # assert all variables are on the right device
        grad_f_u_ = torch.zeros(self.batch_size, self.control_dim, self.state_dim, device=self.device)
        grad_f_u = torch.vmap(self.compute_grad_f_u_, in_dims=(0,0,0))(z, u, grad_f_u_)

        return grad_f_u
    
    def compute_grad_f_z_(self, z, u, grad_f_z_):
        """
        Computes the gradient of the dynamics f with respect to the state vector z
        for one sample in a batch

        Args:
            z (torch.Tensor): State vector of shape (state_dim)
            u (torch.Tensor): Control vector of shape (control_dim)
            grad_f_z_ (torch.Tensor): Output tensor of shape (state_dim, state_dim)

        Return:
            torch.Tensor: Gradient of f with respect to z of shape (state_dim, state_dim)
        """
        grad_f_z_[2:self.state_dim:4,0:self.state_dim:4] = torch.diag(-z[3:self.state_dim:4]*torch.sin(z[2:self.state_dim:4]))
        grad_f_z_[2:self.state_dim:4,1:self.state_dim:4] = torch.diag(z[3:self.state_dim:4]*torch.cos(z[2:self.state_dim:4]))

        grad_f_z_[3:self.state_dim:4,0:self.state_dim:4] = torch.diag(torch.cos(z[2:self.state_dim:4]))
        grad_f_z_[3:self.state_dim:4,1:self.state_dim:4] = torch.diag(torch.sin(z[2:self.state_dim:4]))
        grad_f_z_[3:self.state_dim:4,2:self.state_dim:4] = torch.diag((1.0/self.l)*torch.tan(u[0:self.control_dim:2]))

        return grad_f_z_
    
    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient of the dynamics f with respect to the state vector z

        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control vector of shape (batch_size, control_dim) 

        Return:
            torch.Tensor: Gradient of f with respect to z of shape (batch_size, state_dim, state_dim)
        """
        # assert all variables are on the right device
        #assert z.device.type == self.device; assert u.device.type == self.device
        grad_f_z_ = torch.zeros(self.batch_size, self.state_dim, self.state_dim, device=self.device)
        grad_f_z = torch.vmap(self.compute_grad_f_z_, in_dims=(0,0,0))(z, u, grad_f_z_)
        #assert grad_f_z.shape == (self.batch_size, self.state_dim, self.state_dim)

        return grad_f_z
    
    def sample_initial_condition(self):
        """
        Generates a batch of initial conditions for the Quadcopter optimal control problem 
        """
        z0 = self.ic_var*torch.randn(self.batch_size, self.state_dim, device=self.device) + self.ic_mean
        z0[:,2:self.state_dim:4] = 1e-1
        z0[:,3:self.state_dim:4] = 1e-1
        return z0
    
    def generate_trajectory(self, u, z0, nt, return_full_trajectory=False):
        """
        Generates a trajectory using Euler integration. This method is general
        and works for any system defined by a compute_f method.
        """
        batch_size = z0.shape[0]
        z = torch.zeros(batch_size, self.state_dim, nt + 1, device=self.device)
        z[:, :, 0] = z0
        h = (self.t_final - self.t_initial) / nt
        ti = self.t_initial
        for i in range(nt):
            # The control u can be a neural network or a pre-computed tensor
            if hasattr(u, 'forward'):
                current_u = u(z[:, :, i], ti)
            else: # Assumes u is a tensor of shape [batch, control_dim, nt]
                current_u = u[:, :, i]
            
            z[:, :, i + 1] = z[:, :, i] + h * self.compute_f(ti, z[:, :, i], current_u)
            ti = ti + h
            
        return z if return_full_trajectory else z[:, :, -1]

    def plot_position_trajectories(self, z_traj, title_str="Multi-Agent Bicycle Trajectories", save_path=None):
        """ Plots trajectories for the first batch element of the swarm. """
        batch_size, _, nt = z_traj.shape
        plt.rc('axes', labelsize=26)
        plt.rc('legend', fontsize=26)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for i in range(self.num_agents):
            s = i * self.single_state_dim
            positions = z_traj[0, s:s+2, :].cpu().numpy()
            x, y = positions[0], positions[1]
            
            line, = ax.plot(x, y, linewidth=4)
            color = line.get_color()
            
            ax.scatter(x[0], y[0], color=color, marker='o', s=200)#s=50, label=f'Agent {i+1} Start')
            
            target_pos = self.z_target[0, s:s+2].cpu().numpy()
            ax.scatter(target_pos[0], target_pos[1], color='k', marker='x', s=1000, zorder=10, linewidths=6)
        
        ax.set_xlabel('X Position'); ax.set_ylabel('Y Position');
        #ax.set_title(title_str)
        ax.legend([ax.get_lines()[0], ax.get_children()[1], ax.get_children()[2]], ['Trajectory', 'Start', 'Target'])
        plt.grid(True)
        if save_path: plt.savefig(save_path, bbox_inches='tight'); plt.close()
        else: plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2 
    nt = 250
    
    problem = MultiBicycleOC(batch_size=batch_size, nt=nt, device=device, n_b=4)
    
    # =======================================================
    # --- GRADIENT CHECKING CODE FOR THE INTERACTION TERM ---
    # =======================================================
    print("\n" + "="*50)
    print("Performing Gradient Check for Interaction Term")
    
    from utils import GradientTester_Taylors as GradientTester
    gradient_tester = GradientTester()

    z_test = problem.sample_initial_condition() 

    def interaction_cost_wrapper(z):
        cost, _ = problem._compute_interaction_term(z)
        return torch.sum(cost)

    def interaction_grad_wrapper(z):
        _, grad = problem._compute_interaction_term(z)
        return grad

    gradient_tester.check_gradient(
        cost_func=interaction_cost_wrapper,
        grad_func=interaction_grad_wrapper,
        x=z_test
    )
    print("="*50 + "\n")
    # =======================================================

if __name__ == "__main__":
    main()

