"""
models.Quadcopter
-----------------
Quadcopter (and multi-quadcopter) stabilisation as an ImplicitOC problem.

State: 12D per vehicle (position, Euler angles, velocities, angular rates).
Control: 4D (total thrust and three angular accelerations). Supports single-
and multi-agent variants via `QuadcopterOC` and `MultiQuadcopterOC`.
"""
from ImplicitNets import ImplicitNetOC
from ImplicitNets import Phi
from ImplicitOC import ImplicitOC, TimeLike
import torch
import matplotlib.pyplot as plt
import time
# Workaround for the "OMP: Error #15"
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class QuadcopterOC(ImplicitOC):
    """
    Single Quadcopter Optimal Control problem (12D State).
    This class now contains helper methods for its dynamics that can be reused.
    """

    def __init__(self, batch_size=50, t_initial=0.0, t_final=5.0, nt=250, 
                 alphaL=1.0, alphaG=5.0, device='cpu', ic_mean=0.0, ic_var=0.1, pen_pos=False):
        state_dim = 12
        control_dim = 4
        super().__init__(state_dim, control_dim, batch_size, t_initial, t_final, nt, alphaL, alphaG, device, pen_pos=pen_pos)
        self.oc_problem_name = "Single Quadcopter"
       
        # Mean and variance for sampling initial condition
        self.ic_mean = ic_mean
        self.ic_var = ic_var

        self.z_target = torch.zeros(batch_size, state_dim, device=self.device)
        self.z_target[:, 0:2] = 2.0
        self.z_target[:, 2] = 1.0

        self.mass = 0.5
        self.g = 1.0

    def _compute_f_single(self, t, z, u):
        """ Computes dynamics for a batch of single quadcopters. """
        batch_size = z.shape[0]
        y4, y5, y6, y7, y8, y9, y10, y11, y12 = [z[:, i].view(batch_size, 1) for i in range(3, 12)]
        omega_1, omega_2, omega_3, omega_4 = [u[:, i].view(batch_size, 1) for i in range(4)]

        y1_dot, y2_dot, y3_dot = y7, y8, y9
        y4_dot, y5_dot, y6_dot = y10, y11, y12
        y7_dot = (omega_1/self.mass) * (torch.sin(y4)*torch.sin(y6) + torch.cos(y4)*torch.sin(y5)*torch.cos(y6))
        y8_dot = (omega_1/self.mass) * (-torch.cos(y4)*torch.sin(y6) + torch.sin(y4)*torch.sin(y5)*torch.cos(y6))
        y9_dot = (omega_1/self.mass) * (torch.cos(y5)*torch.cos(y6)) - self.g
        y10_dot, y11_dot, y12_dot = omega_2, omega_3, omega_4
        
        return torch.cat([y1_dot, y2_dot, y3_dot, y4_dot, y5_dot, y6_dot, y7_dot, y8_dot, y9_dot, y10_dot, y11_dot, y12_dot], dim=1)

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_f_single(t, z, u)

    def _compute_grad_f_z_single(self, t, z, u):
        """ Computes dynamics gradient w.r.t. z for a batch of single quadcopters. """
        batch_size = z.shape[0]
        # Use hard-coded dimensions for a single quadcopter
        single_state_dim = 12
        grad_f_z = torch.zeros(batch_size, single_state_dim, single_state_dim, device=self.device)
        omega_1 = u[:, 0].view(batch_size, 1)
        y4, y5, y6 = [z[:, i].view(batch_size, 1) for i in range(3, 6)]
        
        # Derivatives w.r.t. orientation (y4, y5, y6)
        grad_f_z[:, 3, 6] = ((omega_1/self.mass)*(torch.cos(y4)*torch.sin(y6) - torch.sin(y4)*torch.sin(y5)*torch.cos(y6))).view(batch_size)
        grad_f_z[:, 3, 7] = ((omega_1/self.mass)*(torch.sin(y4)*torch.sin(y6) + torch.cos(y4)*torch.sin(y5)*torch.cos(y6))).view(batch_size)
        grad_f_z[:, 4, 6] = ((omega_1/self.mass)*(torch.cos(y4)*torch.cos(y5)*torch.cos(y6))).view(batch_size)
        grad_f_z[:, 4, 7] = ((omega_1/self.mass)*(torch.sin(y4)*torch.cos(y5)*torch.cos(y6))).view(batch_size)
        grad_f_z[:, 4, 8] = ((omega_1/self.mass)*(-torch.sin(y5)*torch.cos(y6))).view(batch_size)
        grad_f_z[:, 5, 6] = ((omega_1/self.mass)*(torch.sin(y4)*torch.cos(y6) - torch.cos(y4)*torch.sin(y5)*torch.sin(y6))).view(batch_size)
        grad_f_z[:, 5, 7] = ((omega_1/self.mass)*(-torch.cos(y4)*torch.cos(y6) - torch.sin(y4)*torch.sin(y5)*torch.sin(y6))).view(batch_size)
        grad_f_z[:, 5, 8] = ((omega_1/self.mass)*(-torch.cos(y5)*torch.sin(y6))).view(batch_size)
        
        # Derivatives of velocities w.r.t. positions (which are zero) and velocities (which are identity)
        grad_f_z[:, 6:single_state_dim, 0:6] = torch.eye(6,6, device=self.device)
        return grad_f_z

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_grad_f_z_single(t, z, u)

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

    def _compute_grad_f_u_single(self, t, z, u):
        """ Computes dynamics gradient w.r.t. u for a batch of single quadcopters. """
        batch_size = z.shape[0]
        # Use hard-coded dimensions for a single quadcopter
        single_state_dim = 12
        single_control_dim = 4
        grad_f_u = torch.zeros(batch_size, single_control_dim, single_state_dim, device=self.device)
        y4, y5, y6 = [z[:, i].view(batch_size, 1) for i in range(3, 6)]
        
        grad_f_u[:, 0, 6] = (1.0/self.mass)*(torch.sin(y4)*torch.sin(y6) + torch.cos(y4)*torch.sin(y5)*torch.cos(y6)).view(batch_size)
        grad_f_u[:, 0, 7] = (1.0/self.mass)*(-torch.cos(y4)*torch.sin(y6) + torch.sin(y4)*torch.sin(y5)*torch.cos(y6)).view(batch_size)
        grad_f_u[:, 0, 8] = (1.0/self.mass)*(torch.cos(y5)*torch.cos(y6)).view(batch_size)
        
        grad_f_u[:, 1, 9] = 1.0
        grad_f_u[:, 2, 10] = 1.0
        grad_f_u[:, 3, 11] = 1.0
        return grad_f_u

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_grad_f_u_single(t, z, u)

    def compute_grad_f_u_(self, z, u, grad_f_u):
        """
        Non-batched version of compute_grad_f_u
        """
        y4, y5, y6 = [z[i] for i in range(3, 6)]
        grad_f_u[0, 6] = (1.0/self.mass)*(torch.sin(y4)*torch.sin(y6) + torch.cos(y4)*torch.sin(y5)*torch.cos(y6))
        grad_f_u[0, 7] = (1.0/self.mass)*(-torch.cos(y4)*torch.sin(y6) + torch.sin(y4)*torch.sin(y5)*torch.cos(y6))
        grad_f_u[0, 8] = (1.0/self.mass)*(torch.cos(y5)*torch.cos(y6))
        grad_f_u[1, 9] = 1.0
        grad_f_u[2, 10] = 1.0
        grad_f_u[3, 11] = 1.0
        return grad_f_u

    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        # batch_size = u.shape[0]
        u = u.view(-1, self.control_dim) # Works for both single and multi
        loss = torch.exp(0.5 * torch.norm(u, dim=1)**2)
        return loss

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        u = u.view(-1, self.control_dim)
        grad_lagrangian_term = (torch.exp(0.5 * torch.norm(u, dim=1)**2)[:,None])*u
        return grad_lagrangian_term

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
        grad_lagrangian_term = torch.exp(0.5 * torch.linalg.norm(u)**2)*u
        return grad_lagrangian_term

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        diff = None
        if self.pen_pos:
            diff = z[:,:3] - self.z_target[:,:3]
        else:
            diff = z - self.z_target
        G = 0.5 * torch.norm(diff, dim=1)**2
        return G
    
    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(-1, self.state_dim)
        return z - self.z_target

    def sample_initial_condition(self):
        z0 = self.ic_var*torch.randn(self.batch_size, self.state_dim, device=self.device) + self.ic_mean
        z0[:, 6:] = 0.0
        return z0
        
class MultiQuadcopterOC(QuadcopterOC):
    """
    High-Dimensional problem with 8 quadcopters (96D State).
    This version uses vectorized operations instead of for-loops for performance.
    """
    def __init__(self, batch_size=50, t_initial=0.0, t_final=5.0, nt=250, 
                 alphaL=1.0, alphaG=5.0, device='cpu', num_quadcopters=8,
                 alpha_interaction=0.1, ic_mean=0.0, ic_var=0.1, pen_pos=False):
        self.num_agents = num_quadcopters
        self.single_state_dim = 12
        self.single_control_dim = 4
        
        state_dim = num_quadcopters * self.single_state_dim
        control_dim = num_quadcopters * self.single_control_dim

        # -- Parameters for interaction cost ---
        # This defines a repulsive potential field around each quadcopter.
        self.r = 1.0  
        self.interaction_range = 2.0 * self.r 
        self.alpha_interaction = alpha_interaction 

        super(QuadcopterOC, self).__init__(state_dim, control_dim, batch_size, t_initial, t_final, nt, alphaL, alphaG, device, pen_pos=pen_pos)
        self.oc_problem_name = "Multi Quadcopter"

        # Mean and variance for sampling initial condition
        self.ic_mean = ic_mean
        self.ic_var = ic_var
        
        self.mass = 0.5
        self.g = 1.0
        
        self._setup_targets()

    def _cvt(self, tensor):
        """Helper to move tensor to the correct device and type."""
        return tensor.to(device=self.device, dtype=torch.float32)

    # --- generate target positions ---
    def _setup_targets(self, original=True):
        if original:
            single_target = torch.zeros(1, self.single_state_dim, device=self.device)
            single_target[:, 0:3] = 2.0
            self.z_target = single_target.repeat(1, self.num_agents).expand(self.batch_size, -1)
        else:
            if self.num_agents % 2 != 0:
                raise ValueError("Number of quadcopters must be even for this target setup.")
            
            half_agents = self.num_agents // 2
            
            # Create a grid for the first half of the agents
            n_rows = int(torch.sqrt(torch.tensor(half_agents, dtype=torch.float32)))
            n_cols = (half_agents + n_rows - 1) // n_rows
            
            y_coords = torch.linspace(-3.0, 3.0, n_cols) #[-2.5,2.5] for 6 agents
            z_coords = torch.linspace(0.0, 6.0, n_rows) #[2.0, 5.0] for 6 agents
            grid_y, grid_z = torch.meshgrid(y_coords, z_coords, indexing='xy')
            
            targets1_2d = torch.stack([grid_y.flatten(), grid_z.flatten()], dim=1)
            targets1_pos = self._cvt(targets1_2d[:half_agents])
            
            targets1 = torch.zeros(half_agents, 3, device=self.device)
            targets1[:, 0] = 3.0 # Set a fixed x-offset 2.0 for 6 agents
            targets1[:, 1] = targets1_pos[:, 0]
            targets1[:, 2] = targets1_pos[:, 1]

            # Second half of agents have shifted targets
            shift = self._cvt(torch.tensor([1.0, -2.0, -1.5])) # [0.5, -1.0, -1.5] for 6 agents
            targets2 = targets1 + shift
            
            self.target_positions = torch.cat((targets1, targets2), dim=0) # Shape: [num_quads, 3]

            # Embed positions into the full 12D state vector
            z_target_template = torch.zeros(self.num_agents, self.single_state_dim, device=self.device)
            z_target_template[:, :3] = self.target_positions
            
            z_target_flat = z_target_template.view(1, self.state_dim)
            self.z_target = z_target_flat.expand(self.batch_size, -1)

    
    def sample_initial_condition(self):
        """
        Generates a batch of initial conditions for the Multi-Quadcopter problem.
        The quadcopters in each swarm start at different random positions
        sampled uniformly from a circle on the XY plane.
        """
        
        z0 = self.ic_var*torch.randn(self.batch_size * self.num_agents, self.single_state_dim, device=self.device) + self.ic_mean
        z0[:, 6:] = 0.0
        return z0.view(-1, self.state_dim)
    
    def sample_initial_condition_cir(self, variance=0.1):
        half_batch = self.batch_size // 2
        
        # 1. Create the "far" starting positions
        transform = self._cvt(torch.tensor([1.0, -1.0, -1.0]))
        shift = self._cvt(torch.tensor([0.0, 0.0, 10.0]))
        init_pos_far = self.target_positions * transform + shift
        
        # 2. Create the "near" starting positions (just the targets themselves)
        init_pos_near = self.target_positions

        # 3. Create the batch with noise
        noise_far = self._cvt(variance * torch.randn(half_batch, self.num_agents, 3))
        batch_pos_far = init_pos_far.unsqueeze(0) + noise_far
        
        # For the second half, ensure we have the correct batch size if it's odd
        remaining_batch = self.batch_size - half_batch
        noise_near = self._cvt(variance * torch.randn(remaining_batch, self.num_agents, 3))
        batch_pos_near = init_pos_near.unsqueeze(0) + noise_near
        
        # Combine the two halves
        batch_positions = torch.cat((batch_pos_far, batch_pos_near), dim=0)
        
        # 4. Embed into the full 12D state vector
        z0 = torch.zeros(self.batch_size, self.state_dim, device=self.device)
        z0_reshaped = z0.view(self.batch_size, self.num_agents, self.single_state_dim)
        z0_reshaped[:, :, :3] = batch_positions
        
        return z0

    # new function
    def _compute_interaction_term(self, z):
        """
        Taken from NeuralOC
        """
        batch_size = z.shape[0]
        positions = z.view(batch_size, self.num_agents, self.single_state_dim)[:, :, :3]
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dists_sq = pos_diff.pow(2).sum(dim=-1)
        mask = (dists_sq < self.interaction_range**2) & (dists_sq > 1e-9)
        potential = torch.exp(-dists_sq / (2 * self.r**2))
        interaction_cost = torch.sum(potential * mask, dim=[1, 2]) / 2.0
        
        grad_potential = (-1 / (2 * self.r**2)) * potential
        term1 = (grad_potential * mask).unsqueeze(-1)
        grad_pos = 2 * torch.sum(term1 * pos_diff, dim=2)
        grad_z = torch.zeros_like(z)
        grad_z_reshaped = grad_z.view(batch_size, self.num_agents, self.single_state_dim)
        grad_z_reshaped[:, :, :3] = grad_pos
        return interaction_cost / (self.num_agents*2), (grad_z_reshaped/ (self.num_agents*2)).view(batch_size, self.state_dim)


    # modified
    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        # batch_size = u.shape[0]
        u = u.view(-1, self.control_dim) 
        loss = torch.exp(0.5 * (torch.norm(u, dim=1)**2/self.num_agents)) # working for AD and JFB 
        # loss = torch.exp(0.5 * (torch.norm(u, dim=1)**2)) # working for AD and JFB 
        interaction_cost, _ = self._compute_interaction_term(z)
        return loss + self.alpha_interaction * interaction_cost

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        u = u.view(-1, self.control_dim)
        grad_lagrangian_term = (torch.exp(0.5 * (torch.norm(u, dim=1)**2)/self.num_agents)[:,None])*u/self.num_agents

        # grad_lagrangian_term = (torch.exp(0.5 * (torch.norm(u, dim=1)**2))[:,None])*u
        return grad_lagrangian_term
    
    # new
    def compute_grad_lagrangian_z(self, t, z, u):
        _, grad_interaction_z = self._compute_interaction_term(z)
        return self.alpha_interaction * grad_interaction_z

    # new (Overrides base (ImplicitOC) class) ---
    def compute_grad_H_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        grad_L_z = self.compute_grad_lagrangian_z(t, z, u)
        grad_H_z_from_f = super().compute_grad_H_z(t, z, u, p)

        return grad_L_z + grad_H_z_from_f

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(-1, self.state_dim)
        if self.pen_pos:
            batch_size = z.shape[0]
            z_temp = z.view(batch_size*self.num_agents, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num_agents, -1)
            diff_p = (z_temp[:,:3] - z_target_temp[:,:3]).view(batch_size,-1)
            diff_np = (z_temp[:,3:] - z_target_temp[:,3:]).view(batch_size, -1)
            G_p = 0.5 * (torch.norm(diff_p, dim=1)**2)/(self.num_agents*2)
            G_np = 0.5 * (torch.norm(diff_np, dim=1)**2)/(self.num_agents*2)
            return G_p#, G_np
        else:   
            diff = z - self.z_target
            G = 0.5 * (torch.norm(diff, dim=1)**2)/(self.num_agents*2)
            return G
    
    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(-1, self.state_dim)
        if self.pen_pos:
            batch_size = z.shape[0]
            z_temp = z.view(batch_size*self.num_agents, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num_agents, -1)
            diff_p = (z_temp[:,:3] - z_target_temp[:,:3]).view(batch_size,-1)
            diff_np = (z_temp[:,3:] - z_target_temp[:,3:]).view(batch_size, -1)
            return diff_p/(self.num_agents*2)#, diff_np/self.num_agents
        else:   
            return (z - self.z_target)/(self.num_agents*2)

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """ Vectorized dynamics computation. """
        batch_size = z.shape[0]
        
        # Reshape [batch, num quads * 12] -> [batch * num quads, 12]
        z_reshaped = z.view(batch_size * self.num_agents, self.single_state_dim)
        # Reshape [batch, num quads * 4] -> [batch * num quads, 4]
        u_reshaped = u.view(batch_size * self.num_agents, self.single_control_dim)
        
        # Compute dynamics for all quadcopters at once
        dz_dt_reshaped = self._compute_f_single(t, z_reshaped, u_reshaped)
        
        # Reshape back [batch * num quads, 12] -> [batch, num quads * 12]
        return dz_dt_reshaped.view(batch_size, self.state_dim)

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """ Vectorized gradient computation for f w.r.t. z. """
        batch_size = z.shape[0]
        
        z_reshaped = z.view(batch_size * self.num_agents, self.single_state_dim)
        u_reshaped = u.view(batch_size * self.num_agents, self.single_control_dim)
        
        # Compute all 12x12 gradient blocks at once
        # Shape: [batch * num quads, 12, 12]
        grad_blocks = self._compute_grad_f_z_single(t, z_reshaped, u_reshaped)
        
        # Reshape to separate batch and quadcopter dimensions
        # Shape: [batch, num quads, 12, 12]
        grad_blocks = grad_blocks.view(batch_size, self.num_agents, self.single_state_dim, self.single_state_dim)
        
        # Create the final block-diagonal matrix without a loop
        grad_f_z = torch.zeros(batch_size, self.state_dim, self.state_dim, device=self.device)
        # Create a view that is a strided version of the original tensor
        # This allows us to write the blocks in a single operation
        strided = grad_f_z.as_strided(
            (batch_size, self.num_agents, self.single_state_dim, self.single_state_dim),
            (self.state_dim * self.state_dim, self.single_state_dim * self.state_dim + self.single_state_dim, self.state_dim, 1)
        )
        strided.copy_(grad_blocks)
        
        return grad_f_z

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """ Vectorized gradient computation for f w.r.t. u. """
        batch_size = z.shape[0]
        
        z_reshaped = z.view(batch_size * self.num_agents, self.single_state_dim)
        u_reshaped = u.reshape(batch_size * self.num_agents, self.single_control_dim)
        
        # Compute all 4x12 gradient blocks at once
        # Shape: [batch * num quads, 4, 12]
        grad_blocks = self._compute_grad_f_u_single(t, z_reshaped, u_reshaped)
        
        # Reshape to separate batch and quadcopter dimensions
        # Shape: [batch, num quads, 4, 12]
        grad_blocks = grad_blocks.view(batch_size, self.num_agents, self.single_control_dim, self.single_state_dim)
        
        # Create the final block-diagonal matrix
        grad_f_u = torch.zeros(batch_size, self.control_dim, self.state_dim, device=self.device)
        
        # This is a bit harder to vectorize elegantly, but a simple loop is okay here
        # since the expensive computation is already vectorized.
        for i in range(self.num_agents):
            s_u, e_u = i * self.single_control_dim, (i + 1) * self.single_control_dim
            s_z, e_z = i * self.single_state_dim, (i + 1) * self.single_state_dim
            grad_f_u[:, s_u:e_u, s_z:e_z] = grad_blocks[:, i]
            
        return grad_f_u
    
    def plot_initial_condition(self, z0, title_str="Initial Condition Trajectory", save_path=None):
        """ Plots the initial condition for the first batch element of the swarm. """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.num_agents):
            s = i * self.single_state_dim
            positions = z0[0, s:s+3].cpu().numpy()
            x, y, z = positions[0], positions[1], positions[2]
            
            ax.scatter(x, y, z, marker='o', s=50, label=f'Copter {i+1} Start')

        target_pos = self.z_target[0, 0:3].cpu().numpy()
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', marker='*', s=200, label='Target', zorder=10)

        ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_zlabel('Z Position')
        ax.set_title(title_str)
        # plt.legend()
        plt.grid(True)
        plt.ion()
        if save_path: plt.savefig(save_path); plt.close()
        else: 
            plt.show()


    def plot_position_trajectories(self, z_traj, title_str="Multi-Quadcopter Trajectories", save_path=None):
        """ Plots 3D trajectories for the first batch element of the swarm. """
        batch_size, _, nt = z_traj.shape
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.num_agents):
            s = i * self.single_state_dim
            positions = z_traj[0, s:s+3, :].cpu().numpy()
            x, y, z = positions[0], positions[1], positions[2]
            
            line, = ax.plot(x, y, z, linewidth=2)
            color = line.get_color()
            
            ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=50, label=f'Copter {i+1} Start')
            
            target_pos = self.z_target[0, s:s+3].cpu().numpy()
            ax.scatter(target_pos[0], target_pos[1], target_pos[2], color=color, marker='*', s=200, zorder=10)
            

        # target_pos = self.z_target[0, 0:3].cpu().numpy()
        # ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', marker='*', s=200, label='Target', zorder=10)

        ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_zlabel('Z Position')
        ax.set_title(title_str)
        ax.legend([ax.get_lines()[0], ax.get_children()[1]], ['Trajectory', 'Start/Target'])
        plt.grid(True)
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2 
    nt = 250
    
    problem = MultiQuadcopterOC(batch_size=batch_size, nt=nt, device=device, num_quadcopters=4)
    
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
    state_dim = problem.state_dim
    control_dim = problem.control_dim
    print(f"Running on device: {device}")
    print(f"Running high-dimensional experiment with {problem.num_agents} quadcopters.")
    print(f"State Dimension: {state_dim}, Control Dimension: {control_dim}")

    z0 = problem.sample_initial_condition()
    
    # Increase network capacity for the more complex problem
    Phi_net = Phi(3, 10, state_dim, dev=device)
    u_net_implicit = ImplicitNetOC(state_dim=state_dim, control_dim=control_dim, hidden_dim=512, tol=1e-3, oc_problem=problem, p_net=Phi_net, use_aa=False).to(device)
    
    z_traj = problem.generate_trajectory(u_net_implicit, z0, nt, return_full_trajectory=True)
    problem.plot_position_trajectories(z_traj.detach())
    
    optimizer = torch.optim.Adam(u_net_implicit.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15)

    plot_frequency = 50
    best_loss = float('inf')
    max_epochs = int(1)

    for i in range(1, max_epochs + 1):
        start_time = time.time()
        optimizer.zero_grad()
        loss, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin = problem.compute_loss(u_net_implicit, z0)

        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     torch.save(u_net_implicit.state_dict(), 'best_implicit_model_96d_vectorized.pth')

        # if i % plot_frequency == 0:
        z_traj = problem.generate_trajectory(u_net_implicit, z0, nt, return_full_trajectory=True)
        problem.plot_position_trajectories(z_traj.detach())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(u_net_implicit.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step(loss)

        grad_norm = sum(torch.norm(p.grad)**2 for p in u_net_implicit.parameters() if p.grad is not None)**0.5
        iter_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']

        print(f"iter {i}, loss {loss.item():.3e}, grad {grad_norm.item():.3e}, L {running_cost.item():.3e}, G {terminal_cost.item():.3e}, "
              f"HJB {cHJB.item():.3e}, HJBfin {cHJBfin.item():.3e}, Adj {cadj.item():.3e}, Adjfin {cadjfin.item():.3e}, "
              f"time: {iter_time:.3e}, lr: {lr:.3e}")

if __name__ == "__main__":
    main()
