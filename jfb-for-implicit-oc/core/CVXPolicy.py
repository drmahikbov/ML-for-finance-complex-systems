"""
core.CVXPolicy
--------------
CVXPY-based explicit policy networks for problems with tractable optimality
conditions. Wraps a `CvxpyLayer` to solve the Hamiltonian minimisation as a
small QP at each forward call, with a neural costate network supplying p.
"""
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from ImplicitNets import DefaultPNet
import numpy as np


class CVXPolicy_MC(nn.Module):
    """
    A policy network that uses a CvxpyLayer to solve for the optimal control `u*`
    for the Mountain Car problem.
    """
    def __init__(self, state_dim, control_dim, power_val=0.0015, p_net=None):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

        # 1. The Neural Network for the Costate (p)
        if p_net is not None:
            self.p_net = p_net  
        else:
            self.p_net = DefaultPNet(state_dim=state_dim, hidden_dim=100)

        # 2. The CVXPY Layer for the Optimal Control (u*)
        # We solve for u* that minimizes the Hamiltonian: H = L(u) + p'f(z,u)
        # For Mountain Car, H_u = 0.01*u^2 + p[1]*power*u. This is a convex QP.
        
        u_cp = cp.Variable(control_dim)
        p_cp = cp.Parameter(state_dim) # Full costate vector [p0, p1]

        # Objective derived from the Hamiltonian
        # Note: In CVXPY, matmul (@) with a scalar is element-wise multiplication
        l2_reg = 1e-7 
        obj = cp.Minimize((0.01 + l2_reg) * cp.sum_squares(u_cp) + (p_cp[1] * power_val) * u_cp)
        
        self.u_layer = CvxpyLayer(cp.Problem(obj), [p_cp], [u_cp])

    def forward(self, z, t):
        # 1. Predict the costate `p` using the neural network
        p = self.p_net(t, z)

        # 2. Solve for the optimal control `u*` using the CVXPY layer
        ustar, = self.u_layer(p, solver_args={'eps': 1e-3})
        # print(f"Computed control u*: {torch.norm(ustar)}")
        # ustar, = self.u_layer(p, solver_args={'eps': 1e-1, 'max_iters': 1})
        
        # Apply control limits (optional but good practice)
        return torch.clamp(ustar, -1.0, 1.0)

class CVXPolicy_LT(nn.Module):
    """
    CVXPolicy for the reformulated Linear Tangent Steering problem.
    """
    def __init__(self, state_dim, control_dim, p_net=None):
        super().__init__()
        raise NotImplementedError("LinearTangentSteering CVXPolicy is not implemented yet.")
    
class CVXPolicy_Quadcopter(nn.Module):
    """
    A policy network that uses CVXPYLayers to solve for the optimal control u* using 
    Jacobian-based backpropagation for the Quadcopter problem 
    """
    def __init__(self, state_dim, control_dim, p_net=None, m=0.5, g=1.0, tol=1.0e-3, dev='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.mass = m
        self.g = g
        self.tol = tol

        # Neural network for costate p
        if p_net is not None:
            self.p_net = p_net
        else:
            self.p_net = DefaultPNet(state_dim, hidden_dim=100, dev=dev)

        # Use CVXPY Layer to solve for control u* that minimizes the Hamiltonian
        # H = (f^T)p + L(u)
        # Ignore state z in dynamics dz/dt = f(t,z,u) so CVXPY layers works
        u_cp = cp.Variable(control_dim)# control
        p_cp = cp.Parameter(state_dim)# costate
        # Objective function
        obj_cp = cp.Minimize((p_cp[6]*(u_cp[0]/self.mass) + p_cp[7]*(u_cp[0]/self.mass) + p_cp[8]*((u_cp[0]/self.mass) - self.g) + p_cp[9]*u_cp[1] + p_cp[10]*u_cp[2] + p_cp[11]*u_cp[3]) + cp.exp(0.5*cp.sum_squares(u_cp)))
        # CVXPY layer
        self.u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp], [u_cp])

    def forward(self, z, t, track_all_fp_iters=False):
        # 1. Predict the costate `p` using the neural network
        p = self.p_net(t, z)

        # 2. Solve for the optimal control `u*` using the CVXPY layer
        ustar, = self.u_layer(p, solver_args={'eps': self.tol})
        return ustar

class CVXPolicy_MultiQuadcopter(nn.Module):
    """
    A policy network that uses CVXPYLayers to solve for the optimal control u* using
    Jacobian-based backpropagation for the multiple Quadcopter problem
    """
    def __init__(self, num_agents, p_net=None, m=0.5, g=1.0, tol=1.0e-3, dev='cpu'):
        super().__init__()
        self.single_state_dim = 12
        self.single_control_dim = 4
        self.num_agents = num_agents
        self.state_dim = int(num_agents*self.single_state_dim)
        self.control_dim = int(num_agents*self.single_control_dim)
        self.mass = m
        self.g = g
        self.tol = tol
        self.tracked_iters=0 # Not applicable for CVXPYLayers

        # Neural network for costate p
        if p_net is not None:
            self.p_net = p_net
        else:
            self.p_net = DefaultPNet(state_dim, hidden_dim=100, dev=dev)

        # Use CVXPY Layer to solve for control u* that minimizes the Hamiltonian
        # H = (f^T)p + L(u)
        # Ignore state z in dynamics dz/dt = f(t,z,u) so CVXPY layers works
        u_cp = cp.Variable(self.control_dim)# control
        p_cp = cp.Parameter(self.state_dim)# costate
        a_x = np.arange(6, self.state_dim, self.single_state_dim)
        a_y = np.arange(7, self.state_dim, self.single_state_dim)
        a_z = np.arange(8, self.state_dim, self.single_state_dim)
        a_psi = np.arange(9, self.state_dim, self.single_state_dim)
        a_theta = np.arange(10, self.state_dim, self.single_state_dim)
        a_phi = np.arange(11, self.state_dim, self.single_state_dim)
        u_0 = np.arange(0, self.control_dim, self.single_control_dim)
        u_1 = np.arange(1, self.control_dim, self.single_control_dim)
        u_2 = np.arange(2, self.control_dim, self.single_control_dim)
        u_3 = np.arange(3, self.control_dim, self.single_control_dim)
        # Objective function
        obj_cp = cp.Minimize(((p_cp[a_x].T)@(u_cp[u_0]/self.mass) + (p_cp[a_y].T)@(u_cp[u_0]/self.mass) + (p_cp[a_z].T)@((u_cp[u_0]/self.mass) - self.g) + (p_cp[a_psi].T)@u_cp[u_1] + (p_cp[a_theta].T)@u_cp[u_2] + (p_cp[a_phi].T)@u_cp[u_3]) + cp.exp((0.5/self.num_agents)*cp.sum_squares(u_cp)))
        # CVXPY layer
        self.u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp], [u_cp])

    def forward(self, z, t, track_all_fp_iters=False):
        # 1. Predict the costate `p` using the neural network
        p = self.p_net(t, z)

        # 2. Solve for the optimal control `u*` using the CVXPY layer
        ustar, = self.u_layer(p, solver_args={'eps': self.tol})
        return ustar

    
class CVXPolicy_Integrator(nn.Module):
    """
    A policy network that uses CVXPYLayers to solve for the optimal control u* using 
    Jacobian-based backpropagation for the Integrator problem 
    """
    def __init__(self, d, p_net=None, tol=1.0e-3, dev='cpu'):
        super().__init__()
        self.state_dim = d
        self.control_dim = d
        self.tol = tol

        # Neural network for costate p
        if p_net is not None:
            self.p_net = p_net
        else:
            self.p_net = DefaultPNet(d, hidden_dim=100, dev=dev)

        # Use CVXPY Layer to solve for control u* that minimizes the Hamiltonian
        # H = (f^T)p + L(u)
        u_cp = cp.Variable(d)# control
        p_cp = cp.Parameter(d)# costate
        # Objective function
        obj_cp = cp.Minimize((p_cp.T)@u_cp + cp.exp(0.5*cp.sum_squares(u_cp)))
        # CVXPY layer
        self.u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp], [u_cp])

    def forward(self, z, t, track_all_fp_iters=False):
        # 1. Predict the costate `p` using the neural network
        p = self.p_net(t, z)

        # 2. Solve for the optimal control `u*` using the CVXPY layer
        ustar, = self.u_layer(p, solver_args={'eps': self.tol})
        return ustar
    
class CVXPolicy_MultiBicycle(nn.Module):
    """
    A policy network that uses CVXPYLayers to solve for the optimal control u* using 
    Jacobian-based backpropagation for the MultiBicycle problem 
    """
    def __init__(self, n_b, p_net=None, l_w=0.5, tol=1.0e-3, dev='cpu'):
        super().__init__()
        raise NotImplementedError("Hamiltonian of MultiBicycle OC problem is non-convex!!!")
    
class CVXPolicy_DoubleIntegrator(nn.Module):
    """
    A policy network that uses CVXPYLayers to solve for the optimal control u* using 
    Jacobian-based backpropagation for the multi-agent double integrator problem 
    """
    def __init__(self, num_agents, p_net=None, tol=1.0e-3, dev='cpu'):
        super().__init__()
        self.single_state_dim = 6
        self.single_control_dim = 3
        self.state_dim = int(num_agents*self.single_state_dim)
        self.control_dim = int(num_agents*self.single_control_dim)
        self.n_agents = num_agents
        self.tol = tol

        # Neural network for costate p
        if p_net is not None:
            self.p_net = p_net
        else:
            self.p_net = DefaultPNet(self.state_dim, hidden_dim=100, dev=dev)

        # Use CVXPY Layer to solve for control u* that minimizes the Hamiltonian
        # H = (f^T)p + L(u)
        # Ignore state z in dynamics dz/dt = f(t,z,u) so CVXPY layers works
        u_cp = cp.Variable(self.control_dim)# control
        p_cp = cp.Parameter(self.state_dim)# costate
        u_x_r = int(3)*np.arange(1,self.n_agents+1)
        u_y_r = int(4)*np.arange(1,self.n_agents+1)
        u_z_r = int(5)*np.arange(1,self.n_agents+1)
        u_idx = np.arange(self.n_agents)
        # Objective function
        obj_cp = cp.Minimize((p_cp[u_x_r].T)@u_cp[u_idx] +  (p_cp[u_y_r].T)@u_cp[int(2)*u_idx] + (p_cp[u_z_r].T)@u_cp[int(3)*u_idx] + 0.5*cp.sum_squares(u_cp) + 0.25*(cp.sum_squares(u_cp)**2))
        # CVXPY layer
        self.u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp], [u_cp])

    def forward(self, z, t, track_all_fp_iters=False):
        # 1. Predict the costate `p` using the neural network
        p = self.p_net(t, z)

        # 2. Solve for the optimal control `u*` using the CVXPY layer
        ustar, = self.u_layer(p, solver_args={'eps': self.tol})
        return ustar

def mountain_car_example(batch_size, state_dim, control_dim, z0, t0=1.0, power_val=1.0):
    """
    Mountain car CVPYLayers example
    """
    # test the CVXPolicy class
    policy = CVXPolicy_MC(state_dim=state_dim, control_dim=control_dim)
    u_star = policy(z0, t0)

    # test without the class
        
    u_cp = cp.Variable(control_dim+1)
    p_cp = cp.Parameter(state_dim) # Full costate vector [p0, p1]
    # Objective derived from the Hamiltonian
    # Note: In CVXPY, matmul (@) with a scalar is element-wise multiplication
    l2_reg = 1e-7 
    obj = cp.Minimize((0.01 + l2_reg) * cp.sum_squares(u_cp) + (p_cp[1] * power_val) * u_cp)
        
    u_layer = CvxpyLayer(cp.Problem(obj), [p_cp], [u_cp])
    p = policy.p_net(t0, z0)

    ustar_dir, = u_layer(p, solver_args={"eps": 1e-3}) 

    print(f'difference: {torch.norm(ustar_dir - u_star)}')

def quadcopter_example(batch_size, state_dim, control_dim, z0, t0=5.0, m = 0.5, g = 1.0):
    """
    Mountain car CVPYLayers example
    """
    # test the CVXPolicy class
    policy = CVXPolicy_Quadcopter(state_dim=state_dim, control_dim=control_dim)
    u_star = policy(z0, t0)

    # test without the class
        
    u_cp = cp.Variable(control_dim+1)
    p_cp = cp.Parameter(state_dim) # Full costate vector 
    # Objective derived from the Hamiltonian
    obj_cp = cp.Minimize(cp.sum(p[6]*(u_cp[0]/m) + p_cp[7]*(u_cp[0]/m) + p_cp[8]*((u_cp[0]/m) - g) + p_cp[9]*u_cp[1] + p_cp[10]*u_cp[2] + p_cp[11]*u_cp[3]) + cp.exp(0.5*cp.sum_squares(u_cp)))
        
    u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp], [u_cp])
    p = policy.p_net(t0, z0)

    ustar_dir, = u_layer(p, solver_args={"eps": 1e-3}) 

    print(f'difference: {torch.norm(ustar_dir - u_star)}')

def main():
    batch_size = 10
    state_dim = 2
    control_dim = 1
    z0 = torch.randn(batch_size, state_dim)

    # Mountain Car
    mountain_car_example(batch_size, state_dim, control_dim, z0)

    # Quadcopter
    quadcopter_example(batch_size, state_dim, control_dim, z0)



if __name__ == '__main__':
    main()

    
