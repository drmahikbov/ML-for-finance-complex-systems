"""
models.Consumption
------------------
Multi-dimensional consumption-savings problem with habit formation.

State: z = (x, h₁, …, h_m) — wealth and habit vector. Control: u ∈ ℝ^m.
Dynamics: dx/dt = rx − Σuᵢ, dh/dt = A u^η − B h^θ.
Running cost: e^{-δt} Σ (uᵢ − hᵢ)^{1-γ}/(1-γ). Terminal cost: ε x(T)^{1-γ}/(1-γ).
"""
# from ImplicitOC import ImplicitOC
from ImplicitOC import ImplicitOC, TimeLike
import torch
from utils import GradientTester

class ConsumptionSavingsOC(ImplicitOC):
    """
    Optimal consumption-savings with habit formation.
    State z = [x, h_1, ..., h_m], control u = [u_1, ..., u_m].
    Dynamics:
        dx/dt = r x - sum_i u_i
        dh/dt = A * u^{\circ eta} - B * h^{\circ theta}
    Running cost: e^{-delta t} * sum_i (u_i - h_i)^{1-gamma} / (1-gamma)
    Terminal cost: epsilon * x(T)^{1-gamma} / (1-gamma)
    """

    def __init__(
        self,
        m,
        A,
        B,
        eta=1.0,
        theta=1.0,
        batch_size=10,
        t_initial=0.0,
        t_final=2.0,
        nt=100,
        r=0.3,
        delta=0.01,
        gamma=2.0,
        epsilon=5.0,
        device='cpu',
    ):
        state_dim = 1 + m
        control_dim = m
        super().__init__(state_dim, control_dim, batch_size,
                         t_initial, t_final, nt, alphaL=1.0, alphaG=1.0, device=device)
        self.oc_problem_name = "Multi Consumption"

        # Habit matrices
        self.m = m
        self.A = A.to(device)
        self.B = B.to(device)
        self.eta = eta
        self.theta = theta
        # Economic params
        self.r = r
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the running cost (Lagrangian).

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim) [wealth x, habit h]
            u (torch.Tensor): Control input of shape (batch_size, control_dim) [consumption u]

        Returns:
            torch.Tensor: Lagrangian values of shape (batch_size,)
        """


        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        h = z[:, 1:1+self.m] # habit level
        diff = (u - h).clamp(min=1e-6)
        util = diff.pow(1 - self.gamma).sum(dim=1) / (1 - self.gamma)
        tau = t if torch.is_tensor(t) else torch.tensor(t, dtype=z.dtype, device=z.device)
        lag = torch.exp(-self.delta * tau) * util
        return lag[0] if squeeze else lag

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the Lagrangian with respect to control.
        Args:
            u (torch.Tensor): Control inputs of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of Lagrangian of shape (batch_size, control_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        h = z[:, 1:1+self.m]
        raw = u - h
        # clamp for utility, but for gradient respect clamp derivative
        diff = raw.clamp(min=1e-6)
        tau = t if torch.is_tensor(t) else torch.tensor(t, dtype=z.dtype, device=z.device)
        exp_term = torch.exp(-self.delta * tau)
        # gradient zero where raw < eps
        mask = (raw > 1e-6).to(z.dtype)
        grad = exp_term * mask * diff.pow(-self.gamma)
        return grad[0] if squeeze else grad

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the state dynamics.
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        x = z[:, 0:1]
        h = z[:, 1:1+self.m]
        dx = self.r * x - u.sum(dim=1, keepdim=True)
        u_pow = u.clamp(min=1e-6).pow(self.eta)
        h_pow = h.clamp(min=1e-6).pow(self.theta)
        dh = u_pow @ self.A.T - h_pow @ self.B.T
        result = torch.cat((dx, dh), dim=1)
        return result[0] if squeeze else result

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the system dynamics f with respect to the state u.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        batch = z.shape[0]
        D = 1 + self.m
        grad = torch.zeros(batch, self.m, D, device=z.device)
        grad[:, :, 0] = -1.0
        u_p = u.clamp(min=1e-6).pow(self.eta - 1) * self.eta
        for i in range(self.m):
            grad[:, i, 1:] = self.A[:, i].unsqueeze(0) * u_p[:, i].unsqueeze(1)
        return grad[0] if squeeze else grad

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        D = 1 + self.m
        grad = torch.zeros(batch, D, D, device=z.device)
        # df1/dx = r
        grad[:, 0, 0] = self.r
        # dh/dh: -B * theta * h^{theta-1}
        h = z[:, 1:1+self.m]
        h_p = h.clamp(min=1e-6).pow(self.theta - 1) * self.theta  # (batch, m)
        # coeff[b,i,k] = B[i,k] * h_p[b,k]
        coeff = h_p.unsqueeze(1) * self.B.unsqueeze(0)  # (batch, m, m)
        grad[:, 1:1+self.m, 1:1+self.m] -= coeff
        return grad[0] if squeeze else grad

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the terminal cost (without discounting).

        Args:
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Terminal cost values of shape (batch_size,)
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        xT = z[:, 0:1]
        G = self.epsilon * xT.clamp(min=1e-6).pow(1 - self.gamma) / (1 - self.gamma)
        return G[0] if squeeze else G

    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        batch = z.shape[0]
        D = 1 + self.m
        grad = torch.zeros(batch, D, device=z.device)
        x = z[:, 0:1].clamp(min=1e-6)
        grad[:, 0] = self.epsilon * x.pow(-self.gamma).squeeze(1)
        return grad[0] if squeeze else grad

    def sample_initial_condition(self):
        x0 = 25.0 + 0.1 * torch.rand(self.batch_size, 1)
        h0 = 0.01 + 0.1 * torch.rand(self.batch_size, self.m)
        return torch.cat((x0, h0), dim=1).to(self.device)

    def generate_trajectory(self, u, z0, nt, return_full_trajectory=False):
        batch = z0.shape[0]
        D = 1 + self.m
        traj = torch.zeros(batch, D, nt+1, device=z0.device)
        traj[:, :, 0] = z0
        dt = (self.t_final - self.t_initial) / nt
        t = self.t_initial
        for i in range(nt):
            if torch.is_tensor(u):
                curr = u[:, :, i]
            else:
                curr = u(traj[:, :, i], t)
            traj[:, :, i+1] = traj[:, :, i] + dt * self.compute_f(t, traj[:, :, i], curr)
            t += dt
        return traj if return_full_trajectory else traj[:, :, -1]

# Example usage
if __name__ == "__main__":

    device = 'cpu'
    batch_size = 10
    nt = 100
    m = 2
    A = torch.eye(m)
    B = torch.eye(m)

    prob = ConsumptionSavingsOC(m=m, A=A, B=B,
                                 eta=1.0, theta=1.0,
                                 batch_size=batch_size,
                                 t_initial=0.0, t_final=2.0,
                                 nt=nt,
                                 r=0.3, delta=0.01,
                                 gamma=2.0, epsilon=5.0,
                                 device=device)
    
    u_rand = torch.randn(batch_size, m, nt, device=device)

    # Compute various losses
    total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin,_ = prob.compute_loss(u_rand, prob.sample_initial_condition())
    print(f"Total Cost: {total_cost.item()}")
    print(f"Running Cost: {running_cost.item()}")
    print(f"Terminal Cost: {terminal_cost.item()}")

    # Gradient tests
    test_z = torch.tensor([[25.0, 0.05, 0.05], [30.0, 0.02, 0.02]], dtype=torch.float32)
    test_u = torch.tensor([[0.1, 0.05], [0.2, 0.1]], dtype=torch.float32)
    test_z = test_z.repeat(batch_size//2, 1)
    test_u = test_u.repeat(batch_size//2, 1)
    print("Running gradient tests...")
    GradientTester.run_all_tests(prob, test_z, test_u)
