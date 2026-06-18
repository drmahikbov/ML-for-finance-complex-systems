"""
core.DirectControlNets
----------------------
Direct-control baseline policy: outputs u = π(z, t) without any fixed-point
iteration. Used to compare against implicit JFB methods. Uses very small weight
initialisation (0.001×) to keep initial controls near zero.
"""
import torch
import torch.nn as nn
from ImplicitNets import Phi, DefaultPNet, ResNN


class DirectControlPolicy(nn.Module):
    """
    Direct control parameterization policy network.

    Unlike implicit methods that parameterize the costate p and solve for u,
    this network directly outputs the control u = π(z, t) without any
    implicit solving step (no fixed-point iteration, no CVX solver).

    This provides a simpler baseline for comparison with implicit optimal control methods.
    """

    def __init__(self, state_dim, control_dim, p_net=None, hidden_dim=100, dev='cpu',
                 u_min=None, u_max=None):
        """
        Initialize the direct control policy network.

        Args:
            state_dim (int): Dimension of the state vector
            control_dim (int): Dimension of the control vector
            p_net (PNet, optional): Feature extraction network. If None, uses Phi network.
            hidden_dim (int): Hidden dimension for default network
            dev (str): Device to use ('cpu' or 'cuda')
            u_min (float, optional): Minimum steering angle value (for clamping)
            u_max (float, optional): Maximum steering angle value (for clamping)
        """
        super(DirectControlPolicy, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.device = dev
        self.u_min = u_min
        self.u_max = u_max

        # Feature extraction network - reuse same architecture as implicit methods
        # for fair comparison
        if p_net is not None:
            self.p_net = p_net
        else:
            # Use Phi network by default to match implicit methods' architecture
            self.p_net = Phi(nTh=3, m=10, d=state_dim, dev=dev)

        # Output layer to map features to control
        self.control_head = nn.Linear(state_dim, control_dim, device=dev)

        # ================================================================
        # CRITICAL: Very small weight initialization (0.001 scale)
        # ================================================================
        # Direct control outputs u = W*features + b, where velocities are
        # unbounded. With standard initialization (~0.01), the initial
        # controls can be large enough to cause immediate instability.
        #
        # JFB doesn't need this because:
        # - It outputs p (costate), not u (control)
        # - Controls are computed via a = -p_v, which couples to value function
        # - This constraint naturally limits control magnitudes
        #
        # For direct control, we use 100x smaller initialization (0.001)
        # to ensure initial controls are small while network learns.
        # ================================================================
        nn.init.xavier_uniform_(self.control_head.weight)
        with torch.no_grad():
            self.control_head.weight.mul_(0.001)  # 100x smaller than typical 0.01
        nn.init.zeros_(self.control_head.bias)

    def forward(self, z, t, track_all_fp_iters=False):
        """
        Forward pass: directly compute control from state and time.

        Args:
            z (torch.Tensor): State tensor of shape (batch_size, state_dim)
            t (float or torch.Tensor): Time value
            track_all_fp_iters (bool): Ignored, included for API compatibility with ImplicitNetOC

        Returns:
            torch.Tensor: Control tensor of shape (batch_size, control_dim)
        """
        # Extract features using the p_net (even though it outputs features, not costate here)
        features = self.p_net(t, z)  # Shape: (batch_size, state_dim)

        # Map features to control
        u = self.control_head(features)  # Shape: (batch_size, control_dim)

        # Clamp controls if limits specified
        # For multi-bicycle: u = [steering_1, velocity_1, steering_2, velocity_2, ...]
        # Match the implicit method's approach: create new tensor to avoid autograd issues
        if self.u_min is not None and self.u_max is not None:
            u_clamped = torch.zeros_like(u)
            u_clamped[:, 0::2] = torch.clamp(u[:, 0::2], self.u_min, self.u_max)
            u_clamped[:, 1::2] = u[:, 1::2]  # Copy velocities unchanged
            return u_clamped

        return u

    def get_convergence_stats(self):
        """
        Return dummy convergence stats for API compatibility with ImplicitNetOC.
        Since there's no fixed-point iteration, these are all trivial.
        """
        return {
            'fp_depth': 0,
            'max_res_norm': 0.0,
            'converged': True
        }

    # For compatibility with the trainer that may access this attribute
    tracked_iters = 0

    # Flag to indicate this is direct control (no HJB residuals should be computed)
    # This prevents compute_loss from trying to access p_net.getPhi()
    is_direct_control = True


class DirectControlPolicyMLP(nn.Module):
    """
    Alternative implementation using ResNN architecture similar to Phi.
    Useful for ablation studies with configurable nTh and m parameters.
    """

    def __init__(self, state_dim, control_dim, nTh=3, m=10, dev='cpu',
                 u_min=None, u_max=None):
        """
        Initialize the ResNN-based direct control policy.

        Args:
            state_dim (int): Dimension of the state vector
            control_dim (int): Dimension of the control vector
            nTh (int): Number of ResNet layers (default: 3)
            m (int): Hidden dimension for ResNN (default: 10)
            dev (str): Device to use ('cpu' or 'cuda')
            u_min (float, optional): Minimum steering angle value (for clamping)
            u_max (float, optional): Maximum steering angle value (for clamping)
        """
        super(DirectControlPolicyMLP, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.nTh = nTh
        self.m = m
        self.device = dev
        self.u_min = u_min
        self.u_max = u_max

        # ResNN feature extractor similar to Phi
        self.resnn = ResNN(d=state_dim, m=m, nTh=nTh).to(dev)

        # Output layer to map ResNN features to control
        self.control_head = nn.Linear(m, control_dim, device=dev)

        # ================================================================
        # CRITICAL: Very small weight initialization (0.001 scale)
        # See DirectControlPolicy.__init__ for detailed explanation.
        # ================================================================
        nn.init.xavier_uniform_(self.control_head.weight)
        with torch.no_grad():
            self.control_head.weight.mul_(0.001)  # 100x smaller than typical 0.01
        nn.init.zeros_(self.control_head.bias)

    def forward(self, z, t, track_all_fp_iters=False):
        """
        Forward pass through ResNN.

        Args:
            z (torch.Tensor): State tensor of shape (batch_size, state_dim)
            t (float or torch.Tensor): Time value
            track_all_fp_iters (bool): Ignored, included for API compatibility

        Returns:
            torch.Tensor: Control tensor of shape (batch_size, control_dim)
        """
        from torch.nn.functional import pad

        batch_size = z.shape[0]

        # Pad state with time to create input for ResNN
        if isinstance(t, float):
            x = pad(z, [0, 1, 0, 0], value=t)
        else:
            # If t is a tensor, need to handle it properly
            t_expanded = t if len(t.shape) == 2 else t.view(-1, 1)
            x = torch.cat([z, t_expanded], dim=1)

        # Forward through ResNN
        features = self.resnn(x)  # Shape: (batch_size, m)

        # Map features to control
        u = self.control_head(features)  # Shape: (batch_size, control_dim)

        # Clamp controls if limits specified
        # For multi-bicycle: u = [steering_1, velocity_1, steering_2, velocity_2, ...]
        # Match the implicit method's approach: create new tensor to avoid autograd issues
        if self.u_min is not None and self.u_max is not None:
            u_clamped = torch.zeros_like(u)
            u_clamped[:, 0::2] = torch.clamp(u[:, 0::2], self.u_min, self.u_max)
            u_clamped[:, 1::2] = u[:, 1::2]  # Copy velocities unchanged
            return u_clamped

        return u

    def get_convergence_stats(self):
        """Return dummy convergence stats for API compatibility."""
        return {
            'fp_depth': 0,
            'max_res_norm': 0.0,
            'converged': True
        }

    tracked_iters = 0

    # Flag to indicate this is direct control (no HJB residuals should be computed)
    is_direct_control = True


if __name__ == '__main__':
    # Simple test
    device = 'cpu'
    state_dim = 12
    control_dim = 4
    batch_size = 10

    print("Testing DirectControlPolicy...")
    policy = DirectControlPolicy(state_dim, control_dim, dev=device)
    z = torch.randn(batch_size, state_dim, device=device)
    t = 0.5

    u = policy(z, t)
    print(f"Input state shape: {z.shape}")
    print(f"Output control shape: {u.shape}")
    print(f"Control values (first sample): {u[0]}")

    print("\nTesting DirectControlPolicyMLP...")
    policy_mlp = DirectControlPolicyMLP(state_dim, control_dim, nTh=3, m=10, dev=device)
    u_mlp = policy_mlp(z, t)
    print(f"MLP output control shape: {u_mlp.shape}")
    print(f"MLP control values (first sample): {u_mlp[0]}")

    print("\nAll tests passed!")
