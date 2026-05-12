"""
models.VanDerPolOC_RL
---------------------
Van der Pol oscillator stabilisation in the **RL setting** (dynamics unknown
to the agent).

    State:   z = (x₁, x₂) ∈ ℝ²     (oscillator position and velocity)
    Control: u ∈ ℝ                   (scalar force input, clamped to [-3, 3])

    Dynamics:
        ẋ₁ = x₂
        ẋ₂ = (1 - x₁²) x₂ − x₁ + u      (Van der Pol with additive control)

    Modified Running cost:   L(z, u) = x₁² + x₂² + 0.5 u²
    Terminal cost:  G(z)    = x₁² + x₂²

Goal: steer the state to the origin (0, 0) from initial conditions near the
limit cycle of the uncontrolled system.

Optimality structure
~~~~~~~~~~~~~~~~~~~~~
The Hamiltonian is

    H(z, u, p) = L + p · f = (x₁² + x₂²) + 0.5 u² + p₁ x₂ + p₂ ((1−x₁²)x₂ − x₁ + u)

so ∇_u H = u + p₂, giving u*(z, p) = −p₂.

This is *linear* in p₂ and trivially solvable, yet the implicit-policy
formulation correctly recovers it via the fixed-point iteration
    T(u) = u − α(u + p₂) → u* = −p₂
while JFB-RL estimates the Jacobians needed to compute p₂ from rollout data.

Why this problem is useful for a JFB-RL vs Autodiff comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. The dynamics are nonlinear (presence of (1-x₁²)x₂), so Autodiff-BPTT must
   propagate gradients through 60 non-trivial Euler steps.
2. There is no known closed-form optimal policy (p₂ is the solution of the
   two-point boundary value problem for the HJB PDE).
3. The problem is non-finance, making it a clean general-purpose benchmark.
"""

from __future__ import annotations

import torch

from core_RL.ImplicitOC_RL import ImplicitOC_RL, TimeLike


class HardVanDerPolOC_RL(ImplicitOC_RL):
    """Van der Pol stabilisation problem (RL setting).

    Parameters
    ----------
    x10_min, x10_max
        Initial x₁ distribution: x₁₀ ~ U[x10_min, x10_max].
    x20_min, x20_max
        Initial x₂ distribution: x₂₀ ~ U[x20_min, x20_max].
    batch_size, t_initial, t_final, nt, alphaL, alphaG, device
        Forwarded to :class:`ImplicitOC_RL`.
    """

    def __init__(
        self,
        x10_min: float = 1.5,
        x10_max: float = 2.5,
        x20_min: float = -0.5,
        x20_max: float = 0.5,
        batch_size: int = 64,
        t_initial: float = 0.0,
        t_final: float = 3.0,
        nt: int = 60,
        alphaL: float = 1.0,
        alphaG: float = 5.0,
        lambda_u: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__(
            state_dim=2,
            control_dim=1,
            batch_size=batch_size,
            t_initial=t_initial,
            t_final=t_final,
            nt=nt,
            alphaL=alphaL,
            alphaG=alphaG,
            device=device,
        )
        self.oc_problem_name = "Hard Van der Pol Stabilisation (RL)"
        self.x10_min = x10_min
        self.x10_max = x10_max
        self.x20_min = x20_min
        self.x20_max = x20_max
        self.lambda_u = lambda_u

    # ================================================================== #
    # Designer-side: what the agent uses                                 #
    # ================================================================== #

    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """L = x₁² + x₂² + 0.5 u². Shape (B,)."""
        if z.dim() == 1:
            z, u = z.unsqueeze(0), u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        x1 = z[:, 0]
        x2 = z[:, 1]
        uu = u[:, 0]

        L = x1.pow(2) + x2.pow(2) + self.lambda_u * (torch.exp(uu**2) - 1.0)

        return L[0] if squeeze else L

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """∂L/∂u = u. Shape (B, 1)."""
        if z.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        grad = 2.0 * self.lambda_u * u * torch.exp(u**2)

        return grad[0] if squeeze else grad

    def compute_grad_lagrangian_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """∂L/∂z = (2x₁, 2x₂). Shape (B, 2)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        grad = 2.0 * z
        return grad[0] if squeeze else grad

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """G = x₁² + x₂². Shape (B,)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        G = z[:, 0].pow(2) + z[:, 1].pow(2)
        return G[0] if squeeze else G

    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """∂G/∂z = (2x₁, 2x₂). Shape (B, 2)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        grad = 2.0 * z
        return grad[0] if squeeze else grad

    def sample_initial_condition(self) -> torch.Tensor:
        """Sample (x₁₀, x₂₀) ~ U[x10_min, x10_max] × U[x20_min, x20_max].
        Shape (batch_size, 2).
        """
        x1 = self.x10_min + (self.x10_max - self.x10_min) * torch.rand(
            self.batch_size, 1, device=self.device
        )
        x2 = self.x20_min + (self.x20_max - self.x20_min) * torch.rand(
            self.batch_size, 1, device=self.device
        )
        return torch.cat([x1, x2], dim=1)

    # ================================================================== #
    # Simulator side: ground-truth f and its Jacobians                   #
    # ================================================================== #
    # These are NOT abstract methods of ImplicitOC_RL. They are exposed
    # here because the experimenter knows the dynamics, even though the
    # agent does not.  Consumed by AnalyticalEnvironment and
    # OracleJacobianEstimator, never by compute_loss_RL.

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """f(z, u) = [x₂, (1−x₁²)x₂ − x₁ + u]. Shape (B, 2)."""
        if z.dim() == 1:
            z, u = z.unsqueeze(0), u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        x1 = z[:, 0:1]
        x2 = z[:, 1:2]
        dx1 = x2
        dx2 = (1.0 - x1.pow(2)) * x2 - x1 + u[:, 0:1]
        f = torch.cat([dx1, dx2], dim=1)
        return f[0] if squeeze else f

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """b_k in (B, m, n) layout: b_k[:, 0, 0]=0, b_k[:, 0, 1]=1.
        Shape (B, 1, 2).
        Only u appears in ẋ₂, not in ẋ₁.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B = z.shape[0]
        b = torch.zeros(B, 1, 2, device=z.device, dtype=z.dtype)
        b[:, 0, 1] = 1.0
        return b[0] if squeeze else b

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """a_k in standard layout (B, n, n): a_k[:, i, j] = ∂f_i/∂z_j.
        Shape (B, 2, 2).

        Row 0: [0,          1     ]  (∂ẋ₁/∂x₁, ∂ẋ₁/∂x₂)
        Row 1: [−2x₁x₂−1,  1−x₁²]  (∂ẋ₂/∂x₁, ∂ẋ₂/∂x₂)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B = z.shape[0]
        x1 = z[:, 0]
        x2 = z[:, 1]
        a = torch.zeros(B, 2, 2, device=z.device, dtype=z.dtype)
        a[:, 0, 1] = 1.0
        a[:, 1, 0] = -2.0 * x1 * x2 - 1.0
        a[:, 1, 1] = 1.0 - x1.pow(2)
        return a[0] if squeeze else a


# ============================================================================ #
# Smoke test                                                                   #
# ============================================================================ #
if __name__ == "__main__":
    prob = HardVanDerPolOC_RL(batch_size=8)

    z = torch.rand(8, 2) * 2 - 1
    u = torch.randn(8, 1) * 0.5
    z.requires_grad_(True)
    u.requires_grad_(True)

    L = prob.compute_lagrangian(0.0, z, u).sum()
    gL_u = torch.autograd.grad(L, u, retain_graph=True)[0]
    gL_z = torch.autograd.grad(L, z, retain_graph=True)[0]
    print(f"∂L/∂u err = {(gL_u - prob.compute_grad_lagrangian(0.0, z, u)).abs().max():.2e}")
    print(f"∂L/∂z err = {(gL_z - prob.compute_grad_lagrangian_z(0.0, z, u)).abs().max():.2e}")

    G = prob.compute_G(z).sum()
    gG_z = torch.autograd.grad(G, z, retain_graph=True)[0]
    print(f"∂G/∂z err = {(gG_z - prob.compute_grad_G_z(z)).abs().max():.2e}")

    f = prob.compute_f(0.0, z, u)
    for i, (row, name) in enumerate([(f[:, 0].sum(), "f0"), (f[:, 1].sum(), "f1")]):
        gf_u = torch.autograd.grad(row, u, retain_graph=True)[0]
        gf_z = torch.autograd.grad(row, z, retain_graph=True)[0]
        print(f"∂{name}/∂u: {gf_u[:, 0].mean().item():.3f} "
              f"(analytical b_k[:, 0, {i}] = {prob.compute_grad_f_u(0.0, z, u)[:, 0, i].mean().item():.3f})")

    print("Smoke test passed.")
