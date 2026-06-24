"""
models.PortfolioOC_RL
---------------------
Merton-type portfolio optimisation with exponential risk penalty (RL setting).

State: W (wealth). Control: π (portfolio fraction).
Dynamics: dW/dt = r W + π(μ − r)W. Running cost: L = λ(exp(π²) − 1).
Terminal cost: G = −log(W/W_ref).

∇_π H = 2λπ exp(π²) + p(μ−r)W is transcendental — no closed-form π*. The
agent never sees μ or r; they appear only in `compute_f` (used by the
environment) and in `OracleJacobianEstimator`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

# core-RL/ is on sys.path (the runner adds it); flat import.
from core_RL.ImplicitOC_RL import ImplicitOC_RL, TimeLike

from benchmarking import Trajectory
from benchmarking.plotter import Panel


def _portfolio_panels() -> List[Panel]:
    """Two-panel layout: wealth ``W(t)`` and portfolio fraction ``π(t)``.

    State layout assumed: ``z = [W]``. Control: scalar ``π`` (component 0).
    """
    def _W_extract(traj: Trajectory):
        # z has shape (N, state_dim) for deterministic trajectories.
        return traj.t, traj.z[..., 0]

    def _pi_extract(traj: Trajectory):
        if traj.u is None:
            return np.empty(0), np.empty(0)
        return traj.t[:-1], traj.u[..., 0]

    return [
        Panel("Wealth  W(t)", _W_extract, "W(t)"),
        Panel("Portfolio fraction  π(t)", _pi_extract, "π(t)"),
    ]


class PortfolioOC_RL(ImplicitOC_RL):
    """Merton portfolio with exponential risk penalty (PDF §4).

    Parameters
    ----------
    mu_true, r_true
        Ground-truth drift and risk-free rate. Stored on the class **for
        the simulator and the oracle baseline only**. The agent never
        queries them.
    lam
        Risk-aversion weight λ in the running cost ``λ(e^{π²} - 1)``.
        Default ``0.5``. The PDF observes ``λ ≤ 1`` keeps the FP operator
        contractive for ``α ∈ [0.01, 0.1]``.
    W_ref
        Reference wealth in the terminal cost ``-log(W/W_ref)``.
    W0_min, W0_max
        Initial-wealth distribution; ``W₀ ~ U[W0_min, W0_max]``.
    W_floor
        Lower clamp applied before any ``log(W)`` or ``1/W`` to prevent
        NaN if the policy briefly drives wealth to (numerically) zero.
    batch_size, t_initial, t_final, nt, alphaL, alphaG, device
        Forwarded to :class:`ImplicitOC_RL`.
    """

    def __init__(
        self,
        mu_true: float = 0.10,
        r_true: float = 0.03,
        lam: float = 0.5,
        W_ref: float = 1.0,
        W0_min: float = 0.8,
        W0_max: float = 1.2,
        W_floor: float = 1e-4,
        batch_size: int = 32,
        t_initial: float = 0.0,
        t_final: float = 1.0,
        nt: int = 50,
        alphaL: float = 1.0,
        alphaG: float = 1.0,
        device: str = "cpu",
    ):
        # State = wealth (scalar); control = portfolio fraction (scalar).
        super().__init__(
            state_dim=1,
            control_dim=1,
            batch_size=batch_size,
            t_initial=t_initial,
            t_final=t_final,
            nt=nt,
            alphaL=alphaL,
            alphaG=alphaG,
            device=device,
        )
        self.oc_problem_name = "Merton Portfolio (RL)"

        # Designer-known parameters of the cost.
        self.lam = lam
        self.W_ref = W_ref
        self.W0_min = W0_min
        self.W0_max = W0_max
        self.W_floor = W_floor

        # Hidden ground-truth dynamics — used only by the simulator and
        # the oracle baseline. The agent's training pipeline never sees
        # these.
        self.mu_true = mu_true
        self.r_true = r_true

    # ================================================================== #
    # Designer-side: what the agent uses                                 #
    # ================================================================== #
    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``L(t, W, π) = λ (e^{π²} - 1)``. Shape ``(B,)`` (or scalar in the
        unbatched branch used by Jacobian checks).

        The running cost depends only on the control. ``z`` is accepted to
        match the signature, but unused.
        """
        if z.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        # u : (B, 1)
        pi = u[:, 0]
        L = self.lam * (torch.exp(pi.pow(2)) - 1.0)        # (B,)
        return L[0] if squeeze else L

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∂L/∂π = 2 λ π e^{π²}``. Shape ``(B, 1)``."""
        if z.dim() == 1:
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        pi = u[:, 0:1]                                      # (B, 1)
        grad = 2.0 * self.lam * pi * torch.exp(pi.pow(2))   # (B, 1)
        return grad[0] if squeeze else grad

    def compute_grad_lagrangian_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∂L/∂W = 0`` — the running cost has no state dependence.

        Override the autograd-based default in :class:`ImplicitOC_RL` for
        speed and to avoid building a graph we don't need.
        """
        if z.dim() == 1:
            return torch.zeros(self.state_dim, device=z.device, dtype=z.dtype)
        return torch.zeros_like(z)

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """``G(W) = -log(W / W_ref) = -log(W) + log(W_ref)``.

        Shape ``(B,)``. ``W`` is clamped below by ``W_floor`` before the
        log to prevent NaN propagation; this clamp never fires under any
        sensible training trajectory.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        W = z[:, 0].clamp(min=self.W_floor)                 # (B,)
        G = -torch.log(W / self.W_ref)                      # (B,)
        return G[0] if squeeze else G

    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """``∂G/∂W = -1/W``. Shape ``(B, 1)``."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        W = z[:, 0:1].clamp(min=self.W_floor)               # (B, 1)
        grad = -1.0 / W                                     # (B, 1)
        return grad[0] if squeeze else grad

    def sample_initial_condition(self) -> torch.Tensor:
        """``W₀ ~ U[W0_min, W0_max]``. Shape ``(batch_size, 1)``."""
        W0 = self.W0_min + (self.W0_max - self.W0_min) * torch.rand(
            self.batch_size, 1, device=self.device
        )
        return W0

    # ================================================================== #
    # Simulator side: ground-truth f and its Jacobians                   #
    # ================================================================== #
    # These are *not* abstract methods of ImplicitOC_RL. We expose them as
    # ordinary methods because we (the experimenter) know them. They are
    # consumed by:
    #   - AnalyticalEnvironment(f_callable=prob.compute_f, ...)
    #   - OracleJacobianEstimator(grad_f_z=..., grad_f_u=..., ...)
    # but never by the agent's training path.
    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``dW/dt = r W + π (μ - r) W``. Shape ``(B, 1)``."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        W = z[:, 0:1]                                       # (B, 1)
        pi = u[:, 0:1]                                      # (B, 1)
        dW = self.r_true * W + pi * (self.mu_true - self.r_true) * W
        return dW[0] if squeeze else dW

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``(∂f/∂π)`` in the repo's transpose layout: shape ``(B, m, n) = (B, 1, 1)``
        with entry ``b_k[:, 0, 0] = (μ - r) W``.

        Used only by the :class:`OracleJacobianEstimator` baseline.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B = z.shape[0]
        W = z[:, 0]                                         # (B,)
        grad = torch.zeros(B, 1, 1, device=z.device, dtype=z.dtype)
        grad[:, 0, 0] = (self.mu_true - self.r_true) * W
        return grad[0] if squeeze else grad

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∂f/∂W`` in the repo's standard layout: shape ``(B, n, n) = (B, 1, 1)``
        with entry ``a_k[:, 0, 0] = r + π (μ - r)``.

        Used only by the :class:`OracleJacobianEstimator` baseline.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B = z.shape[0]
        pi = u[:, 0]                                        # (B,)
        grad = torch.zeros(B, 1, 1, device=z.device, dtype=z.dtype)
        grad[:, 0, 0] = self.r_true + pi * (self.mu_true - self.r_true)
        return grad[0] if squeeze else grad

    # ================================================================== #
    # Plotting hooks                                                     #
    # ================================================================== #
    def panels(self) -> List[Panel]:
        """Two-panel layout: ``W(t)`` and ``π(t)``."""
        return _portfolio_panels()

    def to_trajectory(
        self,
        z_traj: torch.Tensor,
        policy=None,
        path_index: int = 0,
        label: str = "JFB-RL",
    ) -> Trajectory:
        """Pack a rolled-out tensor into a :class:`benchmarking.Trajectory`.

        Mirrors :meth:`LiquidationPortfolioOC.to_trajectory`. If a ``policy``
        is supplied, the controls are reconstructed by evaluating the policy
        along the path; this requires the policy to have a current ``b_k``
        estimate, since :class:`ImplicitNetOC_RL` raises if ``b_k`` is unset.
        Callers that pass a ``policy`` should ensure they've populated
        ``policy._current_b_k`` (the trainer's plotting dispatch handles
        this via the ``jac_setter`` hook in :meth:`Environment.rollout`).

        For simplicity, when reconstructing controls from a policy we fall
        back to whatever ``b_k`` is currently set on the policy (typically
        the last step of the most recent rollout). For diagnostic plots
        this is acceptable; for publication-quality reconstructions, prefer
        passing ``u_traj`` directly via a separate API or extending this
        method to take a Jacobian estimator.
        """
        z_traj = z_traj.detach()
        batch, state_dim, nt1 = z_traj.shape
        if not 0 <= path_index < batch:
            raise IndexError(f"path_index={path_index} out of range for batch={batch}")
        nt = nt1 - 1

        t_np = np.linspace(self.t_initial, self.t_final, nt1)
        z_np = z_traj[path_index].transpose(0, 1).cpu().numpy()      # (nt+1, state_dim)

        u_np = None
        if policy is not None:
            dt = (self.t_final - self.t_initial) / nt
            u_buf = torch.zeros(self.control_dim, nt, device=z_traj.device)
            try:
                with torch.no_grad():
                    z_path = z_traj[path_index : path_index + 1]      # keep batch axis
                    for i in range(nt):
                        t_i = self.t_initial + i * dt
                        u_i = policy(z_path[:, :, i], t_i).view(1, self.control_dim)
                        u_buf[:, i] = u_i[0]
                u_np = u_buf.transpose(0, 1).cpu().numpy()            # (nt, control_dim)
            except RuntimeError:
                # Most likely cause: policy has no current b_k set. Skip
                # control reconstruction and let the panel render an
                # empty series — better than crashing the trainer at
                # plot time.
                u_np = None

        return Trajectory(
            t=t_np,
            z=z_np,
            u=u_np,
            label=label,
            style={"color": "#d6604d", "lw": 2.0},
        )


# ============================================================================ #
# Smoke test                                                                   #
# ============================================================================ #
if __name__ == "__main__":
    # Minimal sanity check: derivatives of L and G match autograd.
    device = "cpu"
    prob = PortfolioOC_RL(batch_size=8, device=device)

    z = torch.rand(8, 1, requires_grad=True) + 0.5      # W ~ [0.5, 1.5]
    u = torch.randn(8, 1, requires_grad=True) * 0.3     # π near 0

    # ∂L/∂π
    L = prob.compute_lagrangian(0.0, z, u).sum()
    grad_L_u_auto = torch.autograd.grad(L, u, retain_graph=True)[0]
    grad_L_u_anal = prob.compute_grad_lagrangian(0.0, z, u)
    print(f"∂L/∂π  err = {(grad_L_u_auto - grad_L_u_anal).abs().max().item():.3e}")

    # ∂G/∂W
    G = prob.compute_G(z).sum()
    grad_G_auto = torch.autograd.grad(G, z, retain_graph=True)[0]
    grad_G_anal = prob.compute_grad_G_z(z)
    print(f"∂G/∂W  err = {(grad_G_auto - grad_G_anal).abs().max().item():.3e}")

    # ∂f/∂π
    f = prob.compute_f(0.0, z, u).sum()
    grad_f_u_auto = torch.autograd.grad(f, u, retain_graph=True, create_graph=False)[0]
    grad_f_u_anal = prob.compute_grad_f_u(0.0, z, u)[:, 0, 0:1]
    print(f"∂f/∂π  err = {(grad_f_u_auto - grad_f_u_anal).abs().max().item():.3e}")

    # ∂f/∂W
    f = prob.compute_f(0.0, z, u).sum()
    grad_f_z_auto = torch.autograd.grad(f, z, retain_graph=True)[0]
    grad_f_z_anal = prob.compute_grad_f_z(0.0, z, u)[:, 0, 0:1]
    print(f"∂f/∂W  err = {(grad_f_z_auto - grad_f_z_anal).abs().max().item():.3e}")

    print("All gradient checks passed (errors should be ~1e-7).")