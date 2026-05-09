from __future__ import annotations

from typing import List

import numpy as np
import torch

from ImplicitOC import ImplicitOC, TimeLike
from utils import GradientTester
from benchmarking import Trajectory
from benchmarking.plotter import Panel, liquidation_panels

class LiquidationPortfolioOC(ImplicitOC):
    """
    Multi-asset Almgren-Chriss liquidation as a finite-horizon OC problem,
    in its **reduced** formulation (cash account ``X`` removed from the
    OC state).

    OC state ``z = [q_1..q_n, S_1..S_n] in R^{2n}`` with ``n = n_assets``.
    Control ``u in R^n`` is the per-asset selling rate.

    Continuous-time dynamics (right-hand side of the controlled ODE):

        dq_i/dt = -u_i                       (no z-dependence)
        dS_i/dt = -kappa_i * u_i             (linear permanent impact)

    Running cost (Lagrangian) — absorbs the impact term into ``L``:

        L'(t, z, u) = 1/2 * sum_i sigma_i^2 q_i^2
                      - sum_i S_i u_i
                      + sum_i eta_i * (u_i^2 + epsilon)^(gamma/2)

    Terminal cost (no ``-X`` term):

        G'(z(T)) = alpha * sum_i q_i(T)^2

    Mathematical equivalence with the legacy 3-state ``[q, S, X]`` problem
    is exact (no martingale assumption, no approximation):

        J_B  = - X(0) + integral [ 1/2 sigma^2 q^2 - S u + eta (u^2+eps)^{g/2} ] dt
                       + alpha q(T)^2
             = - X(0) + J_B'

    The constant ``-X(0)`` is independent of the policy, so
    ``argmin J_B == argmin J_B'``.

    Why this matters for JFB
    ------------------------
    With the impact term inside ``L'``, the Hamiltonian curvature in ``u``
    is a **problem constant** (``2 eta`` for ``gamma = 2``) instead of
    being the learned costate ``-2 eta * p_X``. The fixed-point operator
    ``T(u) = u - alpha_fp * grad_u H`` is then contractive from epoch zero
    with ``alpha_fp = 1/(2 eta)``: T(u) becomes one-step exact and equals
    the closed-form ``u* = (S + p_q + kappa p_S)/(2 eta)``. This is the
    same one-shot FP behaviour as the canonical Almgren-Chriss in
    ``jfb-new-copy/AlmgrenChriss.py``.

    Observer cash account
    ---------------------
    ``X`` is integrated in parallel via :meth:`compute_cash_flow`. It is
    **never** an OC state, never enters ``Phi``, ``compute_grad_H_u``,
    the FP iteration, the JFB unrolled tail, or the loss. It exists only
    on the plotting / financial-reporting boundary so figures and reports
    can still display ``X(t)`` exactly. ``self.X0`` stores the initial
    cash level used by the observer.
    """

    def __init__(
        self,
        batch_size=64,
        t_initial=0.0,
        t_final=2.0,
        nt=100,
        n_assets=2,
        sigma=(0.02, 0.02),
        kappa=(1.0e-4, 1.0e-4),
        eta=(0.1, 0.1),
        gamma=2.0,
        epsilon=1.0e-2,
        alpha=30,
        q0_min=(0.5, 0.5),
        q0_max=(1.5, 1.5),
        S0=(1.0, 1.0),
        X0=0.0,
        device="cpu",
        alphaHJB=(0.0, 0.0),
        alphaadj=(0.0, 0.0),
    ):
        # OC state: (q_1..q_n, S_1..S_n) ∈ R^{2n}. The cash account X is NOT
        # part of the OC state — it is integrated separately by the observer
        # `compute_cash_flow` so it can be plotted/reported without inflating
        # the OC dimension or polluting the Hamiltonian curvature in u.
        state_dim = 2 * n_assets
        control_dim = n_assets  # (u1, u2, ...)
        # Time discretization and batch: horizon [t_initial, t_final], nt steps, parallel trajectories.
        super().__init__(
            state_dim,
            control_dim,
            batch_size,
            t_initial,
            t_final,
            nt,
            alphaL=1.0,
            alphaG=1.0,
            alphaHJB=list(alphaHJB),
            alphaadj=list(alphaadj),
            device=device,
        )
        self.oc_problem_name = "Liquidation Portfolio"

        self.n_assets = n_assets

        # Terminal-impact smoothing in (u^2 + epsilon)^(gamma/2); also used in dX/dt.
        self.epsilon = epsilon

        

        # function to convert scalar or vector parameters into asset-aligned vectors of shape (n_assets,) on the correct device
        def _to_asset_vector(x, n_assets, device, name):
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            if x_t.ndim == 0:
                x_t = x_t.repeat(n_assets)
            elif x_t.ndim == 1 and x_t.numel() == n_assets:
                pass
            else:
                raise ValueError(f"{name} must be a scalar or a vector of length {n_assets}")
            return x_t

        
        
        # # Market model: inventory risk scale sigma; linear price impact kappa; nonlinear cash friction eta, gamma.
        # self.sigma = sigma
        # self.kappa = kappa
        # self.eta = eta
        self.gamma = gamma
        self.sigma = _to_asset_vector(sigma, n_assets, device, "sigma")
        self.kappa = _to_asset_vector(kappa, n_assets, device, "kappa")
        self.eta = _to_asset_vector(eta, n_assets, device, "eta")


        # Terminal penalty weight on leftover inventory (with -X term in G).
        self.alpha = alpha



        # Initial-condition distribution / levels for sampling z0 at episode start.
        # self.q0_min = q0_min
        # self.q0_max = q0_max
        # self.S0 = S0
        self.q0_min = _to_asset_vector(q0_min, n_assets, device, "q0_min")
        self.q0_max = _to_asset_vector(q0_max, n_assets, device, "q0_max")
        self.S0 = _to_asset_vector(S0, n_assets, device, "S0")
        self.X0 = X0


    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Running cost L'(t, z, u) of the **reduced** Almgren-Chriss problem.

        ``L'`` absorbs the negated cash-flow term and the nonlinear impact
        cost that used to live in ``dX/dt`` of the legacy formulation:

            L'(t, z, u) = 1/2 sum_i sigma_i^2 q_i^2
                          - sum_i S_i u_i
                          + sum_i eta_i * (u_i^2 + epsilon)^(gamma/2).

        Equivalence with the legacy 3-state J_B is exact:
        ``J_B = -X(0) + integral L' dt + alpha sum_i q_i(T)^2`` (see class
        docstring). Returns one scalar per batch element, shape
        ``(batch,)`` (or 0-dim under the unbatched ``vmap`` branch).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = self.n_assets
        q = z[:, :n]
        S = z[:, n:2 * n]

        inv_risk    = 0.5 * torch.sum((self.sigma ** 2) * (q ** 2), dim=1)
        # Cash flow appears with a MINUS sign: the OC minimises the running
        # cost, and trading revenue S·u is income (negative cost).
        cash_flow   = -torch.sum(S * u, dim=1)
        # Smooth impact cost — same epsilon-regularised form that previously
        # lived in dX/dt; entering L' is what makes ∂²H/∂u² a problem
        # constant (2η for γ=2) rather than a learned quantity.
        impact_cost = torch.sum(
            self.eta * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0), dim=1
        )

        lag = inv_risk + cash_flow + impact_cost
        return lag[0] if squeeze else lag

    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        ``∂L'/∂u``: nonzero in the reduced formulation (this is the whole
        point — it gives the FP iteration a strong, well-conditioned
        gradient signal that does NOT depend on the learned costate).

        Per asset i:

            ∂L'/∂u_i = -S_i + eta_i * gamma * u_i * (u_i^2 + eps)^(gamma/2 - 1).

        For γ = 2 this collapses to ``-S + 2η u`` and ``∂²L'/∂u²|_{γ=2} = 2η``,
        the constant Hessian that drives one-step contractivity of the
        T-operator at ``alpha_fp = 1/(2η)``.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = self.n_assets
        S = z[:, n:2 * n]

        grad_impact = (
            self.eta * self.gamma * u
            * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0 - 1.0)
        )
        grad = -S + grad_impact

        return grad[0] if squeeze else grad

    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Right-hand side ``f'(t, z, u) = dz/dt`` of the **reduced** controlled
        ODE on ``z = (q, S)``:

            dq_i/dt = -u_i,        dS_i/dt = -kappa_i u_i.

        ``f'`` is independent of ``z`` (so ``∂f'/∂z ≡ 0``) and linear in
        ``u``. Cash dynamics ``dX/dt = S u - η (u²+ε)^{γ/2}`` are handled
        outside the OC by :meth:`compute_cash_flow`.

        Returns ``(batch, state_dim) = (batch, 2n)`` — or the unbatched
        slice under the ``vmap`` branch.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        dq = -u                       # (batch, n)
        dS = -self.kappa * u          # (batch, n) — kappa broadcasts per asset

        result = torch.cat((dq, dS), dim=1)   # (batch, 2n)
        return result[0] if squeeze else result

    def compute_grad_lagrangian_(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Unbatched ``∂L'/∂u`` for vmap/jacrev (single-sample variant of
        :meth:`compute_grad_lagrangian`).

        Same closed-form as the batched routine — ``-S + η γ u (u² + ε)^{γ/2-1}``
        per asset — on ``(state_dim,) / (control_dim,)`` shapes, matching
        the contract of :meth:`ImplicitOC.compute_grad_H_u_`.
        """
        n = self.n_assets
        S = z[n:2 * n]                                            # (n,)
        grad_impact = (
            self.eta * self.gamma * u
            * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0 - 1.0)
        )
        return -S + grad_impact                                   # (n,)

    def compute_grad_f_u_(
        self, z: torch.Tensor, u: torch.Tensor, grad_f_u_: torch.Tensor
    ) -> torch.Tensor:
        """Unbatched ``∂f'/∂u`` writer for vmap/jacrev (single-sample
        variant of :meth:`compute_grad_f_u`).

        Buffer shape is ``(control_dim, state_dim) = (n, 2n)``. Layout
        ``grad[i, s] = ∂f_s/∂u_i``:

            grad[i, i]     = -1               (∂(dq_i/dt)/∂u_i)
            grad[i, n + i] = -kappa_i         (∂(dS_i/dt)/∂u_i)

        No cash row anymore — ``X`` is observer-only.
        """
        n = self.n_assets
        for i in range(n):
            grad_f_u_[i, i] = -1.0
            grad_f_u_[i, n + i] = -self.kappa[i]
        return grad_f_u_

    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Jacobian ``∂f'/∂u`` of the reduced dynamics with respect to ``u``.

        Shape ``(batch, control_dim, state_dim) = (B, n, 2n)``. Block layout:

            grad[:, i, i]       = -1                  (∂(dq_i/dt)/∂u_i)
            grad[:, i, n + i]   = -kappa_i            (∂(dS_i/dt)/∂u_i)

        Constant in ``z`` and ``u`` (linear dynamics) — this is the other
        half of why the FP iteration is one-step exact for γ=2: ``∂_u² H``
        has no ``u``-dependence either.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        n = self.n_assets
        grad = torch.zeros(batch, self.control_dim, self.state_dim, device=z.device)

        for i in range(n):
            grad[:, i, i] = -1.0
            grad[:, i, n + i] = -self.kappa[i]

        return grad[0] if squeeze else grad

    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Jacobian ``∂f'/∂z`` of the reduced dynamics with respect to ``z``.

        Shape ``(batch, state_dim, state_dim) = (B, 2n, 2n)``. **Identically
        zero** in the reduced problem: neither ``dq/dt = -u`` nor ``dS/dt =
        -κ u`` depends on ``z``. The legacy ``∂f_X/∂z_{S_j} = u_j`` slot
        no longer exists — that coupling now sits inside ``L'`` via the
        ``-S·u`` term and is consumed automatically by ``compute_grad_H_z``
        through ``compute_grad_lagrangian_z`` (default zero in this class
        for the inventory subgrid; the price-row gradient comes from
        autograd-on-L if upstream code asks for it).

        Layout convention matches :meth:`compute_grad_f_u` /
        :meth:`ImplicitOC.compute_grad_H_z`: ``grad[b, i, s] = ∂f_s/∂z_i``.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        grad = torch.zeros(batch, self.state_dim, self.state_dim, device=z.device)

        return grad[0] if squeeze else grad

    # ------------------------------------------------------------------
    # State gradient of the running cost — needed because L' now depends
    # on z (through both q and S). Mirrors the pattern used by
    # MultiBicycleOC / QuadcopterOC: provide ``compute_grad_lagrangian_z``
    # and override ``compute_grad_H_z`` so adjoint / HJB consistency
    # diagnostics stay correct when the user enables them
    # (``alphaadj > 0``).
    # ------------------------------------------------------------------
    def compute_grad_lagrangian_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """``∂L'/∂z`` of shape ``(batch, state_dim)``.

        Per asset i:

            ∂L'/∂q_i = sigma_i^2 * q_i      (inventory-risk gradient)
            ∂L'/∂S_i = -u_i                  (cash-flow gradient: trading
                                              one unit at price S_i is
                                              one unit of revenue)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = self.n_assets
        q = z[:, :n]

        grad = torch.zeros(z.shape[0], self.state_dim, device=z.device)
        grad[:, :n]      = (self.sigma ** 2) * q
        grad[:, n:2 * n] = -u

        return grad[0] if squeeze else grad

    def compute_grad_H_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """``∂H'/∂z = ∂L'/∂z + ∂(p^T f')/∂z``.

        The parent ``ImplicitOC.compute_grad_H_z`` only returns the
        ``∂(p^T f)/∂z`` half; we add the L'-gradient here so the
        adjoint sweep / HJB residual stays correct in this model.
        """
        grad_L_z      = self.compute_grad_lagrangian_z(t, z, u)
        grad_H_z_pTf  = super().compute_grad_H_z(t, z, u, p)
        return grad_L_z + grad_H_z_pTf

    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reduced terminal cost ``G'(z(T)) = alpha * sum_i q_i(T)^2``.

        The legacy ``-X(T)`` term has been absorbed into the running cost
        as ``-S·u`` (via ``J_B = -X(0) + J_B'``), so ``G'`` is now purely
        quadratic in ``q`` — a perfect match for the quadratic head of the
        ``Phi`` / ``TerminalAnchoredPhi`` value-function network.

        Args:
            z: Terminal state ``(q, S)``, shape ``(batch, 2n)`` or
               ``(2n,)``.

        Returns:
            Scalar terminal cost per trajectory, shape ``(batch,)`` or
            0-dim when unbatched.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        q = z[:, :self.n_assets]
        G = self.alpha * torch.sum(q ** 2, dim=1)
        return G[0] if squeeze else G

    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient ``∂G'/∂z`` of the reduced terminal cost.

        Only the ``q`` block is nonzero: ``∂G'/∂q_i = 2 alpha q_i`` and
        ``∂G'/∂S_i = 0``. The legacy ``∂G/∂X = -1`` slot no longer exists
        (no ``X`` in the OC state). Used by :class:`TerminalAnchoredPhi`
        to hard-anchor ``p_q(T) = 2α q(T)``, ``p_S(T) = 0``.

        Returns ``(batch, state_dim)`` (or 1D under the unbatched branch).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        grad = torch.zeros(batch, self.state_dim, device=z.device)

        q = z[:, :self.n_assets]
        grad[:, :self.n_assets] = 2.0 * self.alpha * q

        return grad[0] if squeeze else grad

    # ------------------------------------------------------------------
    # Closed-form PMP optimum (γ=2 Almgren-Chriss)
    # ------------------------------------------------------------------
    def has_closed_form_u_star(self) -> bool:
        """γ=2 makes ``∂_u H`` linear in u, so a closed-form minimiser exists."""
        return abs(float(self.gamma) - 2.0) < 1e-6

    def optimal_u_from_costate(
        self, t: TimeLike, z: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form ``argmin_u H'`` at γ=2 in the **reduced** problem.

        With ``H' = L' + p^T f'`` and ``L'`` carrying the
        ``-S·u + η u²`` terms, stationarity ``∂_u H' = 0`` gives, per
        asset:

            -S_i + 2 η_i u_i  -  p_{q_i}  -  κ_i p_{S_i}  =  0
            =>  u*_i = (S_i + p_{q_i} + κ_i p_{S_i}) / (2 η_i).

        No ``p_X`` appears anywhere — the formula is a strictly
        well-conditioned linear function of the learned costate ``p ∈
        R^{2n}``. Inputs:

        * ``z``: ``(B, 2n)`` (or ``(2n,)``)
        * ``p``: ``(B, 2n)`` (or ``(2n,)``)

        Returns ``(B, n)`` (or ``(n,)`` unbatched).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            p = p.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = self.n_assets
        S   = z[:, n:2 * n]               # (B, n)
        p_q = p[:, :n]                    # (B, n)
        p_S = p[:, n:2 * n]               # (B, n)

        u = (S + p_q + self.kappa * p_S) / (2.0 * self.eta)
        return u[0] if squeeze else u

    # ------------------------------------------------------------------
    # Costate / value-function network factory override
    # ------------------------------------------------------------------
    def make_p_net(
        self,
        hidden_dim: int = 50,
        n_resnet_layers: int = 3,
        device: "str | None" = None,
    ):
        """Return a :class:`TerminalAnchoredPhi` wrapping a generic ``Phi`` backbone.

        The reduced Almgren-Chriss terminal cost ``G'(z) = α ‖q‖²`` is
        purely quadratic in ``q`` and analytically known, so we hard-anchor
        the architecture via ``phi(t, z) = G'(z) + (T - t) N_theta(t, z)``.
        This guarantees ``phi(T, z) = G'(z)`` and consequently
        ``p_q(T) = 2α q(T)``, ``p_S(T) = 0`` by construction — no soft
        adjoint penalty needed. The wrapper is generic and only depends on
        :meth:`compute_G` / :meth:`compute_grad_G_z`.
        """
        from ImplicitNets import Phi, TerminalAnchoredPhi
        dev = device or self.device
        backbone = Phi(n_resnet_layers, hidden_dim, self.state_dim, dev=dev)
        return TerminalAnchoredPhi(backbone, self, dev=dev)

    def sample_initial_condition(self):
        """
        Sample initial OC states for training rollouts.

        Returns shape ``(batch, 2n) = (batch, state_dim)``:
        ``z0 = concat(q0, S0)``. ``q0`` is uniform on
        ``[q0_min, q0_max]``; ``S0`` is constant across the batch.

        ``X0`` is **not** part of the OC state. It is reserved for the
        observer cash account and consumed by :meth:`compute_cash_flow`
        callers (e.g. the rollout adapter in
        :class:`benchmarking.solvers.JFBPolicyRollout`).
        """
        q0 = (
            self.q0_min.unsqueeze(0)
            + (self.q0_max - self.q0_min).unsqueeze(0)
            * torch.rand(self.batch_size, self.n_assets, device=self.device)
        )
        S0 = self.S0.unsqueeze(0).expand(self.batch_size, -1)

        return torch.cat((q0, S0), dim=1).to(self.device)

    # ------------------------------------------------------------------
    # Observer cash account (NOT part of the OC state)
    # ------------------------------------------------------------------
    def compute_cash_flow(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Right-hand side of the cash-account ODE, integrated in parallel
        to ``compute_f`` for plotting / financial reporting only.

            dX/dt = sum_i [ S_i u_i  -  eta_i (u_i^2 + epsilon)^(gamma/2) ].

        This is exactly the legacy ``dX/dt`` row that was removed from
        :meth:`compute_f` when ``X`` was eliminated from the OC state.
        It is **never** called by ``ImplicitOC`` or the JFB / FP
        machinery; it exists purely so the rollout adapters can produce a
        ``(batch, 1)`` cash trajectory with the same Euler discretisation
        as the OC state.

        Returns ``(batch, 1)`` (or ``(1,)`` unbatched).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n = self.n_assets
        S = z[:, n:2 * n]
        trading_revenue = S * u
        impact_cost = self.eta * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0)
        dX = torch.sum(trading_revenue - impact_cost, dim=1, keepdim=True)
        return dX[0] if squeeze else dX

    # ------------------------------------------------------------------
    # Legacy plotting (kept for reference, superseded by panels/to_trajectory)
    # ------------------------------------------------------------------
    # def plot_position_trajectories(
    #     self,
    #     z_traj: torch.Tensor,
    #     policy=None,
    #     save_path: str | None = None,
    #     n_show: int = 5,
    #     title_str: str = "Liquidation policy rollout",
    # ):
    #     """
    #     Plot ``q(t), u(t), S(t), X(t)`` for the first ``n_show`` trajectories of
    #     ``z_traj`` (shape ``(batch, state_dim, nt+1)``).
    #
    #     This method matched the calling convention of
    #     :meth:`OptimalControlTrainer.train`, which invoked::
    #
    #         self.oc_problem.plot_position_trajectories(z_traj.detach(), self.policy)
    #
    #     every ``plot_frequency`` epochs. ``policy`` is optional: when supplied the
    #     trading rate ``u(t)`` is reconstructed by evaluating ``policy(z, t)`` along
    #     the rolled-out state trajectory (exactly what ``generate_trajectory`` did
    #     during the Euler march), otherwise the control panel is left empty.
    #
    #     ``save_path`` defaults to an auto-numbered PNG under
    #     ``results_<class_name>/standard_mode/plots/`` so mid-training snapshots
    #     accumulate rather than overwrite each other. The figure is always closed
    #     after writing, so the call never blocks training.
    #     """
    #     import os
    #     import matplotlib.pyplot as plt
    #
    #     z_traj = z_traj.detach()
    #     batch, _, nt1 = z_traj.shape
    #     nt = nt1 - 1
    #     n_show = max(1, min(n_show, batch))
    #
    #     t = torch.linspace(self.t_initial, self.t_final, nt1).cpu().numpy()
    #
    #     q = z_traj[:n_show, 0, :].cpu().numpy()
    #     S = z_traj[:n_show, 1, :].cpu().numpy()
    #     X = z_traj[:n_show, 2, :].cpu().numpy()
    #
    #     u_arr = None
    #     if policy is not None:
    #         dt = (self.t_final - self.t_initial) / nt
    #         u_buf = torch.zeros(n_show, self.control_dim, nt, device=z_traj.device)
    #         with torch.no_grad():
    #             for i in range(nt):
    #                 z_i = z_traj[:n_show, :, i]
    #                 t_i = self.t_initial + i * dt
    #                 u_buf[:, :, i] = policy(z_i, t_i).view(n_show, self.control_dim)
    #         u_arr = u_buf[:, 0, :].cpu().numpy()
    #
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    #     for b in range(n_show):
    #         axes[0, 0].plot(t, q[b], alpha=0.75, label=f"traj {b}" if n_show > 1 else None)
    #         axes[1, 0].plot(t, S[b], alpha=0.75)
    #         axes[1, 1].plot(t, X[b], alpha=0.75)
    #         if u_arr is not None:
    #             axes[0, 1].plot(t[:-1], u_arr[b], alpha=0.75)
    #
    #     axes[0, 0].set_title("Inventory  q(t)")
    #     axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel("q"); axes[0, 0].grid(True)
    #     axes[0, 1].set_title("Trading rate  u(t)")
    #     axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel("u"); axes[0, 1].grid(True)
    #     axes[1, 0].set_title("Impacted price  S(t)")
    #     axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel("S"); axes[1, 0].grid(True)
    #     axes[1, 0].ticklabel_format(style="plain", axis="y", useOffset=False)
    #     axes[1, 1].set_title("Cash  X(t)")
    #     axes[1, 1].set_xlabel("t"); axes[1, 1].set_ylabel("X"); axes[1, 1].grid(True)
    #
    #     if n_show > 1:
    #         axes[0, 0].legend(fontsize=8, loc="best")
    #
    #     fig.suptitle(title_str)
    #     fig.tight_layout(rect=[0, 0, 1, 0.96])
    #
    #     if save_path is None:
    #         if not hasattr(self, "_plot_counter"):
    #             self._plot_counter = 0
    #         self._plot_counter += 1
    #         from core.paths import results_dir
    #         plot_dir = results_dir(type(self).__name__, "training", "training-plots")
    #         save_path = os.path.join(plot_dir, f"rollout_{self._plot_counter:04d}.png")
    #
    #     fig.savefig(save_path, dpi=150, bbox_inches="tight")
    #     plt.close(fig)
    #     print(f"    -> saved rollout figure to {os.path.abspath(save_path)}")

    # ------------------------------------------------------------------
    # BenchmarkPlotter integration
    # ------------------------------------------------------------------
    def panels(self) -> List[Panel]:
        """Multi-asset Almgren-Chriss panel layout (``q_i, u*_i, S_i`` per
        asset, plus a shared ``X(t)``).

        Consumed by :class:`benchmarking.BenchmarkPlotter` (see the trainer's
        plotting dispatch). For ``n_assets == 1`` this collapses to the
        legacy four-panel ``q, u*, S, X`` layout (same indices, just with no
        subscript in the titles).
        """
        return liquidation_panels(self.n_assets)

    def to_trajectory(
        self,
        z_traj: torch.Tensor,
        policy=None,
        path_index: int = 0,
        label: str = "JFB",
        x_traj: "torch.Tensor | None" = None,
    ) -> Trajectory:
        """Pack a rolled-out tensor into a :class:`benchmarking.Trajectory`.

        Parameters
        ----------
        z_traj
            Output of :meth:`generate_trajectory(..., return_full_trajectory=True)`,
            shape ``(batch, state_dim, nt+1) = (batch, 2n, nt+1)``. Carries
            only the OC state ``(q, S)``.
        policy
            Optional callable ``policy(z, t) -> u``. If supplied, the control
            ``u(t)`` is reconstructed by evaluating the policy along the
            selected trajectory exactly as :meth:`generate_trajectory` does
            during the forward Euler march.
        path_index
            Which sample of the batch to package (default ``0``).
        label
            Legend label propagated into the resulting figure.
        x_traj
            Optional observer cash trajectory of shape ``(batch, 1, nt+1)``
            produced by :meth:`generate_trajectory(..., return_cash=True)`.
            When supplied, the returned ``Trajectory.z`` is packed as
            ``concat([q, S, X], axis=1)`` of shape ``(nt+1, 2n+1)`` so the
            existing plotter (which extracts ``X`` at column ``2n``) keeps
            working unchanged. When ``x_traj`` is ``None`` and ``policy``
            is supplied, ``X`` is reconstructed by trapezoidal quadrature
            from the policy outputs and the rolled-out ``(q, S)`` so the
            cash panel is always populated.
        """
        z_traj = z_traj.detach()
        batch, state_dim, nt1 = z_traj.shape
        if not 0 <= path_index < batch:
            raise IndexError(f"path_index={path_index} out of range for batch={batch}")
        nt = nt1 - 1

        t_np = np.linspace(self.t_initial, self.t_final, nt1)
        z_np = z_traj[path_index].transpose(0, 1).cpu().numpy()  # (nt+1, 2n)

        u_np = None
        if policy is not None:
            dt = (self.t_final - self.t_initial) / nt
            u_buf = torch.zeros(self.control_dim, nt, device=z_traj.device)
            with torch.no_grad():
                z_path = z_traj[path_index : path_index + 1]  # keep batch axis = 1
                for i in range(nt):
                    t_i = self.t_initial + i * dt
                    u_i = policy(z_path[:, :, i], t_i).view(1, self.control_dim)
                    u_buf[:, i] = u_i[0]
            u_np = u_buf.transpose(0, 1).cpu().numpy()  # (nt, control_dim)

        # Cash column: prefer the supplied observer; otherwise integrate
        # `compute_cash_flow` from the policy controls (left-endpoint Euler,
        # matching `generate_trajectory(..., return_cash=True)`).
        n = self.n_assets
        x_col = None
        if x_traj is not None:
            x_col = x_traj.detach()[path_index, 0].cpu().numpy()        # (nt+1,)
        elif u_np is not None:
            dt = (self.t_final - self.t_initial) / nt
            x_col = np.zeros(nt1, dtype=np.float64)
            x_col[0] = float(self.X0)
            S_path = z_np[:, n:2 * n]
            for i in range(nt):
                u_i = u_np[i]
                S_i = S_path[i]
                rev = float(np.sum(S_i * u_i))
                imp = float(np.sum(
                    np.asarray(self.eta.detach().cpu().numpy())
                    * (u_i ** 2 + self.epsilon) ** (self.gamma / 2.0)
                ))
                x_col[i + 1] = x_col[i] + dt * (rev - imp)

        if x_col is not None:
            z_packed = np.concatenate([z_np, x_col[:, None]], axis=1)   # (nt+1, 2n+1)
        else:
            z_packed = z_np

        return Trajectory(
            t=t_np,
            z=z_packed,
            u=u_np,
            label=label,
            style={"color": "#d6604d", "lw": 2.0},
        )

    def generate_trajectory(
        self,
        u,
        z0,
        nt,
        return_full_trajectory: bool = False,
        return_cash: bool = False,
    ):
        """
        Forward-simulate the **reduced** controlled ODE with explicit Euler.

        **OC state.** ``z0`` has shape ``(batch, state_dim) = (batch, 2n)``
        and the trajectory buffer ``traj`` has shape ``(batch, 2n, nt + 1)``.
        ``traj[:, :, 0]`` is the initial condition.

        **Control input.** Same contract as before:

        1. tensor ``(batch, control_dim, nt)`` for open-loop, or
        2. callable ``u(z, t) -> (batch, control_dim)`` for feedback.

        **Cash observer (optional).** When ``return_cash=True`` the method
        also integrates ``compute_cash_flow`` in parallel — same Euler
        discretisation, started from ``self.X0`` — and returns
        ``(traj, x_traj)`` where ``x_traj`` has shape
        ``(batch, 1, nt + 1)``. ``X`` is **never** read by the OC update,
        so this branch is purely additive and OC training behaviour stays
        bit-for-bit identical when ``return_cash=False``.

        **Return value.**

        * ``return_full_trajectory=False, return_cash=False``: terminal
          ``traj[:, :, -1]`` of shape ``(batch, 2n)`` (legacy default).
        * ``return_full_trajectory=True,  return_cash=False``: full
          ``traj`` of shape ``(batch, 2n, nt+1)``.
        * ``return_cash=True``: tuple ``(out, x_traj)`` where ``out`` is
          either the terminal slice or the full ``traj`` per the flag
          above, and ``x_traj`` is the parallel cash buffer.
        """
        batch = z0.shape[0]
        D = self.state_dim

        traj = torch.zeros(batch, D, nt + 1, device=z0.device)
        traj[:, :, 0] = z0
        dt = (self.t_final - self.t_initial) / nt
        t = self.t_initial

        if return_cash:
            x_traj = torch.zeros(batch, 1, nt + 1, device=z0.device)
            x_traj[:, :, 0] = float(self.X0)

        for i in range(nt):
            if torch.is_tensor(u):
                curr = u[:, :, i]
            else:
                curr = u(traj[:, :, i], t)
            traj[:, :, i + 1] = traj[:, :, i] + dt * self.compute_f(
                t, traj[:, :, i], curr
            )
            if return_cash:
                x_traj[:, :, i + 1] = x_traj[:, :, i] + dt * self.compute_cash_flow(
                    t, traj[:, :, i], curr
                )
            t += dt

        out = traj if return_full_trajectory else traj[:, :, -1]
        if return_cash:
            return out, x_traj
        return out


# Example usage / structural smoke test
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Local smoke test: derivative consistency + reduced-formulation
    # structural diagnostics.
    # ------------------------------------------------------------------
    device = "cpu"
    batch_size = 10
    nt = 100

    prob = LiquidationPortfolioOC(
        batch_size=batch_size,
        t_initial=0.0,
        t_final=2.0,
        nt=nt,
        n_assets=2,
        sigma=(0.02, 0.03),
        kappa=(1.0e-4, 2.0e-4),
        eta=(0.1, 0.15),
        gamma=2.0,
        epsilon=1.0e-2,
        alpha=30,
        q0_min=(0.5, 0.5),
        q0_max=(1.5, 1.5),
        S0=(1.0, 1.05),
        X0=0.0,
        device=device,
    )

    n = prob.n_assets

    # Build (z, u) seed batch matching the new state layout [q, S] (no X).
    q_seed = torch.tensor([[1.0] * n, [0.8] * n], dtype=torch.float32)
    S_seed = torch.tensor([[1.0] * n, [1.1] * n], dtype=torch.float32)
    test_z = torch.cat([q_seed, S_seed], dim=1)
    test_u = torch.tensor([[0.1] * n, [0.2] * n], dtype=torch.float32)
    test_z = test_z.repeat(batch_size // 2, 1).to(device)
    test_u = test_u.repeat(batch_size // 2, 1).to(device)

    print("Running gradient tests...")
    GradientTester.run_all_tests(prob, test_z, test_u)

    # ------------------------------------------------------------------
    # Structural diagnostics for the reduced formulation
    # ------------------------------------------------------------------
    print()
    print("=== Reduced LiquidationPortfolio diagnostics ===")

    # (1) state_dim must be 2 * n_assets
    assert prob.state_dim == 2 * n, (
        f"state_dim mismatch: got {prob.state_dim}, expected {2 * n}"
    )
    print(f"(1) state_dim == 2 * n_assets : OK  ({prob.state_dim})")

    # (2) ∂_u H residual at u* is zero (per closed-form formula)
    eta = prob.eta
    kappa = prob.kappa
    z_pt = test_z[:1].clone()                           # single sample
    p_pt = torch.randn(1, prob.state_dim) * 0.1
    t0 = float(prob.t_initial)
    t_pt = torch.tensor([t0])

    u_star = prob.optimal_u_from_costate(t0, z_pt, p_pt)
    grad_H_at_ustar = prob.compute_grad_H_u(t_pt, z_pt, u_star, p_pt)
    res_norm = float(torch.linalg.vector_norm(grad_H_at_ustar))
    print(
        f"(2) ||∂_u H(u*)||                 : {res_norm:.3e} "
        f"(should be ≈ 0)"
    )

    # (3) One-shot T-operator exactness: with α_fp = 1/(2η), one
    # gradient-descent step from any u_init lands on u*.
    alpha_fp = 1.0 / (2.0 * eta)                        # (n,)
    u_init = torch.randn_like(u_star)
    grad_H_init = prob.compute_grad_H_u(t_pt, z_pt, u_init, p_pt)
    u_after_one_step = u_init - alpha_fp * grad_H_init
    one_shot_err = float(torch.linalg.vector_norm(u_after_one_step - u_star))
    print(
        f"(3) ||T(u_init) - u*|| (1 step)   : {one_shot_err:.3e} "
        f"(should be ≈ 0 at γ=2)"
    )

    # (4) J_B  ==  -X(0) + J_B'  along a random open-loop rollout.
    #
    # Uses generate_trajectory(..., return_cash=True) so the same Euler grid
    # carries both the OC state (q, S) and the observer cash X. The cost
    # equality is exact up to Euler discretisation.
    z0 = prob.sample_initial_condition()                # (batch, 2n)
    u_open = 0.05 * torch.ones(prob.batch_size, prob.control_dim, prob.nt)
    z_traj_full, x_traj_full = prob.generate_trajectory(
        u_open, z0, prob.nt, return_full_trajectory=True, return_cash=True,
    )
    dt = (prob.t_final - prob.t_initial) / prob.nt

    # J_B' = ∫ L' dt + α‖q(T)‖²
    JB_prime = torch.zeros(prob.batch_size)
    t_loop = prob.t_initial
    for i in range(prob.nt):
        z_i = z_traj_full[:, :, i]
        u_i = u_open[:, :, i]
        JB_prime = JB_prime + dt * prob.compute_lagrangian(t_loop, z_i, u_i)
        t_loop += dt
    JB_prime = JB_prime + prob.compute_G(z_traj_full[:, :, -1])

    # J_B (legacy 3-state) = -X(T) + α‖q(T)‖² + ∫ ½σ²q² dt
    inv_risk_only = torch.zeros(prob.batch_size)
    t_loop = prob.t_initial
    for i in range(prob.nt):
        q_i = z_traj_full[:, :prob.n_assets, i]
        inv_risk_only = inv_risk_only + dt * 0.5 * torch.sum(
            (prob.sigma ** 2) * (q_i ** 2), dim=1,
        )
        t_loop += dt
    q_T = z_traj_full[:, :prob.n_assets, -1]
    X_T = x_traj_full[:, 0, -1]
    JB_legacy = -X_T + prob.alpha * torch.sum(q_T ** 2, dim=1) + inv_risk_only

    X0_b = torch.full((prob.batch_size,), float(prob.X0))
    equiv_err = float(torch.max(torch.abs(JB_legacy - (-X0_b + JB_prime))))
    print(
        f"(4) max |J_B - (-X(0) + J_B')|    : {equiv_err:.3e} "
        f"(should be O(Euler error))"
    )
