import torch
import numpy as np
from abc import ABC, abstractmethod
from utils import GradientTester
import matplotlib.pyplot as plt

TimeLike = float | torch.Tensor

class ImplicitOC(ABC):
    """
    A general class for Implicit Optimal Control problems.
    This abstract base class provides the structure for solving optimal control problems
    using implicit methods.
    """
    
    def __init__(self, state_dim, control_dim, batch_size, t_initial, t_final, nt, 
                 alphaL, alphaG,  device='cpu', alphaHJB = [0.0,0.0], alphaadj = [0.0,0.0],
                 track_all_fp_iters = False, pen_pos=False): 
        """
        Initialize the Implicit Optimal Control problem.
        
        Args:
            state_dim (int): Dimension of the state vector
            control_dim (int): Dimension of the control vector
            batch_size (int): Batch size for trajectory optimization
            t_initial (float): Initial time
            t_final (float): Final time
            nt (int): Number of time steps
            device (str): Device to perform computation on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.batch_size = batch_size
        self.t_initial = t_initial
        self.t_final = t_final
        self.nt = nt
        self.device = device
        self.h = (t_final - t_initial) / nt
        self.pen_pos = pen_pos

        self.oc_problem_name = ""

        # Loss function weights
        self.alphaL = alphaL  # Running cost weight
        self.alphaG = alphaG  # Terminal cost weight
        self.alphaHJB = alphaHJB  # HJB weight
        self.alphaadj = alphaadj  # adjoint weight
        self.use_HJB = True if (self.alphaHJB[0] + self.alphaHJB[1]) > 0.0 else False
        
        # Gradient tracking of fixed point iterations
        self.track_all_fp_iters = track_all_fp_iters

        # ------------------------------------------------------------------
        # Stochastic dynamics (state SDE) — deterministic by default.
        # ------------------------------------------------------------------
        # Number of independent Brownian motions driving the state. 0 means
        # the dynamics are a pure ODE (every legacy problem keeps this).
        # Subclasses with a diffusion term set this to a positive integer
        # and override ``compute_sigma`` / ``has_diffusion``.
        self.n_brownian = 0
        # Default number of Hutchinson probe vectors for the trace-of-Hessian
        # estimator in ``compute_trace_sigma_hess_phi``.
        self.n_hutch = 1
        # Monte-Carlo paths per initial condition used by ``compute_loss``.
        # 1 keeps the deterministic behaviour (and means MC averaging happens
        # across the batch / epochs). >1 tiles each z0 row so the objective is
        # an explicit expectation over independent Brownian paths per IC.
        self.n_paths_per_ic = 1
        # Subclasses with stochastic dynamics may set a dedicated RNG here;
        # the base class leaves it unset (``getattr(..., None)``).
        self.noise_generator = None

    @abstractmethod
    def compute_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Lagrangian (running cost).
        
        Args:
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Lagrangian values of shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def compute_grad_lagrangian(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the Lagrangian with respect to control.
        
        Args:
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of Lagrangian of shape (batch_size, control_dim)
        """
        pass
    
    @abstractmethod
    def compute_G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the terminal cost.
        
        Args:
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Terminal cost values of shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def compute_f(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the system dynamics.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Time derivative of z (dz/dt) of shape (batch_size, state_dim)
        """
        pass
    
    @abstractmethod
    def compute_grad_f_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the dynamics with respect to control.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of f w.r.t. u of shape (batch_size, control_dim, state_dim)
        """
        pass
    
    @abstractmethod
    def compute_grad_f_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the dynamics with respect to state.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of f w.r.t. z of shape (batch_size, state_dim, state_dim)
        """
        pass
    
    @abstractmethod
    def compute_grad_G_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the terminal cost G with respect to state.

        Args:
            z (torch.Tensor): State vector of shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Gradient of G w.r.t. z of shape (batch_size, state_dim)
        """
        pass

    # ------------------------------------------------------------------
    # Optional closed-form PMP optimum
    # ------------------------------------------------------------------
    # Problems whose Hamiltonian admits an explicit minimiser in u (e.g. the
    # γ=2 Almgren-Chriss case where ∂_u H is linear in u) can opt in by
    # overriding both methods below. Diagnostic adapters in
    # ``benchmarking.policies`` consume this hook to evaluate u*(t, z)
    # directly from the learned costate, bypassing the inner FP solver.
    # The defaults keep every other problem class strictly opt-out so we
    # don't have to touch them.
    def has_closed_form_u_star(self) -> bool:
        """Return ``True`` iff :meth:`optimal_u_from_costate` is implemented."""
        return False

    def optimal_u_from_costate(
        self, t: TimeLike, z: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form ``argmin_u H(t, z, u, p)`` when available.

        Override only when the Hamiltonian admits an explicit minimiser.
        Shape contract: ``z`` ``(batch, state_dim)``, ``p`` ``(batch, state_dim)``;
        returns ``(batch, control_dim)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form u*; use the FP policy."
        )

    # ------------------------------------------------------------------
    # Costate / value-function network factory
    # ------------------------------------------------------------------
    # Each problem can advertise the architecture best suited to its known
    # value-function structure. The default returns a plain ``Phi`` so
    # call-sites that have always built ``Phi(3, 50, state_dim)`` keep
    # working untouched.  Subclasses override only when they want to inject
    # an architectural prior (e.g. a terminal-anchored wrapper).
    def make_p_net(
        self,
        hidden_dim: int = 50,
        n_resnet_layers: int = 3,
        device: "str | None" = None,
    ):
        """Construct the recommended costate / value-function network for this problem.

        Default: a generic :class:`Phi` matching the historical
        ``Phi(3, 50, state_dim)`` shape used across the examples. Override
        in subclasses to return a problem-specific architecture (e.g.
        :class:`TerminalAnchoredPhi` wrapping a backbone).
        """
        from ImplicitNets import Phi
        return Phi(
            n_resnet_layers,
            hidden_dim,
            self.state_dim,
            dev=device or self.device,
        )


    def compute_general_H(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the generalized Hamiltonian H = L + p^T f.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Hamiltonian values of shape (batch_size,)
        """
        f_val = self.compute_f(t, z, u)
        
        # Compute Lagrangian
        L_val = self.compute_lagrangian(t, z, u)
        
        # Compute inner product p^T f
        inner_product = torch.sum(p * f_val, dim=1)
        
        # Compute Hamiltonian
        H_val = L_val + inner_product
        
        return H_val

    # ------------------------------------------------------------------
    # Stochastic dynamics: diffusion + second-order (trace) generator
    # ------------------------------------------------------------------
    # The default implementations below make every existing (deterministic)
    # problem behave exactly as before: ``has_diffusion()`` is ``False`` and
    # ``compute_sigma`` returns zeros. A subclass models a state SDE
    #
    #     dz = f(t, z, u) dt + sigma(t, z, u) dW,   W in R^{n_brownian}
    #
    # by setting ``self.n_brownian`` and overriding ``compute_sigma`` (and,
    # if the diffusion can be switched off at runtime, ``has_diffusion``).
    def has_diffusion(self) -> bool:
        """Return ``True`` iff the state dynamics carry a diffusion term.

        Default ``False`` so all deterministic problems and every integrator
        guarded by this flag stay byte-for-byte unchanged.
        """
        return False

    def compute_sigma(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Diffusion matrix ``sigma(t, z, u)`` of the state SDE.

        Shape contract ``(batch, state_dim, n_brownian)``. The default is an
        all-zeros tensor (no noise). Subclasses with stochastic dynamics
        override this; for *additive* (control-independent) noise the ``u``
        argument is ignored.
        """
        batch = z.shape[0]
        n_bm = max(int(self.n_brownian), 1)
        return torch.zeros(batch, self.state_dim, n_bm, device=z.device, dtype=z.dtype)

    def diffusion_increment(
        self,
        t: TimeLike,
        z: torch.Tensor,
        u: torch.Tensor,
        dt: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler-Maruyama state increment ``sqrt(dt) * sigma(t,z,u) dW``.

        Returns a ``(batch, state_dim)`` tensor of zeros when the problem is
        deterministic, so callers can unconditionally add it to the Euler
        drift step ``z + dt f``. ``dW`` is a standard Gaussian in
        ``R^{n_brownian}`` scaled by ``sqrt(dt)``.

        The increment is intentionally detached from the autograd graph: the
        Brownian shock is exogenous data, not a function of the parameters,
        which keeps the JFB Term-I gradient structure intact (additive noise
        is a constant offset per step).
        """
        if not self.has_diffusion():
            return torch.zeros_like(z)

        batch = z.shape[0]
        n_bm = max(int(self.n_brownian), 1)
        with torch.no_grad():
            sigma = self.compute_sigma(t, z, u)            # (B, state_dim, n_bm)
            dW = torch.randn(
                batch, n_bm, device=z.device, dtype=z.dtype, generator=generator
            ) * float(dt) ** 0.5
            incr = torch.bmm(sigma, dW.unsqueeze(-1)).squeeze(-1)   # (B, state_dim)
        return incr

    def compute_trace_sigma_hess_phi(
        self,
        t: TimeLike,
        z: torch.Tensor,
        p_net,
        u: torch.Tensor | None = None,
        n_hutch: int | None = None,
        generator: torch.Generator | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        r"""Hutchinson estimate of ``Tr( Sigma * Hess_zz phi )``.

        With ``Sigma = sigma sigma^T`` and ``H = nabla^2_{zz} phi`` (both
        symmetric), use the variance-reduced symmetric identity

            Tr(Sigma H) = Tr(sigma^T H sigma) = E_v[ (sigma v)^T H (sigma v) ],

        with ``v`` a Rademacher vector in ``R^{n_brownian}``. Each probe costs
        one Hessian-vector product, obtained by differentiating
        ``grad_z phi`` (which is itself one backward pass through ``p_net``).

        Args
        ----
        z : (batch, state_dim) tensor. Must allow autograd w.r.t. ``z``;
            this routine internally re-attaches a leaf copy so callers do not
            have to.
        p_net : value-function network exposing ``getPhi(t, z) -> (batch, 1)``.
        n_hutch : number of probe vectors (default ``self.n_hutch``).
        generator : optional ``torch.Generator`` for reproducible probes.
        create_graph : keep the graph so the estimate is differentiable in
            the network parameters (needed when used inside a training loss).

        Returns
        -------
        (batch,) tensor: the (unbiased) estimate of ``Tr(Sigma H)``.
        """
        if not self.has_diffusion():
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        if n_hutch is None:
            n_hutch = int(self.n_hutch)

        batch = z.shape[0]
        # Work on a leaf that requires grad so we can take HVPs even if the
        # incoming ``z`` is detached (e.g. JFB stop-gradient state).
        z_leaf = z.detach().clone().requires_grad_(True)

        # sigma(t, z, u): (B, state_dim, n_brownian)
        if u is None:
            u = torch.zeros(batch, self.control_dim, device=z.device, dtype=z.dtype)
        sigma = self.compute_sigma(t, z_leaf, u)

        # grad_z phi: (B, state_dim), graph retained for the HVP / params.
        phi = p_net.getPhi(t, z_leaf)                      # (B, 1)
        grad_phi = torch.autograd.grad(
            phi.sum(), z_leaf, create_graph=True, retain_graph=True
        )[0]                                               # (B, state_dim)

        trace_est = torch.zeros(batch, device=z.device, dtype=z.dtype)
        for _ in range(n_hutch):
            # Rademacher probe in Brownian space, mapped to state space.
            v = torch.randint(
                0, 2, (batch, max(int(self.n_brownian), 1)),
                device=z.device, generator=generator, dtype=z.dtype,
            ) * 2.0 - 1.0                                  # (B, n_brownian)
            w = torch.bmm(sigma, v.unsqueeze(-1)).squeeze(-1)   # (B, state_dim)

            # HVP: H w = grad_z( (grad_phi . w) ).
            is_last = create_graph
            Hw = torch.autograd.grad(
                (grad_phi * w.detach()).sum(),
                z_leaf,
                create_graph=create_graph,
                retain_graph=True,
            )[0]                                           # (B, state_dim)
            trace_est = trace_est + (Hw * w).sum(dim=1)

        trace_est = trace_est / float(n_hutch)
        return trace_est

    def compute_general_H_stoch(
        self,
        t: TimeLike,
        z: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        p_net=None,
        n_hutch: int | None = None,
        generator: torch.Generator | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Second-order Hamiltonian ``H = L + p^T f + 0.5 Tr(Sigma Hess phi)``.

        Falls back exactly to :meth:`compute_general_H` when the problem is
        deterministic or ``p_net`` is not supplied (so the diffusion trace
        cannot be evaluated). ``p`` is used for the first-order ``p^T f``
        term; the trace term needs the full value network ``p_net``.
        """
        H_det = self.compute_general_H(t, z, u, p)
        if (not self.has_diffusion()) or (p_net is None):
            return H_det
        trace = self.compute_trace_sigma_hess_phi(
            t, z, p_net, u=u, n_hutch=n_hutch,
            generator=generator, create_graph=create_graph,
        )
        return H_det + 0.5 * trace
    
    def compute_grad_H_u(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the Hamiltonian with respect to control.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Gradient of H w.r.t. u of shape (batch_size, control_dim)
        """

        batch_size = z.shape[0]
        
        # Compute gradient of Lagrangian
        grad_term1 = self.compute_grad_lagrangian(t, z, u)
        
        # Compute gradient of dynamics
        grad_f_u_term = self.compute_grad_f_u(t, z, u)
        
        # Compute gradient of p^T f w.r.t. u
        p = p.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
        grad_term2 = torch.bmm(grad_f_u_term, p).view(batch_size, self.control_dim)
        
        # Compute total gradient
        grad_H_u_val = grad_term1 + grad_term2
        
        return grad_H_u_val

    def compute_grad_H_u_(
        self,
        t: TimeLike,
        z: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        grad_f_u_term: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the non-batch gradient of the Hamiltonian with respect to control.

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (state_dim,)
            u (torch.Tensor): Control input of shape (control_dim,)
            p (torch.Tensor): Costate vector of shape (state_dim,)
            grad_f_u_term: gradient of f with respect to u of shape (control_dim, state_dim)

        Returns:
            torch.Tensor: Gradient of H w.r.t. u of shape (control_dim,)
        """
        # Compute gradient of Lagrangian
        grad_term1 = -1.0*self.compute_grad_lagrangian_(t, z, u)

        # Compute gradient of dynamics
        grad_f_u_term = -1.0*self.compute_grad_f_u_(z, u, grad_f_u_term)

        # Compute gradient of p^T f w.r.t. u
        grad_term2 = torch.matmul(grad_f_u_term, p)

        # Compute total gradient
        grad_H_u_val = grad_term1 + grad_term2

        return grad_H_u_val
    
    def compute_grad_H_z(
        self, t: TimeLike, z: torch.Tensor, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the Hamiltonian with respect to state.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Gradient of H w.r.t. z of shape (batch_size, state_dim)
        """

        batch_size = z.shape[0]
        
        # Compute gradient of dynamics w.r.t. state
        grad_f_z_term = self.compute_grad_f_z(t, z, u)
        
        # Compute gradient of p^T f w.r.t. z
        p = p.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
        output = torch.bmm(grad_f_z_term, p).view(batch_size, self.state_dim)
        
        return output
    
    def solve_adjoint_eq(
        self, z: torch.Tensor, u: torch.Tensor | torch.nn.Module
    ) -> torch.Tensor:
        """
        Compute the adjoint equation for the Hamiltonian.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            pT (torch.Tensor): Terminal costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Adjoint variable of shape (batch_size, state_dim)

        Stochastic dynamics note
        ------------------------
        Under *additive* (control- and state-independent) diffusion such as
        the price noise in :class:`LiquidationPortfolioOC`
        (``dS = -kappa u dt + sigma_S dW``), the costate satisfies the BSDE

            dp = -( (d f / d z)^T p + grad_z L ) dt + q_t dW,     p(T) = grad_z G,

        whose *drift* is identical to the deterministic adjoint ODE solved
        here (the diffusion ``sigma_S`` does not depend on ``z`` or ``u``, so
        it contributes no extra drift term). The additional martingale
        integrand ``q_t`` only affects the path-wise (realised) costate, not
        its conditional mean. Hence this routine still returns the correct
        *conditional-mean* costate ``E[p_t | z_t]`` for the additive-noise
        case and is used purely as a diagnostic. A full second-adjoint / BSDE
        solver (needed when the diffusion depends on ``z`` or ``u``) is out of
        scope for the current state-only additive extension.
        """
        batch_size, nt = z.shape[0], z.shape[2]
        p = torch.zeros(batch_size, self.state_dim, nt)

        p[:, :, -1] = self.compute_grad_G_z(z[:,:,-1])
        h = (self.t_final - self.t_initial)/nt

        ti = self.t_final
        for i in range(nt - 2, -1, -1):
            
            if torch.is_tensor(u):
                assert nt == u.shape[2]
                current_u = u[:,:,i].view(batch_size, self.control_dim)
            elif hasattr(u, 'forward'):
                current_u = u(z[:,:,i], ti).view(batch_size, self.control_dim)
            p[:,:,i] = p[:,:,i+1] - h*self.compute_grad_H_z(ti, z[:,:,i+1], current_u, p[:,:,i+1])
            ti = ti - h
        return p

    def compute_loss(self, u, z0, z_t = None, p_t = None, phi_t = None, jac_based=False):
        """
        Compute the total cost of a trajectory.
        
        Args:
            u (torch.Tensor or callable): Control inputs of shape (batch_size, control_dim, nt)
                                         or a policy function that takes (z, t) and returns control
            z0 (torch.Tensor): Initial states of shape (batch_size, state_dim)
            
        Returns:
            tuple: (total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin)
        """
        # Monte-Carlo over Brownian paths: replicate each initial condition
        # ``n_paths_per_ic`` times so the rolled-out objective is an explicit
        # expectation over independent noise realisations per IC. The
        # ``diffusion_increment`` draws independent shocks per row, so the tiled
        # rows become independent paths. Deterministic problems (or
        # ``n_paths_per_ic == 1``) are untouched. Only applies to the feedback
        # (policy) rollout, not the ``jac_based`` / open-loop tensor paths.
        n_paths_per_ic = int(getattr(self, "n_paths_per_ic", 1))
        if (
            n_paths_per_ic > 1
            and not jac_based
            and hasattr(u, "forward")
            and self.has_diffusion()
        ):
            z0 = z0.repeat_interleave(n_paths_per_ic, dim=0)

        batch_size = z0.shape[0]
        running_cost = 0.0
        cHJB, cHJBfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        cadj, cadjfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        largest_grad_H_u = -1.0
        avg_grad_H_u = 0.0
        
        z = z0
        # ti = 0.0 * torch.ones(1, device=self.device)
        ti = 0.0
        # Integrate system using Euler's method
        if jac_based:
            assert self.nt == u.shape[2] and self.nt+1 == z_t.shape[2] \
            and self.nt+1 == p_t.shape[2] and self.nt+1 == phi_t.shape[2]
            for i in range(self.nt):
                current_u = u[:, :, i]
                z = z_t[:,:,i+1]
                gradPhi = p_t[:,:,i]
                running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                cadj = cadj + torch.mean(gradPhi[:,:self.state_dim]  -
                                        self.h*self.compute_grad_H_z(ti, z, current_u, gradPhi[:,:self.state_dim] ))

                    # double check sign
                cHJB = cHJB + self.h*torch.mean(torch.linalg.vector_norm(phi_t[:,:,i] - self.compute_general_H(ti, z, current_u, gradPhi[:,:self.state_dim]).view(-1,1), ord=2, dim=1)) 
                
                ti = ti + self.h

                # Calculate terminal cost
            temp_final_cost = self.compute_G(z)
            terminal_cost = torch.mean(temp_final_cost)
            gradPhi = p_t[:,:,i+1]
            z_temp = z.view(batch_size*self.num, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num, -1)
            diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(batch_size,-1)
            G = 0.5*torch.norm(diff_p, dim=1)**2            
            cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )
            # Terminal HJB residual is the boundary condition phi(T, z) = G(z).
            # Multiplying G(z) by self.alphaG (the loss weight) is a bug: the
            # residual is a property of the value function, not of the trainer's
            # regularisation preference.
            cHJBfin = torch.mean(torch.linalg.vector_norm(phi_t[:,:,i+1] - temp_final_cost.view(-1, 1), ord=2, dim=1))

        else:    
            if torch.is_tensor(u):
                assert self.nt == u.shape[2]
                for i in range(self.nt):
                    current_u = u[:, :, i].view(batch_size, self.control_dim)
                    z = z + self.h * self.compute_f(ti, z, current_u) \
                        + self.diffusion_increment(ti, z, current_u, self.h,
                                                   generator=getattr(self, "noise_generator", None))
                    running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                    ti = ti + self.h
                # Calculate terminal cost
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)
            elif hasattr(u, 'forward'):
                # Check if this is a direct control policy (no HJB computation needed)
                is_direct_control = getattr(u, 'is_direct_control', False)

                for i in range(self.nt):
                    current_u = u(z, ti, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                    # Euler-Maruyama drift + diffusion (zero increment when the
                    # problem is deterministic, so JFB training is unchanged).
                    z = z + self.h * self.compute_f(ti, z, current_u) \
                        + self.diffusion_increment(ti, z, current_u, self.h,
                                                   generator=getattr(self, "noise_generator", None))
                    running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)

                    # Only compute HJB and adjoint for implicit control methods
                    if not is_direct_control:
                        gradPhi = u.p_net(ti, z, full_grad=True)
                        cadj = cadj + torch.mean(gradPhi[:,:self.state_dim] - self.h*self.compute_grad_H_z(ti, z, current_u, gradPhi[:,:self.state_dim]))
                        if hasattr(u.p_net, "getPhi"):
                            # double check sign
                            #assert gradPhi[:,-1:].shape == self.compute_general_H(ti, z, current_u, gradPhi[:,:self.state_dim]).view(-1,1).shape
                            # Stochastic HJB: the generalized Hamiltonian gains
                            # the second-order term 1/2 Tr(Sigma Hess_zz phi),
                            # estimated by Hutchinson HVP. For deterministic
                            # problems ``compute_general_H_stoch`` reduces
                            # exactly to ``compute_general_H``. The trace is
                            # made differentiable only when it actually enters
                            # the optimizer (alphaHJB[0] > 0), so the default
                            # diagnostic mode (weight 0) stays cheap.
                            _hjb_create_graph = bool(self.alphaHJB[0] > 0) and torch.is_grad_enabled()
                            H_stoch = self.compute_general_H_stoch(
                                ti, z, current_u, gradPhi[:, :self.state_dim],
                                p_net=u.p_net, n_hutch=getattr(self, "n_hutch", 1),
                                generator=getattr(self, "noise_generator", None),
                                create_graph=_hjb_create_graph,
                            )
                            cHJB = cHJB + self.h*torch.mean(torch.linalg.vector_norm(gradPhi[:,-1:] - H_stoch.view(-1,1), ord=2, dim=1))

                        grad_H_u = self.compute_grad_H_u(ti, z, current_u, gradPhi[:,:self.state_dim])
                        max_norm_grad_H_u = torch.max(torch.linalg.vector_norm(grad_H_u, ord=2, dim=1)).item()
                        avg_grad_H_u += torch.mean(torch.linalg.vector_norm(grad_H_u, ord=2, dim=1)).item()
                        if max_norm_grad_H_u > largest_grad_H_u:
                            largest_grad_H_u = max_norm_grad_H_u

                    ti = ti + self.h

                # Calculate terminal cost
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)

                # Only compute terminal HJB and adjoint for implicit control methods
                if not is_direct_control:
                    if self.pen_pos:
                        if (self.oc_problem_name == "Double Integrator") or (self.oc_problem_name == "Multi Quadcopter"):
                            gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size*self.num_agents, -1))[:,:3]
                            cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                        elif self.oc_problem_name == "Multi Bicycle":
                            gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size*self.num_agents, -1))[:,:2]
                            cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                        elif self.oc_problem_name == "Single Quadcopter":
                            gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size, -1))[:,:3]
                            cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                    else:
                        cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )

                    if hasattr(u.p_net, "getPhi"):
                        #assert u.p_net.getPhi(ti,z).shape == temp_final_cost.view(-1, 1).shape
                        # Terminal HJB residual phi(T, z) - G(z); the loss weight
                        # alphaG must NOT scale the comparator G(z).
                        cHJBfin = torch.mean(torch.linalg.vector_norm(u.p_net.getPhi(ti,z) - temp_final_cost.view(-1, 1), ord=2, dim=1))

        # Calculate mean running cost
        running_cost = torch.mean(running_cost)
        
        # Calculate total cost
        total_cost = (self.alphaL * running_cost + self.alphaG * terminal_cost 
                      + self.alphaHJB[0] * cHJB + self.alphaHJB[1] * cHJBfin
                      + self.alphaadj[0] * cadj + self.alphaadj[1] * cadjfin)
        avg_grad_H_u = avg_grad_H_u / self.nt
        return total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u
    
    def compute_loss_verify(self, u, z0, z_t = None, p_t = None, phi_t = None, jac_based=False):
        """
        Compute the total cost of a trajectory as well as numerically verify certain 
        theoretical assumptions
        
        Args:
            u (torch.Tensor or callable): Control inputs of shape (batch_size, control_dim, nt)
                                         or a policy function that takes (z, t) and returns control
            z0 (torch.Tensor): Initial states of shape (batch_size, state_dim)
            
        Returns:
            tuple: (total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin)
        """
        batch_size = z0.shape[0]
        running_cost = 0.0
        cHJB, cHJBfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        cadj, cadjfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        smallest_M_sdval = torch.inf # Smallest singular value of M = dT/dtheta over all samples in batch and over all time steps 
        largest_M_sdval = -1.0 # Largest singular value of M over all samples in batch and over all time steps 
        smallest_lambda_min = torch.inf # Batchwise-largest largest eigenvalue of (MM^{T})^{-1} over all time steps 
        largest_lambda_max = -1.0 # Batchwise-smallest smallest eigenvalue of (MM^{T})^{-1} over all time steps
        avg_grad_T_u = 0.0
        largest_grad_T_u_batch = -1.0*torch.ones(batch_size, device=self.device) # Largest norm of grad of T with respect to u, for each sample
        largest_grad_H_u = -1.0
        avg_grad_H_u = 0.0
        
        z = z0
        ti = 0.0
        # Integrate system using Euler's method
        if jac_based:
            assert self.nt == u.shape[2] and self.nt+1 == z_t.shape[2] \
            and self.nt+1 == p_t.shape[2] and self.nt+1 == phi_t.shape[2]
            for i in range(self.nt):
                current_u = u[:, :, i]
                z = z_t[:,:,i+1]
                gradPhi = p_t[:,:,i]
                running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                cadj = cadj + torch.mean(gradPhi[:,:self.state_dim]  -
                                        self.h*self.compute_grad_H_z(ti, z, current_u, gradPhi[:,:self.state_dim] ))

                    # double check sign
                cHJB = cHJB + torch.mean(phi_t[:,:,i] -
                                    self.h*self.compute_general_H(ti, z, current_u, -gradPhi[:,:self.state_dim]).view(-1,1)) 
                
                ti = ti + self.h

                # Calculate terminal cost
            temp_final_cost = self.compute_G(z)
            terminal_cost = torch.mean(temp_final_cost)
            gradPhi = p_t[:,:,i+1]
            z_temp = z.view(batch_size*self.num, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num, -1)
            diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(batch_size,-1)
            G = 0.5*torch.norm(diff_p, dim=1)**2            
            cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )
            cHJBfin = torch.mean(torch.abs(phi_t[:,:,i+1] - temp_final_cost.view(-1, 1)))
        
        else:    
            if torch.is_tensor(u):
                assert self.nt == u.shape[2]
                for i in range(self.nt):
                    current_u = u[:, :, i].view(batch_size, self.control_dim)
                    z = z + self.h * self.compute_f(ti, z, current_u)
                    running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                    ti = ti + self.h
                # Calculate terminal cost
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)
            elif hasattr(u, 'forward'):
                for i in range(self.nt):
                    current_u = u(z, ti, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                    z = z + self.h * self.compute_f(ti, z, current_u)
                    running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                    gradPhi = u.p_net(ti, z, full_grad=True)
                    cadj = cadj + torch.mean(gradPhi[:,:self.state_dim] -
                                            self.h*self.compute_grad_H_z(ti, z, current_u, gradPhi[:,:self.state_dim]))
                    if hasattr(u.p_net, "getPhi"):
                        # double check sign
                        cHJB = cHJB + torch.mean(u.p_net.getPhi(ti,z) -
                                            self.h*self.compute_general_H(ti, z, current_u, -gradPhi[:,:self.state_dim]).view(-1,1)) 
                    grad_H_u = self.compute_grad_H_u(ti, z, current_u, gradPhi[:,:self.state_dim])
                    max_norm_grad_H_u = torch.max(torch.norm(grad_H_u, dim=1)).item()
                    avg_grad_H_u += torch.mean(torch.norm(grad_H_u, dim=1)).item()
                    if max_norm_grad_H_u > largest_grad_H_u:
                        largest_grad_H_u = max_norm_grad_H_u

                    # Verify Assumption 2 and Hypothesis of Lemma 1 in End-to-end
                    # training paper
                    M_theta, theta0, metadata = self.compute_grad_T_theta(u, z, ti)
                    batch_sdvals = torch.linalg.svdvals(M_theta)
                    if torch.min(batch_sdvals[:,-1]).item() < smallest_M_sdval:
                        smallest_M_sdval = torch.min(batch_sdvals[:,-1]).item()
                    if torch.max(batch_sdvals[:,0]).item() > largest_M_sdval:
                        largest_M_sdval = torch.max(batch_sdvals[:,0]).item()
                    lambda_min = 1.0/(batch_sdvals[:,0]*batch_sdvals[:,0])
                    if torch.min(lambda_min).item() < smallest_lambda_min:
                        smallest_lambda_min = torch.min(lambda_min).item()
                    lambda_max = 1.0/(batch_sdvals[:,-1]*batch_sdvals[:,-1])
                    if torch.max(lambda_max).item() > largest_lambda_max:
                        largest_lambda_max = torch.max(lambda_max).item()
                    grad_T_u = self.compute_grad_T_u(current_u, z, ti, gradPhi[:,:self.state_dim], u.alpha)
                    norm_grad_T_u = torch.linalg.matrix_norm(grad_T_u, ord=2, dim=(1,2))
                    idx_max = torch.argwhere(norm_grad_T_u > largest_grad_T_u_batch).flatten()
                    largest_grad_T_u_batch[idx_max] = norm_grad_T_u[idx_max]
                    avg_grad_T_u += torch.mean(norm_grad_T_u).item()

                    ti = ti + self.h

                # Calculate terminal cost
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)

                if self.pen_pos:
                    if (self.oc_problem_name == "Double Integrator") or (self.oc_problem_name == "Multi Quadcopter"):
                        gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size*self.num_agents, -1))[:,:3]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                    elif self.oc_problem_name == "Multi Bicycle":
                        gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size*self.num_agents, -1))[:,:2]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                    elif self.oc_problem_name == "Single Quadcopter":
                        gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size, -1))[:,:3]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                else:
                    cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )

                if hasattr(u.p_net, "getPhi"):
                    # Terminal HJB residual phi(T, z) - G(z); the loss weight
                    # alphaG must NOT scale the comparator G(z).
                    cHJBfin = torch.mean(torch.linalg.vector_norm(u.p_net.getPhi(ti,z) - temp_final_cost.view(-1, 1),ord=2,dim=1))
        
        # Calculate mean running cost
        running_cost = torch.mean(running_cost)
        
        # Calculate total cost
        total_cost = (self.alphaL * running_cost + self.alphaG * terminal_cost 
                      + self.alphaHJB[0] * cHJB + self.alphaHJB[1] * cHJBfin
                      + self.alphaadj[0] * cadj + self.alphaadj[1] * cadjfin)

        # Verify assumptions
        avg_grad_H_u = avg_grad_H_u / self.nt
        avg_grad_T_u = avg_grad_T_u / self.nt
        sd_grad_T_u = torch.std(largest_grad_T_u_batch).item()
        largest_grad_T_u = torch.max(largest_grad_T_u_batch).item()

        return total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u, smallest_M_sdval, largest_M_sdval, smallest_lambda_min, largest_lambda_max, largest_grad_T_u, avg_grad_T_u, sd_grad_T_u

    def compute_loss_consumcheck(self, policy, z0, z_t=None, p_t=None, phi_t=None, jac_based=False):
        
        """
        JFB compute_loss with closed-form adjoint check.
        """
        B = z0.shape[0]
        dt = (self.t_final - self.t_initial) / self.nt
        running_cost = torch.tensor(0.0, device=z0.device)
        cHJB         = torch.tensor(0.0, device=z0.device)
        cadj         = torch.tensor(0.0, device=z0.device)
        max_grad_u_H = torch.tensor(-1.0, device=z0.device)
        z = z0.clone().requires_grad_(True)
        t = self.t_initial

        # Compute Phi0 and its gradient dPhi0 = D_zPhi0
        Phi  = policy.p_net.getPhi(t, z)              # (B,1)
        dPhi = torch.autograd.grad(
                Phi.sum(), z,
                create_graph=True,
                retain_graph=True
            )[0]                                  # (B, D)

        for k in range(self.nt):

            u_k = policy(z, t)                       # (B, m)
            f_k = self.compute_f(t, z, u_k)          # (B, D)
            # print('shapes', u_k.shape, f_k.shape, z.shape, t)
            
            # Forward‐Euler to next state
            z1 = z + dt * f_k
            t1 = t + dt
            z1 = z1.requires_grad_(True)

            # Next‐step Phi1 and gradient dPhi1
            Phi1  = policy.p_net.getPhi(t1, z1)      # (B,1)
            dPhi1 = torch.autograd.grad(
                    Phi1.sum(), z1,
                    create_graph=True,
                    retain_graph=True
                )[0]                             # (B, D)

            # Running cost at (t, z1, u_k)
            running_cost = running_cost + dt * torch.mean(
                self.compute_lagrangian(t, z1, u_k)
            )

            # HJB residual: Phi_k - [Phi_{k+1} - dt*(L + dPhi*f)]
            H_val     = self.compute_general_H(t, z1, u_k, dPhi)
            resid_hjb = Phi.view(B) - (Phi1.view(B) - dt * H_val)
            cHJB      = cHJB + torch.mean(resid_hjb.pow(2))

            # closed-form adjoint check
            #    finite-difference of dPhi
            dp    = (dPhi1 - dPhi) / dt                       # (B, D)
            # print(f"[HJB step {k}, p_prev: {dPhi.shape}, f_val: {f_k.shape}] ")
            #    closed-form RHS:
            #      dot p_x = -r p_x
            rhs_px = -self.r * dPhi[:, 0]                     # (B,)
            #      dot p_h = e^{-delta t} (u-h)^{-gamma} + (dPhi_h @ B)
            h_k    = z[:, 1:1+self.m]                         # (B,m)
            rhs_ph = (
                torch.exp(-self.delta * torch.tensor(t1, device=z0.device, dtype=z0.dtype)) 
                * (u_k - h_k).pow(-self.gamma) #.clamp_min(1e-6)
                + dPhi[:, 1:1+self.m] @ self.B
            )                                                 # (B,m)
            rhs    = torch.cat([rhs_px.unsqueeze(1), rhs_ph], dim=1)  # (B,D)
            residA = dp - rhs                                 # (B,D)
            cadj   = cadj + torch.mean(residA.pow(2).sum(dim=1))
            # print('cadj', cadj)

            # 7) Track max ||D_u H||
            grad_uH = self.compute_grad_H_u(t, z, u_k, dPhi)
            max_norm = grad_uH.norm(dim=1).max()
            max_grad_u_H = torch.maximum(max_grad_u_H, max_norm)

            # 8) Next step
            z, t, Phi, dPhi = z1, t1, Phi1, dPhi1

        # Terminal penalties
        terminal_cost = torch.mean(self.compute_G(z))

        # adjoint terminal: p_T − DG(z_T)
        gradG   = self.compute_grad_G_z(z)                # (B,D)
        cadjfin = torch.mean((dPhi - gradG).pow(2).sum(dim=1))

        # terminal HJB: Phi_T − G(z_T)
        resid_hjb_fin = Phi.view(B) - self.compute_G(z)
        cHJBfin = torch.mean(resid_hjb_fin.pow(2))

        total_cost = (
            self.alphaL * running_cost
        + self.alphaG * terminal_cost
        + self.alphaHJB[0] * cHJB
        + self.alphaadj[1]   * cadjfin
        + self.alphaHJB[1] * cHJBfin
        )

        return (total_cost, running_cost, terminal_cost,
                cHJB, cHJBfin, cadj, cadjfin, max_grad_u_H)
    
    def compute_grad_T_theta(self, model, z, ti, create_graph=False):
        """
        Compute the Jacobian of the model output w.r.t. all parameters, treating the
        parameters theta as a single flattened tensor.

        Args:
            model (torch.nn.Module):
                Your network. It will be run in training mode so that gradients flow
                through the final differentiable operations.
            z (torch.Tensor):
                Current state
            ti (torch.float):
                Current time
            create_graph (bool, default=False):
                If True, build a graph that allows higher-order derivatives of J.
        Returns:
            J (torch.Tensor):
                Full Jacobian with shape (batch, out_dim, P), where P is the number
                of scalar parameters.
            theta0 (torch.Tensor):
                The flattened parameter vector at which J is evaluated, requires_grad=True.
            meta (dict):
                Metadata to reconstruct parameter structure:
                - names:  list of parameter names (ordered)
                - shapes: list of torch.Size for each parameter
                - idx:    1D tensor of cumulative indices for slicing theta
                Also includes:
                - unflatten(theta): function to map flat theta back to a {name: tensor} dict.
        """
        model.train()
        # Named parameters and buffers (buffers may be empty; kept for generality)
        params_dict = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        names  = list(params_dict.keys())
        shapes = [p.shape for p in params_dict.values()]
        sizes  = [p.numel() for p in params_dict.values()]
        idx = torch.tensor([0] + sizes, device=next(model.parameters()).device).cumsum(0)

        def pack(pdict):
            return torch.cat([p.reshape(-1) for p in pdict.values()])

        def unflatten(theta):
            out = {}
            for i, k in enumerate(names):
                start, end = idx[i].item(), idx[i+1].item()
                out[k] = theta[start:end].view(shapes[i])
            return out

        # Flattened parameter vector theta
        theta0 = pack(params_dict).detach().requires_grad_(True)

        # Wrap the model so theta is the differentiable argument
        def T_of_theta(theta, z, ti):
            pdict = unflatten(theta)
            y = torch.func.functional_call(model, (pdict, buffers), args=(z, ti))
            return y  # shape: (batch, out_dim)

        # jacrev returns J with shape (*y_shape, *theta_shape)
        J = torch.func.jacrev(lambda th: T_of_theta(th, z, ti), has_aux=False)(theta0)

        if create_graph:
            # ensure graph is retained for higher-order derivatives
            J.retain_grad()

        meta = {
            "names": names,
            "shapes": shapes,
            "idx": idx,
            "unflatten": unflatten,
        }
        return J, theta0, meta
    
    def compute_grad_T_u(self, u, z, t, grad_phi, alpha, create_graph=False):
        """
        Compute J(u) = dT_theta/du for a batch of inputs u.

        Parameters
        ----------
        u : torch.Tensor
            Either shape [m] or [B, m]. Will compute a m x m Jacobian per sample.
            u should be floating and on the same device/dtype as used by T_theta.
        z : Any
            Additional input to T_theta (fixed during differentiation).
        t : Any
            Additional input to T_theta (fixed during differentiation).
        grad_phi : Any
            Additional input to T_theta (fixed during differentiation).
        alpha: torch.float
            Additional input to T_theta (fixed during differentiation).
        create_graph : bool
            If True, builds a graph for higher-order derivatives.

        Returns
        -------
        torch.Tensor
            If u is [m], returns [m, m].
            If u is [B, m], returns [B, m, m].

        Notes
        -----
        - Assumes a globally available callable `T_theta(u, z, t)` that
          maps a single u:[m] → [m].
        - Differentiates w.r.t. u only (theta is treated as a constant here).
        """
        if u.ndim == 1:
            # Single sample: shape [p]
            u_single = u.detach().requires_grad_(True)

            def _T_single(u_vec):
                # T_theta should return shape [p]
                return u_vec + alpha*self.compute_grad_H_u(t, z, u_vec, grad_phi)

            # jacrev computes d(T)/d(u) with respect to input u
            J = torch.func.jacrev(_T_single)(u_single)
            if create_graph:
                # The jacrev above already respects create_graph semantics via autograd;
                # but we ensure the requires_grad chain is kept if requested.
                J = J.clone()
            return J  # [p, p]
        elif u.ndim == 2:
            # Batch: shape [B, p]
            B, p = u.shape
            u_batch = u.detach().requires_grad_(True)

            def _T_single(u_vec, z_vec, p_vec, grad_f_u_term):
                return u_vec + alpha*self.compute_grad_H_u_(t, z_vec, u_vec, p_vec, grad_f_u_term)

            # Vectorized jacobian across batch
            # Note: each vmap call needs (control_dim, state_dim) not (B, control_dim, state_dim)
            grad_f_u = torch.zeros(B, p, z.shape[1], device=self.device)
            J_batched = torch.func.vmap(torch.func.jacrev(_T_single), in_dims=(0,0,0,0))(u_batch, z, grad_phi, grad_f_u)  # [B, p, p]
            if create_graph:
                J_batched = J_batched.clone()
            return J_batched

        else:
            raise ValueError(f"`u` must have shape [p] or [B, p], got {tuple(u.shape)}.")

