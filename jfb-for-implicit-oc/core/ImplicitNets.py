import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import copy
import torch.nn.init as init
from torch.nn.functional import pad, softplus
# import MultiBicycle

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))
    # return torch.log(torch.exp(x) + torch.exp(-x)) # numerically unstable

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        :param dev: str, device
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        first_layer = nn.Linear(d + 1, m, bias=True)
        init.xavier_uniform_(first_layer.weight)
        init.zeros_(first_layer.bias)
        self.layers.append(first_layer) # opening layer
        
        resnet_layer = nn.Linear(m, m, bias=True)
        init.xavier_uniform_(resnet_layer.weight)
        init.zeros_(resnet_layer.bias)
        self.layers.append(resnet_layer) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x


class PNet(nn.Module, ABC):
    """Abstract base class for P-networks used in implicit networks."""
    
    def __init__(self, d, dev='cpu'):
        super(PNet, self).__init__()
        self.d = d
        self.device=dev
        
    @abstractmethod
    def forward(self, t, z):
        """
        Forward pass of the P-network.
        
        Args:
            t (torch.Tensor): Time tensor.
            z (torch.Tensor): State tensor.
            
        Returns:
            torch.Tensor: Network output.
        """
        pass

class DefaultPNet(PNet):
    """Default implementation of a P-network with fully connected layers."""
    
    def __init__(self, state_dim, hidden_dim=100, dev='cpu'):
        super(DefaultPNet, self).__init__(state_dim, dev)
        
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim, device=dev)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=dev)
        self.fc3 = nn.Linear(hidden_dim, state_dim, device=dev)
        self.relu = nn.ReLU()
        
    def forward(self, t, z, full_grad=False):
        """
        Forward pass of the default P-network.
        
        Args:
            t (torch.Tensor): Time tensor.
            z (torch.Tensor): State tensor.
            
        Returns:
            torch.Tensor: Network output.
        """
        zt = pad(z, [0,1,0,0], value=t)
        x = self.relu(self.fc1(zt))
        x = self.relu(self.fc2(x))
        p = self.fc3(x)
        return p
    
class Phi(PNet):
    def __init__(self, nTh, m, d, r=10, dev='cpu'):
        """
            neural network approximating Phi
            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        :para dev:   str, device being used
        """
        super().__init__(d, dev)

        self.m    = m
        self.nTh  = nTh

        r = min(r,d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+1, device=dev) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=False, device=dev)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False, device=dev)

        self.N = ResNN(d, m, nTh=nTh).to(dev)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape, device=dev)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape, device=dev)
        # self.c.bias.data   = torch.zeros(self.c.bias.data.shape)



    def forward(self, t, z, full_grad=False):
        """ calculating Phi(s, theta) """
        x = pad(z, [0,1,0,0], value=t) # pad z with t to get x

        # assumes specific N.act as the antiderivative of tanh
        N    = self.N
        symA = torch.matmul(self.A.t(), self.A)
        u = [] # hold the u_0,u_1,...,u_M for the forward pass

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        accGrad = 0.0 # accumulate the gradient as we step backwards through the network
        # compute analytic gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = accGrad # z_{i+1}

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            accGrad = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        tanhopen = torch.tanh(opening)  # act'( K_0 * S + b_0 )
        # z_0 = K_0' diag(...) z_1
        accGrad = torch.mm( N.layers[0].weight.t() , tanhopen.t() * accGrad )
        grad = accGrad + torch.mm(symA, x.t() ) + self.c.weight.t()

        if full_grad:
            # return the full gradient
            return grad.t() 
        else:
            return (grad[:self.d].t())

    def getPhi(self, t, z):
        
        # force A to be symmetric
        x = pad(z, [0,1,0,0], value=t)
        symA = torch.matmul(torch.t(self.A), self.A) # A'A
        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)

class ImplicitNetOC(nn.Module, ABC):
    """
    Abstract base class for implicit neural networks.
    """
    
    def __init__(self, 
                 state_dim=2, 
                 control_dim=1, 
                 hidden_dim=100, 
                 alpha=1e-1, 
                 max_iters=int(2e2), 
                 tol=1e-2, 
                 tracked_iters=5,
                 p_net=None,
                 oc_problem=None,
                 use_control_limits=False,
                 u_min=-1,
                 u_max=1,
                 dev='cpu',
                 use_aa=False,
                 beta=0.5):
        super(ImplicitNetOC, self).__init__()
        
        # Define network architecture parameters
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.oc_problem = oc_problem
        # Control limits
        self.use_control_limits = use_control_limits
        self.u_min = u_min
        self.u_max = u_max
        self.device = ""
        if oc_problem is None:
            self.device = dev
        else:
            self.device = oc_problem.device
        
        # Define P-network
        if p_net is None:
            self.p_net = DefaultPNet(state_dim, hidden_dim, self.device)
        else:
            self.p_net = p_net
            
        # Optimization parameters
        self.alpha = alpha
        self.max_iters = max_iters
        self.tol = tol
        self.tracked_iters = tracked_iters
        
        # Convergence tracking (new additions)
        self.last_fp_depth = 0
        self.last_max_res_norm = 0.0
        self.last_converged = True
        self.track_convergence = True
        # Per-call residual trace populated by ``forward()`` when
        # ``record_trace=True``. List of floats, one per inner iter.
        self.last_residual_trace: list[float] = []

        # Anderson Acceleration
        self.use_anderson = use_aa
        self.beta = beta

        
    def T(self, u, x, t):
        """
        T-operator for the fixed-point iteration.
        
        Args:
            u (torch.Tensor): Control tensor.
            x (torch.Tensor): State tensor.
            t (torch.Tensor): Time tensor.
            
        Returns:
            torch.Tensor: Updated control.
        """
        batch_size = x.shape[0]
        t_scalar = torch.ones(1, device=x.device) * t

        assert x.shape == (batch_size, self.state_dim)
        assert t_scalar.shape == (1,)

        p = self.p_net(t, x)
        grad_H_u_val = self.oc_problem.compute_grad_H_u(t_scalar, x, u, p)
        assert grad_H_u_val.shape == u.shape
        # The sign difference is because the code uses the convention (H = L + p^\top f) 
        # (minimization Hamiltonian), so gradient ascent on (\mathcal{H}) becomes gradient descent on (H).
        return u - self.alpha * grad_H_u_val 
    
    def apply_control_limits(self, u):
        """
        Apply control limits to the control tensor if enabled.
        
        Args:
            u (torch.Tensor): Control tensor.
            
        Returns:
            torch.Tensor: Clamped control tensor if limits are enabled, otherwise unchanged.
        """
        if self.use_control_limits:
            return torch.clamp(u, self.u_min, self.u_max)
        return u
    
    def forward(self, x, t, verbose=False, max_res_out = False, track_all_fp_iters=False,
                record_trace: bool = False):
        """
        Forward pass of the implicit network.

        Args:
            x (torch.Tensor): State tensor.
            t (torch.Tensor): Time tensor.
            verbose (bool): Whether to print convergence information.
            record_trace (bool): If ``True``, the per-iteration residual
                norms of the inner fixed-point solver are recorded into
                ``self.last_residual_trace`` (one float per iter). The
                trace is reset at the beginning of every call.
        Returns:
            torch.Tensor: Optimal control.
        """
        batch_size = x.shape[0]
        # t = torch.ones(1) * t

        
        u = torch.zeros(batch_size, self.control_dim, device=x.device, dtype=x.dtype)  # default initial guess
        converged = False
        max_res_norm = float('inf')
        n_max_iters = 0
        if record_trace:
            self.last_residual_trace = []

        # Determine if we should run the fixed-point iteration with or without gradients
        def find_fixed_point():
            nonlocal u, converged, max_res_norm, n_max_iters
            if not self.use_anderson:
                # Check if torch grad is enabled
                # if torch.is_grad_enabled():
                #     print("torch grad is enabled")
                # else:
                #     print("torch grad is NOT enabled")
                for i in range(self.max_iters):
                    u_old = u.clone()
                    u = self.T(u, x, t)
                    assert u.shape == (batch_size, self.control_dim)
                    
                    # Compute maximum residual norm over batches
                    max_res_norm = (torch.norm(u - u_old, dim=1).max())/self.alpha

                    if record_trace:
                        self.last_residual_trace.append(
                            max_res_norm.item() if torch.is_tensor(max_res_norm) else float(max_res_norm)
                        )

                    if verbose:
                        print(f'iter {i+1}, max res norm {max_res_norm:.5e}')

                    n_max_iters = i 
                    if max_res_norm < self.tol:
                        if verbose:
                            print(f'converged in {i+1} iterations with res norm {max_res_norm:.5e}')
                        converged = True
                        break

                    """   
                    if i == self.max_iters - 1:
                        print("DID NOT CONVERGE!")
                    """

                if verbose and not converged:
                    print(f'did not converge in {self.max_iters} iterations with res norm {max_res_norm:.5e}')
            else:
                _trace = self.last_residual_trace if record_trace else None
                u, max_res_norm, num_itr = self.anderson_direct(
                    u, x, t, self.tol, self.max_iters,
                    m=10, beta=self.beta, trace=_trace,
                )
                #u, max_res_norm, num_itr = self.anderson_qr(u, x, t, self.tol, self.max_iters, m=20)
                assert u.shape == (batch_size, self.control_dim)

                # Compute maximum residual norm over batches
                #max_res_norm = (torch.norm(u - u_prev, dim=1).max()) / self.alpha
                n_max_iters = num_itr

                if verbose:
                    print(f'iter {num_itr}, max res norm {max_res_norm:.5e}')

                if max_res_norm < self.tol:
                    if verbose:
                        print(f'converged in {num_itr} iterations with res norm {max_res_norm:.5e}')
                    converged = True
                        
                #if num_itr == self.max_iters:
                #    print("DID NOT CONVERGE!")

                if verbose and not converged:
                    print(f'did not converge in {self.max_iters} iterations with res norm {max_res_norm:.5e}')

        if not track_all_fp_iters or not self.training:
            with torch.no_grad():
                find_fixed_point()
        else:
            find_fixed_point()

        # Store convergence statistics for trainer
        if self.track_convergence:
            self.last_fp_depth = n_max_iters + 1
            self.last_max_res_norm = max_res_norm.item() if torch.is_tensor(max_res_norm) else max_res_norm
            self.last_converged = converged

        if self.training:
            for i in range(self.tracked_iters):
                u = self.apply_control_limits(self.T(u, x, t))
            output = self.apply_control_limits(self.T(u, x, t))
            if max_res_out:
                return output, max_res_norm, n_max_iters # accumulate everything in info
            else:
                return output
        else:
            output = u
            return output

    def set_convergence_tracking(self, track=True):
        """Enable or disable convergence tracking."""
        self.track_convergence = track

    def get_convergence_stats(self):
        """Get the latest convergence statistics.

        ``residual_trace`` is populated only when ``forward(..., record_trace=True)``
        is called; it lists the per-inner-iteration residual norms of the
        most recent forward pass. Empty otherwise.
        """
        return {
            'fp_depth': self.last_fp_depth,
            'max_res_norm': self.last_max_res_norm,
            'converged': self.last_converged,
            'residual_trace': list(self.last_residual_trace),
        }
    
    def anderson_direct(self, u0, x, t, tol=1.0e-3, max_iters=100, m=5, beta=0.5, lam=1.0e-4,
                        trace: list | None = None):
        """
        Fixed-Point Iteration with Anderson acceleration 

        Parameters:
            T (callable): Operator being learned
            u0 (torch.tensor): Initial guess
            x,t (torch.tensor, float): Data 
            tol (float): Error tolerance for convergence
            m (int): Number of previous iterations to use in least-squares 
               optimization problem
            beta (float): Parameter in Anderson acceleration iteration, must be > 0
            lam (float): Regularization parameter
            trace (list, optional): If provided, the per-iteration residual
                ``res_k/self.alpha`` is appended after every step. Used for
                inner-FP convergence diagnostics.

        Return:
            Fixed point of T_eval, last residual, and number of iterations
        """
        batch_sz, d = u0.shape
        u_hist = torch.zeros(batch_sz, m, d, dtype=u0.dtype, device=u0.device)
        T_hist = torch.zeros(batch_sz, m, d, dtype=u0.dtype, device=u0.device)
        u_hist[:,0] = u0.view(batch_sz, -1)
        T_hist[:,0] = self.T(u0, x, t).view(batch_sz,-1)
        u_hist[:,1] = T_hist[:,0]
        T_hist[:,1] = self.T(T_hist[:,0].view_as(u0), x, t).view(batch_sz,-1)
        H = torch.zeros(batch_sz, m+1, m+1, dtype=u0.dtype, device=u0.device)
        H[:,0,1:] = 1.0
        H[:,1:,0] = 1.0
        Batch_RHS = torch.zeros(batch_sz, m+1, 1, dtype=u0.dtype, device=u0.device)
        Batch_RHS[:,0] = 1.0 

        k = 1
        #res_k = ((T_hist[:,k%m] - u_hist[:,(k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
        res_k = torch.norm(T_hist[:,k%m] - u_hist[:,k%m], dim=1).max().item()
        if trace is not None:
            trace.append(res_k / self.alpha)
        k += 1
        while ((res_k/self.alpha) > tol and k < max_iters):
            M = min(k,m)
            G = T_hist[:,:M] - u_hist[:,:M]
            H[:,1:(M+1),1:(M+1)] = torch.bmm(G, G.transpose(1,2)) + lam*torch.eye(M, dtype=u0.dtype, device=u0.device)[None]

            #Solve for alpha
            alpha = None
            try:
                alpha = torch.linalg.solve(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[:,1:(M+1),0]#Result is batch_sz x n
            except RuntimeError:  # H singular: fall back to least-squares (silent; rare).
                alpha = torch.linalg.lstsq(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[0][:,1:(M+1),0]

            #Update data structures
            u_hist[:,k%m] = (1.0-beta)*((alpha[:,None]@u_hist[:,:M])[:,0]) + beta*((alpha[:,None]@T_hist[:,:M])[:,0])
            T_hist[:,k%m] = self.T(u_hist[:,k%m].view_as(u0), x, t).view(batch_sz, -1)
            #res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
            res_k = torch.norm(T_hist[:,k%m] - u_hist[:,k%m], dim=1).max().item()
            if trace is not None:
                trace.append(res_k / self.alpha)
            k += 1

        return u_hist[:,k%m].view_as(u0), res_k/self.alpha, k

    def givensQRdelete(self,Q,R):
        """
        Update QR decomp. of a matrix after 1st column has been deleted. 
        Computation is done batchwise and in-place. 
        """
        m = R.shape[2]
        for i in range(m-1):
            j = i+1
            denom = torch.sqrt(R[:,i,j]**2 + R[:,j,j]**2)
            c = R[:,i,j]/denom
            s = R[:,j,j]/denom
            R[:,i,j] = denom
            R[:,j,j] = 0.0
            if i < m-2:
                temp = c.unsqueeze(-1)*R[:,i,i+2:] + s.unsqueeze(-1)*R[:,j,i+2:]
                R[:,j,i+2:] = -s.unsqueeze(-1)*R[:,i,i+2:] + c.unsqueeze(-1)*R[:,j,i+2:]
                R[:,i,i+2:] = temp
            temp = c.unsqueeze(-1)*Q[:,:,i] + s.unsqueeze(-1)*Q[:,:,j]
            Q[:,:,j] = -s.unsqueeze(-1)*Q[:,:,i] + c.unsqueeze(-1)*Q[:,:,j]
            Q[:,:,i] = temp
        return Q, torch.roll(R,-1,2)

    def anderson_qr(self, u0, z, t, tol=1.0e-3, max_iters=100, m=5):
        """
        Unconstrained QR implementation of anderson acceleration

        Parameters:                                                                                                                 T (callable): Operator being learned                                                                                    u0 (torch.tensor): Initial guess                                                                                        x,t (torch.tensor, float): Data                                                                                         tol (float): Error tolerance for convergence                                                                            m (int): Number of previous iterations to use in least-squares                                                             optimization problem                                                                                             
        Return:                                                                                                                     Fixed point of T_eval, last residual, and number of iterations
        """
        batch_sz, d  = u0.shape
        T = torch.zeros(batch_sz, d, m, dtype=u0.dtype, device=u0.device)
        M = 0
        res_k = 1.0e10
        k = 0
        uk = u0.view(batch_sz,-1)
        fk_1 = None
        Tk_1 = None
        Q = torch.zeros(batch_sz, d, m, dtype=u0.dtype, device=u0.device)
        R = torch.zeros(batch_sz, m, m, dtype=u0.dtype, device=u0.device)
        #print(res_k/self.alpha)
        while (res_k > tol and k < max_iters):
            #print(res_k/self.alpha)
            Tk = self.T(uk.view_as(u0), z, t).view(batch_sz,d)
            fk = Tk - uk
            if k > 0:
                df = fk - fk_1
                dT = Tk - Tk_1
                if M < m:
                    T[:,:,M] = dT
                else:
                    T = torch.roll(T,-1,2)
                    T[:,:,M-1] = dT
                M += 1
            fk_1 = fk
            Tk_1 = Tk
            if M == 0:
                uk = Tk
            else:
                if M == 1:
                    Q[:,:,0] = df/(torch.norm(df,dim=1).unsqueeze(-1))
                    R[:,0,0] = torch.norm(df,dim=1)
                else:
                    if M > m:
                        Q,R = self.givensQRdelete(Q,R) #Q[:,:,:(m-1)], R[:,:(m-1),1:] are the only valid entries after this line
                        M -= 1
                    # Iterative update of df needed
                    for i in range(M-1):
                        R[:,i,M-1] = (Q[:,:,i]*df).sum(dim=1)
                        df = df - R[:,i,M-1].unsqueeze(-1)*Q[:,:,i]
                    Q[:,:,M-1] = df/(torch.norm(df,dim=1).unsqueeze(-1))
                    R[:,M-1,M-1] = torch.norm(df,dim=1)
                b = torch.matmul(Q[:,:,:M].transpose(1,2), fk.unsqueeze(-1))
                gamma = torch.linalg.solve_triangular(R[:,:M,:M], b, upper=True)[:,:,0]
                uk = Tk_1 - torch.matmul(T[:,:,:M], gamma.unsqueeze(-1)).squeeze(-1)
                #uk = beta*uk - (1.0-beta)*(fk - torch.matmul(Q[:,:,:M], torch.matmul(R[:,:M,:M], gamma.unsqueeze(-1))).squeeze(-1))
            res_k = torch.norm(fk, dim=1).max().item()
            k += 1
        return uk, res_k, k

class ImplicitNetOC_pos(ImplicitNetOC, ABC):
    """
    Similar to ImplicitNetOC, but enforces   u – h > 0 for the optimal consumption example
    by optimising in a r := u − h and 
    mapping u = h + softplus_beta(r) with beta > 0.
    """

    def __init__(
        self,
        *args,
        beta: float = 5.0,          # temperature of the soft-plus
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta         

    #  T-operator (operates on raw r, not on u)
    def T(self, r, x, t):
        """
        r-space update:  r_new = r - alpha * (dH/dr).

        Parameters
        ----------
        r : tensor (B, m)   current raw variable  (u − h)
        x : tensor (B, state_dim)
        t : scalar(float/tensor)
        """
        batch_size = x.shape[0]
        t_scalar   = torch.ones(1, device=x.device) * t

        # current habit
        h = x[:, 1 : 1 + self.oc_problem.m]                   # (B, m)

        # projected control (guaranteed > h)
        u_proj = h + softplus(r, beta=self.beta)

        # dH/du at projected control
        p = self.p_net(t, x)
        grad_H_u = self.oc_problem.compute_grad_H_u(
            t_scalar, x, u_proj, p
        )                                                     # (B, m)

        # chain-rule factor
        sigma = torch.sigmoid(self.beta * r)

        # gradient in raw space
        grad_H_r = grad_H_u * sigma                           # (B, m)

        return r - self.alpha * grad_H_r                     

    def forward(
        self,
        x,
        t,
        verbose: bool = False,
        max_res_out: bool = False,
        track_all_fp_iters: bool = False,
    ):
        batch_size = x.shape[0]
        m          = self.oc_problem.m

        #u = h + softplus(0) > h
        r = torch.zeros(batch_size, m, device=x.device, dtype=x.dtype)

        converged     = False
        max_res_norm  = float("inf")
        n_max_iters   = 0

        def find_fixed_point():
            nonlocal r, converged, max_res_norm, n_max_iters
            if not self.use_anderson:
                for i in range(self.max_iters):
                    r_old = r.clone()
                    r = self.T(r, x, t)

                    max_res_norm = torch.norm(r - r_old, dim=1).max()
                    n_max_iters  = i
                    if verbose:
                        print(
                            f"iter {i+1}, max raw res {max_res_norm:.5e}"
                        )

                    if max_res_norm < self.tol:
                        converged = True
                        if verbose:
                            print(
                                f"converged in {i+1} iters "
                                f"res {max_res_norm:.5e}"
                            )
                        break
                if verbose and not converged:
                    print(
                        f"did not converge in {self.max_iters} iters "
                        f"res {max_res_norm:.5e}"
                    )
            else:
                # Anderson acceleration on raw variable
                r, r_prev, num_itr = self.anderson(
                    r, x, t, self.tol, self.max_iters, beta=1.5
                )
                max_res_norm = torch.norm(r - r_prev, dim=1).max()
                n_max_iters  = num_itr
                if verbose:
                    print(
                        f"iter {num_itr}, max raw res {max_res_norm:.5e}"
                    )
                converged = max_res_norm < self.tol

        if not track_all_fp_iters or not self.training:
            with torch.no_grad():
                find_fixed_point()
        else:
            find_fixed_point()

        # store stats
        if self.track_convergence:
            self.last_fp_depth      = n_max_iters + 1
            self.last_max_res_norm  = (
                max_res_norm.item()
                if torch.is_tensor(max_res_norm)
                else max_res_norm
            )
            self.last_converged     = converged

        # project to control space
        h      = x[:, 1 : 1 + m]
        u_star = h + softplus(r, beta=self.beta)
        u_star = self.apply_control_limits(u_star)

        if self.training:
            for _ in range(self.tracked_iters):
                r = self.T(r, x, t).detach()          

            r = self.T(r, x, t)         
            u_star = h + softplus(r, beta=self.beta)
            u_star = self.apply_control_limits(u_star)
        else:
            u_star = h + softplus(r, beta=self.beta)
            u_star = self.apply_control_limits(u_star)

        if max_res_out:
            return u_star, max_res_norm, n_max_iters
        return u_star

    
    def anderson(self, u0, x, t, tol=1.0e-3, max_iters=100, m=5, beta=0.5, lam=1.0e-6):
        """
        Fixed-Point Iteration with Anderson acceleration 

        Parameters:
            T (callable): Operator being learned
            u0 (torch.tensor): Initial guess
            x,t (torch.tensor, float): Data 
            tol (float): Error tolerance for convergence
            m (int): Number of previous iterations to use in least-squares 
               optimization problem
            beta (float): Parameter in Anderson acceleration iteration, must be > 0
            lam (float): Regularization parameter

        Return:
            Fixed point of T_eval, value of u after previous iteration, and number 
            of iterations
        """
        batch_sz, d = u0.shape
        u_hist = torch.zeros(batch_sz, m, d, dtype=u0.dtype, device=u0.device)
        T_hist = torch.zeros(batch_sz, m, d, dtype=u0.dtype, device=u0.device)
        u_hist[:,0] = u0.view(batch_sz, -1)
        T_hist[:,0] = self.T(u0, x, t).view(batch_sz,-1)
        u_hist[:,1] = T_hist[:,0]
        T_hist[:,1] = self.T(T_hist[:,0].view_as(u0), x, t).view(batch_sz,-1)
        H = torch.zeros(batch_sz, m+1, m+1, dtype=u0.dtype, device=u0.device)
        H[:,0,1:] = 1.0
        H[:,1:,0] = 1.0
        Batch_RHS = torch.zeros(batch_sz, m+1, 1, dtype=u0.dtype, device=u0.device)
        Batch_RHS[:,0] = 1.0 

        k = 1
        res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
        k += 1
        while (res_k > tol and k < max_iters):
            M = min(k,m)
            G = T_hist[:,:M] - u_hist[:,:M]
            H[:,1:(M+1),1:(M+1)] = torch.bmm(G, G.transpose(1,2)) + lam*torch.eye(M, dtype=u0.dtype, device=u0.device)[None]

            #Solve for alpha
            alpha = None
            try:
                alpha = torch.linalg.solve(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[:,1:(M+1),0]#Result is batch_sz x n
            except RuntimeError:#If matrix is singular solve using Householder QR least squares
                alpha = torch.linalg.lstsq(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[0][:,1:(M+1)]

            #Update data structures
            u_hist[:,k%m] = (1.0-beta)*((alpha[:,None]@u_hist[:,:M])[:,0]) + beta*((alpha[:,None]@T_hist[:,:M])[:,0])
            T_hist[:,k%m] = self.T(u_hist[:,k%m].view_as(u0), x, t).view(batch_sz, -1)
            res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
            k += 1

        return u_hist[:,k%m].view_as(u0), u_hist[:,(k-1)%m].view_as(u0), k
    
class ImplicitNetOC_MB(ImplicitNetOC):
    """
    Similar to ImplicitNetOC, but with a different apply_control_limits()
    function for use with the multi bicycleOC problem
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_control_limits(self, u):
        """
        Apply control limits to the control tensor if enabled.
        For multibicycle problem, only angle of handlebars needs
        to be clamped. Acceleration does not.
        
        Args:
            u (torch.Tensor): Control tensor.
            
        Returns:
            torch.Tensor: Clamped control tensor if limits are enabled, otherwise unchanged.
        """
        if self.use_control_limits:
            u_clamped = torch.zeros_like(u)
            u_clamped[:,0:self.oc_problem.control_dim:self.oc_problem.single_control_dim] = torch.clamp(u[:,0:self.oc_problem.control_dim:self.oc_problem.single_control_dim], self.u_min, self.u_max)
            u_clamped[:,1:self.oc_problem.control_dim:self.oc_problem.single_control_dim] = u[:,1:self.oc_problem.control_dim:self.oc_problem.single_control_dim]
            return u_clamped
        else:
            return u
    





