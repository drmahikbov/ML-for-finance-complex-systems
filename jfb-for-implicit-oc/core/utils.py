"""
core.utils
----------
Development utilities. `GradientTester` checks `compute_grad_f_u` and
`compute_grad_f_z` against finite differences, producing Taylor-convergence
plots.
"""
import torch
import matplotlib.pyplot as plt
plt.ion()
import numpy as np


class GradientTester:
    """
    A utility class for checking the correctness of gradient implementations
    using finite difference approximations.
    """
    
    @staticmethod
    def check_grad_f_u(oc_problem, z=None, u=None, t=None, verbose=True):
        """
        Check the gradient of dynamics with respect to control.
        
        Args:
            oc_problem: An instance of ImplicitOC
            z (torch.Tensor, optional): State vector
            u (torch.Tensor, optional): Control vector
            t (float, optional): Time
            epsilon (float, optional): Step size for finite difference
            verbose (bool, optional): Whether to print details
            
        Returns:
            tuple: (analytical_grad, numerical_grad, relative_error)
        """
        batch_size = oc_problem.batch_size
        state_dim = oc_problem.state_dim
        control_dim = oc_problem.control_dim
        device = oc_problem.device
        # Create random vectors if not provided
        if z is None:
            z = torch.randn(batch_size, state_dim, device=device)
        if u is None:
            u = torch.randn(batch_size, control_dim, device=device)
        if t is None:
            t = 0.0
        
        # Get analytical gradient
        analytical_grad = oc_problem.compute_grad_f_u(t, z, u)
        
        # Create a copy of z that requires gradients
        u_autograd = u.clone().detach().requires_grad_(True)
        
        autograd_gradf_u = torch.vmap(torch.func.jacrev(oc_problem.compute_f, argnums = 2))(t,z,u_autograd)
        
        # Compute error
        error = torch.norm(analytical_grad.permute(0,2,1) - autograd_gradf_u) / (torch.norm(analytical_grad) + 1e-8)
        
        if verbose:
            print("-" * 40)
            print(f"Gradient f_u check (autograd):")
            print(f"  Analytical norm: {torch.norm(analytical_grad).item()}")
            print(f"  Autograd norm: {torch.norm(autograd_gradf_u).item()}")
            print(f"  Relative error: {error.item()}")
        
        return analytical_grad, autograd_gradf_u, error
    
    @staticmethod
    def check_grad_f_z(oc_problem, z=None, u=None, t = None,  verbose=True):
        """
        Check the gradient of dynamics with respect to state using PyTorch's autograd.
        
        Args:
            oc_problem: An instance of ImplicitOC
            z (torch.Tensor, optional): State vector
            u (torch.Tensor, optional): Control vector
            t (float, optional): Time
            verbose (bool, optional): Whether to print details
            
        Returns:
            tuple: (analytical_grad, autograd_grad, relative_error)
        """
        
        batch_size = oc_problem.batch_size
        state_dim = oc_problem.state_dim
        control_dim = oc_problem.control_dim
        device = oc_problem.device
        # Create random vectors if not provided
        if z is None:
            z = torch.randn(batch_size, state_dim, device=device)
        if u is None:
            u = torch.randn(batch_size, control_dim, device=device)
        
        # Get analytical gradient
        analytical_grad = oc_problem.compute_grad_f_z(t, z, u)
        
        # Create a copy of z that requires gradients
        z_autograd = z.clone().detach().requires_grad_(True)
        
        autograd_gradf_z = torch.vmap(torch.func.jacrev(oc_problem.compute_f, argnums = 1))(t,z_autograd,u)
        
        # Compute error
        error = torch.norm(analytical_grad - autograd_gradf_z) / (torch.norm(analytical_grad) + 1e-8)
        
        if verbose:
            print("-" * 40)
            print(f"Gradient f_z check (autograd):")
            print(f"  Analytical norm: {torch.norm(analytical_grad).item()}")
            print(f"  Autograd norm: {torch.norm(autograd_gradf_z).item()}")
            print(f"  Relative error: {error.item()}")
        
        return analytical_grad, autograd_gradf_z, error
    
    @staticmethod
    def check_grad_lagrangian(oc_problem, z=None, u=None, t=None, verbose=True):
        """
        Check the gradient of the Lagrangian with respect to control.
        
        Args:
            oc_problem: An instance of ImplicitOC
            u (torch.Tensor, optional): Control vector
            epsilon (float, optional): Step size for finite difference
            verbose (bool, optional): Whether to print details
            
        Returns:
            tuple: (analytical_grad, numerical_grad, relative_error)
        """
        batch_size = oc_problem.batch_size
        state_dim = oc_problem.state_dim
        control_dim = oc_problem.control_dim
        device = oc_problem.device
        # Create random vectors if not provided
        if z is None:
            z = torch.randn(batch_size, state_dim, device=device)
        if u is None:
            u = torch.randn(batch_size, control_dim, device=device)
        if t is None:
            t = 0.0
        
        # Get analytical gradient
        analytical_grad = oc_problem.compute_grad_lagrangian(t, z, u)
        
        # Create a copy of z that requires gradients
        u_autograd = u.clone().detach().requires_grad_(True)
        
        autograd_gradL_u = torch.vmap(torch.func.jacrev(oc_problem.compute_lagrangian, argnums = 2))(t,z,u_autograd)
        
        # Compute error
        error = torch.norm(analytical_grad - autograd_gradL_u.view(*analytical_grad.shape)) / (torch.norm(analytical_grad) + 1e-8)
        
        if verbose:
            print("-" * 40)
            print(f"Gradient L_u check (autograd):")
            print(f"  Analytical norm: {torch.norm(analytical_grad).item()}")
            print(f"  Autograd norm: {torch.norm(autograd_gradL_u).item()}")
            print(f"  Relative error: {error.item()}")
        
        return analytical_grad, autograd_gradL_u, error

    
    @staticmethod
    def run_all_tests(oc_problem, z=None, u=None, t=None, epsilon=1e-7):
        """
        Run all gradient tests.
        
        Args:
            oc_problem: An instance of ImplicitOC
            z (torch.Tensor, optional): State vector
            u (torch.Tensor, optional): Control vector
            t (float, optional): Time
            epsilon (float, optional): Step size for finite difference
            
        Returns:
            dict: Dictionary of test results
        """
        batch_size = oc_problem.batch_size
        state_dim = oc_problem.state_dim
        control_dim = oc_problem.control_dim
        device = oc_problem.device
        
        # Create random vectors if not provided
        if z is None:
            z = torch.randn(batch_size, state_dim, device=device)
        if u is None:
            u = torch.randn(batch_size, control_dim, device=device)
        if t is None:
            t = torch.zeros(batch_size, 1, device=device)
        
        # Run gradient tests
        _, _, error_grad_f_u = GradientTester.check_grad_f_u(oc_problem, z, u, t)
        _, _, error_grad_f_z = GradientTester.check_grad_f_z(oc_problem, z, u, t)
        _, _, error_grad_lagrangian = GradientTester.check_grad_lagrangian(oc_problem, z, u, t)
        
        print("-" * 40)
        print("Summary of gradient tests:")
        print(f"  Gradient f_u error: {error_grad_f_u.item()}")
        print(f"  Gradient f_z error: {error_grad_f_z.item()}")
        print(f"  Gradient Lagrangian error: {error_grad_lagrangian.item()}")
        
        # Check if errors are acceptable
        threshold = 1e-7
        all_passed = (error_grad_f_u < threshold and 
                      error_grad_f_z < threshold and 
                      error_grad_lagrangian < threshold
                     )
        
        print("-" * 40)
        if all_passed:
            print("All gradient tests PASSED!")
        else:
            print("Some gradient tests FAILED!")
        
        return {
            "grad_f_u_error": error_grad_f_u.item(),
            "grad_f_z_error": error_grad_f_z.item(),
            "grad_lagrangian_error": error_grad_lagrangian.item(),
            "all_passed": all_passed
        }



class GradientTester_Taylors:
    """
    A utility class to verify analytical gradients using finite differences
    based on Taylor series expansion.
    """

    def check_gradient(self, cost_func, grad_func, x, num_checks=5, h_values=None):
        """
        Checks the analytical gradient against a numerical one.

        Args:
            cost_func (callable): A function that takes a tensor x and returns a scalar cost.
            grad_func (callable): A function that takes a tensor x and returns its analytical gradient.
            x (torch.Tensor): The point at which to check the gradient.
            num_checks (int): The number of random directions to test.
            h_values (list, optional): A list of step sizes for finite differencing. 
                                     Defaults to a logarithmic range.
        """
        if h_values is None:
            h_values = 2.0 ** -np.arange(1, 21)

        print("--- Starting Gradient Check ---")
        errors_for_h = []
        # Calculate the analytical directional derivative: ∇f(x)ᵀv
        v = torch.randn_like(x)
        v = v / torch.norm(v)

        fx = cost_func(x)
        num_grad = grad_func(x)
        num_grad_deriv = torch.sum(num_grad * v)

        for h in h_values:
            f_plus_h = cost_func(x + h * v)
            
            # Compute the absolute error
            error = torch.abs(f_plus_h - fx - h * num_grad_deriv).item()
            errors_for_h.append(error)
            print(f"h = {h:.2e} | grad Error = {error:.2e}")

        self._plot_results(h_values, errors_for_h)
        print("--- Gradient Check Finished ---")


    def _plot_results(self, h_values, errors):
        """
        Plots the error vs. h on a log-log scale to verify the order of convergence.
        """
        h_values = np.array(h_values)
        errors = np.array(errors)

        # Filter out zero errors for log plot
        valid_indices = errors > 0
        if not np.any(valid_indices):
            print("All errors are zero. Cannot plot.")
            return
            
        h_values = h_values[valid_indices]
        errors = errors[valid_indices]

        # Fit a line to the log-log data to find the slope
        log_h = np.log10(h_values)
        log_err = np.log10(errors)
        coeffs = np.polyfit(log_h, log_err, 1)
        slope = coeffs[0]

        plt.figure(figsize=(8, 6))
        plt.loglog(h_values, errors, 'o-', label='Finite Difference Error')
        # Plot reference lines with slopes 1 and 2
        plt.loglog(h_values, (errors[0] / h_values[0]) * h_values, 'r--', label='O(h) Reference')
        plt.loglog(h_values, (errors[0] / h_values[0]**2) * h_values**2, 'g--', label='O(h^2) Reference')
        plt.gca().invert_xaxis()
        plt.title(f'Gradient Check (Log-Log Plot)\nEstimated Slope: {slope:.2f}')
        plt.xlabel('Step Size (h)')
        plt.ylabel('Approximation Error')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show(block=True)  