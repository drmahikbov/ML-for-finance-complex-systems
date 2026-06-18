"""
benchmarking.gradient_checks
----------------------------
Taylor-convergence check for `compute_grad_f_u` via finite differences.
Evaluates the directional derivative at a single (z, u) sample (batch index 0)
and returns a log-log error curve vs. step size h.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def _plot_taylor(
    h_vals: np.ndarray,
    errors: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Log-log Taylor-convergence plot with O(h) and O(h²) reference lines."""
    valid = errors > 0
    if not valid.any():
        return
    h_v, e_v = h_vals[valid], errors[valid]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(h_v, e_v, "o-", lw=1.8, color="#1a1a2e", label="FD error")
    ax.loglog(h_v, (e_v[0] / h_v[0] ** 2) * h_v ** 2, "r--", lw=1.2, label=r"$O(h^2)$")
    ax.loglog(h_v, (e_v[0] / h_v[0]) * h_v,           "b--", lw=1.2, label=r"$O(h)$")
    ax.invert_xaxis()
    ax.set_xlabel("Step size  h")
    ax.set_ylabel("Error")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def taylor_check_compute_f_u(
    prob: Any,
    z: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    sample_index: int = 0,
    h_vals: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Taylor-convergence test for :meth:`ImplicitOC.compute_grad_f_u`.

    The test evaluates the directional derivative ``(∂f/∂u · v)`` at a
    **single** ``(z, u)`` sample, then compares a symmetric central
    finite difference ``(f(u+hv) - f(u-hv)) / (2h)`` against it for a
    sequence of step sizes ``h``.  The quadratic regime is the expected
    behaviour of a correct central-FD scheme.

    Parameters
    ----------
    prob : ImplicitOC
        Problem providing ``compute_f`` and ``compute_grad_f_u``.
    z, u : torch.Tensor, optional
        Test state and control.  If ``None`` they are sampled from
        ``prob.sample_initial_condition()`` and ``U[0, 2]``.  A
        single-sample slice is taken via ``sample_index``.
    sample_index : int
        Which batch row to use.  Defaults to 0.
    h_vals : np.ndarray, optional
        Step sizes.  Defaults to ``2.0 ** -arange(1, 18)``.
    save_path : str, optional
        Destination PNG for the Taylor plot.
    verbose : bool
        Whether to print the h / error pairs.

    Returns
    -------
    dict
        Keys ``"h_vals"``, ``"errors"``, ``"analytical_dir_deriv"``.
    """
    if z is None:
        z = prob.sample_initial_condition()
    if u is None:
        u = torch.rand(prob.batch_size, prob.control_dim, device=prob.device) * 2.0
    if h_vals is None:
        h_vals = 2.0 ** -np.arange(1, 18)

    z_single = z[sample_index:sample_index + 1]
    u_single = u[sample_index:sample_index + 1]

    v = torch.randn_like(u_single)
    v = v / v.norm()

    grad_fu = prob.compute_grad_f_u(0.0, z_single, u_single)  # (1, control_dim, state_dim)
    ana_dir = (grad_fu[0] * v[0].unsqueeze(-1)).sum().item()

    errors = []
    for h in h_vals:
        u_p = u_single + h * v
        u_m = u_single - h * v
        f_p = prob.compute_f(0.0, z_single, u_p)  # (1, state_dim)
        f_m = prob.compute_f(0.0, z_single, u_m)
        fd = ((f_p - f_m) / (2.0 * h)).sum().item()
        err = abs(fd - ana_dir)
        errors.append(err)
        if verbose:
            print(f"  h={h:.2e}  |FD - analytic| = {err:.3e}")
    errors = np.asarray(errors)

    _plot_taylor(
        np.asarray(h_vals), errors,
        r"$\partial f/\partial u$ Taylor convergence",
        save_path=save_path,
    )

    return {
        "h_vals": np.asarray(h_vals),
        "errors": errors,
        "analytical_dir_deriv": ana_dir,
    }


def gradient_check(
    prob: Any,
    z: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    sample_index: int = 0,
) -> Dict[str, Any]:
    """Run the framework's full gradient test suite + Taylor check for ``f_u``.

    Parameters
    ----------
    prob : ImplicitOC
    z, u : torch.Tensor, optional
        Batched test tensors; sampled if ``None``.
    save_path : str, optional
        Destination PNG for the Taylor plot.
    sample_index : int
        Batch row used by the Taylor check.

    Returns
    -------
    dict
        The dictionary returned by
        :meth:`utils.GradientTester.run_all_tests`.
    """
    from utils import GradientTester

    if z is None:
        z = prob.sample_initial_condition()
    if u is None:
        u = torch.rand(prob.batch_size, prob.control_dim, device=prob.device) * 2.0

    print("\n-- Analytical gradient tests (autograd vs hand-coded) --")
    results = GradientTester.run_all_tests(prob, z, u)

    print("\n-- Taylor convergence: compute_grad_f_u --")
    taylor_check_compute_f_u(
        prob, z=z, u=u,
        sample_index=sample_index,
        save_path=save_path,
    )
    return results
