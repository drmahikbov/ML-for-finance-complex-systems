"""
core_RL.Environment
-------------------
Abstract environment interface for the RL pipeline.

Two classes: `Environment` (abstract, one required `step` method + a batched
`rollout` helper) and `AnalyticalEnvironment` (wraps a known `f` callable
behind `step`, with `.detach()` to sever autograd through the dynamics).
The agent only ever calls `step`; it never touches `f` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch


class Environment(ABC):
    """Abstract environment.

    The contract is intentionally minimal: one ``step`` method that maps
    ``(z, u)`` to ``z'`` deterministically.

    Parameters
    ----------
    state_dim, control_dim
        Match the dimensions declared by the corresponding
        :class:`ImplicitOC_RL` problem.
    t_initial, t_final, nt
        Discretization grid. ``dt = (t_final - t_initial) / nt``.
    device
        ``"cpu"`` / ``"cuda"`` / ...
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        t_initial: float,
        t_final: float,
        nt: int,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.t_initial = t_initial
        self.t_final = t_final
        self.nt = nt
        self.dt = (t_final - t_initial) / nt
        self.device = device

    # Required interface
    @abstractmethod
    def step(self, z: torch.Tensor, u: torch.Tensor, t: float) -> torch.Tensor:
        """One environment step.

        Parameters
        ----------
        z : ``(batch, state_dim)``
        u : ``(batch, control_dim)``
        t : scalar wall time (helps non-stationary environments).

        Returns
        -------
        z_next : ``(batch, state_dim)``, **detached** (no autograd graph).

        Subclasses must guarantee that the returned tensor has no grad
        history: the whole point of using an environment instead of
        ``compute_f`` is to break the autograd path through the dynamics.
        """
        ...

    # Convenience: batched rollout under a policy
    @torch.no_grad()
    def rollout(
        self,
        policy: Callable,
        z0: torch.Tensor,
        jac_setter: Optional[Callable[[int], None]] = None,
        return_full_trajectory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out ``policy`` in this environment for ``nt`` steps.

        ``policy(z, t) -> u`` is whatever the trainer passes (in practice an
        :class:`ImplicitNetOC_RL`).

        ``jac_setter(k)`` is an optional hook called *before* the policy is
        evaluated at step ``k``. We use it during training to push the
        current ``B_k`` estimate into the implicit policy via
        ``policy.set_step_jacobian(...)``. Without it, this method also serves
        as a clean way to roll out a *converged* policy for plotting (the
        policy reads stale ``B_k`` if the setter is omitted, which is fine at
        eval time as long as the policy was trained with up-to-date ``B_k``).

        Returns
        -------
        z_traj : ``(batch, state_dim, nt+1)`` if ``return_full_trajectory``,
                 else ``(batch, state_dim)`` (terminal state only)
        u_traj : ``(batch, control_dim, nt)`` controls applied along the way
        """
        batch = z0.shape[0]
        z_traj = torch.zeros(batch, self.state_dim, self.nt + 1, device=z0.device)
        u_traj = torch.zeros(batch, self.control_dim, self.nt, device=z0.device)
        z_traj[:, :, 0] = z0

        z = z0
        t = self.t_initial
        for k in range(self.nt):
            if jac_setter is not None:
                jac_setter(k)
            u = policy(z, t).view(batch, self.control_dim)
            z_next = self.step(z, u, t)
            z_traj[:, :, k + 1] = z_next
            u_traj[:, :, k] = u
            z = z_next
            t = t + self.dt

        if return_full_trajectory:
            return z_traj, u_traj
        return z_traj[:, :, -1], u_traj


class AnalyticalEnvironment(Environment):
    """Environment backed by an analytical right-hand side ``f``.

    Used when the *experimenter* knows the dynamics (e.g. a Merton problem
    with hidden but known ``mu`` and ``r``) but wants to expose only the
    ``step`` interface to the agent. Internally takes one explicit-Euler
    step::

        z_next = (z + dt * f(t, z, u)).detach()

    The ``.detach()`` is the critical line: it severs any autograd path
    through ``f``, preventing the trainer from accidentally differentiating
    through the true dynamics.

    Parameters
    ----------
    f_callable
        Function ``f(t, z, u) -> dz/dt`` of shape ``(batch, state_dim)``.
        This is typically ``oc_problem.compute_f`` from a
        :class:`ImplicitOC_RL`-side model that *also* implements ``compute_f``
        for simulation purposes (we keep the analytical ``f`` on the
        ground-truth-aware *simulator* class, and instantiate the env with
        it). The agent never sees this callable.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        t_initial: float,
        t_final: float,
        nt: int,
        f_callable: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
        device: str = "cpu",
    ):
        super().__init__(state_dim, control_dim, t_initial, t_final, nt, device)
        self._f = f_callable

    @torch.no_grad()
    def step(self, z: torch.Tensor, u: torch.Tensor, t: float) -> torch.Tensor:
        # Explicit Euler. Every output is detached because we run under
        # @torch.no_grad(); no autograd graph leaks back through f.
        return z + self.dt * self._f(t, z, u)