from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchdyn.numerics
from torchdiffeq import odeint, odeint_adjoint
from torchode import (
    AutoDiffAdjoint,
    Dopri5,
    InitialValueProblem,
    IntegralController,
    ODETerm,
)

from .data import TimeEncoder


class TorchODEStats:
    def __init__(self, stats: dict[str, torch.Tensor]):
        self.stats = stats

    def forward_steps(self):
        return self.stats["n_steps"].cpu().numpy()


class TorchODESolver(nn.Module):
    def __init__(
        self,
        f,
        method: str,
        atol: float,
        rtol: float,
        options: Optional[dict] = None,
        adjoint: bool = False,
        adjoint_options: Optional[dict] = None,
    ):
        super().__init__()

        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.options = options
        self.adjoint = adjoint
        self.adjoint_options = adjoint_options

        assert not adjoint
        assert method == "dopri5"

        term = ODETerm(f, with_args=True, with_stats=False)
        step_method = Dopri5(term=term)
        step_size_controller = IntegralController(self.atol, self.rtol, term=term)
        self.adjoint = AutoDiffAdjoint(step_method, step_size_controller)

    def forward(
        self, y0: torch.Tensor, t: torch.Tensor, args: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        problem = InitialValueProblem(y0, t_eval=t)
        solution = self.adjoint.solve(problem, args=args)

        return solution.ys[:, 1:], solution.stats


class TorchDiffEqWrapper(nn.Module):
    def __init__(self, f, t, args):
        super().__init__()

        self.f = f
        self.t = t
        self.args = args
        t_min = t.amin(dim=1)
        t_range = t.amax(dim=1) - t_min
        # Map linearly from the range of t[0] to the range of the other instances
        self.t_slope = t_range / t_range[0]
        self.t_intercept = t_min - t_min[0] * self.t_slope
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        instance_time = t * self.t_slope + self.t_intercept
        return self.f(instance_time, y, self.args)


class TorchDiffEqStats:
    def __init__(self, nfe):
        self.nfe = nfe

    def forward_steps(self):
        return np.array((self.nfe - 2) / 6)


class ODESolver(nn.Module):
    def __init__(
        self,
        f,
        method: str,
        atol: float,
        rtol: float,
        options: Optional[dict] = None,
        adjoint: bool = False,
        adjoint_options: Optional[dict] = None,
    ):
        super().__init__()

        self.f = f
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.options = options
        self.adjoint = adjoint
        self.adjoint_options = adjoint_options

        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, y0: torch.Tensor, t: torch.Tensor, args: Any) -> torch.Tensor:
        kwargs = dict(
            rtol=self.rtol, atol=self.atol, method=self.method, options=self.options
        )
        if self.adjoint:
            kwargs["adjoint_options"] = self.adjoint_options

        wrapper = TorchDiffEqWrapper(self.f, t, args)
        y = self.odeint(wrapper, y0, t[0], **kwargs)

        stats = TorchDiffEqStats(wrapper.nfe)
        return y[1:].transpose(0, 1), stats


class TorchDynSolver(nn.Module):
    def __init__(self, f, method: str, atol: float, rtol: float):
        super().__init__()

        self.f = f
        self.method = method
        self.atol = atol
        self.rtol = rtol

    def forward(self, y0: torch.Tensor, t: torch.Tensor, args: Any) -> torch.Tensor:
        wrapper = TorchDiffEqWrapper(self.f, t, args)
        _, y = torchdyn.numerics.odeint(
            wrapper,
            y0,
            t[0],
            solver=self.method,
            interpolator="4th",
            atol=self.atol,
            rtol=self.rtol,
        )

        stats = TorchDiffEqStats(wrapper.nfe)
        return y[1:].transpose(0, 1), stats
