import numpy as np
import torch
import torch.nn as nn
import torchdyn.core
from torchdiffeq import odeint_adjoint as odeint
from torchode import (
    BacksolveAdjoint,
    InitialValueProblem,
    IntegralController,
    JointBacksolveAdjoint,
    ODETerm,
    Tsit5,
)
from torchode_utils.timing import cuda_timing

from .odefunc import sample_rademacher_like
from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class Reshaper(nn.Module):
    def __init__(self, odefunc, z_shape):
        super().__init__()

        self.odefunc = odefunc
        self.z_shape = z_shape
        self.times = []

    def forward(self, t, y, e=None):
        t = torch.atleast_1d(t)
        y = torch.atleast_2d(y)

        z_shape = self.z_shape
        if e is not None:
            e = torch.atleast_2d(e).unflatten(-1, z_shape)

        z = (
            y[..., : np.prod(z_shape)].unflatten(-1, z_shape),
            y[..., np.prod(z_shape) :],
        )
        with cuda_timing() as time:
            self.times.append(time)
            dz, dlogpz = self.odefunc(t, z, e)

        return torch.cat((dz.flatten(start_dim=-len(self.z_shape)), dlogpz), dim=-1)


class Timing(nn.Module):
    def __init__(self, f):
        super().__init__()

        self.f = f
        self.times = []

    def forward(self, *args, **kwargs):
        with cuda_timing() as time:
            self.times.append(time)
            return self.f(*args, **kwargs)


class CNF(nn.Module):
    def __init__(
        self,
        odefunc,
        T=1.0,
        train_T=False,
        regularization_fns=None,
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
        library="torchdiffeq",
    ):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter(
                "sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T)))
            )
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

        self.library = library

        self.stats = None

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor(
                [0.0, self.sqrt_end_time * self.sqrt_end_time]
            ).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.library.startswith("torchode"):
            if self.training:
                assert self.nreg == 0

            with cuda_timing() as solver_time:
                reshaper = Reshaper(self.odefunc, z.shape[1:])
                term = ODETerm(reshaper, with_args=True, with_stats=False)
                step_method = Tsit5()
                controller = IntegralController(self.atol, self.rtol)
                if self.library == "torchode":
                    adjoint = BacksolveAdjoint(
                        term, step_method, controller, vmap_args_dims=0
                    )
                elif self.library == "torchode-joint":
                    adjoint = JointBacksolveAdjoint(term, step_method, controller)

                batch_size = z.shape[0]
                y0 = torch.cat((z.reshape((batch_size, -1)), _logpz), dim=1)
                t_eval = integration_times.expand(batch_size, -1).to(z)
                problem = InitialValueProblem(
                    y0, t_start=t_eval[:, 0], t_end=t_eval[:, -1]
                )
                e = sample_rademacher_like(z).reshape((batch_size, -1))
                solution = adjoint.solve(problem, args=e)

            # In torchdiffeq, we include the reshaping in the solver time, so we have to
            # do the same here
            solution.stats["fw_model_times"] = reshaper.times
            # Ensure that model evaluations during the backward pass do not get counted
            # towards the forward pass
            reshaper.times = []

            self.stats = solution.stats
            self.stats["fw_solver_time"] = solver_time

            z_end = solution.ys[:, -1]
            z_shape = z.shape[1:]
            z_t = z_end[:, : np.prod(z_shape)].reshape((-1, *z_shape))
            if logpz is None:
                return z_t
            else:
                return z_t, z_end[:, np.prod(z_shape) :]
        elif self.library == "torchdiffeq":
            with cuda_timing() as solver_time:
                # Refresh the odefunc statistics.
                self.odefunc.before_odeint(sample_rademacher_like(z))

                odefunc = Timing(self.odefunc)
                if self.training:
                    state_t, self.stats = odeint(
                        odefunc,
                        (z, _logpz) + reg_states,
                        integration_times.to(z),
                        atol=self.atol,
                        rtol=self.rtol,
                        method=self.solver,
                        options=self.solver_options,
                    )
                else:
                    state_t, self.stats = odeint(
                        odefunc,
                        (z, _logpz),
                        integration_times.to(z),
                        atol=self.test_atol,
                        rtol=self.test_rtol,
                        method=self.test_solver,
                    )
            self.stats["fw_solver_time"] = solver_time
            self.stats["fw_model_times"] = odefunc.times
            # Ensure that model evaluations during the backward pass do not get counted
            # towards the forward pass
            odefunc.times = []

            if len(integration_times) == 2:
                state_t = tuple(s[1] for s in state_t)

            z_t, logpz_t = state_t[:2]
            self.regularization_states = state_t[2:]

            if logpz is not None:
                return z_t, logpz_t
            else:
                return z_t
        elif self.library == "torchdyn":
            with cuda_timing() as solver_time:
                # Refresh the odefunc statistics.
                self.odefunc.before_odeint(sample_rademacher_like(z))

                batch_size = z.shape[0]
                y0 = torch.cat((z.reshape((batch_size, -1)), _logpz), dim=1)

                reshaper = Reshaper(self.odefunc, z.shape[1:])
                _, ys, self.stats = torchdyn.core.ODEProblem(
                    reshaper,
                    solver=self.solver,
                    interpolator="4th",
                    sensitivity="adjoint",
                    atol=self.atol,
                    rtol=self.rtol,
                    atol_adjoint=self.atol,
                    rtol_adjoint=self.rtol,
                ).odeint(y0, integration_times.to(y0))
            self.stats["fw_solver_time"] = solver_time
            self.stats["fw_model_times"] = reshaper.times
            # Ensure that model evaluations during the backward pass do not get counted
            # towards the forward pass
            reshaper.times = []

            z_end = ys[-1]
            z_shape = z.shape[1:]
            z_t = z_end[:, : np.prod(z_shape)].reshape((-1, *z_shape))
            if logpz is None:
                return z_t
            else:
                return z_t, z_end[:, np.prod(z_shape) :]

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]
