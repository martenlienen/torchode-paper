# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inspect import getfullargspec
import torch
from torch.autograd import Function, grad
from torchcde import CubicSpline, natural_cubic_coeffs
from torchdyn.numerics.odeint import odeint, odeint_mshooting
from torchode_utils.timing import cuda_timing


def generic_odeint(problem_type, vf, x, t_span, solver, atol, rtol, interpolator, B0=None,
                  return_all_eval=False, maxiter=4, fine_steps=4, save_at=()):
    "Dispatches to appropriate `odeint` function depending on `Problem` class (ODEProblem, MultipleShootingProblem)"
    if problem_type == 'standard':
        return odeint(vf, x, t_span, solver, atol=atol, rtol=rtol, interpolator=interpolator, return_all_eval=return_all_eval,
                      save_at=save_at)
    elif problem_type == 'multiple_shooting':
        return odeint_mshooting(vf, x, t_span, solver, B0=B0, fine_steps=fine_steps, maxiter=maxiter)


# TODO: optimize and make conditional gradient computations w.r.t end times
# TODO: link `seminorm` arg from `ODEProblem`
def _gather_odefunc_adjoint(vf, vf_params, solver, atol, rtol, interpolator, solver_adjoint,
                            atol_adjoint, rtol_adjoint, integral_loss, problem_type, maxiter=4, fine_steps=4):
    "Prepares definition of autograd.Function for adjoint sensitivity analysis of the above `ODEProblem`"
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span, stats_out, B=None, save_at=()):
            ctx.stats = stats_out
            t_sol, sol = generic_odeint(problem_type, vf, x, t_span, solver, atol, rtol, interpolator, B,
                                        False, maxiter, fine_steps, save_at)
            ctx.save_for_backward(sol, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            with cuda_timing() as solver_time:
                sol, t_sol = ctx.saved_tensors
                vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
                # initialize flattened adjoint state
                xT, ??T, ??T = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
                xT_nel, ??T_nel, ??T_nel = xT.numel(), ??T.numel(), ??T.numel()
                xT_shape, ??T_shape, ??T_shape = xT.shape, ??T.shape, ??T.shape

                ??T_flat = ??T.flatten()
                ??tT = ??T_flat @ vf(t_sol[-1], xT).flatten()
                # concatenate all states of adjoint system
                A = torch.cat([xT.flatten(), ??T_flat, ??T.flatten(), ??tT[None]])

                model_times = []

                def adjoint_dynamics(t, A):
                    if len(t.shape) > 0: t = t[0]
                    x, ??, ?? = A[:xT_nel], A[xT_nel:xT_nel+??T_nel], A[-??T_nel-1:-1]
                    x, ??, ?? = x.reshape(xT.shape), ??.reshape(??T.shape), ??.reshape(??T.shape)
                    with torch.set_grad_enabled(True):
                        x, t = x.requires_grad_(True), t.requires_grad_(True)
                        with cuda_timing() as time:
                            model_times.append(time)
                            dx = vf(t, x)
                            d??, dt, *d?? = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -??,
                                                allow_unused=True, retain_graph=False))

                            if integral_loss:
                                dg = torch.autograd.grad(integral_loss(t, x).sum(), x, allow_unused=True, retain_graph=True)[0]
                                d?? = d?? - dg

                        d?? = torch.cat([el.flatten() if el is not None else torch.zeros(1)
                                        for el in d??], dim=-1)
                        if dt == None: dt = torch.zeros(1).to(t)
                        if len(t.shape) == 0: dt = dt.unsqueeze(0)
                    return torch.cat([dx.flatten(), d??.flatten(), d??.flatten(), dt.flatten()])

                # solve the adjoint equation
                n_elements = (xT_nel, ??T_nel, ??T_nel)
                dLdt = torch.zeros(len(t_sol)).to(xT)
                dLdt[-1] = ??tT
                for i in range(len(t_sol) - 1, 0, -1):
                    t_adj_sol, A = odeint(adjoint_dynamics, A, t_sol[i - 1:i + 1].flip(0),
                                          solver_adjoint, atol=atol_adjoint, rtol=rtol_adjoint,
                                          seminorm=(True, xT_nel+??T_nel))
                    # prepare adjoint state for next interval
                    #TODO: reuse vf_eval for dLdt calculations
                    xt = A[-1, :xT_nel].reshape(xT_shape)
                    dLdt_ = A[-1, xT_nel:xT_nel + ??T_nel]@vf(t_sol[i], xt).flatten()
                    A[-1, -1:] -= grad_output[0][i - 1]
                    dLdt[i-1] = dLdt_

                    A = torch.cat([A[-1, :xT_nel], A[-1, xT_nel:xT_nel + ??T_nel], A[-1, -??T_nel-1:-1], A[-1, -1:]])
                    A[xT_nel:xT_nel + ??T_nel] += grad_output[-1][i - 1].flatten()

                ??, ?? = A[xT_nel:xT_nel + ??T_nel], A[-??T_nel-1:-1]
                ??, ?? = ??.reshape(??T.shape), ??.reshape(??T.shape)
                ??_tspan = torch.stack([dLdt[0], dLdt[-1]])

            ctx.stats["bw_solver_time"] = solver_time
            ctx.stats["bw_model_times"] = model_times

            return (??, ??, ??_tspan, None, None, None, None)

    return _ODEProblemFunc


#TODO: introduce `t_span` grad as above
def _gather_odefunc_interp_adjoint(vf, vf_params, solver, atol, rtol, interpolator, solver_adjoint,
                                   atol_adjoint, rtol_adjoint, integral_loss, problem_type, maxiter=4, fine_steps=4):
    "Prepares definition of autograd.Function for interpolated adjoint sensitivity analysis of the above `ODEProblem`"
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span, B=None, save_at=()):
            t_sol, sol = generic_odeint(problem_type, vf, x, t_span, solver, atol, rtol, interpolator, B,
                                        True, maxiter, fine_steps, save_at)
            ctx.save_for_backward(sol, t_span, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_span, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

            # initialize adjoint state
            xT, ??T, ??T = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            ??T_nel, ??T_nel = ??T.numel(), ??T.numel()
            xT_shape, ??T_shape, ??T_shape = xT.shape, ??T.shape, ??T.shape
            A = torch.cat([??T.flatten(), ??T.flatten()])

            spline_coeffs = natural_cubic_coeffs(x=sol.permute(1, 0, 2).detach(), t=t_sol)
            x_spline = CubicSpline(coeffs=spline_coeffs, t=t_sol)

            # define adjoint dynamics
            def adjoint_dynamics(t, A):
                if len(t.shape) > 0: t = t[0]
                x = x_spline.evaluate(t).requires_grad_(True)
                t = t.requires_grad_(True)
                ??, ?? = A[:??T_nel], A[-??T_nel:]
                ??, ?? = ??.reshape(??T.shape), ??.reshape(??T.shape)
                with torch.set_grad_enabled(True):
                    dx = vf(t, x)
                    d??, dt, *d?? = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -??,
                                        allow_unused=True, retain_graph=False))

                    if integral_loss:
                        dg = torch.autograd.grad(integral_loss(t, x).sum(), x, allow_unused=True, retain_graph=True)[0]
                        d?? = d?? - dg

                    d?? = torch.cat([el.flatten() if el is not None else torch.zeros(1)
                                    for el in d??], dim=-1)
                return torch.cat([d??.flatten(), d??.flatten()])

            # solve the adjoint equation
            n_elements = (??T_nel, ??T_nel)
            for i in range(len(t_span) - 1, 0, -1):
                t_adj_sol, A = odeint(adjoint_dynamics, A, t_span[i - 1:i + 1].flip(0), solver, atol=atol, rtol=rtol)
                # prepare adjoint state for next interval
                A = torch.cat([A[-1, :??T_nel], A[-1, -??T_nel:]])
                A[:??T_nel] += grad_output[-1][i - 1].flatten()

            ??, ?? = A[:??T_nel], A[-??T_nel:]
            ??, ?? = ??.reshape(??T.shape), ??.reshape(??T.shape)
            return (??, ??, None, None, None)

    return _ODEProblemFunc
