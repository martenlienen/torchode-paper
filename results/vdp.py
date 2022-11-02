import os

# In pytorch 1.12.1, nvfuser has a problem compiling torchode and while the built-in
# fallback compiler achieves a speed-up, it is larger if we go with the "old" JIT
# compiler NNC right away.
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

import argparse
import time
from pathlib import Path

import diffrax
import einops as eo
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Memory
from scipy.integrate import solve_ivp
from tqdm import tqdm

import torchdiffeq
import torchdyn.numerics
from torchode import (
    AutoDiffAdjoint,
    Dopri5,
    InitialValueProblem,
    IntegralController,
    ODETerm,
)

memory = Memory("cache")


class VdP(nn.Module):
    def __init__(self, mu: float):
        super().__init__()

        self.mu = mu
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        x, y = y[..., 0], y[..., 1]
        return torch.stack((y, self.mu * (1 - x**2) * y - x), dim=-1)


class TorchODE:
    @property
    def name(self):
        if self.jit:
            return "torchode-jit"
        else:
            return "torchode"

    def __init__(self, device, dtype, *, jit=False):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype is np.float32 else torch.float64
        self.jit = jit

    def init(self, mu, y0, t0, t1, t_eval, atol, rtol):
        y0 = torch.from_numpy(y0).to(device=self.device, dtype=self.dtype)
        t0 = torch.tensor(t0, device=self.device, dtype=self.dtype)
        t1 = torch.tensor(t1, device=self.device, dtype=self.dtype)
        t_eval = torch.from_numpy(t_eval).to(device=self.device, dtype=self.dtype)

        batch_size = y0.shape[0]
        t0 = eo.repeat(t0, "-> b", b=batch_size)
        t1 = eo.repeat(t1, "-> b", b=batch_size)
        t_eval = eo.repeat(t_eval, "t -> b t", b=batch_size)

        f = torch.jit.script(VdP(mu))
        term = ODETerm(f, with_stats=False)
        controller = IntegralController(atol=atol, rtol=rtol, term=term)
        step = Dopri5(term)
        adjoint = AutoDiffAdjoint(step, controller).to(self.device)
        problem = InitialValueProblem(y0=y0, t_start=t0, t_end=t1, t_eval=t_eval)

        if self.jit:
            adjoint = torch.jit.script(adjoint)

        return adjoint, problem

    def solve(self, state):
        adjoint, problem = state
        sol = adjoint.solve(problem)
        return sol

    def synchronize(self, out):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def results(self, sol):
        assert (sol.status == 0).all()

        return sol.ys.cpu().numpy(), sol.stats["n_steps"].max().item()


class TorchDiffEq:
    name = "torchdiffeq"

    def __init__(self, device, dtype):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype is np.float32 else torch.float64

    def init(self, mu, y0, t0, t1, t_eval, atol, rtol):
        assert t_eval[0] == t0
        assert t_eval[-1] == t1

        y0 = torch.from_numpy(y0).to(device=self.device, dtype=self.dtype)
        t_eval = torch.from_numpy(t_eval).to(device=self.device, dtype=self.dtype)
        f = torch.jit.script(VdP(mu)).to(device=self.device, dtype=self.dtype)

        return f, y0, t_eval, atol, rtol

    def solve(self, state):
        f, y0, t_eval, atol, rtol = state
        f.nfe = 0
        y = torchdiffeq.odeint(f, y0, t_eval, atol=atol, rtol=rtol, method="dopri5")
        return y, f.nfe

    def synchronize(self, out):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def results(self, out):
        y, nfe = out
        n_steps = (nfe - 2) // 6
        return eo.rearrange(y.cpu().numpy(), "t b f -> b t f"), n_steps


class TorchDyn:
    name = "torchdyn"

    def __init__(self, device, dtype):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype is np.float32 else torch.float64

    def init(self, mu, y0, t0, t1, t_eval, atol, rtol):
        assert t_eval[0] == t0
        assert t_eval[-1] == t1

        y0 = torch.from_numpy(y0).to(device=self.device, dtype=self.dtype)
        t_eval = torch.from_numpy(t_eval).to(device=self.device, dtype=self.dtype)
        f = torch.jit.script(VdP(mu)).to(device=self.device, dtype=self.dtype)

        return f, y0, t_eval, atol, rtol

    def solve(self, state):
        f, y0, t_eval, atol, rtol = state
        f.nfe = 0
        _, y = torchdyn.numerics.odeint(
            f, y0, t_eval, solver="dopri5", interpolator="4th", atol=atol, rtol=rtol
        )
        return y, f.nfe

    def synchronize(self, out):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def results(self, out):
        y, nfe = out
        n_steps = (nfe - 2) // 6
        return eo.rearrange(y.cpu().numpy(), "t b f -> b t f"), n_steps


def jax_vdp(t, y, args):
    mu = args
    x, y = y[..., 0], y[..., 1]
    return jnp.stack([y, mu * (1 - x**2) * y - x], axis=-1)


def diffrax_solve_vdp(mu, y0, t0, t1, t_eval, atol, rtol):
    term = diffrax.ODETerm(jax_vdp)
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController(
        atol=atol, rtol=rtol, pcoeff=0.0, icoeff=1.0, dcoeff=0.0
    )
    solution = diffrax.diffeqsolve(
        term,
        solver,
        stepsize_controller=controller,
        dt0=None,
        t0=t0,
        t1=t1,
        y0=y0,
        args=mu,
        saveat=diffrax.SaveAt(ts=t_eval),
    )
    return solution


class Diffrax:
    @property
    def name(self):
        if self.jit:
            return "diffrax-jit"
        else:
            return "diffrax"

    def __init__(self, device, dtype, *, jit=False):
        super().__init__()

        self.device = jax.devices(device)[0]
        self.dtype = dtype
        self.jit = jit

    def init(self, mu, y0, t0, t1, t_eval, atol, rtol):
        batch_size = y0.shape[0]
        y0 = jnp.array(y0).astype(self.dtype)
        t0 = eo.repeat(jnp.array(t0), "-> b", b=batch_size).astype(self.dtype)
        t1 = eo.repeat(jnp.array(t1), "-> b", b=batch_size).astype(self.dtype)
        t_eval = eo.repeat(jnp.array(t_eval), "t -> b t", b=batch_size).astype(
            self.dtype
        )

        y0 = jax.device_put(y0, self.device)
        t0 = jax.device_put(t0, self.device)
        t1 = jax.device_put(t1, self.device)
        t_eval = jax.device_put(t_eval, self.device)

        @jax.vmap
        def solve(y0, t0, t1, t_eval):
            return diffrax_solve_vdp(mu, y0, t0, t1, t_eval, atol, rtol)

        if self.jit:
            solve = jax.jit(solve)

        return solve, y0, t0, t1, t_eval

    def solve(self, state):
        solve, *args = state

        sol = solve(*args)
        return sol

    def synchronize(self, sol):
        sol.ys.block_until_ready()

    def results(self, sol):
        return np.array(sol.ys), sol.stats["num_steps"].max().item()


def vdp_cycle_length(mu):
    """Approximate cycle length of the VdP solution.

    I found the formula on some blog [1]. The original source might be [2],
    though I cannot access it. Numerical verification showed that this formula
    good enough for all our purposes. Even for smaller mu < 10, the error was
    less than 1.

    [1] https://www.johndcook.com/blog/2019/12/26/van-der-pol-period/
    [2] https://academic.oup.com/jlms/article-abstract/s2-36/1/102/861033
    """
    return (3 - 2 * np.log(2)) * mu + 2 * np.pi * mu ** (-1 / 3)


def vdp(t, x_, mu: float):
    x, xdot = x_.reshape((2, -1))
    return np.stack((xdot, mu * (1 - x**2) * xdot - x), axis=0).ravel()


@memory.cache
def solve_vdp(mu, t0, t1, y0, t_eval, atol, rtol, method="RK45"):
    sol = solve_ivp(
        lambda t, y: vdp(t, y, mu=mu),
        [t0, t1],
        y0,
        t_eval=t_eval,
        method=method,
        atol=atol,
        rtol=rtol,
    )

    return sol


@memory.cache
def vdp_cycle_length_numeric(mu, *, atol=1e-10, rtol=1e-6, n_cycles=10):
    # The approximate formula underestimates the cycle length by about 1% in the mu
    # ranges we care about, so we estimate it more accurately numerically
    sol = solve_vdp(
        mu,
        t0=0.0,
        t1=n_cycles * vdp_cycle_length(mu),
        y0=np.array([1.0, 0.0]),
        t_eval=None,
        atol=atol,
        rtol=rtol,
    )
    y = sol.y
    change_xdot_sign = np.concatenate(((y[1, 1:] * y[1, :-1]) < 0.0, [False]))
    ts = sol.t[change_xdot_sign & (y[0] > 0.0)]

    return np.median(np.diff(ts))


@memory.cache
def good_starting_point(mu, atol, rtol, *, n_cycles=5, n_steps=1000):
    assert n_cycles >= 2

    cycle_length = vdp_cycle_length(mu)

    t0 = 0.0
    t1 = n_cycles * cycle_length
    y0 = np.array([1, 0])
    t_eval = None
    sol = solve_vdp(mu, t0, t1, y0, t_eval, atol, rtol)

    return sol.y[:, sol.y[0].argmax()]


@memory.cache
def evenly_distributed_starting_points(
    n, mu, cycle_length, *, atol, rtol, n_steps, y0=None
):
    t_eval = np.linspace(0.0, cycle_length, n_steps)
    if y0 is None:
        y0 = good_starting_point(mu, atol=atol, rtol=rtol, n_steps=5_000)
    sol = solve_vdp(
        mu, t0=0.0, t1=cycle_length, y0=y0, t_eval=t_eval, atol=atol, rtol=rtol
    )

    offsets = [np.abs(sol.t - cycle_length * i / (2 * n)).argmin() for i in range(n)]
    y0n = sol.y[:, offsets]

    return y0n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", default=1e-5, type=float)
    parser.add_argument("--rtol", default=1e-5, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--mu", required=True, nargs="+", type=float)
    parser.add_argument("--batch-size", required=True, nargs="+", type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--steps", nargs="+", required=True, type=int)
    parser.add_argument("--methods", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    atol, rtol = args.atol, args.rtol
    device = args.device
    dtype = getattr(np, args.dtype)
    mus = args.mu
    batch_sizes = args.batch_size
    out_path = Path(args.out)

    steps = args.steps
    n_warmup = args.warmup
    n_trials = args.trials
    all_methods = {
        "torchode": TorchODE(device, dtype),
        "torchode-jit": TorchODE(device, dtype, jit=True),
        "torchdiffeq": TorchDiffEq(device, dtype),
        "torchdyn": TorchDyn(device, dtype),
        "diffrax": Diffrax(device, dtype),
        "diffrax-jit": Diffrax(device, dtype, jit=True),
    }
    methods = [all_methods[m] for m in args.methods]

    columns = ["mu", "batch-size", "n_eval", "method", "steps", "times", "max-error"]

    mu_bar = tqdm(mus, desc="mu", position=0)
    for mu in mu_bar:
        mu_bar.set_description(f"mu = {mu:.1f}")

        cycle_length = vdp_cycle_length_numeric(mu)
        y0 = good_starting_point(mu, atol, rtol, n_steps=5000)
        t0 = 0.0
        t1 = cycle_length

        bs_bar = tqdm(batch_sizes, desc="batch size", leave=False)
        for batch_size in bs_bar:
            bs_bar.set_description(f"batch size {batch_size}")

            y0_batch = evenly_distributed_starting_points(
                batch_size,
                mu,
                cycle_length,
                y0=y0,
                atol=atol,
                rtol=rtol,
                n_steps=250_000,
            ).T

            steps_bar = tqdm(steps, desc="step", leave=False)
            for n_steps in steps_bar:
                t_eval = np.linspace(t0, t1, n_steps)
                sol_batch = solve_vdp(
                    mu, t0, t1, y0_batch.T.ravel(), t_eval, atol, rtol
                )
                y_batch = sol_batch.y.reshape((2, -1, n_steps)).transpose((1, 2, 0))

                values = []
                m_bar = tqdm(methods, desc="method", leave=False)
                for method in m_bar:
                    m_bar.set_description(method.name)

                    state = method.init(mu, y0_batch, t0, t1, t_eval, atol, rtol)

                    # Warmup
                    for _ in range(n_warmup):
                        out = method.solve(state)
                    if n_warmup > 0:
                        method.synchronize(out)

                    times = []
                    for i in range(n_trials):
                        start = time.monotonic_ns()
                        out = method.solve(state)
                        method.synchronize(out)
                        end = time.monotonic_ns()
                        times.append(end - start)
                    times = np.array(times) / 10**6

                    y_out, method_steps = method.results(out)

                    max_error = np.abs(y_out - y_batch).max()
                    values.append(
                        (
                            mu,
                            batch_size,
                            n_steps or 0,
                            method.name,
                            method_steps,
                            times,
                            max_error,
                        )
                    )

                df = pd.DataFrame(values, columns=columns)
                if out_path.is_file():
                    old_df = pd.read_pickle(out_path)
                    df = pd.concat([old_df, df])
                df.to_pickle(out_path)


if __name__ == "__main__":
    main()
