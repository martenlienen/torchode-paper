import itertools
import os

# In pytorch 1.12.1, nvfuser has a problem compiling torchode and while the built-in
# fallback compiler achieves a speed-up, it is larger if we go with the "old" JIT
# compiler NNC right away.
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

import time

import einops as eo
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torchode import (
    AutoDiffAdjoint,
    Dopri5,
    InitialValueProblem,
    IntegralController,
    ODETerm,
    PIDController,
)
from vdp import VdP, good_starting_point, vdp_cycle_length_numeric


class TorchODE:
    def __init__(self, device, dtype, *, pid=None, jit=False):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype is np.float32 else torch.float64
        self.pid = pid
        self.jit = jit

    def init(self, mu, y0, t0, t1, atol, rtol, max_steps=None):
        y0 = torch.from_numpy(y0).to(device=self.device, dtype=self.dtype)
        t0 = torch.tensor(t0, device=self.device, dtype=self.dtype)
        t1 = torch.tensor(t1, device=self.device, dtype=self.dtype)

        batch_size = y0.shape[0]
        t0 = eo.repeat(t0, "-> b", b=batch_size)
        t1 = eo.repeat(t1, "-> b", b=batch_size)

        f = torch.jit.script(VdP(torch.tensor(mu)))
        term = ODETerm(f)
        if self.pid is None:
            controller = IntegralController(atol=atol, rtol=rtol, term=term)
        else:
            p, i, d = self.pid
            controller = PIDController(
                atol=atol, rtol=rtol, pcoeff=p, icoeff=i, dcoeff=d, term=term
            )
        step = Dopri5(term)
        adjoint = AutoDiffAdjoint(step, controller, max_steps=max_steps).to(self.device)
        problem = InitialValueProblem(y0=y0, t_start=t0, t_end=t1, t_eval=None)

        if self.jit:
            adjoint = torch.jit.script(adjoint)

        return adjoint, problem

    def solve(self, state):
        adjoint, problem = state
        return adjoint.solve(problem)

    def results(self, sol):
        return sol.stats["n_steps"].max().item(), (sol.status == 0).all()


atol, rtol = 1e-5, 1e-5
device = "cuda"
dtype = np.float32
mus = np.arange(1, 101, dtype=dtype)
batch_size = 1


def dejl_to_pid(beta1, beta2, beta3=0.0):
    return beta1 + beta2 + beta3, -beta2 - beta3, beta3, "diffeq.jl"


pids = [
    # I controller
    (0.0, 1.0, 0.0, "I"),
    # Recommended by DifferentialEquations.jl:
    # https://diffeq.sciml.ai/stable/extras/timestepping/
    dejl_to_pid(0.6, -0.2),
    dejl_to_pid(2 / 3, -1 / 3),
    dejl_to_pid(0.7, -0.4),
    dejl_to_pid(1 / 6, 1 / 6),
    dejl_to_pid(1 / 18, 1 / 9, 1 / 9),
    # From diffrax
    (0.4, 1.0, 0.0, "diffrax"),
    (0.4, 0.3, 0.0, "diffrax"),
    (0.3, 0.3, 0.0, "diffrax"),
    (0.2, 0.4, 0.0, "diffrax"),
]

columns = ["mu", "P", "I", "D", "source", "steps", "finished"]
values = []

mu_bar = tqdm(mus, desc="mu", position=0)
for mu in mu_bar:
    mu_bar.set_description(f"mu = {mu:.1f}")

    y0 = good_starting_point(mu, atol, rtol, n_steps=5000)[None]
    t0, t1 = 0.0, vdp_cycle_length_numeric(mu)

    method = TorchODE(device, dtype)
    state = method.init(mu, y0, t0, t1, atol, rtol)
    i_steps, _ = method.results(method.solve(state))
    max_steps = i_steps * 1.5

    pid_bar = tqdm(pids, desc="PID")
    for p, i, d, source in pid_bar:
        pid_bar.set_description(f"PID ({p:.3f}, {i:.3f}, {d:.3f})")

        method = TorchODE(device, dtype, pid=(p, i, d))
        state = method.init(mu, y0, t0, t1, atol, rtol, max_steps=max_steps)
        method_steps, finished = method.results(method.solve(state))

        values.append((mu, p, i, d, source, method_steps, bool(finished)))

        df = pd.DataFrame(values, columns=columns)
        df.to_pickle("pid.pickle")
