from typing import Any
import torch
from torchode import InitialValueProblem, Status, construct_solver

def vdp(t, y, mu: Any):
    assert isinstance(mu, float)
    x, xdot = y[..., 0], y[..., 1]
    return torch.stack((xdot, mu * (1 - x**2) * xdot - x), dim=-1)

batch_size, mu = 5, 10.0
y0 = torch.randn((batch_size, 2))
t_eval = torch.linspace(0.0, 10.0, steps=50)

solver = torch.jit.script(construct_solver(vdp, method="tsit5"))
problem = InitialValueProblem(y0, t_eval=t_eval.expand((batch_size, -1)))
sol = solver.solve(problem, args=mu)
