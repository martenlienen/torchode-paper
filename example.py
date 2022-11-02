import torch
from torchode import Status, solve_ivp

def vdp(t, y, mu):
    x, xdot = y[..., 0], y[..., 1]
    return torch.stack((xdot, mu * (1 - x**2) * xdot - x), dim=-1)

batch_size, mu = 5, 10.0
y0 = torch.randn((batch_size, 2))
t_eval = torch.linspace(0.0, 10.0, steps=50)
sol = solve_ivp(vdp, y0, t_eval, method="tsit5", args=mu)

print(sol.status)  # => tensor([0, 0, 0, 0, 0])
assert all(sol.status == Status.SUCCESS.value)
print(sol.stats)
# => {'n_f_evals': tensor([1412, 1412, 1412, 1412, 1412]),
#     'n_steps': tensor([201, 230, 227, 235, 220]),
#     'n_accepted': tensor([197, 223, 222, 229, 214]),
#     'n_initialized': tensor([50, 50, 50, 50, 50])}
