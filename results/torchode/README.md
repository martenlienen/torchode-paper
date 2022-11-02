# A Parallel ODE Solver for PyTorch

## Usage

```python
import matplotlib.pyplot as pp
import torch
from torchode import solve_ivp

def f(t, y):
    return -0.5 * y

y0 = torch.tensor([[1.2, 5.0]])
t_eval = torch.linspace(0, 5, 10).unsqueeze(dim=0)
sol = solve_ivp(f, y0, t_eval)

print(sol.stats)
# => {'n_steps': tensor([17]), 'n_accepted': tensor([17]), 'n_f_evals': tensor([105])}

pp.plot(sol.ts[0], sol.ys[0])
```
