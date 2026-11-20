import random

import numpy as np
import torch

from engine import Value

N = 10


def make_data(seed=0, n=N):
    """Random inputs, kept away from the ReLU kink (|a*b+c| > 0.05) so numerical gradients are valid."""
    rng = random.Random(seed)
    while True:
        a = [rng.uniform(-3, 3) for _ in range(n)]
        b = [rng.uniform(-3, 3) for _ in range(n)]
        c = [rng.uniform(-3, 3) for _ in range(n)]
        if all(abs(x * y + z) > 0.05 for x, y, z in zip(a, b, c)):
            return a, b, c


def engine_grads(fn, a0, b0, c0):
    a, b, c = ([Value(x) for x in v] for v in (a0, b0, c0))
    out = sum(fn(x, y, z) for x, y, z in zip(a, b, c))   # Python sum() = the lab's .sum()
    out.backward()
    return out.data, [np.array([v.grad for v in vs]) for vs in (a, b, c)]


def torch_grads(fn, a0, b0, c0):
    ts = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (a0, b0, c0)]
    out = fn(*ts).sum()
    out.backward()
    return out.item(), [t.grad.numpy() for t in ts]


def numeric_grads(fn, a0, b0, c0, eps=1e-5):
    """Central differences: (f(x+eps) - f(x-eps)) / (2 eps), one input at a time."""
    base = [np.array(v, dtype=np.float64) for v in (a0, b0, c0)]

    def f(arrs):
        return fn(*[torch.tensor(x, dtype=torch.float64) for x in arrs]).sum().item()

    grads = []
    for k in range(3):
        g = np.zeros(len(base[k]))
        for i in range(len(g)):
            up = [x.copy() for x in base]
            dn = [x.copy() for x in base]
            up[k][i] += eps
            dn[k][i] -= eps
            g[i] = (f(up) - f(dn)) / (2 * eps)
        grads.append(g)
    return grads


def maxdiff(g1, g2):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(g1, g2))


# The same lambda works for Value scalars and for torch tensors (both have * + ** and .relu()).
EXPRS = {
    "lab: ((a*b + c)**2).relu().sum()": lambda a, b, c: ((a * b + c) ** 2).relu(),
    "extra: (a*b + c).relu().sum()   ": lambda a, b, c: (a * b + c).relu(),
}

if __name__ == "__main__":
    a0, b0, c0 = make_data()
    active = sum(x * y + z > 0 for x, y, z in zip(a0, b0, c0))
    print(f"{N} inputs per variable; ReLU active on {active}, blocking on {N - active} (extra expression)\n")
    print(f"{'expression':36s} {'|eng-torch|':>12s} {'|eng-numeric|':>14s} {'|torch-numeric|':>16s}")
    worst = 0.0
    for name, fn in EXPRS.items():
        v_e, g_e = engine_grads(fn, a0, b0, c0)
        v_t, g_t = torch_grads(fn, a0, b0, c0)
        g_n = numeric_grads(fn, a0, b0, c0)
        d1, d2, d3 = maxdiff(g_e, g_t), maxdiff(g_e, g_n), maxdiff(g_t, g_n)
        worst = max(worst, d1, d2)
        print(f"{name:36s} {d1:12.2e} {d2:14.2e} {d3:16.2e}   forward diff {abs(v_e - v_t):.1e}")
    assert worst < 1e-6, worst
    print(f"\nmax abs difference (engine vs torch / numeric) = {worst:.2e} < 1e-6  -> PASS")