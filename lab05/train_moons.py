import statistics
import time

import numpy as np
import torch
from sklearn.datasets import make_moons

from engine import Value

N_SAMPLES, NOISE = 200, 0.2
EPOCHS, LR, EPS = 100, 0.5, 1e-12
N_RUNS = 5

X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=0)
X, y = X.astype(np.float64), y.astype(np.float64)
X_list, y_list = X.tolist(), y.tolist()          # plain Python floats for the scalar engine


def init_params(seed=0):
    """2 -> 8 -> 1. Both implementations start from exactly these numbers."""
    rng = np.random.default_rng(seed)
    k1, k2 = 1 / np.sqrt(2), 1 / np.sqrt(8)
    return (rng.uniform(-k1, k1, (8, 2)), rng.uniform(-k1, k1, 8),
            rng.uniform(-k2, k2, (1, 8)), rng.uniform(-k2, k2, 1))


def accuracy(params, X, y):
    W1, b1, W2, b2 = params
    h = np.maximum(0.0, X @ W1.T + b1)
    p = 1 / (1 + np.exp(-((h @ W2.T)[:, 0] + b2[0])))
    return float(((p > 0.5) == (y == 1)).mean())


# ---------------- our engine ----------------
def train_engine(params0):
    W1, b1, W2, b2 = params0
    W1v = [[Value(w) for w in row] for row in W1.tolist()]
    b1v = [Value(b) for b in b1.tolist()]
    W2v = [Value(w) for w in W2[0].tolist()]
    b2v = Value(float(b2[0]))
    params = [v for row in W1v for v in row] + b1v + W2v + [b2v]
    n, last_loss = len(X_list), None
    for _ in range(EPOCHS):
        losses = []
        for (x0, x1), t in zip(X_list, y_list):
            h = [(W1v[j][0] * x0 + W1v[j][1] * x1 + b1v[j]).relu() for j in range(8)]
            z = sum(W2v[j] * h[j] for j in range(8)) + b2v
            p = (1 + (-z).exp()) ** -1                       # sigmoid
            losses.append(-(t * (p + EPS).log() + (1 - t) * (1.0 - p + EPS).log()))
        loss = sum(losses) * (1.0 / n)                        # mean BCE
        for q in params:
            q.grad = 0.0                                      # zero grads: graph is rebuilt each epoch
        loss.backward()
        last_loss = loss.data
        for q in params:
            q.data -= LR * q.grad
    out = (np.array([[v.data for v in row] for row in W1v]), np.array([v.data for v in b1v]),
           np.array([[v.data for v in W2v]]), np.array([b2v.data]))
    return last_loss, out


# ---------------- PyTorch, same maths, same start ----------------
def train_torch(params0):
    W1, b1, W2, b2 = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in params0]
    Xt, yt = torch.tensor(X), torch.tensor(y)
    last_loss = None
    for _ in range(EPOCHS):
        h = torch.relu(Xt @ W1.T + b1)
        p = torch.sigmoid((h @ W2.T)[:, 0] + b2)
        loss = -(yt * torch.log(p + EPS) + (1 - yt) * torch.log(1 - p + EPS)).mean()
        for t in (W1, b1, W2, b2):
            t.grad = None
        loss.backward()
        last_loss = loss.item()
        with torch.no_grad():
            for t in (W1, b1, W2, b2):
                t -= LR * t.grad
    return last_loss, tuple(t.detach().numpy() for t in (W1, b1, W2, b2))


def median_time(fn, params0):
    times, result = [], None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = fn(params0)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


if __name__ == "__main__":
    p0 = init_params()
    print(f"make_moons n={N_SAMPLES} noise={NOISE}; 2->8->1 ReLU/sigmoid, BCE; "
          f"{EPOCHS} epochs full-batch SGD lr={LR}; median of {N_RUNS} runs\n")
    t_eng, (loss_eng, p_eng) = median_time(train_engine, p0)
    t_tor, (loss_tor, p_tor) = median_time(train_torch, p0)

    print(f"{'':10s} {'median time (s)':>16s} {'final-epoch loss':>17s} {'train accuracy':>15s}")
    print(f"{'engine':10s} {t_eng:16.4f} {loss_eng:17.6f} {accuracy(p_eng, X, y):15.3f}")
    print(f"{'PyTorch':10s} {t_tor:16.4f} {loss_tor:17.6f} {accuracy(p_tor, X, y):15.3f}")
    print(f"\nengine is {t_eng / t_tor:.0f}x slower than PyTorch")
    print(f"max |param difference| engine vs torch after training: "
          f"{max(float(np.abs(a - b).max()) for a, b in zip(p_eng, p_tor)):.2e}")