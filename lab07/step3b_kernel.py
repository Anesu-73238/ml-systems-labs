import statistics
import time

import scipy.sparse as sp
import torch

from common import BaselineCNN

torch.set_num_threads(1)            # one thread everywhere: scipy's sparse kernel is single-threaded


def bench(fn, warmup=20, runs=200):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def fc1_weight(path):
    m = BaselineCNN()
    m.load_state_dict(torch.load(path))
    return m.classifier[1].weight.detach().contiguous()      # (256, 2304)


CASES = [("baseline (dense)", "results/baseline.pt"),
         ("global 0.5", "results/pruned/global_0.5.pt"),
         ("global 0.9", "results/pruned/global_0.9.pt"),
         ("global 0.95", "results/pruned/global_0.95.pt"),
         ("layerwise 0.9", "results/pruned/layerwise_0.9.pt")]

if __name__ == "__main__":
    print("fc1 = Linear(2304 -> 256), 589,824 weights, single thread, median time in microseconds\n")
    print(f"{'weights':18s} {'fc1 zeros':>9s} {'batch':>6s} {'dense torch':>12s} {'CSR scipy':>10s} {'CSR torch':>10s} {'best sparse / dense':>20s}")
    for name, path in CASES:
        W = fc1_weight(path)
        zeros = float((W == 0).float().mean())
        Wn = W.numpy()
        Wc = sp.csr_matrix(Wn)
        try:
            Wt = W.to_sparse_csr()
        except Exception:
            Wt = None
        for b in (1, 128):
            X = torch.randn(2304, b)
            Xn = X.numpy()
            t_dense = bench(lambda: torch.mm(W, X))
            t_scipy = bench(lambda: Wc @ Xn)
            try:
                t_tcsr = bench(lambda: torch.sparse.mm(Wt, X))
            except Exception:
                t_tcsr = float("nan")
            best = min(t_scipy, t_tcsr) if t_tcsr == t_tcsr else t_scipy
            print(f"{name:18s} {zeros:9.2f} {b:6d} {t_dense:12.1f} {t_scipy:10.1f} {t_tcsr:10.1f} {best / t_dense:19.2f}x")