import json
import statistics
import time

import torch
import torch.nn as nn

torch.manual_seed(0)


def median_ms(fn, warmup=20, runs=100):
    """Median wall time in ms after untimed warm-up calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def info(name, t):
    print(f"  {name:14s} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} "
          f"contiguous={t.is_contiguous()}")


def walk(fn, depth=0):
    """Print the autograd graph: grad_fn and its next_functions, output -> inputs."""
    if fn is None:
        return
    print("  " + "  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)


results = {}

# ------------------------------------------------------------------ A. graph of a small model
print("=== A. Autograd graph of 2 -> 8 -> 1 (ReLU, sigmoid) ===")
tiny = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
xin = torch.randn(5, 2)
out = tiny(xin)
info("input", xin)
info("output", out)
info("weight 0", tiny[0].weight)
print("  graph (output -> inputs):")
walk(out.grad_fn)

# ------------------------------------------------------------------ B. eager vs traced vs compiled
print("\n=== B. Eager vs torch.jit.trace vs torch.compile (inference, batch 64) ===")
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                      nn.Linear(256, 10)).eval()
x = torch.randn(64, 128)
with torch.no_grad():
    results["eager_ms"] = median_ms(lambda: model(x))

    traced = torch.jit.trace(model, x)
    print("  traced graph:\n" + "\n".join("    " + ln for ln in str(traced.graph).splitlines()[:12]) + "\n    ...")
    assert torch.allclose(traced(x), model(x), atol=1e-6)
    results["trace_ms"] = median_ms(lambda: traced(x))

    try:
        t0 = time.perf_counter()
        compiled = torch.compile(model)
        compiled(x)                                   # first call does the compiling
        results["compile_first_call_s"] = time.perf_counter() - t0
        assert torch.allclose(compiled(x), model(x), atol=1e-5)
        results["compile_ms"] = median_ms(lambda: compiled(x))
    except Exception as e:                            # e.g. no C++ compiler on Windows
        results["compile_ms"] = None
        results["compile_error"] = str(e).strip().splitlines()[0][:200]

print(f"  eager          : {results['eager_ms']:.4f} ms")
print(f"  jit.trace      : {results['trace_ms']:.4f} ms")
if results["compile_ms"] is None:
    print(f"  torch.compile  : UNSUPPORTED here -> {results['compile_error']}")
else:
    print(f"  torch.compile  : {results['compile_ms']:.4f} ms "
          f"(first call incl. compile: {results['compile_first_call_s']:.1f} s)")

# ------------------------------------------------------------------ C. contiguity
print("\n=== C. Contiguous vs non-contiguous tensors (4096 x 4096 float32) ===")
n = 4096
a, b = torch.randn(n, n), torch.randn(n, n)
at = a.t()                      # a view: same memory, strides swapped
sl = a[:, ::2]                  # a view: every second column
sl_c = sl.contiguous()          # same values, packed
info("a", a)
info("a.t()", at)
info("a[:, ::2]", sl)
try:
    at.view(-1)
except RuntimeError as e:
    print("  at.view(-1) fails:", str(e).splitlines()[0][:90])
print("  at.reshape(-1) works but copies:", at.reshape(-1).is_contiguous())

cases = {
    "a + b            (both contiguous)": lambda: a + b,
    "a + a.t()        (one transposed)": lambda: a + at,
    "sl_c * 2         (contiguous copy)": lambda: sl_c * 2,
    "a[:, ::2] * 2    (strided view)": lambda: sl * 2,
    "a.t().contiguous()  (cost of the copy)": lambda: at.contiguous(),
}
results["contiguity_ms"] = {}
print(f"\n  {'operation':42s} {'median ms':>10s}")
for name, fn in cases.items():
    ms = median_ms(fn, warmup=3, runs=20)
    results["contiguity_ms"][name.strip()] = ms
    print(f"  {name:42s} {ms:10.2f}")

with open("step6_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved step6_results.json")