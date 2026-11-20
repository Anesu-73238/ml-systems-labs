# Lab 5: Frameworks and automatic differentiation

Machine: TODO (CPU, OS). Python 3.11, PyTorch (version from `pip show torch`), CPU only.
Code: `engine.py` (autograd engine), `test_backward.py`, `verify.py`, `train_moons.py`, `torch_inspect.py`.

## 1. The autograd engine (Steps 1-3)

**Value node.** Each `Value` stores `data`, `grad`, `_parents` (the nodes it was computed from, i.e. the graph edges),
`_backward` (its local derivative rule) and `_op` (a label for inspecting the graph).

**Operations.** `+`, `*`, `**` (constant exponent), `relu`, plus `exp` and `log` (needed for sigmoid and BCE in Step 5).
Subtraction, negation, division and the reversed operators are built from these primitives, so they need no rules of their own.
Every operation creates a new `Value` that records its parents, so the graph is built during the forward pass.
For `((a*b + c)**2).relu()` the graph walk is `relu <- **2 <- + <- (*, leaf)` and `* <- (leaf, leaf)`: 3 leaves + 4 operation nodes = 7 nodes.

| Operation | Local rule (added into the parents' grads) |
|---|---|
| `a + b` | `a.grad += out.grad`, `b.grad += out.grad` |
| `a * b` | `a.grad += b.data * out.grad`, `b.grad += a.data * out.grad` |
| `a ** k` | `a.grad += k * a.data**(k-1) * out.grad` |
| `relu(a)` | `a.grad += (out.data > 0) * out.grad` |
| `exp(a)` | `a.grad += out.data * out.grad` |
| `log(a)` | `a.grad += out.grad / a.data` |

**Backward pass.**
1. Topologically sort the graph (iterative depth-first search; every node comes after its parents).
2. Set the output gradient to 1.
3. Visit the nodes in reverse topological order and call each node's `_backward`.
4. Every rule uses `+=` (accumulate), never `=`, so a node used more than once receives the sum of its contributions.

Tests in `test_backward.py` (all pass):
- `a * a` gives `a.grad = 2a = 6` for `a = 3`; overwriting instead of accumulating would give 3.
- `a*b + a` gives `da = b + 1`, `db = a`.
- ReLU on a negative input blocks the gradient.
- The lab expression for `a=2, b=-3, c=10`: `y = 16`, `dy/da = -24`, `dy/db = 16`, `dy/dc = 8`
  (check by hand: the inner value is 4, so `dy/d(inner) = 2*4 = 8`; then `dc = 8`, `da = 8*b = -24`, `db = 8*a = 16`).
- Every node appears after all of its parents in the topological order.

## 2. Verification against PyTorch and numerical gradients (Step 4)

`verify.py` evaluates `((a*b + c)**2).relu().sum()` on 10 random triples. The scalar engine adds the ten results with Python's `sum()`
(it has no `.sum()` method). PyTorch autograd runs in float64, and the numerical gradient uses central differences with eps = 1e-5.
Because the square is never negative, ReLU never blocks in the lab expression, so I added a second expression,
`(a*b + c).relu().sum()`, on inputs where ReLU is active for 6 of 10 and blocking for 4. The inputs are kept at least 0.05 away
from the ReLU kink, where numerical gradients are not meaningful.

| Expression | max abs diff engine vs PyTorch | engine vs numeric | PyTorch vs numeric |
|---|---|---|---|
| `((a*b + c)**2).relu().sum()` | 0.00e+00 | 1.37e-09 | 1.37e-09 |
| `(a*b + c).relu().sum()` (extra) | 0.00e+00 | 2.14e-10 | 2.14e-10 |

Maximum over both = **1.37e-9 < 1e-6: PASS.** The engine matches PyTorch exactly (the forward values differ by at most 3.6e-15).
The ~1e-9 differences to the numerical estimate are finite-difference rounding error, which is expected.

## 3. Training on make_moons: engine vs PyTorch (Step 5)

Setup: `make_moons(n=200, noise=0.2)`, network 2 -> 8 -> 1 (ReLU hidden layer, sigmoid output), binary cross-entropy (mean),
100 epochs of full-batch gradient descent with lr = 0.5. Both implementations start from the same initial weights.
The engine rebuilds its graph every epoch and zeroes the parameter gradients before `backward()`, because accumulation would otherwise
add old gradients. Timing: median of 5 runs.

| | Median time (s) | Final loss | Train accuracy |
|---|---|---|---|
| Scalar engine | 14.69 | 0.288788 | 0.875 |
| PyTorch | 0.0336 | 0.288788 | 0.875 |

The engine was **438x slower** in this run (387x in an earlier run: 14.37 s vs 0.0371 s), so I report it as roughly 400x.
After training, the largest parameter difference between the two implementations is **3.3e-16**, so they followed the same trajectory.

Why PyTorch is so much faster:
- The engine does one Python-level operation per scalar. One epoch builds roughly 20,000 nodes (about 100 per sample x 200 samples), plus the allocation of
  closures, the topological sort and a Python call per node in the backward pass.
- PyTorch's graph for the same network has about 10 tensor-level nodes (see section 4A), and each node is one optimized matrix operation in compiled code.

## 4. PyTorch graph, eager vs traced vs compiled, contiguity (Step 6)

### A. Autograd graph of the 2 -> 8 -> 1 model

```
SigmoidBackward0
  AddmmBackward0                <- Linear(8 -> 1)
    AccumulateGrad              <- bias b2
    ReluBackward0
      AddmmBackward0            <- Linear(2 -> 8)
        AccumulateGrad          <- bias b1
        TBackward0              <- transposed W1
          AccumulateGrad        <- W1
    TBackward0                  <- transposed W2
      AccumulateGrad            <- W2
```

Each `Linear` becomes one fused `AddmmBackward0` (bias + matrix multiply) with a `TBackward0` for the transposed weight, and the four
`AccumulateGrad` leaves are the parameters, where PyTorch does the same accumulation as `+=` in my engine. The input `x` does not require
a gradient, so it has no node. Tensor facts: input shape (5, 2), float32, stride (2, 1), contiguous; output shape (5, 1), float32;
`weight 0` shape (8, 2), float32, stride (2, 1), contiguous.

### B. Eager vs torch.jit.trace vs torch.compile

Model: Linear(128, 256) - ReLU - Linear(256, 256) - ReLU - Linear(256, 10), input batch 64, inference, `no_grad`, 20 warm-up calls, median of 100 runs.

| Mode | Median (ms) |
|---|---|
| Eager | 0.1798 |
| `torch.jit.trace` | 0.1647 |
| `torch.compile` | not supported on my machine: `InvalidCxxCompiler: Compiler: cl is not found` |

`torch.compile` could not run because its default backend (inductor) needs a C++ compiler (MSVC `cl`) that is not installed on this
Windows machine. As the lab allows, I used `torch.jit.trace` instead (PyTorch also prints a `FutureWarning` that `torch.jit.trace` is
deprecated in favour of `torch.compile` / `torch.export`). The traced model is about 8% faster than eager, which is small enough to be
partly noise on a tiny model.

### C. Contiguity (4096 x 4096 float32)

| Tensor | Shape | Strides | Contiguous |
|---|---|---|---|
| `a` | (4096, 4096) | (4096, 1) | True |
| `a.t()` | (4096, 4096) | (1, 4096) | False |
| `a[:, ::2]` | (4096, 2048) | (4096, 2) | False |

`a.t().view(-1)` fails ("view size is not compatible with input tensor's size and stride"), while `.reshape(-1)` works because it copies.

| Operation | Median (ms) |
|---|---|
| `a + b` (both contiguous) | 6.95 |
| `a + a.t()` (one operand transposed) | 86.73 |
| `sl_c * 2` (contiguous copy of the slice) | 2.17 |
| `a[:, ::2] * 2` (strided view, same values and shape) | 4.30 |
| `a.t().contiguous()` (cost of making a copy) | 29.75 |

TODO: these timings are from one run; a repeat run gave 4.47 ms and 25.74 ms for the last two rows, so rerun once and copy the final values
from `step6_results.json` so the table matches the file.

- The transposed add is about 12x slower than the contiguous add: `a.t()` has stride 4096 in its fast dimension, so the elementwise loop jumps 16 KB between
  neighbouring reads and misses the cache on almost every access.
- The strided slice is about 2x slower than its contiguous copy, which has the same shape and values, so the gap comes purely from memory layout.
- Making `a.t()` contiguous first costs about 30 ms, and the add then costs about 7 ms, so about 37 ms in total, which is less than half
  of the 87 ms for the add on the non-contiguous view, even if the tensor is used only once.

## 5. When does compilation help?

TODO: rewrite this in your own words. Draft:

Compilation (`torch.compile`) captures the computation graph and can fuse operations, for example bias + ReLU + scaling into one kernel,
which removes intermediate tensors and kernel launches. That helps most for models with many small element-wise operations, for GPU
execution where launch overhead matters, and for code that is run many times with fixed shapes. It costs time up front (the first call compiles),
so it does not pay off for one-off calls, tiny models or very short runs, and changing input shapes can trigger recompilation.
My model is a small MLP dominated by matrix multiplications, which PyTorch already runs in optimized library kernels, so there is
little to fuse. This is consistent with the small gain from tracing (0.1647 vs 0.1798 ms), which mainly removes Python overhead.
I could not measure `torch.compile` on my machine (no C++ compiler), so statements about it come from the reading and general behaviour,
not from my measurements.

## 6. Limitations

- The scalar engine supports only the operations needed here, and only constant exponents in `**`.
- Timings are from a laptop CPU and vary between runs (for example 387x vs 438x slowdown), so differences of a few percent should not be over-interpreted.
- The eager vs trace comparison uses one small model; `torch.compile` was not measured.
