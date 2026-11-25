import json
import statistics
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from common import BaselineCNN, latency_ms, state_dict_bytes
from models import CNN

ROUNDS = 3                                    # latency is measured in 3 interleaved rounds; the median is kept
SINGLE_THREAD_SET = ["baseline", "global 0.5", "global 0.9", "global 0.95",
                     "structured 0.1", "structured 0.25", "structured 0.5"]
Path("results/tmp").mkdir(parents=True, exist_ok=True)


def load_models():
    models = {}
    m = BaselineCNN()
    m.load_state_dict(torch.load("results/baseline.pt"))
    models["baseline"] = m
    for mode in ("global", "layerwise"):
        for s in (0.25, 0.5, 0.75, 0.9, 0.95):
            m = BaselineCNN()
            m.load_state_dict(torch.load(f"results/pruned/{mode}_{s}.pt"))
            models[f"{mode} {s}"] = m
    st = json.loads(Path("results/pruning_structured.json").read_text())
    for r in st.values():
        f = r["filters_removed"]
        m = CNN(tuple(r["channels"]))
        m.load_state_dict(torch.load(f"results/pruned/structured_{f}.pt"))
        models[f"structured {f}"] = m
    return models


def measure(models, names, threads):
    torch.set_num_threads(threads)
    lat = {n: {"b1": [], "b128": []} for n in names}
    for _ in range(ROUNDS):
        for n in names:
            lat[n]["b1"].append(latency_ms(models[n], 1, runs=100))
            lat[n]["b128"].append(latency_ms(models[n], 128, runs=30))
    return {n: {k: statistics.median(v) for k, v in d.items()} for n, d in lat.items()}


def storage(name, model):
    """On-disk size of the same weights in three formats."""
    tag = name.replace(" ", "_")
    dense_csr, dense_arrays = {}, {}
    for k, v in model.state_dict().items():
        a = v.numpy()
        key = k.replace(".", "_")
        dense_arrays[key] = a
        if a.ndim > 1:                                         # conv / linear weight -> 2-D CSR matrix
            c = sp.csr_matrix(a.reshape(a.shape[0], -1))
            dense_csr[key + "_data"], dense_csr[key + "_idx"], dense_csr[key + "_ptr"] = c.data, c.indices, c.indptr
        else:
            dense_csr[key] = a                                 # biases stay dense
    p1, p2 = f"results/tmp/{tag}_csr.npz", f"results/tmp/{tag}_dense_zip.npz"
    np.savez(p1, **dense_csr)                                  # uncompressed container, CSR values + indices
    np.savez_compressed(p2, **dense_arrays)                    # dense tensors, zlib-compressed
    return {"dense_pt": state_dict_bytes(model), "csr": Path(p1).stat().st_size,
            "dense_compressed": Path(p2).stat().st_size}


if __name__ == "__main__":
    models = load_models()
    n_threads = torch.get_num_threads()
    print(f"{len(models)} models loaded, default torch threads: {n_threads}\n")

    lat_default = measure(models, list(models), n_threads)
    print("default threads done", flush=True)
    lat_single = measure(models, SINGLE_THREAD_SET, 1)
    torch.set_num_threads(n_threads)

    rows = {}
    for n, m in models.items():
        rows[n] = {"nonzero": int(sum((p != 0).sum() for p in m.parameters())),
                   **storage(n, m), "lat_default": lat_default[n], "lat_single": lat_single.get(n)}
    Path("results/latency_size.json").write_text(json.dumps(rows, indent=2))

    base = rows["baseline"]
    print(f"\n{'model':18s} {'nonzero':>9s} {'dense .pt':>10s} {'CSR':>7s} {'dense zip':>10s}"
          f" {'b=1 (ms)':>9s} {'b=128 (ms)':>11s} {'vs base b128':>13s}")
    for n, r in rows.items():
        print(f"{n:18s} {r['nonzero']:9,d} {r['dense_pt'] / 1e6:9.2f}M {r['csr'] / 1e6:6.2f}M "
              f"{r['dense_compressed'] / 1e6:9.2f}M {r['lat_default']['b1']:9.3f} "
              f"{r['lat_default']['b128']:11.2f} {base['lat_default']['b128'] / r['lat_default']['b128']:12.2f}x")

    print("\nsingle thread (torch.set_num_threads(1)):")
    print(f"{'model':18s} {'b=1 (ms)':>9s} {'b=128 (ms)':>11s} {'vs base b128':>13s}")
    for n in SINGLE_THREAD_SET:
        s = rows[n]["lat_single"]
        print(f"{n:18s} {s['b1']:9.3f} {s['b128']:11.2f} {rows['baseline']['lat_single']['b128'] / s['b128']:12.2f}x")
    print("\nsaved results/latency_size.json")