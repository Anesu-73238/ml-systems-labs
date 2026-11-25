import json
from pathlib import Path

R = Path("results")


def load(name):
    return json.loads((R / name).read_text())


base, un, st = load("baseline.json"), load("pruning_unstructured.json"), load("pruning_structured.json")
lat, dist = load("latency_size.json"), load("distillation.json")


def flops_cnn(ch, hidden=256):
    c1, c2, c3, c4 = ch
    return (2 * 9 * 1 * c1 * 784 + 2 * 9 * c1 * c2 * 196 + 2 * 9 * c2 * c3 * 49
            + 2 * 9 * c3 * c4 * 9 + 2 * (c4 * 9) * hidden + 2 * hidden * 10)


# weights per layer x output positions per weight, for the baseline
LAYER_COST = {"features.0.weight": 288 * 784, "features.3.weight": 18_432 * 196,
              "features.6.weight": 73_728 * 49, "features.9.weight": 294_912 * 9,
              "classifier.1.weight": 589_824, "classifier.3.weight": 2_560}


def effective_flops(per_layer_sparsity):
    return sum(2 * LAYER_COST[n] * (1 - s) for n, s in per_layer_sparsity.items())


BASE_FLOPS = flops_cnn((32, 64, 128, 256))
g = un["global_0.9"]
s50 = st["filters_0.5"]
L = lambda key: lat[key]["lat_single"]
student_lat = dist["latency_single"]["student"]
student_flops = flops_cnn((16, 32, 64, 64), 64)
sb = dist["student_size_bytes"]

rows = [
    dict(name="Baseline (dense CNN)", data="10 epochs", params=base["params"], nonzero=base["params"],
         size=lat["baseline"]["dense_pt"], best=min(lat["baseline"]["dense_pt"], lat["baseline"]["csr"], lat["baseline"]["dense_compressed"]),
         acc=base["test_acc"], eff=BASE_FLOPS, exe=BASE_FLOPS, b1=L("baseline")["b1"], b128=L("baseline")["b128"]),
    dict(name="Unstructured, global 90% (+FT)", data="10 + 3 epochs", params=base["params"], nonzero=g["nonzero_params"],
         size=lat["global 0.9"]["dense_pt"], best=min(lat["global 0.9"]["dense_pt"], lat["global 0.9"]["csr"], lat["global 0.9"]["dense_compressed"]),
         acc=g["test_acc"], eff=effective_flops(g["per_layer_sparsity"]), exe=BASE_FLOPS,
         b1=L("global 0.9")["b1"], b128=L("global 0.9")["b128"]),
    dict(name="Structured, 50% filters (+FT)", data="10 + 3 epochs", params=s50["params"], nonzero=s50["params"],
         size=lat["structured 0.5"]["dense_pt"], best=min(lat["structured 0.5"]["dense_pt"], lat["structured 0.5"]["dense_compressed"]),
         acc=s50["test_acc"], eff=flops_cnn(s50["channels"]), exe=flops_cnn(s50["channels"]),
         b1=L("structured 0.5")["b1"], b128=L("structured 0.5")["b128"]),
    dict(name="Student, plain", data="10 epochs", params=dist["student_params"], nonzero=dist["student_params"],
         size=sb, best=sb, acc=dist["stats"]["plain"]["mean"], eff=student_flops, exe=student_flops,
         b1=student_lat["b1"], b128=student_lat["b128"]),
    dict(name="Student, distilled", data="10 epochs + teacher forward pass", params=dist["student_params"],
         nonzero=dist["student_params"], size=sb, best=sb, acc=dist["stats"]["distilled"]["mean"],
         eff=student_flops, exe=student_flops, b1=student_lat["b1"], b128=student_lat["b128"]),
]

b = rows[0]
hdr = ("| Model | Data: training | Model: params (nonzero) | Model: size dense / smallest format (MB) | Model: test acc | "
       "Compute: FLOPs/img effective / executed (M) | Compute: latency 1 thread b=1 / b=128 (ms) | "
       "Energy proxy: time-based (x baseline) | Energy proxy: FLOPs-based (x baseline) |")
lines = [hdr, "|" + "---|" * 9]
for r in rows:
    lines.append(
        f"| {r['name']} | {r['data']} | {r['params']:,} ({r['nonzero']:,}) | {r['size'] / 1e6:.2f} / {r['best'] / 1e6:.2f} | "
        f"{r['acc']:.4f} | {r['eff'] / 1e6:.2f} / {r['exe'] / 1e6:.2f} | {r['b1']:.3f} / {r['b128']:.1f} | "
        f"{r['b128'] / b['b128']:.2f} | {r['eff'] / b['eff']:.2f} |")
table = "\n".join(lines)
Path("results/summary.md").write_text(table + "\n")
print(table)
print(f"\nstudent mean accuracies: plain {dist['stats']['plain']['mean']:.4f}, distilled {dist['stats']['distilled']['mean']:.4f} (3 seeds)")
print("dense fine-tuned control (no pruning, 3 extra epochs):", f"{un['dense_0.0']['test_acc']:.4f}")