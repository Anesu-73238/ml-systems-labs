import json

r = json.load(open("results/latency_size.json"))
print(f"{'model':18s} {'nonzero':>9s} {'pt MB':>6s} {'csr MB':>7s} {'zip MB':>7s} {'b1 ms':>7s} {'b128 ms':>8s}")
for n, v in r.items():
    l = v["lat_default"]
    print(f"{n:18s} {v['nonzero']:9,d} {v['dense_pt'] / 1e6:6.2f} {v['csr'] / 1e6:7.2f} "
          f"{v['dense_compressed'] / 1e6:7.2f} {l['b1']:7.3f} {l['b128']:8.2f}")