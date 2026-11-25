import gc
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from profile_epoch import run_epoch
from train import CNN, build_optimizer, make_loaders, set_seed

WORKERS = [0, 1, 2, 4, 8]
REPEATS = 3          # measured epochs per setting (the median one is reported)


def measure(cfg, nw):
    cfg["data"]["num_workers"] = nw
    set_seed(cfg["seed"])
    dev = torch.device(cfg["train"]["device"])
    train_loader, _, _ = make_loaders(cfg)
    n_samples = len(train_loader.dataset)
    model = CNN().to(dev)
    opt = build_optimizer(model, cfg["train"])

    _, first, _ = run_epoch(model, train_loader, opt, dev)        # includes worker start-up
    runs = [run_epoch(model, train_loader, opt, dev) for _ in range(REPEATS)]
    runs.sort(key=lambda r: r[1])
    timers, wall, _ = runs[len(runs) // 2]                        # median epoch

    del train_loader
    gc.collect()                                                  # shuts the workers down
    compute = timers["forward"] + timers["backward"] + timers["optimizer_step"]
    return {"num_workers": nw, "first_epoch_s": first, "epoch_s": wall,
            "samples_per_s": n_samples / wall, "data_wait_s": timers["data_loading"],
            "compute_s": compute, "data_share": timers["data_loading"] / sum(timers.values())}


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"CPU cores: {os.cpu_count()}, batch size {cfg['data']['batch_size']}, "
          f"median of {REPEATS} epochs after 1 warm-up epoch\n")
    rows = []
    for nw in WORKERS:
        r = measure(cfg, nw)
        rows.append(r)
        print(f"num_workers={nw}: epoch {r['epoch_s']:.2f} s, {r['samples_per_s']:.0f} samples/s, "
              f"data wait {r['data_share'] * 100:.0f}%", flush=True)

    base = rows[0]["epoch_s"]
    print(f"\n{'workers':>8s} {'1st epoch (s)':>14s} {'epoch (s)':>10s} {'samples/s':>10s} "
          f"{'data wait (s)':>14s} {'compute (s)':>12s} {'data share':>11s} {'speed-up':>9s}")
    for r in rows:
        print(f"{r['num_workers']:8d} {r['first_epoch_s']:14.2f} {r['epoch_s']:10.2f} "
              f"{r['samples_per_s']:10.0f} {r['data_wait_s']:14.2f} {r['compute_s']:12.2f} "
              f"{r['data_share'] * 100:10.0f}% {base / r['epoch_s']:8.2f}x")

    Path("results").mkdir(exist_ok=True)
    Path("results/workers.json").write_text(json.dumps(rows, indent=2))

    ws = [str(r["num_workers"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ws, [r["samples_per_s"] for r in rows], "o-")
    ax[0].set_xlabel("num_workers")
    ax[0].set_ylabel("training throughput (samples/s)")
    ax[0].set_title("Throughput vs num_workers")
    ax[0].grid(alpha=0.3)
    ax[1].bar(ws, [r["compute_s"] for r in rows], label="compute (fwd+bwd+opt)")
    ax[1].bar(ws, [r["data_wait_s"] for r in rows], bottom=[r["compute_s"] for r in rows],
              label="waiting for data")
    ax[1].set_xlabel("num_workers")
    ax[1].set_ylabel("seconds per epoch")
    ax[1].set_title("Where the epoch time goes")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("results/workers.png", dpi=150)
    print("\nsaved results/workers.json and results/workers.png")