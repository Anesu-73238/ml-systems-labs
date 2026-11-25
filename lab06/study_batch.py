import ast
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil

BATCH_SIZES = [32, 64, 128, 256, 512]
WORKERS = 4
N_TRAIN = 55000


def rss(p):
    try:
        return p.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def is_train(p, cache):
    """True for the process running train.py (not for DataLoader worker processes)."""
    if p.pid not in cache:
        try:
            cache[p.pid] = "train.py" in " ".join(p.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    return cache[p.pid]


def run_one(bs):
    name = f"bs{bs}"
    log = Path("results") / f"{name}.log"
    cmd = [sys.executable, "train.py", "--set", f"run_name={name}",
           f"data.batch_size={bs}", f"data.num_workers={WORKERS}"]
    peak_main = peak_total = 0
    cache = {}
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                procs = [root] + root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                procs = []
            peak_total = max(peak_total, sum(rss(p) for p in procs))
            peak_main = max([peak_main] + [rss(p) for p in procs if is_train(p, cache)])
            time.sleep(0.05)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed, see {log}")

    result = None
    for line in reversed(log.read_text().splitlines()):
        if line.startswith("{'run_name'"):
            result = ast.literal_eval(line)
            break
    hist = json.loads((Path("checkpoints") / name / "history.json").read_text())
    return {
        "batch_size": bs,
        "updates_per_epoch": -(-N_TRAIN // bs),
        "samples_per_s": statistics.median(h["samples_per_s"] for h in hist),
        "epoch_s": statistics.median(h["epoch_time_s"] for h in hist),
        "peak_main_mb": peak_main / 2**20,
        "peak_total_mb": peak_total / 2**20,
        "final_val_acc": result["final_val_acc"],
        "test_acc": result["test_acc"],
    }


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    print(f"batch sizes {BATCH_SIZES}, num_workers {WORKERS}, 5 epochs, lr and optimizer from config.yaml\n")
    rows = []
    for bs in BATCH_SIZES:
        r = run_one(bs)
        rows.append(r)
        print(f"bs={bs}: {r['samples_per_s']:.0f} samples/s, peak main {r['peak_main_mb']:.0f} MB, "
              f"test acc {r['test_acc']:.4f}", flush=True)

    print(f"\n{'batch':>6s} {'updates/ep':>11s} {'samples/s':>10s} {'epoch (s)':>10s} "
          f"{'peak main MB':>13s} {'peak total MB':>14s} {'val acc':>8s} {'test acc':>9s}")
    for r in rows:
        print(f"{r['batch_size']:6d} {r['updates_per_epoch']:11d} {r['samples_per_s']:10.0f} "
              f"{r['epoch_s']:10.2f} {r['peak_main_mb']:13.0f} {r['peak_total_mb']:14.0f} "
              f"{r['final_val_acc']:8.4f} {r['test_acc']:9.4f}")

    Path("results/batch_size.json").write_text(json.dumps(rows, indent=2))
    bs = [str(r["batch_size"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(bs, [r["samples_per_s"] for r in rows], "o-")
    ax[0].set_ylabel("training throughput (samples/s)")
    ax[1].plot(bs, [r["peak_main_mb"] for r in rows], "o-", label="training process")
    ax[1].plot(bs, [r["peak_total_mb"] for r in rows], "s--", label="incl. worker processes")
    ax[1].set_ylabel("peak memory (MB, RSS)")
    ax[1].legend()
    ax[2].plot(bs, [r["test_acc"] for r in rows], "o-", label="test")
    ax[2].plot(bs, [r["final_val_acc"] for r in rows], "s--", label="validation")
    ax[2].set_ylabel("accuracy after 5 epochs")
    ax[2].legend()
    for a in ax:
        a.set_xlabel("batch size")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/batch_size.png", dpi=150)
    print("\nsaved results/batch_size.json and results/batch_size.png")