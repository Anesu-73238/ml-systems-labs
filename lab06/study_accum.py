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

from study_batch import is_train, rss      # reuse the memory-sampling helpers

EPOCHS = 10
N_TRAIN = 55000
RUNS = {                                    # label: (run_name, micro-batch size, accumulation steps)
    "bs256 (reference)": ("accum-bs256", 256, 1),
    "bs64 x accum 4":    ("accum-bs64x4", 64, 4),
    "bs64 (no accum)":   ("accum-bs64", 64, 1),
}


def run_one(run_name, bs, accum):
    log = Path("results") / f"{run_name}.log"
    cmd = [sys.executable, "train.py", "--set", f"run_name={run_name}",
           f"train.epochs={EPOCHS}", "data.num_workers=4", f"data.batch_size={bs}",
           f"train.accum_steps={accum}", "train.optimizer=adam", "train.lr=0.001",
           "train.scheduler=none"]
    peak_main, cache = 0, {}
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                procs = [root] + root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                procs = []
            peak_main = max([peak_main] + [rss(p) for p in procs if is_train(p, cache)])
            time.sleep(0.05)
    if proc.returncode != 0:
        raise RuntimeError(f"{run_name} failed, see {log}")
    result = None
    for line in reversed(log.read_text().splitlines()):
        if line.startswith("{'run_name'"):
            result = ast.literal_eval(line)
            break
    hist = json.loads((Path("checkpoints") / run_name / "history.json").read_text())
    return {"history": hist, "test_acc": result["test_acc"], "peak_main_mb": peak_main / 2**20,
            "updates_per_epoch": -(-N_TRAIN // (bs * accum)),
            "samples_per_s": statistics.median(h["samples_per_s"] for h in hist)}


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    out = {}
    for label, (name, bs, accum) in RUNS.items():
        out[label] = run_one(name, bs, accum)
        print(f"{label}: test acc {out[label]['test_acc']:.4f}", flush=True)

    print(f"\n{'run':20s} {'updates/ep':>10s} {'samples/s':>10s} {'peak MB':>8s} "
          f"{'train loss':>10s} {'final val':>10s} {'test acc':>9s}")
    for label, r in out.items():
        h = r["history"][-1]
        print(f"{label:20s} {r['updates_per_epoch']:10d} {r['samples_per_s']:10.0f} "
              f"{r['peak_main_mb']:8.0f} {h['train_loss']:10.4f} {h['val_acc']:10.4f} {r['test_acc']:9.4f}")

    def curve(label, key):
        return [h[key] for h in out[label]["history"]]

    ref = "bs256 (reference)"
    for other in ("bs64 x accum 4", "bs64 (no accum)"):
        dl = max(abs(a - b) for a, b in zip(curve(ref, "train_loss"), curve(other, "train_loss")))
        da = max(abs(a - b) for a, b in zip(curve(ref, "val_acc"), curve(other, "val_acc")))
        print(f"\nvs {ref}: {other}: max |train loss diff| = {dl:.5f}, max |val acc diff| = {da:.5f}")

    json.dump(out, open("results/accumulation.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label in out:
        ep = [h["epoch"] for h in out[label]["history"]]
        ax[0].plot(ep, curve(label, "train_loss"), "o-", label=label)
        ax[1].plot(ep, curve(label, "val_acc"), "o-", label=label)
    ax[0].set_ylabel("train loss")
    ax[1].set_ylabel("validation accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.savefig("results/accumulation.png", dpi=150)
    print("\nsaved results/accumulation.json and results/accumulation.png")