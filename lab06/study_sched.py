import ast
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCHS = 10
BASE = [f"train.epochs={EPOCHS}", "data.num_workers=4", "data.batch_size=128"]
RUNS = {
    "adam-constant": ["train.optimizer=adam", "train.lr=0.001", "train.scheduler=none"],
    "adam-step":     ["train.optimizer=adam", "train.lr=0.001", "train.scheduler=step",
                      "train.step_size=3", "train.gamma=0.5"],
    "adam-cosine":   ["train.optimizer=adam", "train.lr=0.001", "train.scheduler=cosine"],
    "adam-plateau":  ["train.optimizer=adam", "train.lr=0.001", "train.scheduler=plateau",
                      "train.gamma=0.5", "train.patience=1"],
    "sgd-constant":  ["train.optimizer=sgd", "train.lr=0.01", "train.momentum=0.9",
                      "train.scheduler=none"],
}


def run_one(name, overrides):
    run_name = f"sched-{name}"
    log = Path("results") / f"{run_name}.log"
    cmd = [sys.executable, "train.py", "--set", f"run_name={run_name}", *BASE, *overrides]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{run_name} failed, see {log}")
    result = None
    for line in reversed(log.read_text().splitlines()):
        if line.startswith("{'run_name'"):
            result = ast.literal_eval(line)
            break
    hist = json.loads((Path("checkpoints") / run_name / "history.json").read_text())
    best = max(hist, key=lambda h: h["val_acc"])
    return {"name": name, "history": hist, "final_val_acc": hist[-1]["val_acc"],
            "best_val_acc": best["val_acc"], "best_epoch": best["epoch"],
            "final_train_loss": hist[-1]["train_loss"], "final_val_loss": hist[-1]["val_loss"],
            "test_acc": result["test_acc"], "final_lr": hist[-1]["lr"]}


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    print(f"{EPOCHS} epochs per run, batch 128, seed from config.yaml\n")
    rows = []
    for name, ov in RUNS.items():
        r = run_one(name, ov)
        rows.append(r)
        print(f"{name}: final val {r['final_val_acc']:.4f}, test {r['test_acc']:.4f}", flush=True)

    print(f"\n{'run':15s} {'train loss':>10s} {'val loss':>9s} {'final val':>10s} "
          f"{'best val (epoch)':>17s} {'test acc':>9s} {'last lr':>9s}")
    for r in rows:
        print(f"{r['name']:15s} {r['final_train_loss']:10.4f} {r['final_val_loss']:9.4f} "
              f"{r['final_val_acc']:10.4f} {r['best_val_acc']:10.4f} ({r['best_epoch']:2d}) "
              f"{r['test_acc']:9.4f} {r['final_lr']:9.5f}")

    json.dump(rows, open("results/schedulers.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in rows:
        ep = [h["epoch"] for h in r["history"]]
        ax[0].plot(ep, [h["lr"] for h in r["history"]], "o-", label=r["name"])
        ax[1].plot(ep, [h["val_loss"] for h in r["history"]], "o-", label=r["name"])
        ax[2].plot(ep, [h["val_acc"] for h in r["history"]], "o-", label=r["name"])
    ax[0].set_ylabel("learning rate used in the epoch")
    ax[0].set_yscale("log")
    ax[1].set_ylabel("validation loss")
    ax[2].set_ylabel("validation accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/schedulers.png", dpi=150)
    print("\nsaved results/schedulers.json and results/schedulers.png")