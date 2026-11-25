import ast
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCHS, STOP = 6, 3
BASE = [f"train.epochs={EPOCHS}", "data.num_workers=4", "data.batch_size=128",
        "train.optimizer=adam", "train.lr=0.001", "train.scheduler=cosine"]


def train(run_name, *extra):
    log = Path("results") / f"{run_name}.log"
    cmd = [sys.executable, "train.py", "--set", f"run_name={run_name}", *BASE, *extra]
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
    return result, hist


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    print(f"{EPOCHS} epochs with cosine schedule; interruption after epoch {STOP}\n")

    full_res, full = train("resume-full")
    print(f"uninterrupted run: test acc {full_res['test_acc']:.4f}", flush=True)

    r1, part1 = train("resume-part1", f"train.stop_after_epoch={STOP}")
    print(f"part 1: {r1['epochs_done']} epochs done, interrupted={r1['interrupted']}", flush=True)

    r2, resumed = train("resume-part2", "checkpoint.resume=checkpoints/resume-part1/last.pt")
    print(f"part 2 (resumed): test acc {r2['test_acc']:.4f}\n", flush=True)

    assert len(resumed) == EPOCHS
    print("history of epochs 1-3 carried over unchanged:", resumed[:STOP] == part1)
    print("learning-rate schedule identical to the uninterrupted run:",
          all(abs(a["lr"] - b["lr"]) < 1e-12 for a, b in zip(full, resumed)))

    print(f"\n{'epoch':>5s} {'lr':>9s} {'train loss (full)':>18s} {'(resumed)':>10s} "
          f"{'val acc (full)':>15s} {'(resumed)':>10s}")
    for a, b in zip(full, resumed):
        mark = "  <- resumed here" if a["epoch"] == STOP + 1 else ""
        print(f"{a['epoch']:5d} {b['lr']:9.6f} {a['train_loss']:18.4f} {b['train_loss']:10.4f} "
              f"{a['val_acc']:15.4f} {b['val_acc']:10.4f}{mark}")

    jump_full = full[STOP]["train_loss"] - full[STOP - 1]["train_loss"]
    jump_res = resumed[STOP]["train_loss"] - resumed[STOP - 1]["train_loss"]
    print(f"\ntrain-loss change from epoch {STOP} to {STOP + 1}: uninterrupted {jump_full:+.4f}, resumed {jump_res:+.4f}")
    d_loss = max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(full, resumed))
    d_acc = max(abs(a["val_acc"] - b["val_acc"]) for a, b in zip(full, resumed))
    print(f"max |train loss diff| = {d_loss:.4f}, max |val acc diff| = {d_acc:.4f}")
    print(f"final test acc: uninterrupted {full_res['test_acc']:.4f}, resumed {r2['test_acc']:.4f}")

    json.dump({"full": full, "resumed": resumed}, open("results/resume.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for key, a in zip(("train_loss", "val_acc", "lr"), ax):
        a.plot([h["epoch"] for h in full], [h[key] for h in full], "o-", label="uninterrupted")
        a.plot([h["epoch"] for h in resumed], [h[key] for h in resumed], "s--", label="interrupted + resumed")
        a.axvline(STOP + 0.5, color="gray", linestyle=":")
        a.set_xlabel("epoch")
        a.set_ylabel(key)
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.savefig("results/resume.png", dpi=150)
    print("\nsaved results/resume.json and results/resume.png")