import gc
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (BaselineCNN, count_params, evaluate, latency_ms, make_loaders, set_seed,
                    state_dict_bytes)
from models import CNN

T, ALPHA = 4.0, 0.5
EPOCHS, LR = 10, 1e-3
SEEDS = [0, 1, 2]
STUDENT_CH, STUDENT_H = (16, 32, 64, 64), 64


def distill_loss(s_logits, t_logits, y):
    """alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(teacher distribution || student distribution)."""
    ce = F.cross_entropy(s_logits, y)
    kl = F.kl_div(F.log_softmax(s_logits / T, dim=1), F.softmax(t_logits / T, dim=1),
                  reduction="batchmean")
    return ALPHA * ce + (1 - ALPHA) * T * T * kl


def train_student(seed, teacher, loaders, distill, tag):
    train_loader, val_loader, _ = loaders
    set_seed(seed)
    train_loader.sampler.generator.manual_seed(seed)      # different data order per seed, same split
    student = CNN(STUDENT_CH, STUDENT_H)
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    hist = []
    for ep in range(1, EPOCHS + 1):
        student.train()
        for x, y in train_loader:
            opt.zero_grad()
            out = student(x)
            if distill:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = distill_loss(out, t_out, y)
            else:
                loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
        hist.append(evaluate(student, val_loader)[1])
        print(f"{tag} epoch {ep}/{EPOCHS}  val_acc {hist[-1]:.4f}", flush=True)
    return student, hist


def measure_latency(models, threads):
    torch.set_num_threads(threads)
    lat = {n: {"b1": [], "b128": []} for n in models}
    for _ in range(3):                                     # 3 interleaved rounds, median kept
        for n, m in models.items():
            lat[n]["b1"].append(latency_ms(m, 1))
            lat[n]["b128"].append(latency_ms(m, 128, runs=30))
    return {n: {k: statistics.median(v) for k, v in d.items()} for n, d in lat.items()}


if __name__ == "__main__":
    teacher = BaselineCNN()
    teacher.load_state_dict(torch.load("results/baseline.pt"))
    teacher.eval()
    loaders = make_loaders()
    teacher_acc = evaluate(teacher, loaders[2])[1]
    print(f"teacher: {count_params(teacher):,} params, test acc {teacher_acc:.4f}\n")

    res = {"plain": [], "distilled": []}
    last = {}
    for seed in SEEDS:
        for name, distill in (("plain", False), ("distilled", True)):
            student, hist = train_student(seed, teacher, loaders, distill, f"[seed {seed} {name}]")
            acc = evaluate(student, loaders[2])[1]
            res[name].append({"seed": seed, "test_acc": acc, "val_acc": hist[-1], "val_curve": hist})
            print(f"--> seed {seed} {name}: test acc {acc:.4f}\n", flush=True)
            last[name] = student
            if seed == SEEDS[0]:
                torch.save(student.state_dict(), f"results/student_{name}.pt")

    del loaders                                            # stop the DataLoader workers before timing
    gc.collect()
    n_threads = torch.get_num_threads()
    models = {"teacher (baseline)": teacher, "student": last["plain"]}
    lat_default = measure_latency(models, n_threads)
    lat_single = measure_latency(models, 1)
    torch.set_num_threads(n_threads)

    stats = {}
    for name in res:
        a = np.array([r["test_acc"] for r in res[name]])
        stats[name] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "per_seed": a.tolist()}
    diff = np.array(stats["distilled"]["per_seed"]) - np.array(stats["plain"]["per_seed"])

    out = {"teacher_test_acc": teacher_acc, "T": T, "alpha": ALPHA, "epochs": EPOCHS,
           "student_params": count_params(last["plain"]), "student_size_bytes": state_dict_bytes(last["plain"]),
           "teacher_params": count_params(teacher), "teacher_size_bytes": state_dict_bytes(teacher),
           "latency_default": lat_default, "latency_single": lat_single, "stats": stats,
           "paired_diff_per_seed": diff.tolist(), "runs": res}
    Path("results/distillation.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'model':22s} {'params':>9s} {'size MB':>8s} {'test acc (mean +- std, 3 seeds)':>32s} {'per seed':>26s}")
    print(f"{'teacher (baseline)':22s} {out['teacher_params']:9,d} {out['teacher_size_bytes'] / 1e6:8.2f} {teacher_acc:32.4f}")
    for name in ("plain", "distilled"):
        s = stats[name]
        print(f"{'student ' + name:22s} {out['student_params']:9,d} {out['student_size_bytes'] / 1e6:8.2f} "
              f"{s['mean']:20.4f} +- {s['std']:.4f}   " + "  ".join(f"{v:.4f}" for v in s["per_seed"]))
    print(f"\ndistilled minus plain, per seed: " + "  ".join(f"{d:+.4f}" for d in diff) +
          f"   (mean {diff.mean():+.4f})")
    for label, lat in (("all threads", lat_default), ("single thread", lat_single)):
        t, s = lat["teacher (baseline)"], lat["student"]
        print(f"latency, {label}: teacher b=1 {t['b1']:.3f} ms / b=128 {t['b128']:.2f} ms; "
              f"student b=1 {s['b1']:.3f} ms / b=128 {s['b128']:.2f} ms "
              f"({t['b1'] / s['b1']:.1f}x / {t['b128'] / s['b128']:.1f}x faster)")
    print("\nsaved results/distillation.json, results/student_plain.pt, results/student_distilled.pt")