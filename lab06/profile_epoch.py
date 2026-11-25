import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from train import CNN, build_optimizer, make_loaders, set_seed

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def run_epoch(model, loader, opt, dev, max_batches=None):
    """One training epoch. Each phase is labelled for torch.profiler AND timed with perf_counter."""
    timers = dict.fromkeys(PHASES, 0.0)
    model.train()
    it = iter(loader)
    n = 0
    t_epoch = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        with record_function("data_loading"):          # waiting for the next batch + copy to device
            batch = next(it, None)
            if batch is not None:
                x, y = batch[0].to(dev), batch[1].to(dev)
        timers["data_loading"] += time.perf_counter() - t0
        if batch is None:
            break

        t0 = time.perf_counter()
        with record_function("forward"):               # forward pass + loss
            loss = F.cross_entropy(model(x), y)
        timers["forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        with record_function("backward"):
            loss.backward()
        timers["backward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        with record_function("optimizer_step"):        # step + zero_grad
            opt.step()
            opt.zero_grad()
        timers["optimizer_step"] += time.perf_counter() - t0

        n += 1
        if max_batches and n >= max_batches:
            break
    return timers, time.perf_counter() - t_epoch, n


def shares(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}, total


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    dev = torch.device(cfg["train"]["device"])
    train_loader, _, _ = make_loaders(cfg)
    model = CNN().to(dev)
    opt = build_optimizer(model, cfg["train"])

    run_epoch(model, train_loader, opt, dev, max_batches=30)       # warm-up, not measured

    # A) one epoch without the profiler (manual timers only, no profiler overhead)
    timers, wall_plain, n = run_epoch(model, train_loader, opt, dev)

    # B) one epoch under torch.profiler
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        ptimers, wall_prof, _ = run_epoch(model, train_loader, opt, dev)

    events = {e.key: e for e in prof.key_averages() if e.key in PHASES}
    prof_s = {p: events[p].cpu_time_total / 1e6 for p in PHASES}   # microseconds -> seconds

    sh_plain, tot_plain = shares(timers)
    sh_prof, tot_prof = shares(prof_s)

    print(f"\nbatch size {cfg['data']['batch_size']}, num_workers {cfg['data']['num_workers']}, "
          f"{n} batches/epoch, device {dev}\n")
    print(f"{'phase':16s} {'profiler (s)':>13s} {'share':>7s} {'perf_counter (s)':>17s} {'share':>7s}")
    for p in PHASES:
        print(f"{p:16s} {prof_s[p]:13.3f} {sh_prof[p]*100:6.1f}% {timers[p]:17.3f} {sh_plain[p]*100:6.1f}%")
    print(f"{'sum of phases':16s} {tot_prof:13.3f} {'':7s} {tot_plain:17.3f}")
    print(f"\nepoch wall time: {wall_plain:.2f} s without profiler, {wall_prof:.2f} s with profiler "
          f"({wall_prof / wall_plain:.2f}x overhead)")

    print("\nTop operators by self CPU time (profiler):")
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
    print(table)

    Path("results").mkdir(exist_ok=True)
    Path("results/profile_table.txt").write_text(table)
    Path("results/profile.json").write_text(json.dumps({
        "profiler_s": prof_s, "profiler_share": sh_prof,
        "perf_counter_s": timers, "perf_counter_share": sh_plain,
        "epoch_wall_plain_s": wall_plain, "epoch_wall_profiled_s": wall_prof,
        "batch_size": cfg["data"]["batch_size"], "num_workers": cfg["data"]["num_workers"]}, indent=2))
    print("saved results/profile.json and results/profile_table.txt")