import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CNN(nn.Module):
    """The Lab 3 CNN (20,490 parameters)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loaders(cfg):
    d = cfg["data"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(d["root"], train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(d["root"], train=False, download=True, transform=tf)
    train, val = random_split(
        full, [len(full) - d["val_size"], d["val_size"]],
        generator=torch.Generator().manual_seed(cfg["seed"]))
    nw = d["num_workers"]
    kw = dict(batch_size=d["batch_size"], num_workers=nw, persistent_workers=nw > 0,
              worker_init_fn=seed_worker)
    g = torch.Generator().manual_seed(cfg["seed"])
    return (DataLoader(train, shuffle=True, generator=g, **kw),
            DataLoader(val, **kw), DataLoader(test, **kw))


def build_optimizer(model, t):
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                               weight_decay=t["weight_decay"])
    raise ValueError(t["optimizer"])


def build_scheduler(opt, t):
    name = t["scheduler"]
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=t["step_size"], gamma=t["gamma"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=t["gamma"], patience=t["patience"])
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, correct / n


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def apply_overrides(cfg, items):
    """--set train.epochs=3 data.batch_size=64"""
    for item in items:
        key, val = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node[p]
        node[last] = yaml.safe_load(val)


def run(cfg):
    set_seed(cfg["seed"])
    t = cfg["train"]
    dev = torch.device(t["device"])
    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = CNN().to(dev)
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t)
    accum = t["accum_steps"]

    ckdir = Path(cfg["checkpoint"]["dir"]) / cfg["run_name"]
    ckdir.mkdir(parents=True, exist_ok=True)
    start, best_val, history = 0, 0.0, []
    resume = cfg["checkpoint"]["resume"]
    if resume:
        ck = torch.load(resume, map_location=dev, weights_only=False)  # our own file
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck["sched"] is not None:
            sched.load_state_dict(ck["sched"])
        start, best_val, history = ck["epoch"], ck["best_val_acc"], ck["history"]
        print(f"resumed from {resume} after epoch {start}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["experiment"])
    interrupted = False
    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params(flatten(cfg))
        mlflow.log_param("n_params", sum(p.numel() for p in model.parameters()))
        cfg_path = ckdir / "config_used.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        mlflow.log_artifact(str(cfg_path))

        for epoch in range(start, t["epochs"]):
            model.train()
            t0 = time.perf_counter()
            loss_sum, seen = 0.0, 0
            n_batches = len(train_loader)
            opt.zero_grad()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(dev), y.to(dev)
                loss = F.cross_entropy(model(x), y)
                (loss / accum).backward()
                if (i + 1) % accum == 0 or (i + 1) == n_batches:
                    opt.step()
                    opt.zero_grad()
                loss_sum += loss.item() * y.numel()
                seen += y.numel()
            elapsed = time.perf_counter() - t0
            val_loss, val_acc = evaluate(model, val_loader, dev)
            lr = opt.param_groups[0]["lr"]
            if sched is not None:
                sched.step(val_loss) if t["scheduler"] == "plateau" else sched.step()

            rec = {"epoch": epoch + 1, "train_loss": loss_sum / seen, "val_loss": val_loss,
                   "val_acc": val_acc, "lr": lr, "samples_per_s": seen / elapsed,
                   "epoch_time_s": elapsed}
            history.append(rec)
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch + 1)
            print(f"epoch {epoch + 1}/{t['epochs']}  train_loss {rec['train_loss']:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  lr {lr:.5f}  "
                  f"{rec['samples_per_s']:.0f} samples/s")

            is_best = val_acc > best_val
            best_val = max(best_val, val_acc)
            ck = {"epoch": epoch + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict() if sched is not None else None,
                  "best_val_acc": best_val, "history": history}
            torch.save(ck, ckdir / "last.pt")
            if is_best:
                torch.save(ck, ckdir / "best.pt")

            if t["stop_after_epoch"] and epoch + 1 == t["stop_after_epoch"]:
                interrupted = True
                print("-- simulated interruption --")
                break

        result = {"run_name": cfg["run_name"], "epochs_done": len(history),
                  "interrupted": interrupted, "final_val_acc": history[-1]["val_acc"]}
        if not interrupted:
            test_loss, test_acc = evaluate(model, test_loader, dev)
            mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
            result.update(test_loss=test_loss, test_acc=test_acc)
        hist_path = ckdir / "history.json"
        hist_path.write_text(json.dumps(history, indent=2))
        mlflow.log_artifact(str(hist_path))
        if cfg["mlflow"]["log_checkpoint"]:
            mlflow.log_artifact(str(ckdir / "last.pt"))
    return result


if __name__ == "__main__":      # required on Windows when num_workers > 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.set)
    print(run(cfg))