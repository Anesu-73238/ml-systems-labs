import io
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loaders(batch_size=128, num_workers=4, root="data"):
    """Fashion-MNIST, 55k train / 5k validation (seeded) / 10k test."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    train, val = random_split(full, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              persistent_workers=num_workers > 0, worker_init_fn=seed_worker)
    g = torch.Generator().manual_seed(SEED)
    return (DataLoader(train, shuffle=True, generator=g, **kw),
            DataLoader(val, **kw), DataLoader(test, **kw))


class BaselineCNN(nn.Module):
    """Oversized baseline: 980,490 parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14 -> 7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 7 -> 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),                   # 3 x 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 3 * 3, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model, nonzero=False):
    if nonzero:
        return sum(int((p != 0).sum()) for p in model.parameters())
    return sum(p.numel() for p in model.parameters())


def state_dict_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, correct / n


def fit(model, train_loader, val_loader, epochs, lr=1e-3, after_step=None, tag=""):
    """Adam training. after_step() runs after every optimizer step (used later to keep pruning masks fixed)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if after_step is not None:
                after_step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
        val_loss, val_acc = evaluate(model, val_loader)
        rec = {"epoch": ep, "train_loss": loss_sum / n, "val_loss": val_loss, "val_acc": val_acc,
               "epoch_time_s": time.perf_counter() - t0}
        hist.append(rec)
        print(f"{tag}epoch {ep}/{epochs}  train_loss {rec['train_loss']:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  ({rec['epoch_time_s']:.0f} s)", flush=True)
    return hist


@torch.no_grad()
def latency_ms(model, batch, warmup=20, runs=100):
    """Median forward-pass time in ms on CPU, after warm-up (same protocol as Lab 3)."""
    model = model.cpu().eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)