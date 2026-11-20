import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from torchinfo import summary
from thop import profile

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

SEED = 42
BATCH = 128
torch.manual_seed(SEED)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

full_train = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test_set = datasets.FashionMNIST("data", train=False, download=True, transform=tf)

train_set, val_set = random_split(
    full_train, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH)
test_loader = DataLoader(test_set, batch_size=BATCH)

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class AttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(49, 64)                    # 7x7 patch -> 64
        self.pos = nn.Parameter(torch.zeros(1, 16, 64))   # learned positions
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        b = x.shape[0]
        # 28x28 -> 16 patches of 7x7 -> (b, 16, 49)
        x = x.reshape(b, 1, 4, 7, 4, 7).permute(0, 2, 4, 1, 3, 5).reshape(b, 16, 49)
        x = self.embed(x) + self.pos
        x = self.enc(x).mean(dim=1)                       # mean pool over patches
        return self.head(x)


# ---------- Step 3a: parameters + FLOPs ----------
def n_params(m):
    return sum(p.numel() for p in m.parameters())


def hand_flops():
    """FLOPs per single image, 2 x MACs."""
    seq, d, ff = 16, 64, 128
    mlp = {
        "Linear 784->256": 2 * 784 * 256,
        "Linear 256->128": 2 * 256 * 128,
        "Linear 128->10": 2 * 128 * 10,
    }
    cnn = {
        "Conv1": 2 * 3 * 3 * 1 * 16 * 28 * 28,
        "Conv2": 2 * 3 * 3 * 16 * 32 * 14 * 14,
        "Linear 1568->10": 2 * 1568 * 10,
    }
    layer = {
        "QKV proj": 2 * seq * d * 3 * d,
        "scores QK^T": 2 * seq**2 * d,
        "attn x V": 2 * seq**2 * d,
        "out proj": 2 * seq * d * d,
        "FFN": 2 * seq * d * ff + 2 * seq * ff * d,
    }
    attn = {
        "patch embed": 2 * seq * 49 * d,
        "2 encoder layers": 2 * sum(layer.values()),
        "head": 2 * 64 * 10,
    }
    return {"MLP": mlp, "CNN": cnn, "AttnNet": attn}, layer


hand, attn_layer = hand_flops()
models = {"MLP": MLP(), "CNN": CNN(), "AttnNet": AttnNet()}

for name, model in models.items():
    print(f"\n===== {name} =====")
    summary(model, input_size=(1, 1, 28, 28))        # torchinfo (screenshot this)
    h = sum(hand[name].values())
    macs, _ = profile(model.eval(), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    print(f"params (manual count): {n_params(model):,}")
    print(f"hand FLOPs : {h:,}")
    print(f"thop FLOPs : {2 * macs:,.0f}   (thop MACs x 2)")

print("\nAttention, one encoder layer:", attn_layer, "=", f"{sum(attn_layer.values()):,}")
# ---------- Step 3b: latency ----------
import statistics
import time


@torch.no_grad()
def latency_ms(model, bs, warmup=20, runs=100):
    """Median of `runs` timed forward passes after `warmup` untimed ones (CPU)."""
    model = model.cpu().eval()
    x = torch.randn(bs, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


lat = {}
print("\nLatency (ms, median of 100 runs after 20 warm-up)")
print(f"{'model':10s} {'batch 1':>10s} {'batch 128':>12s}")
for name, model in models.items():
    lat[name] = (latency_ms(model, 1), latency_ms(model, 128))
    print(f"{name:10s} {lat[name][0]:10.3f} {lat[name][1]:12.3f}")
    # ---------- Step 4: training ----------
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, LR = 5, 1e-3


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(model):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep}/{EPOCHS}  val acc {accuracy(model, val_loader):.4f}")
    return accuracy(model, test_loader)


results = {}
for name, cls in {"MLP": MLP, "CNN": CNN, "AttnNet": AttnNet}.items():
    print(f"\n== training {name} on {DEVICE} ==")
    torch.manual_seed(SEED)
    model = cls()
    test_acc = train(model)
    macs, _ = profile(cls().eval(), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    results[name] = {
        "params": n_params(model),
        "hand": sum(hand[name].values()),
        "thop": 2 * macs,
        "lat1": lat[name][0],
        "lat128": lat[name][1],
        "acc": test_acc,
    }

print(f"\n{'model':8s} {'params':>9s} {'hand FLOPs':>11s} {'thop FLOPs':>11s} {'b=1 ms':>8s} {'b=128 ms':>9s} {'test acc':>9s}")
for n, r in results.items():
    print(f"{n:8s} {r['params']:9,d} {r['hand']:11,d} {r['thop']:11,.0f} {r['lat1']:8.3f} {r['lat128']:9.3f} {r['acc']:9.4f}")

   # ---------- Step 5: plots ----------
names = list(results)

# accuracy vs parameters
plt.figure(figsize=(5, 4))
for n in names:
    plt.scatter(results[n]["params"], results[n]["acc"] * 100, s=90)
    plt.annotate(n, (results[n]["params"], results[n]["acc"] * 100),
                 textcoords="offset points", xytext=(6, 6))
plt.xscale("log")
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.xticks([2e4, 5e4, 1e5, 2e5], ["20k", "50k", "100k", "200k"])
plt.xlabel("Parameters (log scale)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs parameters")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("acc_vs_params.png", dpi=150)
plt.close()

# accuracy vs latency (both batch sizes)
plt.figure(figsize=(5.5, 4))
for key, marker, label in [("lat1", "o", "batch 1"), ("lat128", "s", "batch 128")]:
    for n in names:
        plt.scatter(results[n][key], results[n]["acc"] * 100, marker=marker, s=90,
                    color="tab:blue" if key == "lat1" else "tab:orange",
                    label=label if n == names[0] else None)
        plt.annotate(n, (results[n][key], results[n]["acc"] * 100),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.xscale("log")
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.xticks([0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
plt.xlabel("Latency per forward pass (ms, log scale)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs latency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("acc_vs_latency.png", dpi=150)
plt.close()
print("saved acc_vs_params.png and acc_vs_latency.png") 