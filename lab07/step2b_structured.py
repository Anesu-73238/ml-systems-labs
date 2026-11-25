import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import (BaselineCNN, count_params, evaluate, fit, make_loaders, set_seed,
                    state_dict_bytes)
from models import CNN

FRACS = [0.10, 0.25, 0.50]
FT_EPOCHS, FT_LR = 3, 5e-4
CONV_IDX = [0, 3, 6, 9]                     # positions of the 4 conv layers in model.features
BASE_PARAMS = 980_490
OUT = Path("results/pruning_structured.json")


def filter_masks(model, frac):
    """In every conv layer, drop the fraction `frac` of filters with the smallest L2 norm."""
    keep = {}
    for i in CONV_IDX:
        w = model.features[i].weight.detach()                 # (out, in, 3, 3)
        norms = w.flatten(1).norm(dim=1)                      # L2 norm of each whole filter
        k = torch.ones(w.shape[0], dtype=torch.bool)
        k[torch.argsort(norms)[: int(round(frac * w.shape[0]))]] = False
        keep[i] = k
    return keep


@torch.no_grad()
def apply_filter_masks(model, keep):
    for i, k in keep.items():
        conv = model.features[i]
        conv.weight[~k] = 0.0
        conv.bias[~k] = 0.0


@torch.no_grad()
def shrink(model, keep):
    """Build the physically smaller network: drop the pruned filters, the matching input channels of
    the next layer, and the matching input columns of the first fully connected layer."""
    small = CNN(tuple(int(keep[i].sum()) for i in CONV_IDX), hidden=256)
    prev = torch.ones(1, dtype=torch.bool)                    # the single input channel
    for i in CONV_IDX:
        k = keep[i]
        small.features[i].weight.copy_(model.features[i].weight[k][:, prev])
        small.features[i].bias.copy_(model.features[i].bias[k])
        prev = k
    cols = prev.unsqueeze(1).expand(-1, 9).reshape(-1)        # 3x3 spatial positions per channel
    small.classifier[1].weight.copy_(model.classifier[1].weight[:, cols])
    small.classifier[1].bias.copy_(model.classifier[1].bias)
    small.classifier[3].load_state_dict(model.classifier[3].state_dict())
    return small


def run_one(frac, base_state):
    set_seed()
    train_loader, val_loader, test_loader = make_loaders()
    model = BaselineCNN()
    model.load_state_dict(base_state)
    keep = filter_masks(model, frac)
    apply_filter_masks(model, keep)
    acc_before = evaluate(model, test_loader)[1]              # one-shot, before fine-tuning
    fit(model, train_loader, val_loader, FT_EPOCHS, lr=FT_LR,
        after_step=lambda: apply_filter_masks(model, keep), tag=f"[filters {frac}] ")
    acc_after = evaluate(model, test_loader)[1]

    small = shrink(model, keep)
    model.eval()
    small.eval()
    with torch.no_grad():
        x = torch.randn(16, 1, 28, 28)
        diff = (model(x) - small(x)).abs().max().item()
    assert diff < 1e-4, f"shrunk network differs from the masked one: {diff}"
    Path("results/pruned").mkdir(parents=True, exist_ok=True)
    torch.save(small.state_dict(), f"results/pruned/structured_{frac}.pt")
    chans = [int(keep[i].sum()) for i in CONV_IDX]
    return {"filters_removed": frac, "channels": chans, "params": count_params(small),
            "params_removed_frac": 1 - count_params(small) / BASE_PARAMS,
            "size_bytes": state_dict_bytes(small), "test_acc_before_finetune": acc_before,
            "test_acc": acc_after, "test_acc_shrunk": evaluate(small, test_loader)[1],
            "max_output_diff_masked_vs_shrunk": diff}


if __name__ == "__main__":
    base_state = torch.load("results/baseline.pt")
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for frac in FRACS:
        key = f"filters_{frac}"
        if key in done:
            print(f"skipping {key} (already done)")
            continue
        done[key] = run_one(frac, base_state)
        OUT.write_text(json.dumps(done, indent=2))
        print(f"--> {key}: test acc {done[key]['test_acc']:.4f}\n", flush=True)

    print(f"\n{'filters removed':>15s} {'channels kept':>20s} {'params':>9s} {'removed':>8s} "
          f"{'size MB':>8s} {'acc before FT':>14s} {'acc after FT':>13s} {'shrunk net':>11s}")
    for r in done.values():
        print(f"{r['filters_removed']:15.2f} {str(r['channels']):>20s} {r['params']:9,d} "
              f"{r['params_removed_frac'] * 100:7.1f}% {r['size_bytes'] / 1e6:8.2f} "
              f"{r['test_acc_before_finetune']:14.4f} {r['test_acc']:13.4f} {r['test_acc_shrunk']:11.4f}")

    # ---- plot: accuracy against the fraction of parameters removed, for all three methods
    un = json.loads(Path("results/pruning_unstructured.json").read_text())
    dense = un["dense_0.0"]

    def unstructured(mode, key):
        rows = sorted((r for r in un.values() if r["mode"] == mode), key=lambda r: r["target_sparsity"])
        return ([0.0] + [1 - r["nonzero_params"] / BASE_PARAMS for r in rows],
                [dense[key]] + [r[key] for r in rows])

    st = sorted(done.values(), key=lambda r: r["filters_removed"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, key, title in [(ax[0], "test_acc", "after 3 epochs of fine-tuning"),
                          (ax[1], "test_acc_before_finetune", "one-shot, before fine-tuning")]:
        for mode, label, style in [("global", "unstructured, global", "o-"),
                                   ("layerwise", "unstructured, layer-wise", "s-")]:
            xs, ys = unstructured(mode, key)
            a.plot(xs, ys, style, label=label)
        a.plot([0.0] + [r["params_removed_frac"] for r in st], [dense[key]] + [r[key] for r in st],
               "^-", label="structured (filters, physically removed)")
        a.set_xlabel("fraction of parameters removed")
        a.set_ylabel("test accuracy")
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.savefig("results/sparsity_curves.png", dpi=150)
    print("\nsaved results/pruning_structured.json and results/sparsity_curves.png")