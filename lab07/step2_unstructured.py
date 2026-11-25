import json
from pathlib import Path

import torch

from common import BaselineCNN, evaluate, fit, make_loaders, set_seed

SPARSITIES = [0.25, 0.5, 0.75, 0.9, 0.95]
FT_EPOCHS, FT_LR = 3, 5e-4
OUT = Path("results/pruning_unstructured.json")


def prunable(model):
    """Weights of conv and linear layers (biases are not pruned)."""
    return [(n, p) for n, p in model.named_parameters() if p.dim() > 1]


def global_masks(model, s):
    """Rank ALL prunable weights together by |w|; the smallest fraction s gets masked."""
    ps = prunable(model)
    allw = torch.cat([p.detach().abs().flatten() for _, p in ps])
    thr = torch.kthvalue(allw, int(s * allw.numel())).values
    return {n: (p.detach().abs() > thr).float() for n, p in ps}


def layer_masks(model, s):
    """Rank the weights of each layer separately; every layer loses the same fraction s."""
    masks = {}
    for n, p in prunable(model):
        thr = torch.kthvalue(p.detach().abs().flatten(), int(s * p.numel())).values
        masks[n] = (p.detach().abs() > thr).float()
    return masks


@torch.no_grad()
def apply_masks(model, masks):
    for n, p in model.named_parameters():
        if n in masks:
            p.mul_(masks[n])


def sparsity_report(model):
    ps = prunable(model)
    zeros = sum(int((p == 0).sum()) for _, p in ps)
    total = sum(p.numel() for _, p in ps)
    return zeros / total, {n: float((p == 0).float().mean()) for n, p in ps}


def run_one(mode, s, base_state):
    set_seed()
    train_loader, val_loader, test_loader = make_loaders()
    model = BaselineCNN()
    model.load_state_dict(base_state)
    masks = None
    if mode != "dense":
        masks = (global_masks if mode == "global" else layer_masks)(model, s)
        apply_masks(model, masks)
    acc_pruned = evaluate(model, test_loader)[1]              # one-shot, before any fine-tuning
    hist = fit(model, train_loader, val_loader, FT_EPOCHS, lr=FT_LR,
               after_step=(lambda: apply_masks(model, masks)) if masks else None,
               tag=f"[{mode} {s}] ")                          # the mask is re-applied after every step
    sp, per_layer = sparsity_report(model)
    if masks:
              assert sp >= s - 1e-5, f"mask drifted: sparsity {sp:.6f} < target {s}"
    test_acc = evaluate(model, test_loader)[1]
    Path("results/pruned").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"results/pruned/{mode}_{s}.pt")
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return {"mode": mode, "target_sparsity": s, "achieved_sparsity": sp, "nonzero_params": nonzero,
            "per_layer_sparsity": per_layer, "test_acc_before_finetune": acc_pruned,
            "test_acc": test_acc, "val_acc": hist[-1]["val_acc"]}


if __name__ == "__main__":
    base_state = torch.load("results/baseline.pt")
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    # "dense" = the baseline fine-tuned for the same 3 epochs with no pruning (a control)
    plan = [("dense", 0.0)] + [(m, s) for m in ("global", "layerwise") for s in SPARSITIES]
    for mode, s in plan:
        key = f"{mode}_{s}"
        if key in done:
            print(f"skipping {key} (already done)")
            continue
        done[key] = run_one(mode, s, base_state)
        OUT.write_text(json.dumps(done, indent=2))          # saved after every run, so it can be resumed
        print(f"--> {key}: test acc {done[key]['test_acc']:.4f}\n", flush=True)

    print(f"\n{'mode':10s} {'target':>7s} {'achieved':>9s} {'nonzero':>9s} {'acc before FT':>14s} {'acc after FT':>13s}")
    for r in done.values():
        print(f"{r['mode']:10s} {r['target_sparsity']:7.2f} {r['achieved_sparsity']:9.3f} "
              f"{r['nonzero_params']:9,d} {r['test_acc_before_finetune']:14.4f} {r['test_acc']:13.4f}")
    print("\nper-layer sparsity (global pruning):")
    for r in done.values():
        if r["mode"] == "global":
            print(f"  target {r['target_sparsity']:.2f}: " +
                  "  ".join(f"{n.split('.')[-2] if False else n}={v:.2f}" for n, v in r["per_layer_sparsity"].items()))