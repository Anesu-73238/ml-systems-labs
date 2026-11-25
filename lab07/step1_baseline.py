import json
from pathlib import Path

import torch

from common import (BaselineCNN, count_params, evaluate, fit, latency_ms, make_loaders,
                    set_seed)

EPOCHS = 10

if __name__ == "__main__":          # required on Windows because the DataLoader uses worker processes
    set_seed()
    train_loader, val_loader, test_loader = make_loaders()
    model = BaselineCNN()
    print(f"baseline parameters: {count_params(model):,}\n")
    hist = fit(model, train_loader, val_loader, EPOCHS)
    test_loss, test_acc = evaluate(model, test_loader)

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/baseline.pt")
    size_bytes = Path("results/baseline.pt").stat().st_size
    lat1, lat128 = latency_ms(model, 1), latency_ms(model, 128)

    res = {"params": count_params(model), "size_bytes": size_bytes, "test_acc": test_acc,
           "val_acc": hist[-1]["val_acc"], "latency_b1_ms": lat1, "latency_b128_ms": lat128,
           "epochs": EPOCHS, "history": hist}
    Path("results/baseline.json").write_text(json.dumps(res, indent=2))
    print(f"\nBaseline: test acc {test_acc:.4f}, params {res['params']:,}, "
          f"size {size_bytes / 1e6:.2f} MB, latency batch 1 {lat1:.3f} ms, batch 128 {lat128:.2f} ms")
    print("saved results/baseline.pt and results/baseline.json")