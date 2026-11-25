| Model | Data: training | Model: params (nonzero) | Model: size dense / smallest format (MB) | Model: test acc | Compute: FLOPs/img effective / executed (M) | Compute: latency 1 thread b=1 / b=128 (ms) | Energy proxy: time-based (x baseline) | Energy proxy: FLOPs-based (x baseline) |
|---|---|---|---|---|---|---|---|---|
| Baseline (dense CNN) | 10 epochs | 980,490 (980,490) | 3.93 / 3.65 | 0.9225 | 21.40 / 21.40 | 0.925 / 91.6 | 1.00 | 1.00 |
| Unstructured, global 90% (+FT) | 10 + 3 epochs | 980,490 (98,721) | 3.93 / 0.74 | 0.9262 | 4.84 / 21.40 | 0.942 / 94.1 | 1.03 | 0.23 |
| Structured, 50% filters (+FT) | 10 + 3 epochs | 394,890 (394,890) | 1.58 / 1.47 | 0.9166 | 5.76 / 5.76 | 0.468 / 36.9 | 0.40 | 0.27 |
| Student, plain | 10 epochs | 97,802 (97,802) | 0.40 / 0.40 | 0.9092 | 4.58 / 4.58 | 0.435 / 36.8 | 0.40 | 0.21 |
| Student, distilled | 10 epochs + teacher forward pass | 97,802 (97,802) | 0.40 / 0.40 | 0.9094 | 4.58 / 4.58 | 0.435 / 36.8 | 0.40 | 0.21 |
