# Lab 6: Training pipelines and project proposal

Machine: TODO (CPU model, OS; 20 logical cores, CPU only). PyTorch: TODO (`pip show torch`). MLflow 3.18.0.
Model: the CNN from Lab 3 (20,490 parameters, 89.5% test accuracy in Lab 3). Data: Fashion-MNIST, 55k train / 5k validation / 10k test, seed 42.
Unless a section says otherwise: Adam, lr 0.001, batch size 128. Every number below comes from a single seed and a laptop CPU, so
differences of a few tenths of a point (or a few percent in timing) should not be over-interpreted.

## 1. YAML-driven training pipeline with MLflow (Step 1)

`train.py` is driven by one YAML object (`config.yaml`): `python train.py` runs the baseline, and `--set key=value` overrides any setting,
for example `python train.py --set run_name=w2 data.num_workers=2`, so every study below is the same script with different settings.

- **Seed:** `seed` sets Python, NumPy and PyTorch seeds, seeds the train/validation split and the DataLoader shuffle generator, and seeds the workers.
- **Checkpoints:** at the end of every epoch `checkpoints/<run_name>/last.pt` is saved (model, optimizer and scheduler state, epoch, history),
  and `best.pt` whenever the validation accuracy improves.
- **MLflow** (local SQLite store `mlflow.db`, experiment `lab6-fashion-mnist`):
  - *params:* the whole flattened config plus `n_params`
  - *metrics per epoch:* `train_loss`, `val_loss`, `val_acc`, `lr`, `samples_per_s`, `epoch_time_s`; and `test_loss`, `test_acc` at the end
  - *artifacts:* `config_used.yaml`, `history.json`, `last.pt`

Baseline run (5 epochs, `num_workers` 0): validation accuracy 0.8974, test accuracy 0.8925, about 3,300 samples/s.
Screenshots of the run page (parameters, metric charts, artifacts) are in `screenshots/`. TODO: list the file names.
All the studies below appear as separate MLflow runs: `baseline`, `bs32` ... `bs512`, `sched-*`, `accum-*`, `resume-*`.

## 2. Profile of one epoch (Step 2)

`profile_epoch.py` runs one epoch (430 batches, batch 128, `num_workers` 0) after a short warm-up, labelling each phase with
`torch.profiler.record_function` and timing it with `perf_counter`. Data loading includes waiting for the next batch and copying it to the device;
forward includes the loss; the optimizer step includes `zero_grad`.

| Phase | perf_counter (s) | Share | torch.profiler (s) | Share |
|---|---|---|---|---|
| Data loading | 10.543 | **66.4%** | 15.721 | 73.4% |
| Forward | 2.572 | 16.2% | 2.689 | 12.6% |
| Backward | 2.371 | 14.9% | 2.471 | 11.5% |
| Optimizer step | 0.382 | 2.4% | 0.525 | 2.5% |

The epoch took 15.87 s without the profiler and 21.45 s with it (1.35x overhead), so I use the `perf_counter` shares as the headline numbers; the
profiler run inflates data loading because it records every small operation in the per-sample transforms. The top profiler entry is
`enumerate(DataLoader)` (73% of CPU time); the largest model operators are `convolution_backward` (8.2%) and `max_pool2d_with_indices` (6.8%).
Per batch, loading costs about 24.5 ms against about 12 ms for forward + backward + optimizer step, so with no workers the pipeline is
**input-bound**: the main process applies `ToTensor` and `Normalize` to every image in Python and only then runs the model, with no overlap.

## 3. num_workers study (Step 3)

Median of 3 epochs after 1 warm-up epoch, batch 128 (`study_workers.py`, plot: `results/workers.png`).

| Workers | Epoch (s) | Samples/s | Data wait (s) | Compute (s) | Data share | Speed-up |
|---|---|---|---|---|---|---|
| 0 | 15.82 | 3,476 | 10.53 | 5.29 | 67% | 1.00x |
| 1 | 17.95 | 3,065 | 11.48 | 6.47 | 64% | 0.88x |
| 2 | 9.13 | 6,023 | 3.24 | 5.89 | 35% | 1.73x |
| 4 | 8.57 | 6,415 | 0.63 | 7.94 | 7% | 1.85x |
| 8 | 8.71 | 6,314 | 0.68 | 8.03 | 8% | 1.82x |

- **Bottleneck:** data loading when there are 0 or 1 workers (about two thirds of the epoch). One worker is slower than none (0.88x): it does the same per-sample
  work as the main process, and every batch also has to be sent between processes.
- **At 4 workers the bottleneck moves to compute:** the main process waits only 0.6 s, so more workers (8) cannot help. I use 4 workers from here on.
- **The ideal speed-up was not reached:** hiding loading completely would give an epoch of about 5.3 s (3x), but I measured 1.85x, because compute time itself rose from 5.3 s
  to about 8 s once workers were running. A likely reason is competition for CPU cores and memory bandwidth, which I did not test.

## 4. Batch-size study (Step 4)

5 epochs, 4 workers, same learning rate (0.001) for every batch size (`study_batch.py`, plot: `results/batch_size.png`).
Peak memory is the peak resident set size (RSS) of the training process, which includes the PyTorch baseline.

| Batch | Updates/epoch | Samples/s | Epoch (s) | Peak memory (MB) | Final val acc | Test acc |
|---|---|---|---|---|---|---|
| 32 | 1,719 | 3,057 | 17.99 | 515 | 0.9066 | 0.9044 |
| 64 | 860 | 4,504 | 12.21 | 521 | 0.8952 | 0.8911 |
| 128 | 430 | 6,062 | 9.07 | 530 | 0.8978 | 0.8907 |
| 256 | 215 | 6,239 | 8.82 | 565 | 0.8902 | 0.8884 |
| 512 | 108 | 7,154 | 7.69 | 639 | 0.8792 | 0.8716 |

- **Throughput** rises 2.3x from batch 32 to 512 with diminishing returns (+35% from 64 to 128, +3% from 128 to 256).
- **Memory** grows only 24% (515 to 639 MB) because most of it is the fixed PyTorch and Python baseline.
- **Accuracy falls with batch size** (0.9044 to 0.8716 test). Because the learning rate was kept fixed, larger batches also make fewer updates in 5 epochs
  (1,719 vs 108); I think this is the main cause, but I did not test it (for example by scaling the learning rate). Batches 64 and 128 differ by only 0.0004
  in test accuracy, which is within noise.
- Batch 128 gives about 85% of the maximum throughput at 0.8907 test accuracy; batch 32 is the most accurate but about half as fast.

## 5. Learning-rate schedules and optimizers (Step 5)

10 epochs each, batch 128, 4 workers (`study_sched.py`, plot: `results/schedulers.png`). The learning rates (Adam 0.001, SGD 0.01) and the step and plateau
settings (StepLR: halve every 3 epochs; ReduceLROnPlateau: halve after 1 epoch without validation-loss improvement) are reasonable defaults and were **not tuned**.

| Run | Train loss | Val loss | Final val acc | Best val acc (epoch) | Test acc | LR in last epoch |
|---|---|---|---|---|---|---|
| Adam, constant | 0.2096 | 0.2602 | 0.9028 | 0.9048 (8) | 0.8999 | 0.00100 |
| Adam + StepLR | 0.2200 | 0.2639 | 0.9052 | 0.9052 (10) | 0.9002 | 0.00013 |
| Adam + CosineAnnealingLR | 0.2139 | 0.2604 | 0.9072 | 0.9076 (9) | 0.9021 | 0.00002 |
| Adam + ReduceLROnPlateau | 0.2096 | 0.2602 | 0.9028 | 0.9048 (8) | 0.8999 | 0.00100 |
| SGD (momentum 0.9), constant | 0.2251 | 0.2679 | 0.9012 | 0.9024 (9) | 0.9008 | 0.01000 |

- **ReduceLROnPlateau never triggered:** its row is identical to constant Adam and its learning rate stayed at 0.001, because the validation loss kept
  improving often enough for patience 1 never to run out.
- The StepLR and cosine final learning rates match their formulas (0.001 x 0.5^3 = 0.000125; cosine decays to about 0.00003).
- Test accuracy spans only 0.8999 to 0.9021 (22 images out of 10,000), so with one seed I treat the schedules as similar. Cosine is nominally best on both validation and test.
- SGD with momentum matches Adam on test accuracy but has a higher final training loss (0.2251 vs 0.2096): it fits more slowly under the same 10-epoch budget.

## 6. Gradient accumulation (Step 6)

Emulating batch 256 with micro-batches of 64 and 4 accumulation steps (the loss is divided by 4 and the optimizer steps every 4th micro-batch), 10 epochs
(`study_accum.py`, plot: `results/accumulation.png`).

| Run | Updates/epoch | Samples/s | Peak memory (MB) | Final train loss | Final val acc | Test acc |
|---|---|---|---|---|---|---|
| Batch 256 (reference) | 215 | 6,304 | 566 | 0.2356 | 0.9036 | 0.8970 |
| Batch 64 x accumulation 4 | 215 | 4,900 | 522 | 0.2358 | 0.9020 | 0.8971 |
| Batch 64, no accumulation | 860 | 4,436 | 522 | 0.1846 | 0.9076 | 0.9060 |

Largest difference from the reference over 10 epochs: accumulation 0.00019 in train loss and 0.0028 in validation accuracy; plain batch 64 0.142 and 0.0236.
Accumulation reproduces the large-batch training behaviour almost exactly (the CNN has no batch-norm or dropout, and the data order is the same), at 22% lower throughput
(more, smaller forward/backward calls) and a slightly lower peak memory (only 64 samples' activations exist at a time). Plain batch 64 makes 4x more updates and so learns faster under the same learning rate.

## 7. Interrupt and resume (Step 7)

6 epochs with a cosine schedule. Run A is uninterrupted; run B is stopped after epoch 3 (`train.stop_after_epoch=3`) and then resumed from `checkpoints/resume-part1/last.pt`,
which restores the model, the Adam state, the scheduler state and the history (`study_resume.py`, plot: `results/resume.png`).

| Epoch | LR | Train loss (A) | Train loss (B) | Val acc (A) | Val acc (B) |
|---|---|---|---|---|---|
| 1 | 0.001000 | 0.5295 | 0.5295 | 0.8484 | 0.8484 |
| 2 | 0.000933 | 0.3486 | 0.3486 | 0.8774 | 0.8774 |
| 3 | 0.000750 | 0.3041 | 0.3041 | 0.8886 | 0.8886 |
| 4 (resumed) | 0.000500 | 0.2802 | 0.2794 | 0.8944 | 0.8884 |
| 5 | 0.000250 | 0.2612 | 0.2613 | 0.8962 | 0.8974 |
| 6 | 0.000067 | 0.2501 | 0.2499 | 0.8982 | 0.8988 |

The history of epochs 1 to 3 carried over unchanged, and the learning-rate schedule is identical to the uninterrupted run, so the scheduler resumed from the right place.
The training-loss change from epoch 3 to 4 is -0.0239 uninterrupted and -0.0247 resumed. The largest differences are 0.0008 in train loss and 0.0060 in validation accuracy
(30 images out of 5,000, at epoch 4, recovering by epoch 5). Final test accuracy: 0.8943 uninterrupted, 0.8942 resumed.
The curves continue smoothly, but the resumed run is not bitwise identical, because the checkpoint does not store the data-shuffling random state:
the DataLoader's generator starts again from the seed in the new process, so I believe the first epoch after resuming sees the same sample order as the original epoch 1 (not verified).
Saving and restoring the generator state would make the resume exact.

## 8. Individual project proposal (Step 8)

`docs/project_proposal.md`: quantized 1D-CNN for on-device human activity recognition on the UCI HAR dataset (latency <= 50 ms, size <= 5 MB, int8 quantization).
Approved by the instructor. TODO: add the date or the way it was approved if you want to record it.

## 9. Conclusions

TODO: write these in your own words. Points that the measurements support:
- With 0 workers, data loading, not the model, is the bottleneck of this small CNN (66% of the epoch); 4 workers remove most of it, after which compute dominates.
- Larger batches raise throughput with diminishing returns but, at a fixed learning rate, lower accuracy in a fixed number of epochs.
- The schedules and optimizers were within noise of each other under equal budgets; gradient accumulation matches the large batch; resuming from a checkpoint continues the curves smoothly.
- Limitations: one seed, one laptop CPU, no learning-rate tuning, RSS as the memory measure, and `torch.cuda` was not used.
