# Lab 7: Compression: pruning and distillation

Machine: 13th Gen Intel Core i9-13900H, Windows 11 Pro, 14 cores / 20 threads; CPU only. PyTorch: 2.15.0.

Data: Fashion-MNIST, 55k train / 5k validation / 10k test (seed 42). Adam throughout. All pruning results are from a single seed;
the student comparison uses 3 seeds. Test accuracy is reported on the 10k test set. Latency is the median forward-pass time on CPU after warm-up.

## 1\. Oversized baseline (Step 1)

Four convolution layers (32, 64, 128, 256 filters) and a 256-unit fully connected layer, trained 10 epochs (Adam, lr 1e-3, batch 128).

|Layer|Parameters|
|-|-|
|conv1 (1 -> 32)|320|
|conv2 (32 -> 64)|18,496|
|conv3 (64 -> 128)|73,856|
|conv4 (128 -> 256)|295,168|
|fc1 (2,304 -> 256)|590,080|
|fc2 (256 -> 10)|2,570|
|**Total**|**980,490**|

Test accuracy **0.9225** (validation 0.9246), file size **3.93 MB** (float32 state dict). The validation loss was lowest at epoch 7 and rose slightly afterwards while the training loss kept falling (mild overfitting).
The latency measured in Step 1 (1.028 ms at batch 1, 33 ms at batch 128) was taken while DataLoader worker processes were still alive, which I believe inflated it; all latency comparisons below use the measurements of Step 3 and Step 4,
where the baseline measures about 0.93 ms and 91.6 ms with one thread and about 0.4 to 0.5 ms and 17 to 18 ms with all threads.

## 2\. Magnitude pruning (Step 2)

**Method.** Weights of the conv and linear layers (979,744 of the 980,490 parameters; biases are not pruned) are ranked by absolute value and the smallest fraction is masked.
*Global:* one ranking over all layers. *Layer-wise:* each layer is ranked separately, so every layer loses the same fraction.
Each sparsity level is pruned one-shot from the original baseline and then fine-tuned for 3 epochs (Adam, lr 5e-4, not tuned) with the mask re-applied after every optimizer step, which keeps pruned weights at exactly zero (checked: the sparsity never dropped below the target).
The sparsity-0 row is a **control**: the baseline fine-tuned for the same 3 epochs without pruning (0.9209), so each pruned row is compared with 0.9209, not with 0.9225.

|Mode|Target sparsity|Nonzero params|Test acc before fine-tuning|Test acc after fine-tuning|
|-|-|-|-|-|
|dense control|0.00|980,490|0.9225|0.9209|
|global|0.25|735,554|0.9228|0.9220|
|global|0.50|490,618|0.9230|0.9238|
|global|0.75|245,682|0.9170|0.9268|
|global|0.90|98,721|0.7045|0.9262|
|global|0.95|49,734|0.3058|0.9192|
|layer-wise|0.25|735,554|0.9155|0.9217|
|layer-wise|0.50|490,618|0.8071|0.9239|
|layer-wise|0.75|245,682|0.3172|0.9202|
|layer-wise|0.90|98,723|0.1271|0.8852|
|layer-wise|0.95|49,736|0.1101|0.8378|

**Structured pruning** removes whole filters by the L2 norm of each filter's weights, in each of the four convolution layers, followed by the same 3-epoch fine-tuning with the filters held at zero.
The surviving filters were then copied into a physically smaller network (fewer channels, a smaller `fc1` input); its outputs match the masked network (checked, difference below 1e-4), and its accuracy is identical.

|Filters removed|Channels kept|Parameters|Removed|File size (MB)|Acc before FT|Acc after FT|
|-|-|-|-|-|-|-|
|10%|29, 58, 115, 230|846,657|13.6%|3.39|0.9159|0.9221|
|25%|24, 48, 96, 192|663,498|32.3%|2.66|0.8749|0.9150|
|50%|16, 32, 64, 128|394,890|59.7%|1.58|0.5240|0.9166|

Sparsity curves (both panels: after fine-tuning and one-shot):

!\[Accuracy against sparsity](results/sparsity\_curves.png)

**Observations.**

* Global pruning loses almost nothing after fine-tuning, even at 95% sparsity (0.9192, 0.2 points under the control). The 0.9268 and 0.9262 values at 75% and 90% are about 50 images above the control on one seed, so I read them as "no loss", not as an improvement.
* Layer-wise pruning collapses much earlier: it is near chance (0.10) right after pruning at 75% to 95%, and after fine-tuning it ends 3.6 points (90%) and 8.3 points (95%) below the control.
The per-layer sparsity explains a large part of it: under global pruning, `fc1` (590k weights, 60% of the model) takes most of the pruning (94% at the 90% target) while the 288-weight first convolution stays the densest (19%).
Layer-wise pruning forces every layer to the same fraction, leaving the first convolution with only 29 of its 288 weights at 90%. I did not isolate this with a separate experiment.
* Fine-tuning does most of the recovery (for example global 90%: 0.7045 to 0.9262), and it uses only 3 epochs with an untuned learning rate.
* Per removed parameter, unstructured pruning is more accurate than structured (global 50%: 0.9238; structured, 59.7% removed: 0.9166), because it can choose individual weights. Whole-filter removal is much harsher one-shot (0.5240 at 50%) but recovers to within 0.4 points of the control.

## 3\. Latency and storage: honest analysis (Step 3)

### Whole-model latency, single thread (median of 3 interleaved rounds)

|Model|b=1 (ms)|b=128 (ms)|Speed-up at b=128|
|-|-|-|-|
|baseline|0.925|91.57|1.00x|
|global 50%|0.927|90.16|1.02x|
|global 90%|0.942|94.06|0.97x|
|global 95%|0.955|94.98|0.96x|
|structured 10%|0.839|85.97|1.07x|
|structured 25%|0.672|60.98|1.50x|
|structured 50%|0.468|36.95|2.48x|

With all threads at batch 128, the structured models were 1.14x, 1.61x and 2.73x faster than the baseline, and the unstructured models showed no speed-up (16.5 to 17.5 ms against 17.41 ms for the baseline); the full table is in the Storage section below.

**Unstructured sparsity gave no speed-up** (50% to 95% sparsity within about 4% of the baseline), because the pruned networks run through the same dense convolution and matrix-multiply kernels, which multiply every stored weight whether it is zero or not.
**Structured pruning did** (up to 2.48x single-threaded), because whole filters disappear and the network does less arithmetic. The measured speed-up is lower than the arithmetic ratio: by my hand count (2 x MACs) the baseline needs 21.40M FLOPs per image, the 50% network 5.76M (3.7x fewer) and the 25% network 12.34M (1.7x fewer).
A plausible reason is that smaller layers use the CPU less efficiently and fixed per-layer overheads remain; I did not test this.

### Storage: dense vs sparse format

Sizes of the same weights in three formats (dense PyTorch file, CSR sparse format, zlib-compressed dense tensors), and latencies with all threads (median of 3 interleaved rounds):

|Model|Nonzero|Dense .pt (MB)|CSR (MB)|zlib dense (MB)|b=1 (ms)|b=128 (ms)|
|-|-|-|-|-|-|-|
|baseline|980,490|3.93|7.85|3.65|0.531|17.41|
|global 0.25|735,554|3.93|5.89|3.02|0.709|16.83|
|global 0.5|490,618|3.93|3.93|2.21|0.470|17.23|
|global 0.75|245,682|3.93|1.97|1.33|0.495|16.65|
|global 0.9|98,721|3.93|0.80|0.74|0.681|17.47|
|global 0.95|49,734|3.93|0.40|0.52|0.495|17.03|
|layerwise 0.25|735,554|3.93|5.89|3.02|0.671|16.59|
|layerwise 0.5|490,618|3.93|3.93|2.23|0.497|17.31|
|layerwise 0.75|245,682|3.93|1.97|1.34|0.485|17.31|
|layerwise 0.9|98,723|3.93|0.80|0.75|0.724|16.61|
|layerwise 0.95|49,736|3.93|0.40|0.53|0.443|16.52|
|structured 0.1|846,657|3.39|6.78|3.15|0.655|15.29|
|structured 0.25|663,498|2.66|5.31|2.47|0.403|10.84|
|structured 0.5|394,890|1.58|3.17|1.47|0.515|6.37|

* **The dense file never shrinks for unstructured pruning:** it stays at 3.93 MB at every sparsity level, because the zeros are still stored.
* **CSR stores about 8 bytes per nonzero weight** (a 4-byte value and a 4-byte index) against 4 bytes per weight for dense, so it only pays off at high sparsity: the unpruned baseline in CSR is 7.85 MB (twice the dense size), CSR equals the dense size at 50% sparsity (3.93 MB),
is half of it at 75% (1.97 MB), and is 4.9x smaller at 90% (0.80 MB) and 9.8x smaller at 95% (0.40 MB).
* **Compressing the dense file competes with CSR:** zlib is smaller than CSR up to 90% sparsity (for example 2.21 against 3.93 MB at 50%, and 0.74 against 0.80 MB at 90%), while CSR wins at 95% (0.40 against 0.52 MB). zlib reduces only the file; the weights still have to be decompressed into dense tensors before running,
whereas a CSR matrix can be used directly by a sparse kernel.
* **The structured models shrink in the dense format** (3.39, 2.66 and 1.58 MB), while CSR makes them about twice as large (6.78, 5.31 and 3.17 MB), because they contain no zeros.
* **Batch-1 latencies with all threads are noisy** (0.40 to 0.72 ms with no relation to sparsity), so I base batch-1 claims on the single-thread table above. The batch-128 values are stable.

### Kernel micro-benchmark: does a sparse kernel help?

Fully connected layer `fc1` (2,304 -> 256), single thread, median time in microseconds, comparing `torch.mm` on the dense weights with sparse CSR matrix products (scipy and torch):

|Weights|Zeros in fc1|Batch|Dense|Best sparse|Sparse vs dense|
|-|-|-|-|-|-|
|baseline|0.00|1|40.9|376.1|9.2x slower|
|baseline|0.00|128|1661.1|5372.2|3.2x slower|
|global 50%|0.60|1|57.2|152.6|2.7x slower|
|global 50%|0.60|128|1677.1|2447.7|1.5x slower|
|global 90%|0.94|1|55.3|23.0|2.4x faster|
|global 90%|0.94|128|1668.5|296.6|5.6x faster|
|global 95%|0.97|1|66.3|14.3|4.6x faster|
|global 95%|0.97|128|1669.2|141.9|11.8x faster|
|layer-wise 90%|0.90|1|44.5|39.0|1.14x faster|
|layer-wise 90%|0.90|128|1654.1|529.3|3.1x faster|

The baseline rows come from a second run of the same script, so the dense batch-1 times differ between rows by tens of percent (the dense batch-1 time of the same global-50% layer was 57.2 microseconds in one run and 42.4 in another).
The batch-128 dense times are stable (about 1.65 to 1.68 ms in every row), and the conclusions below do not depend on the batch-1 noise.

* The dense time at batch 128 is the same at 0% and at 97% zeros: a dense kernel does not care about zeros, which is why the whole-model latency did not move.
* A sparse kernel only pays off at high sparsity: at 0% to 60% zeros it was 1.5x to 9.2x slower, and at 90% it was 1.14x (batch 1) to 3.1x (batch 128) faster. I measured only 0%, 60%, 90%, 94% and 97%, so the break-even point is only known to lie between 60% and 90%.
Which sparse kernel is fastest depends on the batch size (scipy at batch 1, torch at batch 128).
* A fast `fc1` does not make a fast model: `fc1` holds 60% of the weights but only 5.5% of the FLOPs (1.18M of 21.4M per image; my hand count), so even a 12x faster `fc1` barely changes the whole-model latency.
The convolution layers carry most of the compute, and as far as I know PyTorch has no sparse convolution on CPU (I did not benchmark a sparse convolution).

**Why unstructured sparsity rarely speeds up dense CPU kernels:** dense kernels are written for regular, contiguous memory access and use vectorized instructions and cache blocking; a zero weight is stored and multiplied like any other.
A sparse format has to store an index for each nonzero and read memory irregularly, which costs more per nonzero, so it wins only when almost everything is zero, and only in the layers where a sparse kernel exists.

## 4\. Distillation (Step 4)

Teacher: the baseline (980,490 parameters). Student: the same layout with fewer channels, (16, 32, 64, 64) filters and 64 hidden units, **97,802 parameters** (about 10x fewer, 4.58M FLOPs per image against 21.40M).
Loss: `alpha \* CE(student, labels) + (1 - alpha) \* T^2 \* KL(...)` with T = 4 and alpha = 0.5, so the soft term is weighted by T^2 (1 - alpha) = 8.
I use the standard form from Hinton et al., where the teacher's softened distribution is the target, `F.kl\_div(log\_softmax(student/T), softmax(teacher/T))`, i.e. KL(teacher || student); the lab's `KL(a || b)` notation is ambiguous on this point.
Both students: Adam (lr 1e-3), 10 epochs, same data split; 3 seeds each (the seed changes the student's initial weights and the data order). The learning rate, T and alpha were not tuned.

|Model|Params|Size (MB)|Test acc (mean +- std, 3 seeds)|Per seed|
|-|-|-|-|-|
|Teacher (baseline)|980,490|3.93|0.9225|n/a|
|Student, plain|97,802|0.40|0.9092 +- 0.0017|0.9074, 0.9107, 0.9095|
|Student, distilled|97,802|0.40|0.9094 +- 0.0032|0.9066, 0.9088, 0.9129|

Distilled minus plain, per seed: -0.0008, -0.0019, +0.0034 (mean +0.0002). **There is no detectable gain from distillation here**: the effect is smaller than the spread between seeds.
Possible reasons (not tested): the teacher is only 1.3 points better than the student, so there is little to transfer; the teacher overfit slightly (its validation loss rose after epoch 7); and T, alpha and the learning rate were not tuned.

|Latency (median)|Teacher b=1|Student b=1|Teacher b=128|Student b=128|
|-|-|-|-|-|
|All threads|0.435 ms|0.295 ms (1.5x)|18.11 ms|6.40 ms (2.8x)|
|Single thread|0.944 ms|0.435 ms (2.2x)|91.97 ms|36.78 ms (2.5x)|

The student needs 4.7x fewer FLOPs but is 2.2x to 2.8x faster, again below the arithmetic ratio. Compared with the teacher, the student is 9.8x smaller in file size and 1.3 points less accurate.

## 5\. Summary by efficiency pillar (Step 5)

Energy was **not measured**. I use two proxies: (a) *time-based*, the single-thread batch-128 latency relative to the baseline (this assumes similar power draw, which I did not check), and (b) *FLOPs-based*, the effective FLOPs (counting only nonzero weights) relative to the baseline.
FLOPs are my hand counts (2 x MACs). "Executed" FLOPs are what the dense kernels actually compute. "Smallest format" is the smallest of the dense file, CSR and the zlib-compressed dense file.

|Model|Data (training)|Model: params (nonzero)|Model: size, dense / smallest format (MB)|Model: test acc|Compute: FLOPs/img effective / executed (M)|Compute: latency, 1 thread, b=1 / b=128 (ms)|Energy proxy: time (x baseline)|Energy proxy: FLOPs (x baseline)|
|-|-|-|-|-|-|-|-|-|
|Baseline (dense CNN)|10 epochs|980,490 (980,490)|3.93 / 3.65|0.9225|21.40 / 21.40|0.925 / 91.6|1.00|1.00|
|Unstructured, global 90% (+ fine-tuning)|10 + 3 epochs|980,490 (98,721)|3.93 / 0.74|0.9262|4.84 / 21.40|0.942 / 94.1|1.03|0.23|
|Structured, 50% filters (+ fine-tuning)|10 + 3 epochs|394,890 (394,890)|1.58 / 1.47|0.9166|5.76 / 5.76|0.468 / 36.9|0.40|0.27|
|Student, plain|10 epochs|97,802 (97,802)|0.40 / 0.40|0.9092|4.58 / 4.58|0.435 / 36.8|0.40|0.21|
|Student, distilled|10 epochs + teacher forward pass each batch|97,802 (97,802)|0.40 / 0.40|0.9094|4.58 / 4.58|0.435 / 36.8|0.40|0.21|

(The dense fine-tuned control, with no pruning, scores 0.9209; the student rows are 3-seed means. For global 90% the smallest format is the zlib-compressed file, 0.74 MB; the CSR file is 0.80 MB.)

Notes on reading the table:

* The global-90% row shows the difference between the two energy proxies: effective FLOPs fall to 0.23 of the baseline, but time stays at 1.03 because the dense kernels execute all 21.40M FLOPs. It is the only row where "effective" and "executed" differ.
Effective FLOPs fell to 0.23 rather than to about 0.10 because the pruning is concentrated in `fc1`, which carries few FLOPs, while the convolution layers (most of the compute) keep many of their weights.
* The structured 50% network and the plain student have the same latency (36.9 ms and 36.8 ms at batch 128), but the structured one is 0.7 points more accurate (0.9166 against 0.9092, with a seed spread of about 0.2 points) and four times larger on disk (1.58 against 0.40 MB).
* Distillation changed the data/training pillar (about twice the cost per epoch because of the teacher forward pass) and gave no accuracy benefit here.

## 6\. Conclusion

**Speed:** Structured pruning and the student models provided real speed-ups (structured pruning at 50% of the filters was about 2.5x faster with one thread, and the student about 2.2x to 2.8x faster). Unstructured pruning mainly reduced storage size, with little to no runtime improvement,
and even the storage saving needs a sparse or compressed format and high sparsity (at 90% sparsity, 3.93 MB became 0.80 MB in CSR or 0.74 MB zlib-compressed; the dense file stays 3.93 MB).

**Accuracy:** Compared to the dense fine-tuned control (0.9209), global 90% pruning maintained accuracy (0.9262), structured pruning at 50% lost about 0.4 percentage points (0.9166), and the student models were about 1.2 points lower (0.9092 and 0.9094; about 1.3 points below the original baseline of 0.9225).
Distillation gave no detectable gain over the plain student (+0.0002 on average, smaller than the spread between seeds).

**Deployment choice:** If model size is the main constraint, choose the student model (0.40 MB). If accuracy is more important, choose the structured-pruned model (0.9166 accuracy, 1.58 MB), since it is more accurate (about 0.7 points) while keeping similar latency, at about four times the file size of the student.

**Limitations:** Results were obtained using a single random seed for the pruning experiments (three seeds for the students) on one laptop CPU. Key hyperparameters were not fully tuned, energy consumption was not measured directly (two proxies only), and sparse convolution kernels were not evaluated,
which may affect the generalizability of the findings.

