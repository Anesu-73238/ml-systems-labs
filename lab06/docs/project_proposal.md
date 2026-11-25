# Individual Project Proposal: Quantized 1D-CNN for On-Device Human Activity Recognition

**Student:** ANESU CHIGWADA **Student ID:** 73238

## Problem

Always-on activity recognition (walking, sitting, lying and so on) on a phone or wearable has to run on the device itself, because sending raw
motion data to a server costs battery and exposes private behaviour. This project builds a small neural network that classifies the user's
activity from smartphone inertial-sensor windows, and then compresses it so that it fits typical on-device limits without losing accuracy.

## Dataset

**UCI "Human Activity Recognition Using Smartphones"** (Reyes-Ortiz, Anguita, Ghio, Oneto, Parra, 2012; DOI 10.24432/C54S4K; licence CC BY 4.0).

* 30 volunteers (ages 19-48) wore a waist-mounted Samsung Galaxy S II and performed six activities: walking, walking upstairs, walking
downstairs, sitting, standing and laying.
* 3-axis accelerometer and 3-axis gyroscope at 50 Hz, cut into 2.56 s windows (128 readings) with 50% overlap; 10,299 windows; download about 58 MB.
* **Split by subject, not by window.** The windows overlap and each volunteer has a characteristic movement style, so I will use the dataset's own
volunteer-based 70/30 train/test split, and hold out some of the training volunteers for validation. This avoids leakage between train and test.

## Success metric

* **Primary:** macro-F1 and accuracy on the held-out test volunteers.
* **Target:** the compressed model stays within 1 percentage point of the float32 model's accuracy while meeting both deployment limits below.
* **Reference points:** a logistic-regression baseline on the provided 561 hand-crafted features, and my own float32 1D-CNN before compression.

## Deployment constraints

|Constraint|Limit|How I will measure it|
|-|-|-|
|Latency|<= 50 ms per window|batch 1, CPU, median of 100 runs after warm-up (the same protocol as Lab 3)|
|Model size|<= 5 MB|size of the serialized model file on disk|

These are measured on a laptop CPU as a stand-in for a phone, and I will state that limitation in the report.

## Planned optimization

**Post-training int8 quantization** of a 1D-CNN (a few convolution layers on the 9 x 128 input, followed by a small classifier head).
If post-training quantization loses more than 1 point of accuracy, I will use quantization-aware training. I will report size, latency and accuracy
before and after. If time permits, I will add magnitude pruning as a second, smaller comparison. First I will check that the quantization
backend works on my machine, with ONNX Runtime quantization as the fallback.

## Plan

1. Load the data, build the subject-based split and reach the baselines (logistic regression on the 561 features).
2. Train the float32 1D-CNN with my Lab 6 pipeline (YAML config, seed, MLflow logging, checkpoints).
3. Quantize it and measure size, latency and accuracy for both versions.
4. Write the short report with a results table, plus a README with the one command that reproduces the experiment.

## Risks

* The dataset is small and varies between volunteers, so results could be noisy; I will report variation across seeds.
* Quantization may reduce accuracy more than 1 point on a very small network; the fallback is quantization-aware training.
* Latency on a laptop CPU is only a proxy for a phone processor.

