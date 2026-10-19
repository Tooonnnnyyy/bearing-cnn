# Experiment workflow

1. **Define the task.** One prediction is a 1,024-sample window from the first sensor column. Labels map to healthy, inner race and outer race. Row count is not the number of independent machines.
2. **Validate inputs.** Check names, column meaning, finite values, row count, sampling frequency and acquisition context. The current loader treats the first nonnumeric row as a header; review the file schema before running.
3. **Split recordings.** Use the fixed train/validation/test manifest in the script. Adjacent windows from one recording must not be shuffled across splits. Check physical-bearing identity separately when known.
4. **Segment.** `420000 // 1024 = 410`; discard 160 tail samples without overlap or zero padding. If the sampling rate is `fs`, window duration is `1024 / fs` seconds. The sampling rate is not established by the CSV row count.
5. **Standardize.** Fit a single mean and standard deviation to training samples. Apply the same constants to validation, test and future inference. This preserves relative amplitude differences; per-window normalization is a different experiment.
6. **Train.** Tensor shape is `(batch, 1, 1024)`. Two convolutions learn local patterns. Cross-entropy compares logits with class IDs; backpropagation computes gradients; Adam updates weights. An epoch visits all 1,230 training windows, using 39 batches (38 of 32 and one of 14).
7. **Select.** Validation has no gradient updates and dropout is disabled. Save the lowest validation-loss checkpoint. Patience 10 is a stopping rule, not a target accuracy. Training accuracy is measured while dropout is active and weights are changing.
8. **Evaluate once.** Restore the selected checkpoint and inspect the test confusion matrix. Each class percentage is row recall, not one-versus-rest accuracy. Reusing test feedback to tune makes it development evidence; a fresh test boundary is then needed.
9. **Keep evidence.** Version code, split, dependencies, seed, normalization, configuration, checkpoint and metrics. Only sanitized aggregate results belong in this public repository.
10. **Investigate before deployment.** Inspect the weak outer-race recording: acquisition metadata, amplitude/offset, saturation, labels and comparable training coverage. These are hypotheses, not established causes. Use training/development data for experiments and new independent recordings for final evaluation.

## Shapes through the CNN

| Operation | Output shape |
|---|---|
| Input | B × 1 × 1024 |
| Conv1d(1,32,9,padding=4), ReLU | B × 32 × 1024 |
| MaxPool1d(4) | B × 32 × 256 |
| Conv1d(32,64,5,padding=2), ReLU | B × 64 × 256 |
| MaxPool1d(4) | B × 64 × 64 |
| AdaptiveAvgPool1d(8), Flatten | B × 512 |
| Linear(512,64), ReLU, Dropout(0.3) | B × 64 |
| Linear(64,3) | B × 3 logits |

The model has two convolution layers and two dense layers. Softmax is applied only for reported probabilities; `CrossEntropyLoss` already handles logits. A high softmax score is not a guarantee of correct or calibrated confidence.

## What is implemented

Implemented and run: CSV loading, segmentation, scaling, training, validation selection, test inference and saved metrics. Tested with synthetic data: parsing and numerical invariants, gradient updates and the full CLI path. Not implemented: production inference service, streaming buffer, model monitoring, automated retraining, calibrated abstention or field deployment.

For production, the complete input and preprocessing contract must travel with the weights. A known bad outer-race result is a reason to keep the baseline in research, not a reason to silently change the test split.

Optional [experiment tools](experiments.md) add window-local preprocessing, FFT/PSD diagnostics, recording-level split manifests and compatible-checkpoint transfer learning. They use separate outputs and default to development-only evaluation. The original reference script remains unchanged.
