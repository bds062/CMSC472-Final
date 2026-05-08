# EEGNet-Style Cross-Subject EEG Emotion Recognition on SEED-IV

This repository contains a PyTorch implementation of an EEGNet-style model for SEED-IV emotion recognition using pre-extracted differential entropy (DE) features from the `eeg_feature_smooth` directory.

The code supports four training objectives:

1. Cross-entropy classification
2. Supervised contrastive learning
3. Prototype contrastive learning
4. Subject-Excluded Prototype Contrastive learning (SEPC)

The main experimental setting is cross-subject emotion recognition, where one subject is held out for validation/test and the model is trained on the remaining subjects.

---

## Repository Structure

```text
.
├── dataloader.py        # Loads SEED/SEED-IV DE features and builds DataLoaders
├── model.py             # EEGNet-style spatial-spectral model and Trainer
├── losses.py            # CE, SupCon, Prototype, and SEPC losses
├── run_experiment.py    # Single-run training script
├── sweep.py             # Hyperparameter sweep script
├── plot_sweep.py        # Sweep visualization script
├── inspect_data.py      # Dataset shape sanity-check script
├── requirements.txt     # Minimal Python dependencies
└── README.md
```

---

## Data Format

This code assumes the SEED-IV pre-extracted feature files are stored under:

```text
data/eeg_feature_smooth/
  1/
    1_20160518.mat
    2_20150915.mat
    ...
  2/
  3/
```

Each `.mat` file contains keys such as:

```text
de_LDS1, de_LDS2, ..., de_LDS24
```

Each trial array has shape:

```text
(62, T, 5)
```

where:

```text
62 = EEG electrodes/channels
T  = precomputed windows within the trial
5  = frequency bands: delta, theta, alpha, beta, gamma
```

A single model input sample is extracted as:

```python
arr[:, t, :]
```

with shape:

```text
(62, 5)
```

The dataloader then adds the Conv2d channel dimension, giving:

```text
(1, 62, 5)
```

So the model should be built with:

```python
build_model(nb_classes=4, Chans=62, Samples=5)
```

Here, `Samples=5` means frequency bands, not raw temporal samples.

---

## Installation

Recommended Python version:

```text
Python 3.10+
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version using the official PyTorch installation selector. The `requirements.txt` file does not pin a CUDA-specific wheel.

---

## Inspecting the Dataset

Before training, you can verify the `.mat` file structure with:

```bash
python inspect_data.py
```

Or specify a file manually:

```bash
python inspect_data.py \
  --mat-path /path/to/data/eeg_feature_smooth/1/1_20160518.mat
```

This script prints the available feature keys, their shapes, and confirms that the correct sample extraction is:

```python
arr[:, t, :]
```

not:

```python
arr[:, :, 0]
```

The latter is one frequency band across all windows, not one training sample.

---

## Single Experiment

Run a single experiment with:

```bash
python run_experiment.py
```

Inside `run_experiment.py`, select the loss mode:

```python
LOSS_MODE = "ce"
LOSS_MODE = "supcon"
LOSS_MODE = "prototype"
LOSS_MODE = "sepc"
```

The default recommended evaluation setting is held-out-subject validation:

```python
leave_one_out = True
```

This avoids random window-level leakage between train and validation sets.

Outputs are saved under:

```text
checkpoints/<run_name>/
```

Typical saved files include:

```text
model.pt
history.json
config.json
plots.png
confusion_matrix.png
classification_report.txt
```

---

## Hyperparameter Sweep

Run the full sweep with:

```bash
python sweep.py
```

Resume from a later run index with:

```bash
python sweep.py --start-idx 41
```

The sweep compares:

```text
ce
supcon
prototype
sepc
```

and saves each run under:

```text
checkpoints/<run_name>/
```

A summary CSV is written to:

```text
checkpoints/sweep_summary.csv
```

---

## Plotting Sweep Results

After running the sweep, generate summary plots with:

```bash
python plot_sweep.py checkpoints/sweep_summary.csv
```

To specify an output directory:

```bash
python plot_sweep.py checkpoints/sweep_summary.csv --outdir sweep_plots
```

The plotting script produces figures comparing loss types, learning rates, contrastive weights, warmup epochs, temperatures, and top-performing runs.

---

## Loss Functions

The repository defines four losses in `losses.py`.

### 1. Cross-Entropy Loss

```python
ClassificationLoss()
```

Standard multiclass classification loss over emotion labels.

### 2. Supervised Contrastive Loss

```python
ContrastiveLoss(temperature=0.1)
```

Uses class labels to pull together embeddings from the same emotion class and push apart embeddings from different classes.

### 3. Prototype Loss

```python
PrototypeLoss(num_classes=4, temperature=0.1)
```

Builds one class prototype per emotion class from the current batch and trains samples to align with their class prototype.

### 4. Subject-Excluded Prototype Contrastive Loss

```python
SEPCLoss(num_classes=4, temperature=0.1)
```

For each anchor sample, SEPC builds class prototypes using only samples from different subjects. For an anchor from subject `s_i`, the class prototype for class `c` is computed from samples satisfying:

```text
label = c and subject_id != s_i
```

This is intended to encourage emotion-relevant representations that are less dependent on subject-specific structure.

---

## Evaluation Protocol

For cross-subject EEG emotion recognition, the recommended protocol is:

1. Hold out one subject.
2. Train on the remaining subjects.
3. Evaluate on the held-out subject.
4. Repeat for all subjects.
5. Report mean accuracy, standard deviation, macro-F1, and worst-subject accuracy.

The default single-run setup holds out one subject. For publication-quality results, run leave-one-subject-out evaluation across all subjects.

---

## Important Implementation Notes

The code uses pre-extracted DE features, not raw EEG time series. Therefore, the model is best described as an EEGNet-style spatial-spectral network, not a raw temporal EEGNet.

The input shape is:

```text
(B, 1, 62, 5)
```

where:

```text
B  = batch size
1  = Conv2d input channel
62 = EEG electrodes
5  = frequency bands
```

The trainer preserves `subject_ids` from the dataloader so that SEPC can correctly exclude same-subject samples when constructing prototypes.

Train/validation normalization is fit only on the training set and then applied to validation data.

---

## Reproducibility

The training scripts set random seeds for Python, NumPy, and PyTorch. However, exact reproducibility can still depend on GPU hardware, CUDA/cuDNN behavior, PyTorch version, and dataloader worker settings.

For more deterministic runs, use:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings are included in the experiment scripts.

---

## Minimal Example

```python
from dataloader import build_loaders
from model import build_model, Trainer
from losses import ClassificationLoss, SEPCLoss
import torch

train_loader, val_loader = build_loaders(
    data_root="/path/to/data",
    dataset="SEED-IV",
    val_subject=1,
    leave_one_out=True,
    n_per_class=8,
)

model = build_model(
    nb_classes=4,
    Chans=62,
    Samples=5,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

trainer = Trainer(
    model=model,
    cls_loss_fn=ClassificationLoss(),
    con_loss_fn=SEPCLoss(num_classes=4, temperature=0.1),
    optimizer=optimizer,
    lambda_con=0.5,
    warmup_epochs=5,
    device="cuda",
)

history = trainer.fit(train_loader, val_loader, epochs=50)
```

In practice, use `run_experiment.py` or `sweep.py`, which handle optimizer creation, checkpointing, plotting, and saving results.

---

## Notes for Reviewers

This implementation uses pre-extracted SEED-IV DE/LDS features rather than raw EEG waveforms. The width dimension of the model input is therefore the five frequency bands, not time.

The purpose of `inspect_data.py` is to make this data interpretation explicit and reproducible. It is not part of the training pipeline.

The SEPC loss requires subject IDs. The dataloader returns subject IDs, and the trainer passes them to `SEPCLoss` so that prototypes can be constructed from subjects other than the anchor subject.
