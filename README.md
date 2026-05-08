# EEGNet-Style Cross-Subject EEG Emotion Recognition on SEED-IV

This repository contains a PyTorch implementation of an EEGNet-style model for SEED-IV emotion recognition using pre-extracted differential entropy (DE) features from the eeg_feature_smooth directory.

The code supports four training objectives:

Cross-entropy classification
Supervised contrastive learning
Prototype contrastive learning
Subject-Excluded Prototype Contrastive learning (SEPC)
The main experimental setting is cross-subject emotion recognition, where one subject is held out for validation/test and the model is trained on the remaining subjects.


## Repository Structure

```text
project/
|-- README.md
|-- requirements.txt
|-- data/
|-- src/
|   |-- dataloader.py
|   |-- losses.py
|   |-- model.py
|   |-- debug_mat.py
|-- experiments/
|   |-- sweep.py
|   |-- plot_sweep.py
|   |-- model_training.jpynb
|-- results/
|   |-- checkpoints/
|   |-- sweep_plots/
|-- notebook/
|   |-- data_processing.jpynb
```


## Environment Setup

We recommend using Python 3.10 or later.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Data Format

This project uses the SEED-IV pre-extracted EEG differential entropy features from the `eeg_feature_smooth` folder.

```text
data/eeg_feature_smooth/
|-- 1/
|   |-- 1_20160518.mat
|   |-- 2_20150915.mat
|   |-- ...
|-- 2/
|-- 3/
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

- `62` = EEG electrodes/channels
- `T` = precomputed windows within the trial
- `5` = frequency bands: delta, theta, alpha, beta, gamma

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

### 2. Run a single training experiment

Cross-entropy baseline:

```bash
python experiments/run_experiment.py --loss_type ce --epochs 100
```

Supervised contrastive learning:

```bash
python experiments/run_experiment.py --loss_type contrastive --epochs 100 --temperature 0.1 --lambda_contrastive 0.5
```

Global prototype contrast:

```bash
python experiments/run_experiment.py --loss_type prototype --epochs 100 --temperature 0.1 --lambda_contrastive 0.5
```

Leave-one-out / SEPC-style contrastive run:

```bash
python experiments/run_experiment.py --loss_type leave_one_out --epochs 100 --temperature 0.1 --lambda_contrastive 0.5
```

Outputs are saved under:

```text
results/checkpoints/
```

Each run saves training curves, validation metrics, and confusion matrices.

## Reproducing Main Results

All commands should be run from the project root.

### 1. Inspect the dataset

```bash
python experiments/inspect_data.py
```

This checks that the SEED-IV files are in the expected location and verifies the feature shapes.

### 3. Run the hyperparameter sweep

```bash
python experiments/sweep.py
```

This runs the sweep over loss types and hyperparameters used for the main comparison in the report.

### 4. Generate sweep plots

```bash
python experiments/plot_sweep.py
```

This generates the plots used in the report, including loss-type comparisons, learning-rate effects, and top-run summaries.

Plots are saved under:

```text
results/sweep_plots/
```

## Expected Runtime and Hardware

Training time varies by loss function and sweep size. On our setup, a single 100-epoch training run takes approximately 10 minutes. Cross-entropy runs are usually the fastest, while contrastive, prototype, and leave-one-out objectives can take longer because they compute additional embedding similarities or prototype-based losses.

Approximate runtime:

```text
Single 100-epoch run: ~10 minutes
Small hyperparameter sweep: varies by number of configurations
Full sweep: number of runs x ~10 minutes per run
```

Recommended hardware:

```text
GPU recommended
Tested with a single GPU environment
CPU training is possible but slower
```

## Notes

anything else to add like how u processed the data etc
