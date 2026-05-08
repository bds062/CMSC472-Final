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





