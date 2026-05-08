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


## Data Setup

This project uses the SEED-IV pre-extracted EEG differential entropy features from the `eeg_feature_smooth` folder.

Place the data in the following structure:

```text
data/
|-- eeg_feature_smooth/
|   |-- 1/
|   |-- 2/
|   |-- 3/
```

Each session folder should contain the `.mat` files for all subjects.

The dataloader expects each `.mat` file to contain `de_LDS` feature arrays with shape:

```text
(62, T, 5)
```

where:

- `62` = EEG channels
- `T` = number of 4-second feature windows
- `5` = frequency bands: delta, theta, alpha, beta, gamma

Each window is converted into one model input of shape:

```text
(62, 5)
```



