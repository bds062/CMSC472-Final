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


## Project Structure

```text
project/
|-- README.md
|-- requirements.txt
|-- data/
|-- src/
|   |-- dataloader.py
|   |-- losses.py
|   |-- model.py
|-- experiments/
|   |-- run_experiment.py
|   |-- sweep.py
|   |-- plot_sweep.py
|   |-- inspect_data.py
|-- results/
|   |-- checkpoints/
|   |-- sweep_plots/
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


