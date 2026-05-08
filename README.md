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
|-- experiments/
|   |-- run_experiment.py
|   |-- sweep.py
|   |-- plot_sweep.py
|   |-- inspect_data.py
|-- results/
|   |-- checkpoints/
|   |-- sweep_plots/
