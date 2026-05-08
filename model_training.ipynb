"""
run_experiment.py

Single-run training script for the corrected EEGNet-style DE-feature pipeline.

Recommended protocol for paper results:
  - Use leave_one_out=True.
  - Run once for each held-out subject.
  - Report mean ± std, macro-F1, and worst-subject performance.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataloader import build_loaders
from losses import ClassificationLoss, ContrastiveLoss, PrototypeLoss, SEPCLoss
from model import build_model, Trainer


# ----------------------------
# Configuration
# ----------------------------
DATA_ROOT = "/fs/vulcan-projects/fsh_track/jason-bhargav-temp/CMSC472-Final/data"
DATASET = "SEED-IV"
VAL_SUBJECT = 1
NB_CLASSES = 4
CLASS_NAMES = ["Neutral", "Sad", "Fear", "Happy"]

CHANS = 62
SAMPLES = 5  # DE frequency bands, not raw time points.

LOSS_MODE = "ce"  # one of: "ce", "supcon", "prototype", "sepc"

EPOCHS = 50
N_PER_CLASS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_CON = 0.5
TEMPERATURE = 0.1
WARMUP_EPOCHS = 0 if LOSS_MODE == "ce" else 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
RUN_NAME = f"{DATASET}_valsubj={VAL_SUBJECT}_loss={LOSS_MODE}"
SAVE_DIR = Path("checkpoints") / RUN_NAME


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_contrastive_loss(mode: str):
    if mode == "ce":
        return None
    if mode == "supcon":
        return ContrastiveLoss(temperature=TEMPERATURE)
    if mode == "prototype":
        return PrototypeLoss(num_classes=NB_CLASSES, temperature=TEMPERATURE)
    if mode == "sepc":
        return SEPCLoss(num_classes=NB_CLASSES, temperature=TEMPERATURE)
    raise ValueError(f"Unknown LOSS_MODE: {mode}")


def save_training_curves(history: dict, save_dir: Path) -> None:
    epochs = range(1, len(history["train"]) + 1)

    def values(split: str, key: str):
        return [row.get(key, np.nan) for row in history[split]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    specs = [
        ("loss", "Total Loss", "Loss"),
        ("cls_loss", "Classification Loss", "Loss"),
        ("con_loss", "Contrastive Loss", "Loss"),
        ("acc", "Accuracy", "Accuracy"),
    ]

    for ax, (key, title, ylabel) in zip(axes, specs):
        ax.plot(epochs, values("train", key), label="Train")
        if history["val"]:
            ax.plot(epochs, values("val", key), label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_dir / "plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_eval_outputs(model, loader, save_dir: Path) -> None:
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    for batch in loader:
        x = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        logits, _ = model(x)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    with open(save_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ))

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax,
        values_format=".2f",
        colorbar=True,
    )
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": DATASET,
        "val_subject": VAL_SUBJECT,
        "nb_classes": NB_CLASSES,
        "chans": CHANS,
        "samples": SAMPLES,
        "loss_mode": LOSS_MODE,
        "epochs": EPOCHS,
        "n_per_class": N_PER_CLASS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "lambda_con": LAMBDA_CON,
        "temperature": TEMPERATURE,
        "warmup_epochs": WARMUP_EPOCHS,
        "seed": SEED,
        "device": DEVICE,
    }
    with open(SAVE_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_loader, val_loader = build_loaders(
        data_root=DATA_ROOT,
        val_subject=VAL_SUBJECT,
        dataset=DATASET,
        n_per_class=N_PER_CLASS,
        leave_one_out=True,
        augment_train=True,
        seed=SEED,
    )

    model = build_model(nb_classes=NB_CLASSES, Chans=CHANS, Samples=SAMPLES)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        cls_loss_fn=ClassificationLoss(),
        con_loss_fn=build_contrastive_loss(LOSS_MODE),
        optimizer=optimizer,
        lambda_con=LAMBDA_CON,
        warmup_epochs=WARMUP_EPOCHS,
        device=DEVICE,
    )

    history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        SAVE_DIR / "model.pt",
    )
    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    save_training_curves(history, SAVE_DIR)
    save_eval_outputs(model, val_loader, SAVE_DIR)

    best_val = max(row["acc"] for row in history["val"]) if history["val"] else None
    print(f"\nSaved run to: {SAVE_DIR}")
    if best_val is not None:
        print(f"Best validation accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()
