"""
sweep.py

Grid sweep for the corrected EEGNet-style DE-feature pipeline.

This version matches the corrected loss/model code:
  - CE-only, SupCon, Prototype, and SEPC are all supported.
  - Prototype and SEPC use per-subject-class EMA prototypes.
  - CE-only runs do not use contrastive warmup or lambda_con.
  - Resumed runs preserve the global epoch count, so warmup is not restarted.
  - The default evaluation is held-out-subject validation.

Note: EMA prototype state is *not* checkpointed. If a run resumes from a
partial checkpoint, prototypes will re-warm over the first few batches.
This has negligible effect on final results.

Usage:
    python sweep.py
    python sweep.py --start-idx 41
    python sweep.py --val-subject 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
import traceback
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
# Fixed config
# ----------------------------
NB_CLASSES = 4
N_SUBJECTS = 15
CHANS = 62
SAMPLES = 5  # DE frequency bands, not raw time points.
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_ROOT = Path("checkpoints")
CLASS_NAMES = ["Neutral", "Sad", "Fear", "Happy"]

DATA_ROOT = "/fs/vulcan-projects/fsh_track/jason-bhargav-temp/CMSC472-Final/data"
DATASET = "SEED-IV"
N_PER_CLASS = 8
SEED = 42
EMA_ALPHA = 0.9


# ----------------------------
# Sweep grid
# ----------------------------
SWEEP = {
    "loss_type": ["ce", "supcon", "prototype", "sepc"],
    "lambda_con": [0.1, 0.5, 1.0],
    "warmup_epochs": [0, 5, 10],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4],
    "temperature": [0.07, 0.1, 0.2],
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loss(loss_type: str, temperature: float):
    if loss_type == "ce":
        return None
    if loss_type == "supcon":
        return ContrastiveLoss(temperature=temperature)
    if loss_type == "prototype":
        return PrototypeLoss(
            num_classes=NB_CLASSES,
            num_subjects=N_SUBJECTS,
            temperature=temperature,
            ema_alpha=EMA_ALPHA,
        )
    if loss_type == "sepc":
        return SEPCLoss(
            num_classes=NB_CLASSES,
            num_subjects=N_SUBJECTS,
            temperature=temperature,
            ema_alpha=EMA_ALPHA,
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def build_configs() -> list[dict]:
    configs = []

    # CE-only should not be swept over meaningless contrastive parameters.
    for lr, wd in itertools.product(SWEEP["lr"], SWEEP["weight_decay"]):
        configs.append({
            "loss_type": "ce",
            "lambda_con": 0.0,
            "warmup_epochs": 0,
            "lr": lr,
            "weight_decay": wd,
            "temperature": None,
        })

    for loss_type, lam, wu, lr, wd, temp in itertools.product(
        ["supcon", "prototype", "sepc"],
        SWEEP["lambda_con"],
        SWEEP["warmup_epochs"],
        SWEEP["lr"],
        SWEEP["weight_decay"],
        SWEEP["temperature"],
    ):
        configs.append({
            "loss_type": loss_type,
            "lambda_con": lam,
            "warmup_epochs": wu,
            "lr": lr,
            "weight_decay": wd,
            "temperature": temp,
        })

    return configs


def run_name_for(cfg: dict, val_subject: int) -> str:
    temp = cfg["temperature"] if cfg["temperature"] is not None else "na"
    return (
        f"subj={val_subject:02d}"
        f"_loss={cfg['loss_type']}"
        f"_lam={cfg['lambda_con']}"
        f"_wu={cfg['warmup_epochs']}"
        f"_lr={cfg['lr']}"
        f"_wd={cfg['weight_decay']}"
        f"_temp={temp}"
    )


def last_metric(history: dict, split: str, key: str):
    try:
        return history[split][-1][key]
    except (KeyError, IndexError):
        return None


def make_summary_row(cfg: dict, name: str, history: dict, elapsed: float | None = None) -> dict:
    val_accs = [e["acc"] for e in history.get("val", [])]
    return {
        "run_name": name,
        "loss_type": cfg["loss_type"],
        "lambda_con": cfg["lambda_con"],
        "warmup_epochs": cfg["warmup_epochs"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "temperature": cfg.get("temperature"),
        "final_train_acc": round(last_metric(history, "train", "acc") or 0.0, 4),
        "final_val_acc": round(last_metric(history, "val", "acc") or 0.0, 4),
        "best_val_acc": round(max(val_accs) if val_accs else 0.0, 4),
        "final_val_loss": round(last_metric(history, "val", "loss") or 0.0, 4),
        "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
    }


def save_checkpoint(save_dir: Path, model, optimizer, history, filename: str = "model.pt") -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        save_dir / filename,
    )


def save_training_curves(save_dir: Path, history: dict) -> None:
    epochs = range(1, len(history["train"]) + 1)

    def vals(split: str, key: str):
        return [e.get(key, float("nan")) for e in history[split]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    specs = [
        ("loss", "Total Loss", "Loss"),
        ("cls_loss", "Classification Loss", "Loss"),
        ("con_loss", "Contrastive Loss", "Loss"),
        ("acc", "Accuracy", "Accuracy"),
    ]

    for ax, (key, title, ylabel) in zip(axes, specs):
        ax.plot(epochs, vals("train", key), label="Train", linewidth=1.5)
        ax.plot(epochs, vals("val", key), label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_dir / "plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_eval_outputs(save_dir: Path, model, loader) -> None:
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

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax,
        values_format=".2f",
        colorbar=True,
    )
    ax.set_title("Normalized Confusion Matrix (Val)")
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report_text)


def load_partial(save_dir: Path, model, optimizer) -> tuple[dict | None, int]:
    model_path = save_dir / "model_partial.pt"
    hist_path = save_dir / "history_partial.json"

    if not (model_path.exists() and hist_path.exists()):
        return None, 0

    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    with open(hist_path) as f:
        history = json.load(f)

    epochs_done = len(history.get("train", []))
    print(f"    Resuming from epoch {epochs_done}/{EPOCHS}")
    return history, epochs_done


def train_with_epoch_checkpoints(
    trainer: Trainer,
    model,
    optimizer,
    train_loader,
    val_loader,
    save_dir: Path,
    total_epochs: int,
    existing_history: dict | None = None,
    start_epoch: int = 0,
) -> dict:
    history = existing_history or {"train": [], "val": []}

    for epoch_done in range(start_epoch, total_epochs):
        one_epoch = trainer.fit(
            train_loader,
            val_loader,
            epochs=1,
            start_epoch=epoch_done,
        )
        history["train"].extend(one_epoch["train"])
        history["val"].extend(one_epoch["val"])

        save_checkpoint(save_dir, model, optimizer, history, "model_partial.pt")
        with open(save_dir / "history_partial.json", "w") as f:
            json.dump(history, f)

    return history


def write_summary_csv(summary_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    import csv
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(start_idx: int = 1, val_subject: int = 1) -> list[dict]:
    set_seed(SEED)

    print("Building data loaders ...")
    train_loader, val_loader = build_loaders(
        data_root=DATA_ROOT,
        val_subject=val_subject,
        dataset=DATASET,
        n_per_class=N_PER_CLASS,
        leave_one_out=True,
        augment_train=True,
        seed=SEED,
        n_subjects=N_SUBJECTS,
    )
    print("Data loaders ready.\n")

    configs = build_configs()
    print(f"Total runs planned: {len(configs)}")
    print(f"Starting from index: {start_idx}\n")

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = CHECKPOINT_ROOT / f"sweep_summary_subj={val_subject:02d}.csv"

    summary_rows = []
    if summary_path.exists():
        import csv
        with open(summary_path, newline="") as f:
            summary_rows = list(csv.DictReader(f))

    completed_names = {r["run_name"] for r in summary_rows}
    completed, failed = 0, 0

    for idx, cfg in enumerate(configs, start=1):
        if idx < start_idx:
            continue

        name = run_name_for(cfg, val_subject)
        save_dir = CHECKPOINT_ROOT / name
        hist_path = save_dir / "history.json"

        if hist_path.exists():
            print(f"[{idx}/{len(configs)}] SKIP complete: {name}")
            if name not in completed_names:
                with open(hist_path) as f:
                    history = json.load(f)
                summary_rows.append(make_summary_row(cfg, name, history))
            completed += 1
            continue

        print(f"\n[{idx}/{len(configs)}] START: {name}")
        t0 = time.time()

        try:
            save_dir.mkdir(parents=True, exist_ok=True)

            model = build_model(nb_classes=NB_CLASSES, Chans=CHANS, Samples=SAMPLES)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
            con_loss = make_loss(cfg["loss_type"], cfg["temperature"] or 0.1)

            existing_history, epochs_done = load_partial(save_dir, model, optimizer)

            trainer = Trainer(
                model=model,
                cls_loss_fn=ClassificationLoss(),
                con_loss_fn=con_loss,
                optimizer=optimizer,
                lambda_con=cfg["lambda_con"],
                warmup_epochs=cfg["warmup_epochs"],
                device=DEVICE,
            )

            history = train_with_epoch_checkpoints(
                trainer=trainer,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=save_dir,
                total_epochs=EPOCHS,
                existing_history=existing_history,
                start_epoch=epochs_done,
            )

            elapsed = time.time() - t0

            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)

            with open(save_dir / "config.json", "w") as f:
                json.dump({**cfg, "val_subject": val_subject}, f, indent=2)

            save_checkpoint(save_dir, model, optimizer, history, "model.pt")
            save_training_curves(save_dir, history)
            save_eval_outputs(save_dir, model, val_loader)

            for partial in ["model_partial.pt", "history_partial.json"]:
                p = save_dir / partial
                if p.exists():
                    p.unlink()

            row = make_summary_row(cfg, name, history, elapsed)
            summary_rows = [r for r in summary_rows if r.get("run_name") != name]
            summary_rows.append(row)
            write_summary_csv(summary_path, summary_rows)

            completed += 1
            print(f"  done in {elapsed:.0f}s | best val acc={row['best_val_acc']:.4f}")

        except Exception:
            failed += 1
            print(f"  FAILED:\n{traceback.format_exc()}")
            write_summary_csv(summary_path, summary_rows)

    print(f"\nSweep complete: {completed} ok, {failed} failed.")
    print(f"Summary CSV: {summary_path}")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--val-subject", type=int, default=1)
    args = parser.parse_args()
    run_sweep(start_idx=args.start_idx, val_subject=args.val_subject)
