"""
sweep.py
--------
Runs a grid sweep over loss types, lambda_con, temperature, lr, etc.
For each run saves:
  checkpoints/<run_name>/
    ├── model.pt               # final model + optimizer state + history
    ├── model_partial.pt       # overwritten after every epoch (for resume)
    ├── history.json           # full history (only written on completion)
    ├── history_partial.json   # overwritten after every epoch (for resume)
    ├── config.json
    ├── plots.png
    └── confusion_matrix.png

Resuming
--------
  - A *completed* run (has history.json)      → skipped entirely.
  - A *partial*   run (has history_partial.json but no history.json)
      → model weights + history reloaded; training continues from the
        last completed epoch.
  - Runs that never started                   → run normally.

Usage:
    python sweep.py                 # full sweep, auto-resume
    python sweep.py --start-idx 41  # skip to run 41 (1-based), then auto-resume
"""

import os
import sys
import json
import argparse
import itertools
import time
import csv
import traceback

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataloader import build_loaders
from model import build_model, Trainer
from losses import ClassificationLoss, ContrastiveLoss, ContrastivePrototype


# ──────────────────────────────────────────────
# 1.  FIXED CONFIG
# ──────────────────────────────────────────────
NB_CLASSES      = 4
CHANS           = 62
SAMPLES         = 5           # frequency bands, NOT raw timepoints
EPOCHS          = 50
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_ROOT = "checkpoints"
CLASS_NAMES     = ["Neutral", "Sad", "Fear", "Happy"]

DATA_ROOT    = "/fs/vulcan-projects/fsh_track/jason-bhargav-temp/CMSC472-Final/data"
VAL_SUBJECT  = 1
DATASET      = "SEED-IV"
WINDOW_SEC   = 1.0
SFREQ        = 200
N_PER_CLASS  = 8


# ──────────────────────────────────────────────
# 2.  SWEEP GRID
# ──────────────────────────────────────────────
SWEEP = {
    "loss_types": [
        ("none",        lambda _:   None),
        ("contrastive", lambda cfg: ContrastiveLoss(temperature=cfg["temperature"])),
        ("prototype",   lambda _:   ContrastivePrototype(num_classes=NB_CLASSES)),
    ],
    "lambda_con":    [0.1, 0.5, 1.0],
    "warmup_epochs": [5, 10],
    "lr":            [1e-3, 5e-4],
    "weight_decay":  [1e-4],
    "temperature":   [0.07, 0.1, 0.2],   # ContrastiveLoss only
}


# ──────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────

def run_name_for(cfg: dict) -> str:
    temp = cfg["temperature"] if cfg["temperature"] is not None else "na"
    return (
        f"loss={cfg['loss_type']}"
        f"_lam={cfg['lambda_con']}"
        f"_wu={cfg['warmup_epochs']}"
        f"_lr={cfg['lr']}"
        f"_wd={cfg['weight_decay']}"
        f"_temp={temp}"
    )


def build_configs() -> list[dict]:
    configs = []
    for (loss_label, con_factory), lam, wu, lr, wd in itertools.product(
        SWEEP["loss_types"],
        SWEEP["lambda_con"],
        SWEEP["warmup_epochs"],
        SWEEP["lr"],
        SWEEP["weight_decay"],
    ):
        if loss_label == "contrastive":
            for temp in SWEEP["temperature"]:
                configs.append(dict(
                    loss_type=loss_label, con_factory=con_factory,
                    lambda_con=lam, warmup_epochs=wu,
                    lr=lr, weight_decay=wd, temperature=temp,
                ))
        else:
            configs.append(dict(
                loss_type=loss_label, con_factory=con_factory,
                lambda_con=lam, warmup_epochs=wu,
                lr=lr, weight_decay=wd, temperature=None,
            ))
    return configs


def last_metric(history, split, key):
    try:
        return history[split][-1][key]
    except (KeyError, IndexError):
        return None


def _make_summary_row(cfg, name, history, elapsed=None):
    return {
        "run_name":        name,
        "loss_type":       cfg["loss_type"],
        "lambda_con":      cfg["lambda_con"],
        "warmup_epochs":   cfg["warmup_epochs"],
        "lr":              cfg["lr"],
        "weight_decay":    cfg["weight_decay"],
        "temperature":     cfg.get("temperature"),
        "final_train_acc": round(last_metric(history, "train", "acc") or 0, 4),
        "final_val_acc":   round(last_metric(history, "val",   "acc") or 0, 4),
        "best_val_acc":    round(max((e["acc"] for e in history["val"]), default=0), 4),
        "final_val_loss":  round(last_metric(history, "val",   "loss") or 0, 4),
        "elapsed_s":       round(elapsed, 1) if elapsed else None,
    }


# ──────────────────────────────────────────────
# 4.  SAVE helpers
# ──────────────────────────────────────────────

def _save_checkpoint(save_dir, model, optimizer, history, filename="model.pt"):
    path = os.path.join(save_dir, filename)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history":              history,
    }, path)
    return path


def save_confusion_matrix(save_dir, model, loader):
    model.to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            logits, _ = model(x)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=True
    )
    ax.set_title("Normalised Confusion Matrix (val)")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    confusion matrix → {path}")


def save_training_curves(save_dir, history, warmup_epochs):
    start  = warmup_epochs
    total  = len(history["train"])
    epochs = range(start + 1, total + 1)

    def sl(split, key):
        return [e.get(key, float("nan")) for e in history[split]][start:]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    for ax, key, title, ylabel in zip(
        axes,
        ["loss",       "cls_loss",             "con_loss",         "acc"],
        ["Total Loss", "Classification Loss",  "Contrastive Loss", "Accuracy"],
        ["Loss",       "Loss",                 "Loss",             "Accuracy"],
    ):
        ax.plot(epochs, sl("train", key), label="Train", linewidth=1.5)
        ax.plot(epochs, sl("val",   key), label="Val",   linewidth=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    training curves  → {path}")


# ──────────────────────────────────────────────
# 5.  EPOCH-LEVEL TRAINING LOOP
#     Runs one epoch at a time so we can checkpoint after each one.
#     Accumulates history across calls by running trainer.fit(epochs=1)
#     on the same trainer instance (model/optimizer state is preserved).
#
#     NOTE: If Trainer.fit() resets its internal epoch counter each call
#     (affecting warmup), pass already_done_warmup=True and the Trainer
#     will see epoch numbers 1…remaining — warmup will re-apply for those
#     few epochs, which is a minor artefact but harmless.
# ──────────────────────────────────────────────

def train_with_epoch_checkpoints(
    trainer,
    model,
    optimizer,
    train_loader,
    val_loader,
    save_dir,
    total_epochs,
    existing_history=None,
    start_epoch=0,          # number of epochs already done
):
    """
    Train epoch-by-epoch, saving a partial checkpoint after every epoch.
    Returns the full accumulated history dict.
    """
    history = existing_history or {"train": [], "val": []}
    remaining = total_epochs - start_epoch

    if remaining <= 0:
        print(f"    Nothing left to train (already done {start_epoch}/{total_epochs}).")
        return history

    if start_epoch > 0:
        print(f"    Resuming from epoch {start_epoch + 1}/{total_epochs} …")

    for ep in range(remaining):
        # Train exactly one epoch
        ep_history = trainer.fit(train_loader, val_loader, epochs=1)

        # Merge into accumulated history
        history["train"].extend(ep_history["train"])
        history["val"].extend(ep_history["val"])

        # ── partial checkpoint (overwrite each epoch) ──
        _save_checkpoint(save_dir, model, optimizer, history,
                         filename="model_partial.pt")
        partial_hist = os.path.join(save_dir, "history_partial.json")
        with open(partial_hist, "w") as f:
            json.dump(history, f)

    return history


# ──────────────────────────────────────────────
# 6.  LOAD PARTIAL CHECKPOINT
# ──────────────────────────────────────────────

def load_partial(save_dir, model, optimizer):
    """
    If a partial checkpoint exists, load weights into model/optimizer
    and return (partial_history, epochs_done).
    Returns (None, 0) if no partial checkpoint found.
    """
    partial_model = os.path.join(save_dir, "model_partial.pt")
    partial_hist  = os.path.join(save_dir, "history_partial.json")

    if not (os.path.exists(partial_model) and os.path.exists(partial_hist)):
        return None, 0

    print(f"    Found partial checkpoint in {save_dir}")
    ckpt = torch.load(partial_model, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    with open(partial_hist) as f:
        history = json.load(f)

    epochs_done = len(history["train"])
    print(f"    Resuming from epoch {epochs_done}/{EPOCHS}")
    return history, epochs_done


# ──────────────────────────────────────────────
# 7.  MAIN SWEEP LOOP
# ──────────────────────────────────────────────

def run_sweep(start_idx: int = 1):
    print("Building data loaders …")
    train_loader, val_loader = build_loaders(
        data_root   = DATA_ROOT,
        val_subject = VAL_SUBJECT,
        dataset     = DATASET,
        window_sec  = WINDOW_SEC,
        sfreq       = SFREQ,
        n_per_class = N_PER_CLASS,
    )
    print("Data loaders ready.\n")

    configs = build_configs()
    print(f"Total runs planned : {len(configs)}")
    print(f"Starting from index: {start_idx}\n")

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    summary_path = os.path.join(CHECKPOINT_ROOT, "sweep_summary.csv")

    # Pre-load any existing summary rows so we don't lose them
    summary_rows = []
    if os.path.exists(summary_path):
        import csv as _csv
        with open(summary_path, newline="") as f:
            summary_rows = list(_csv.DictReader(f))

    existing_names = {r["run_name"] for r in summary_rows}
    completed, failed = 0, 0

    for i, cfg in enumerate(configs, 1):
        if i < start_idx:
            continue                                   # honour --start-idx

        name      = run_name_for(cfg)
        save_dir  = os.path.join(CHECKPOINT_ROOT, name)
        hist_path = os.path.join(save_dir, "history.json")

        # ── (a) fully completed run ──
        if os.path.exists(hist_path):
            print(f"[{i}/{len(configs)}] SKIP (complete): {name}")
            if name not in existing_names:
                with open(hist_path) as f:
                    history = json.load(f)
                summary_rows.append(_make_summary_row(cfg, name, history))
            completed += 1
            continue

        # ── (b) partial or fresh run ──
        print(f"\n[{i}/{len(configs)}] START: {name}")
        t0 = time.time()

        try:
            os.makedirs(save_dir, exist_ok=True)

            model     = build_model(nb_classes=NB_CLASSES, Chans=CHANS, Samples=SAMPLES)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
            con_loss  = cfg["con_factory"](cfg)

            # Try to resume from a partial checkpoint
            existing_history, epochs_done = load_partial(save_dir, model, optimizer)

            trainer = Trainer(
                model,
                ClassificationLoss(),
                con_loss,
                optimizer,
                lambda_con    = cfg["lambda_con"],
                warmup_epochs = cfg["warmup_epochs"],
                device        = DEVICE,
            )

            history = train_with_epoch_checkpoints(
                trainer, model, optimizer,
                train_loader, val_loader,
                save_dir,
                total_epochs     = EPOCHS,
                existing_history = existing_history,
                start_epoch      = epochs_done,
            )

            elapsed = time.time() - t0

            # ── finalise: write permanent artefacts ──
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)

            cfg_out = {k: v for k, v in cfg.items() if k != "con_factory"}
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(cfg_out, f, indent=2)

            _save_checkpoint(save_dir, model, optimizer, history, "model.pt")
            save_training_curves(save_dir, history, cfg["warmup_epochs"])
            save_confusion_matrix(save_dir, model, val_loader)

            # Clean up partial files now that we're done
            for partial in ("model_partial.pt", "history_partial.json"):
                p = os.path.join(save_dir, partial)
                if os.path.exists(p):
                    os.remove(p)

            row = _make_summary_row(cfg, name, history, elapsed)
            # Replace any old partial row with completed one
            summary_rows = [r for r in summary_rows if r.get("run_name") != name]
            summary_rows.append(row)
            completed += 1

            best_val = max(e["acc"] for e in history["val"])
            print(f"  → done in {elapsed:.0f}s  |  best val acc: {best_val:.4f}")

        except Exception:
            failed += 1
            print(f"  ✗ FAILED:\n{traceback.format_exc()}")

        # Write CSV after every run (so partial sweeps aren't lost)
        if summary_rows:
            import csv as _csv
            fieldnames = list(summary_rows[0].keys())
            with open(summary_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

    print(f"\nSweep complete: {completed} ok, {failed} failed.")
    print(f"Summary CSV → {summary_path}")
    return summary_rows


# ──────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-idx", type=int, default=1,
        help="1-based run index to start from (skip all earlier runs). "
             "Default: 1 (run everything). Pass 41 to jump straight to run 41."
    )
    args = parser.parse_args()
    run_sweep(start_idx=args.start_idx)
