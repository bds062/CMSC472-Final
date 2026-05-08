"""
plot_sweep.py  –  Visualise hyperparameter-sweep results from a TSV/CSV.

Usage:
    python plot_sweep.py results.tsv          # saves PNGs + prints best model
    python plot_sweep.py results.tsv --show   # also pops up interactive windows
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
LOSS_COLORS = {
    "none":        "#4C72B0",
    "contrastive": "#DD8452",
    "prototype":   "#55A868",
}
LOSS_LABELS = {
    "none":        "No contrastive (CE only)",
    "contrastive": "ContrastiveLoss",
    "prototype":   "ContrastivePrototype",
}

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise column names
    df.columns = df.columns.str.strip()
    # numeric coercion
    for col in ["final_train_acc", "final_val_acc", "best_val_acc",
                "final_val_loss", "lambda_con", "warmup_epochs",
                "lr", "weight_decay", "temperature", "elapsed_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bar chart: best_val_acc by loss_type
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_type_comparison(df: pd.DataFrame, out: Path, show: bool):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle("Performance by Loss Type", fontsize=14, fontweight="bold")

    for ax, metric, label in zip(
        axes,
        ["best_val_acc", "final_val_acc"],
        ["Best Val Acc", "Final Val Acc"],
    ):
        grp = df.groupby("loss_type")[metric]
        means = grp.mean()
        stds  = grp.std()
        colors = [LOSS_COLORS.get(lt, "#888") for lt in means.index]
        bars = ax.bar(means.index, means.values, yerr=stds.values,
                      color=colors, capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, min(1.0, means.max() * 1.25))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xticklabels([LOSS_LABELS.get(l, l) for l in means.index],
                           rotation=12, ha="right", fontsize=9)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    _save(fig, out / "01_loss_type_comparison.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LR vs val-acc, grouped by loss_type
# ─────────────────────────────────────────────────────────────────────────────
def plot_lr_effect(df: pd.DataFrame, out: Path, show: bool):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("Learning Rate vs Best Val Acc  (grouped by loss type)",
                 fontsize=13, fontweight="bold")

    for loss_type, sub in df.groupby("loss_type"):
        grp = sub.groupby("lr")["best_val_acc"]
        lrs   = grp.mean().index.astype(float)
        means = grp.mean().values
        stds  = grp.std().fillna(0).values
        color = LOSS_COLORS.get(loss_type, "#888")
        ax.plot(lrs, means, marker="o", color=color,
                label=LOSS_LABELS.get(loss_type, loss_type), linewidth=2)
        ax.fill_between(lrs, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out / "02_lr_effect.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lambda_con vs val-acc (only non-"none" losses actually use it)
# ─────────────────────────────────────────────────────────────────────────────
def plot_lambda_effect(df: pd.DataFrame, out: Path, show: bool):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("λ_con vs Best Val Acc  (grouped by loss type)",
                 fontsize=13, fontweight="bold")

    for loss_type, sub in df.groupby("loss_type"):
        grp = sub.groupby("lambda_con")["best_val_acc"]
        lambdas = grp.mean().index.astype(float)
        means   = grp.mean().values
        stds    = grp.std().fillna(0).values
        color   = LOSS_COLORS.get(loss_type, "#888")
        ax.plot(lambdas, means, marker="s", color=color,
                label=LOSS_LABELS.get(loss_type, loss_type), linewidth=2)
        ax.fill_between(lambdas, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("λ_con")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out / "03_lambda_effect.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Warmup epochs comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_warmup_effect(df: pd.DataFrame, out: Path, show: bool):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("Warmup Epochs vs Best Val Acc  (grouped by loss type)",
                 fontsize=13, fontweight="bold")

    warmup_vals = sorted(df["warmup_epochs"].dropna().unique())
    x = np.arange(len(warmup_vals))
    width = 0.25
    loss_types = df["loss_type"].unique()

    for i, loss_type in enumerate(loss_types):
        sub  = df[df["loss_type"] == loss_type]
        grp  = sub.groupby("warmup_epochs")["best_val_acc"]
        means = [grp.mean().get(w, np.nan) for w in warmup_vals]
        stds  = [grp.std().get(w, 0)       for w in warmup_vals]
        color = LOSS_COLORS.get(loss_type, "#888")
        offset = (i - len(loss_types) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=LOSS_LABELS.get(loss_type, loss_type),
               color=color, capsize=4, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)} epochs" for w in warmup_vals])
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _save(fig, out / "04_warmup_effect.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temperature effect  (contrastive loss only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_effect(df: pd.DataFrame, out: Path, show: bool):
    sub = df[df["loss_type"] == "contrastive"].dropna(subset=["temperature"])
    if sub.empty:
        print("[skip] No contrastive rows with temperature – skipping plot 05.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    fig.suptitle("Temperature vs Best Val Acc  (ContrastiveLoss only)",
                 fontsize=13, fontweight="bold")

    grp   = sub.groupby("temperature")["best_val_acc"]
    temps = grp.mean().index.astype(float)
    means = grp.mean().values
    stds  = grp.std().fillna(0).values

    ax.plot(temps, means, marker="D", color=LOSS_COLORS["contrastive"], linewidth=2)
    ax.fill_between(temps, means - stds, means + stds,
                    alpha=0.2, color=LOSS_COLORS["contrastive"])
    for t, m in zip(temps, means):
        ax.annotate(f"{m:.3f}", (t, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out / "05_temperature_effect.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train vs Val accuracy scatter  (overfitting view)
# ─────────────────────────────────────────────────────────────────────────────
def plot_train_val_scatter(df: pd.DataFrame, out: Path, show: bool):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    fig.suptitle("Train Acc vs Val Acc  (coloured by loss type)",
                 fontsize=13, fontweight="bold")

    for loss_type, sub in df.groupby("loss_type"):
        ax.scatter(sub["final_train_acc"], sub["final_val_acc"],
                   color=LOSS_COLORS.get(loss_type, "#888"), alpha=0.7,
                   label=LOSS_LABELS.get(loss_type, loss_type), s=50, edgecolors="white")

    lo = min(df["final_train_acc"].min(), df["final_val_acc"].min()) - 0.02
    hi = max(df["final_train_acc"].max(), df["final_val_acc"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="y = x (no gap)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Final Train Acc")
    ax.set_ylabel("Final Val Acc")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    _save(fig, out / "06_train_val_scatter.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Heatmap: lr × lambda_con for each loss type  (best_val_acc)
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmaps(df: pd.DataFrame, out: Path, show: bool):
    loss_types = df["loss_type"].unique()
    fig, axes = plt.subplots(1, len(loss_types),
                             figsize=(5 * len(loss_types), 4.5),
                             constrained_layout=True)
    if len(loss_types) == 1:
        axes = [axes]
    fig.suptitle("Heatmap: LR × λ_con  →  Best Val Acc",
                 fontsize=13, fontweight="bold")

    for ax, loss_type in zip(axes, loss_types):
        sub = df[df["loss_type"] == loss_type]
        pivot = sub.pivot_table(index="lr", columns="lambda_con",
                                values="best_val_acc", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn",
                       vmin=df["best_val_acc"].min(),
                       vmax=df["best_val_acc"].max())
        ax.set_title(LOSS_LABELS.get(loss_type, loss_type), fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0e}" for v in pivot.index], fontsize=8)
        ax.set_xlabel("λ_con"); ax.set_ylabel("LR")
        for (r, c), val in np.ndenumerate(pivot.values):
            if not np.isnan(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, out / "07_heatmap_lr_lambda.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Top-N runs bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_top_runs(df: pd.DataFrame, out: Path, show: bool, n: int = 15):
    top = df.nlargest(n, "best_val_acc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"Top {n} Runs by Best Val Acc", fontsize=13, fontweight="bold")

    colors = [LOSS_COLORS.get(lt, "#888") for lt in top["loss_type"]]
    bars = ax.barh(range(len(top)), top["best_val_acc"], color=colors,
                   edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["run_name"], fontsize=7)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Best Val Acc")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=LOSS_LABELS.get(lt, lt))
               for lt, c in LOSS_COLORS.items() if lt in df["loss_type"].values]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    for bar, val in zip(bars, top["best_val_acc"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    _save(fig, out / "08_top_runs.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, path: Path, show: bool):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def print_best(df: pd.DataFrame):
    best = df.loc[df["final_val_acc"].idxmax()]
    print("\n" + "=" * 60)
    print("  ★  BEST MODEL  (by final_val_acc)  ★")
    print("=" * 60)
    print(f"  run_name      : {best['run_name']}")
    print(f"  loss_type     : {best['loss_type']}")
    print(f"  lambda_con    : {best['lambda_con']}")
    print(f"  warmup_epochs : {int(best['warmup_epochs']) if not pd.isna(best['warmup_epochs']) else 'n/a'}")
    print(f"  lr            : {best['lr']}")
    print(f"  weight_decay  : {best['weight_decay']}")
    print(f"  temperature   : {best['temperature'] if not pd.isna(best['temperature']) else 'n/a'}")
    print("-" * 60)
    print(f"  final_train_acc : {best['final_train_acc']:.4f}")
    print(f"  final_val_acc   : {best['final_val_acc']:.4f}  ← ranked by this")
    print(f"  best_val_acc    : {best['best_val_acc']:.4f}")
    print(f"  final_val_loss  : {best['final_val_loss']:.4f}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot sweep results.")
    parser.add_argument("tsv", help="Path to results TSV/CSV file")
    parser.add_argument("--show", action="store_true",
                        help="Also show interactive plot windows")
    parser.add_argument("--outdir", default="sweep_plots",
                        help="Directory to save plots (default: sweep_plots/)")
    args = parser.parse_args()

    df = load(args.tsv)
    print(f"Loaded {len(df)} rows from '{args.tsv}'")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots → {out}/")
    plot_loss_type_comparison(df, out, args.show)
    plot_lr_effect(df, out, args.show)
    plot_lambda_effect(df, out, args.show)
    plot_warmup_effect(df, out, args.show)
    plot_temperature_effect(df, out, args.show)
    plot_train_val_scatter(df, out, args.show)
    plot_heatmaps(df, out, args.show)
    plot_top_runs(df, out, args.show)

    print_best(df)


if __name__ == "__main__":
    main()
