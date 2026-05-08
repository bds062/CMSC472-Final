"""
plot_sweep.py

Visualize hyperparameter-sweep results from a CSV/TSV file.

Usage:
    python plot_sweep.py checkpoints/sweep_summary_subj=01.csv
    python plot_sweep.py checkpoints/sweep_summary_subj=01.csv --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


LOSS_ORDER = ["ce", "supcon", "prototype", "sepc"]

LOSS_COLORS = {
    "ce": "#4C72B0",
    "supcon": "#DD8452",
    "prototype": "#55A868",
    "sepc": "#C44E52",
}

LOSS_LABELS = {
    "ce": "CE only",
    "supcon": "Supervised contrastive",
    "prototype": "Prototype",
    "sepc": "SEPC",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "final_train_acc",
        "final_val_acc",
        "best_val_acc",
        "final_val_loss",
        "lambda_con",
        "warmup_epochs",
        "lr",
        "weight_decay",
        "temperature",
        "elapsed_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "loss_type" in df.columns:
        present_order = [x for x in LOSS_ORDER if x in set(df["loss_type"])]
        df["loss_type"] = pd.Categorical(
            df["loss_type"],
            categories=present_order,
            ordered=True,
        )
        df = df.sort_values("loss_type")

    return df


def _save(fig, path: Path, show: bool) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_loss_type_comparison(df: pd.DataFrame, out: Path, show: bool) -> None:
    metrics = [("best_val_acc", "Best Val Acc"), ("final_val_acc", "Final Val Acc")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle("Performance by Loss Type", fontsize=14, fontweight="bold")

    for ax, (metric, title) in zip(axes, metrics):
        grouped = df.groupby("loss_type", observed=True)[metric]
        means = grouped.mean()
        stds = grouped.std().fillna(0.0)

        x = np.arange(len(means))
        colors = [LOSS_COLORS.get(str(loss), "#888888") for loss in means.index]
        bars = ax.bar(x, means.values, yerr=stds.values, color=colors, capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LOSS_LABELS.get(str(loss), str(loss)) for loss in means.index],
            rotation=15,
            ha="right",
        )
        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, min(1.0, max(0.05, means.max() * 1.25)))

        for bar, val in zip(bars, means.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _save(fig, out / "01_loss_type_comparison.png", show)


def plot_lr_effect(df: pd.DataFrame, out: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("Learning Rate vs Best Val Acc", fontsize=13, fontweight="bold")

    for loss_type, sub in df.groupby("loss_type", observed=True):
        grp = sub.groupby("lr")["best_val_acc"]
        lrs = grp.mean().index.astype(float)
        means = grp.mean().values
        stds = grp.std().fillna(0).values
        color = LOSS_COLORS.get(str(loss_type), "#888888")

        ax.plot(lrs, means, marker="o", label=LOSS_LABELS.get(str(loss_type), str(loss_type)), color=color)
        ax.fill_between(lrs, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, out / "02_lr_effect.png", show)


def plot_lambda_effect(df: pd.DataFrame, out: Path, show: bool) -> None:
    sub_df = df[df["loss_type"].astype(str) != "ce"].copy()
    if sub_df.empty:
        print("skip lambda plot: no contrastive/prototype rows")
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("lambda_con vs Best Val Acc", fontsize=13, fontweight="bold")

    for loss_type, sub in sub_df.groupby("loss_type", observed=True):
        grp = sub.groupby("lambda_con")["best_val_acc"]
        xs = grp.mean().index.astype(float)
        means = grp.mean().values
        stds = grp.std().fillna(0).values
        color = LOSS_COLORS.get(str(loss_type), "#888888")

        ax.plot(xs, means, marker="s", label=LOSS_LABELS.get(str(loss_type), str(loss_type)), color=color)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("lambda_con")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, out / "03_lambda_effect.png", show)


def plot_warmup_effect(df: pd.DataFrame, out: Path, show: bool) -> None:
    sub_df = df[df["loss_type"].astype(str) != "ce"].copy()
    if sub_df.empty:
        print("skip warmup plot: no contrastive/prototype rows")
        return

    warmups = sorted(sub_df["warmup_epochs"].dropna().unique())
    x = np.arange(len(warmups))
    loss_types = list(sub_df["loss_type"].dropna().unique())
    width = 0.8 / max(1, len(loss_types))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("Warmup Epochs vs Best Val Acc", fontsize=13, fontweight="bold")

    for i, loss_type in enumerate(loss_types):
        sub = sub_df[sub_df["loss_type"] == loss_type]
        grp = sub.groupby("warmup_epochs")["best_val_acc"]
        means = [grp.mean().get(w, np.nan) for w in warmups]
        stds = [grp.std().fillna(0).get(w, 0.0) for w in warmups]
        offset = (i - (len(loss_types) - 1) / 2) * width
        color = LOSS_COLORS.get(str(loss_type), "#888888")

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=LOSS_LABELS.get(str(loss_type), str(loss_type)),
            color=color,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(w)) for w in warmups])
    ax.set_xlabel("Warmup Epochs")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, out / "04_warmup_effect.png", show)


def plot_temperature_effect(df: pd.DataFrame, out: Path, show: bool) -> None:
    sub_df = df[df["temperature"].notna() & (df["loss_type"].astype(str) != "ce")].copy()
    if sub_df.empty:
        print("skip temperature plot: no temperature-swept rows")
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle("Temperature vs Best Val Acc", fontsize=13, fontweight="bold")

    for loss_type, sub in sub_df.groupby("loss_type", observed=True):
        grp = sub.groupby("temperature")["best_val_acc"]
        temps = grp.mean().index.astype(float)
        means = grp.mean().values
        stds = grp.std().fillna(0).values
        color = LOSS_COLORS.get(str(loss_type), "#888888")

        ax.plot(temps, means, marker="D", label=LOSS_LABELS.get(str(loss_type), str(loss_type)), color=color)
        ax.fill_between(temps, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Best Val Acc")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, out / "05_temperature_effect.png", show)


def plot_train_val_scatter(df: pd.DataFrame, out: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    fig.suptitle("Train Acc vs Val Acc", fontsize=13, fontweight="bold")

    for loss_type, sub in df.groupby("loss_type", observed=True):
        ax.scatter(
            sub["final_train_acc"],
            sub["final_val_acc"],
            alpha=0.75,
            s=55,
            label=LOSS_LABELS.get(str(loss_type), str(loss_type)),
            color=LOSS_COLORS.get(str(loss_type), "#888888"),
            edgecolors="white",
        )

    lo = min(df["final_train_acc"].min(), df["final_val_acc"].min()) - 0.02
    hi = max(df["final_train_acc"].max(), df["final_val_acc"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Final Train Acc")
    ax.set_ylabel("Final Val Acc")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out / "06_train_val_scatter.png", show)


def plot_heatmaps(df: pd.DataFrame, out: Path, show: bool) -> None:
    sub_df = df[df["loss_type"].astype(str) != "ce"].copy()
    if sub_df.empty:
        print("skip heatmap: no contrastive/prototype rows")
        return

    loss_types = list(sub_df["loss_type"].dropna().unique())
    fig, axes = plt.subplots(
        1,
        len(loss_types),
        figsize=(5 * len(loss_types), 4.5),
        constrained_layout=True,
    )
    if len(loss_types) == 1:
        axes = [axes]

    fig.suptitle("LR x lambda_con -> Best Val Acc", fontsize=13, fontweight="bold")

    vmin = sub_df["best_val_acc"].min()
    vmax = sub_df["best_val_acc"].max()

    for ax, loss_type in zip(axes, loss_types):
        sub = sub_df[sub_df["loss_type"] == loss_type]
        pivot = sub.pivot_table(
            index="lr",
            columns="lambda_con",
            values="best_val_acc",
            aggfunc="mean",
        )
        im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(LOSS_LABELS.get(str(loss_type), str(loss_type)))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0e}" for v in pivot.index])
        ax.set_xlabel("lambda_con")
        ax.set_ylabel("LR")

        for (r, c), val in np.ndenumerate(pivot.values):
            if not np.isnan(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, out / "07_heatmap_lr_lambda.png", show)


def plot_top_runs(df: pd.DataFrame, out: Path, show: bool, n: int = 15) -> None:
    top = df.nlargest(n, "best_val_acc").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"Top {n} Runs by Best Val Acc", fontsize=13, fontweight="bold")

    colors = [LOSS_COLORS.get(str(loss), "#888888") for loss in top["loss_type"]]
    bars = ax.barh(range(len(top)), top["best_val_acc"], color=colors)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["run_name"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Best Val Acc")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, top["best_val_acc"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=7)

    _save(fig, out / "08_top_runs.png", show)


def print_best(df: pd.DataFrame) -> None:
    best = df.loc[df["best_val_acc"].idxmax()]
    print("\n" + "=" * 60)
    print("BEST MODEL by best_val_acc")
    print("=" * 60)
    for key in [
        "run_name",
        "loss_type",
        "lambda_con",
        "warmup_epochs",
        "lr",
        "weight_decay",
        "temperature",
        "final_train_acc",
        "final_val_acc",
        "best_val_acc",
        "final_val_loss",
    ]:
        value = best.get(key, "n/a")
        print(f"{key:16s}: {value}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep results.")
    parser.add_argument("results", help="Path to sweep CSV/TSV.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--outdir", default="sweep_plots")
    args = parser.parse_args()

    df = load(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

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
