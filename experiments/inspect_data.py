"""
inspect_data.py

Sanity-check script for SEED-IV eeg_feature_smooth .mat files.

Purpose:
  - Verify that each de_LDS trial has shape (62, T, 5).
  - Confirm that a single model input sample should be arr[:, t, :],
    with shape (62, 5).
  - Avoid confusing the 5 frequency-band axis with raw EEG time samples.

Usage:
    python inspect_data.py

Optional:
    python inspect_data.py --mat-path /path/to/file.mat
    python inspect_data.py --feature de_LDS
    python inspect_data.py --trial 1
    python inspect_data.py --window 0
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_MAT_PATH = (
    "/fs/vulcan-projects/fsh_track/jason-bhargav-temp/"
    "CMSC472-Final/data/eeg_feature_smooth/1/1_20160518.mat"
)


def numeric_feature_sort_key(key: str, feature_prefix: str) -> int:
    """
    Sort keys like de_LDS1, de_LDS2, ..., de_LDS24 numerically.
    """
    suffix = key.replace(feature_prefix, "")
    return int(suffix) if suffix.isdigit() else -1


def inspect_mat_file(
    mat_path: str,
    feature_prefix: str = "de_LDS",
    trial: int = 1,
    window: int = 0,
) -> None:
    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(f"Could not find .mat file: {mat_path}")

    print("=" * 80)
    print("Inspecting SEED-IV feature file")
    print("=" * 80)
    print(f"Path: {mat_path}")
    print(f"Feature prefix: {feature_prefix}")
    print()

    data = loadmat(mat_path)

    keys = list(data.keys())
    feature_keys = sorted(
        [k for k in keys if k.startswith(feature_prefix)],
        key=lambda k: numeric_feature_sort_key(k, feature_prefix),
    )

    print(f"Total keys in file: {len(keys)}")
    print(f"Number of {feature_prefix} trial keys: {len(feature_keys)}")
    print()

    if not feature_keys:
        raise RuntimeError(f"No keys starting with '{feature_prefix}' found.")

    print("Trial feature shapes:")
    for key in feature_keys:
        arr = data[key]
        print(f"  {key:<12} {arr.shape}")

    print()
    print("=" * 80)
    print("Axis interpretation")
    print("=" * 80)
    print("Expected de_LDS trial shape: (62, T, 5)")
    print("  axis 0: 62 EEG electrodes/channels")
    print("  axis 1: T precomputed windows within the trial")
    print("  axis 2: 5 frequency bands: delta, theta, alpha, beta, gamma")
    print()

    trial_key = f"{feature_prefix}{trial}"
    if trial_key not in data:
        raise KeyError(
            f"{trial_key} not found. Available keys include: {feature_keys[:5]} ..."
        )

    arr = np.asarray(data[trial_key])
    print(f"Selected trial: {trial_key}")
    print(f"Selected trial shape: {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for {trial_key}, got shape {arr.shape}.")

    n_channels, n_windows, n_bands = arr.shape

    if n_channels != 62:
        print(f"[warning] Expected 62 channels, found {n_channels}.")
    if n_bands != 5:
        print(f"[warning] Expected 5 frequency bands, found {n_bands}.")

    if window < 0 or window >= n_windows:
        raise IndexError(
            f"Window index {window} out of range for {trial_key}; "
            f"valid range is 0 to {n_windows - 1}."
        )

    print()
    print("=" * 80)
    print("Correct sample extraction")
    print("=" * 80)

    single_sample = arr[:, window, :]
    print(f"Correct single sample: arr[:, {window}, :]")
    print(f"Shape: {single_sample.shape}")
    print("This should be one model input before adding the Conv2d channel dimension.")
    print("The dataloader should later convert it to shape: (1, 62, 5)")
    print()

    one_band_over_all_windows = arr[:, :, 0]
    print("Common mistake: arr[:, :, 0]")
    print(f"Shape: {one_band_over_all_windows.shape}")
    print(
        "This is NOT one training sample. It is one frequency band across all "
        "windows in the trial."
    )
    print()

    print("=" * 80)
    print("Model configuration implied by this file")
    print("=" * 80)
    print("Use:")
    print("    build_model(nb_classes=4, Chans=62, Samples=5)")
    print()
    print("Here, Samples=5 means frequency bands, not raw time samples.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect SEED-IV eeg_feature_smooth .mat feature shapes."
    )
    parser.add_argument(
        "--mat-path",
        type=str,
        default=DEFAULT_MAT_PATH,
        help="Path to a SEED-IV eeg_feature_smooth .mat file.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="de_LDS",
        help="Feature prefix to inspect, e.g. de_LDS, de_movingAve, psd_LDS.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Trial number to inspect, e.g. 1 for de_LDS1.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Window index inside the selected trial.",
    )

    args = parser.parse_args()

    inspect_mat_file(
        mat_path=args.mat_path,
        feature_prefix=args.feature,
        trial=args.trial,
        window=args.window,
    )


if __name__ == "__main__":
    main()
