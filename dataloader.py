"""
dataloader.py

Dataloader for SEED / SEED-IV pre-extracted DE-LDS EEG features.

For SEED-IV eeg_feature_smooth files, each trial key such as de_LDS1 has shape
(62, T, 5):
  - 62 = EEG electrodes
  - T  = trial windows
  - 5  = DE frequency bands: delta, theta, alpha, beta, gamma

Each model sample is therefore a spatial-spectral matrix of shape (62, 5).
Build the model with:
    build_model(nb_classes=4, Chans=62, Samples=5)

This file avoids the most common evaluation leak:
  - For cross-subject evaluation, a full subject is held out.
  - For non-LOO sanity checks, splitting is trial-level, not window-level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


SEED_IV_LABELS = {
    # 0=neutral, 1=sad, 2=fear, 3=happy
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

SEED_LABELS = {
    # 0=negative, 1=neutral, 2=positive
    1: [1, 0, 2, 0, 1, 1, 2, 0, 1, 2, 2, 1, 0, 2, 0],
    2: [2, 1, 0, 0, 2, 1, 1, 2, 0, 2, 1, 2, 0, 1, 0],
    3: [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
}


def _load_mat(path: str) -> dict:
    """Load a MATLAB file using scipy for v7.2 or h5py for v7.3."""
    import scipy.io as sio

    try:
        return sio.loadmat(path)
    except NotImplementedError:
        import h5py

        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if not k.startswith("#"):
                    out[k] = np.array(f[k])
        return out


def _numeric_suffix(key: str, prefix: str) -> int:
    suffix = key.replace(prefix, "")
    return int(suffix) if suffix else 0


@dataclass
class LoadedSubject:
    X: np.ndarray          # (N, 62, 5)
    y: np.ndarray          # (N,)
    subject_id: np.ndarray # (N,)
    trial_id: np.ndarray   # (N,), unique within subject/session/trial


def load_subject_data(
    data_root: str,
    subject_id: int,
    dataset: str = "SEED-IV",
    sessions: Optional[list[int]] = None,
    feature: str = "de_LDS",
) -> LoadedSubject:
    """
    Load all windows for one subject.

    subject_id is 1-indexed, matching file prefixes in the SEED directories.
    Returned subject_id is 0-indexed for model/loss use.
    """
    if dataset not in {"SEED", "SEED-IV"}:
        raise ValueError("dataset must be 'SEED' or 'SEED-IV'.")

    label_map = SEED_IV_LABELS if dataset == "SEED-IV" else SEED_LABELS
    sessions = sessions or [1, 2, 3]

    Xs, ys, ss, tids = [], [], [], []

    for sess in sessions:
        sess_dir = os.path.join(data_root, "eeg_feature_smooth", str(sess))
        if not os.path.isdir(sess_dir):
            print(f"[warn] missing session directory: {sess_dir}")
            continue

        candidates = sorted(
            f for f in os.listdir(sess_dir)
            if f.startswith(f"{subject_id}_") and f.endswith(".mat")
        )
        if not candidates:
            print(f"[warn] no .mat file for subject {subject_id}, session {sess}")
            continue

        mat_path = os.path.join(sess_dir, candidates[0])
        mat_data = _load_mat(mat_path)
        labels = label_map[sess]

        trial_keys = sorted(
            [k for k in mat_data.keys() if k.startswith(feature)],
            key=lambda k: _numeric_suffix(k, feature),
        )

        for trial_idx, key in enumerate(trial_keys):
            if trial_idx >= len(labels):
                break

            arr = np.array(mat_data[key], dtype=np.float32)

            # Expected shape for scipy loadmat is (62, T, 5).
            # If a v7.3 file is encountered, dimensions can occasionally be reversed.
            if arr.ndim == 3 and arr.shape[-1] == 62 and arr.shape[0] == 5:
                arr = np.transpose(arr, (2, 1, 0))

            if arr.ndim != 3 or arr.shape[0] != 62 or arr.shape[2] != 5:
                print(f"[warn] skipping unexpected array {key}: shape={arr.shape}")
                continue

            label = labels[trial_idx]
            global_trial_id = (subject_id - 1) * 10_000 + sess * 100 + trial_idx

            for t in range(arr.shape[1]):
                Xs.append(arr[:, t, :])  # (62, 5)
                ys.append(label)
                ss.append(subject_id - 1)
                tids.append(global_trial_id)

    if not Xs:
        raise RuntimeError(
            f"No data loaded for subject {subject_id}. Check data_root, dataset, and feature."
        )

    return LoadedSubject(
        X=np.stack(Xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        subject_id=np.asarray(ss, dtype=np.int64),
        trial_id=np.asarray(tids, dtype=np.int64),
    )


class ChannelBandNormalizer:
    """
    Z-score per electrode and frequency band.

    For DE features, normalizing per (channel, band) is cleaner than collapsing
    across all five bands, because the bands have different scales.
    """
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ChannelBandNormalizer":
        self.mean_ = X.mean(axis=0, keepdims=True)              # (1, 62, 5)
        self.std_ = X.std(axis=0, keepdims=True).clip(min=1e-8) # (1, 62, 5)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fit before transform.")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class GaussianNoise:
    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class FreqBandDropout:
    """Randomly zero out one DE frequency band."""
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            band = torch.randint(0, x.shape[-1], (1,)).item()
            x = x.clone()
            x[..., band] = 0.0
        return x


class ComposeTransforms:
    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


class EEGDataset(Dataset):
    """
    Dataset wrapper for pre-extracted DE feature samples.

    X shape: (N, 62, 5).
    __getitem__ returns x with shape (1, 62, 5), plus label and subject_id.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_id: np.ndarray,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subject_id = torch.as_tensor(subject_id, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].unsqueeze(0)  # (1, 62, 5)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx], self.subject_id[idx]


class BalancedBatchSampler(Sampler[list[int]]):
    """
    Draw n_per_class examples from each class per batch.

    This is useful for supervised contrastive learning because each batch is
    likely to contain positives for each class.
    """
    def __init__(
        self,
        labels: np.ndarray | torch.Tensor,
        n_per_class: int = 8,
        nb_classes: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.n_per_class = n_per_class
        self.nb_classes = nb_classes
        self.batch_size = n_per_class * nb_classes
        self.seed = seed
        self.epoch = 0

        self.class_idx = [
            (self.labels == c).nonzero(as_tuple=True)[0].tolist()
            for c in range(nb_classes)
        ]
        empty = [c for c, idx in enumerate(self.class_idx) if len(idx) == 0]
        if empty:
            raise ValueError(f"No samples for classes {empty}; cannot build balanced batches.")

        self.n_batches = min(len(idx) for idx in self.class_idx) // n_per_class

    def __iter__(self):
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        perms = [torch.randperm(len(idx), generator=generator).tolist() for idx in self.class_idx]
        ptr = [0] * self.nb_classes

        for _ in range(self.n_batches):
            batch = []
            for c in range(self.nb_classes):
                chosen = perms[c][ptr[c]: ptr[c] + self.n_per_class]
                batch.extend(self.class_idx[c][j] for j in chosen)
                ptr[c] += self.n_per_class
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def _concat(subjects: list[LoadedSubject]) -> LoadedSubject:
    return LoadedSubject(
        X=np.concatenate([s.X for s in subjects], axis=0),
        y=np.concatenate([s.y for s in subjects], axis=0),
        subject_id=np.concatenate([s.subject_id for s in subjects], axis=0),
        trial_id=np.concatenate([s.trial_id for s in subjects], axis=0),
    )


def cross_subject_split(
    data_root: str,
    val_subject: int,
    dataset: str = "SEED-IV",
    n_subjects: int = 15,
    feature: str = "de_LDS",
) -> tuple[LoadedSubject, LoadedSubject]:
    """Train on all subjects except val_subject; validate on val_subject."""
    train_subjects, val_subjects = [], []

    for sid in range(1, n_subjects + 1):
        print(f"Loading subject {sid:02d}/{n_subjects} ...", end=" ", flush=True)
        subj = load_subject_data(data_root, sid, dataset=dataset, feature=feature)
        print(f"{len(subj.y)} windows")
        if sid == val_subject:
            val_subjects.append(subj)
        else:
            train_subjects.append(subj)

    return _concat(train_subjects), _concat(val_subjects)


def trial_level_mixed_subject_split(
    data_root: str,
    dataset: str = "SEED-IV",
    n_subjects: int = 15,
    feature: str = "de_LDS",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[LoadedSubject, LoadedSubject]:
    """
    Mixed-subject sanity-check split.

    This is not the main cross-subject protocol. It splits by whole trials within
    each subject so that windows from the same trial are not split across train/val.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []

    for sid in range(1, n_subjects + 1):
        print(f"Loading subject {sid:02d}/{n_subjects} ...", end=" ", flush=True)
        subj = load_subject_data(data_root, sid, dataset=dataset, feature=feature)
        print(f"{len(subj.y)} windows")

        train_mask = np.zeros(len(subj.y), dtype=bool)
        val_mask = np.zeros(len(subj.y), dtype=bool)

        for c in sorted(np.unique(subj.y)):
            class_trials = np.unique(subj.trial_id[subj.y == c])
            rng.shuffle(class_trials)
            n_val = max(1, int(round(len(class_trials) * val_fraction)))
            val_trials = set(class_trials[:n_val].tolist())
            class_val_mask = np.array([tid in val_trials for tid in subj.trial_id])
            val_mask |= class_val_mask & (subj.y == c)

        train_mask = ~val_mask

        train_parts.append(LoadedSubject(
            X=subj.X[train_mask],
            y=subj.y[train_mask],
            subject_id=subj.subject_id[train_mask],
            trial_id=subj.trial_id[train_mask],
        ))
        val_parts.append(LoadedSubject(
            X=subj.X[val_mask],
            y=subj.y[val_mask],
            subject_id=subj.subject_id[val_mask],
            trial_id=subj.trial_id[val_mask],
        ))

    return _concat(train_parts), _concat(val_parts)


def build_loaders(
    data_root: str,
    val_subject: int = 1,
    dataset: str = "SEED-IV",
    n_per_class: int = 8,
    num_workers: int = 4,
    augment_train: bool = True,
    feature: str = "de_LDS",
    leave_one_out: bool = True,
    val_fraction: float = 0.15,
    n_subjects: int = 15,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation loaders.

    leave_one_out=True is the recommended cross-subject evaluation protocol.
    """
    nb_classes = 4 if dataset == "SEED-IV" else 3
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    print(
        f"\nBuilding loaders | dataset={dataset} | feature={feature} | "
        f"leave_one_out={leave_one_out}"
    )

    if leave_one_out:
        train, val = cross_subject_split(
            data_root=data_root,
            val_subject=val_subject,
            dataset=dataset,
            n_subjects=n_subjects,
            feature=feature,
        )
    else:
        train, val = trial_level_mixed_subject_split(
            data_root=data_root,
            dataset=dataset,
            n_subjects=n_subjects,
            feature=feature,
            val_fraction=val_fraction,
            seed=seed,
        )

    normalizer = ChannelBandNormalizer()
    X_train = normalizer.fit_transform(train.X)
    X_val = normalizer.transform(val.X)

    print(f"\nTrain: {X_train.shape}, labels={np.bincount(train.y, minlength=nb_classes)}")
    print(f"Val:   {X_val.shape}, labels={np.bincount(val.y, minlength=nb_classes)}")

    train_transform = (
        ComposeTransforms([GaussianNoise(std=0.05), FreqBandDropout(p=0.2)])
        if augment_train else None
    )

    train_ds = EEGDataset(X_train, train.y, train.subject_id, transform=train_transform)
    val_ds = EEGDataset(X_val, val.y, val.subject_id)

    train_sampler = BalancedBatchSampler(
        train.y,
        n_per_class=n_per_class,
        nb_classes=nb_classes,
        seed=seed,
    )
    batch_size = n_per_class * nb_classes

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"Batch size: {batch_size} ({n_per_class}/class x {nb_classes} classes)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}\n")

    return train_loader, val_loader
