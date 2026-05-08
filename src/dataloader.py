"""
dataloader.py
─────────────
Loads SEED-IV eeg_feature_smooth data (pre-extracted DE features).

Data format inside each .mat file:
  de_LDS{1..24}  : shape (62, T, 5)
    62  – EEG channels
    T   – pre-computed 4-second windows (varies per trial, ~12–64)
    5   – frequency bands: delta, theta, alpha, beta, gamma

Each trial window becomes one sample: shape (62, 5)
  → model should be built with Chans=62, Samples=5

Folder layout:
  data/eeg_feature_smooth/
    1/                         ← session 1
      1_20160518.mat           ← subject 1
      2_20150915.mat           ← subject 2
      ...
    2/  3/  ...

Quick-start
───────────
    from dataloader import build_loaders

    train_loader, val_loader = build_loaders(
        data_root   = '/path/to/CMSC472-Final/data',
        val_subject = 1,
        dataset     = 'SEED-IV',
    )
    # then build model with Chans=62, Samples=5
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import EEGDataset, BalancedBatchSampler


# ─────────────────────────────────────────────────────────────────────────────
# SEED-IV emotion label maps
# ─────────────────────────────────────────────────────────────────────────────

# 0=neutral  1=sad  2=fear  3=happy
SEED_IV_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

# 0=negative  1=neutral  2=positive
SEED_LABELS = {
    1: [1,0,2,0,1,1,2,0,1,2,2,1,0,2,0],
    2: [2,1,0,0,2,1,1,2,0,2,1,2,0,1,0],
    3: [1,2,0,1,2,0,1,2,0,1,2,0,1,2,0],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  .mat loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_mat(path: str) -> dict:
    """Load .mat file (scipy for <v7.3, h5py for v7.3+)."""
    import scipy.io as sio
    try:
        return sio.loadmat(path)
    except Exception:
        try:
            import h5py
            with h5py.File(path, 'r') as f:
                return {k: np.array(v) for k, v in f.items()
                        if not k.startswith('#')}
        except ImportError:
            raise ImportError("pip install h5py  (needed for MATLAB v7.3 files)")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-subject loader
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_data(
    data_root  : str,
    subject_id : int,           # 1-indexed (1 … 15)
    dataset    : str = 'SEED-IV',
    sessions   : list = None,   # None → all 3
    feature    : str = 'de_LDS',  # prefix of mat keys to use
) -> tuple:
    """
    Load pre-extracted DE features for one subject.

    Each trial array has shape (62, T, 5).
    We yield T samples per trial, each of shape (62, 5):
        axis-0 (62) = EEG channels
        axis-1  (5) = freq bands [delta, theta, alpha, beta, gamma]

    Returns
    -------
    X : np.ndarray  (N, 62, 5)   float32
    y : np.ndarray  (N,)          int64
    """
    label_map = SEED_IV_LABELS if dataset == 'SEED-IV' else SEED_LABELS
    sessions  = sessions or [1, 2, 3]

    all_X, all_y = [], []

    for sess in sessions:
        # session is the directory; subject_id is the file prefix
        sess_dir = os.path.join(data_root, 'eeg_feature_smooth', str(sess))

        if not os.path.isdir(sess_dir):
            print(f"  [warn] session dir missing: {sess_dir}")
            continue

        candidates = [
            f for f in os.listdir(sess_dir)
            if f.startswith(f'{subject_id}_') and f.endswith('.mat')
        ]
        if not candidates:
            print(f"  [warn] no .mat for subject {subject_id} session {sess}")
            continue

        mat_path = os.path.join(sess_dir, candidates[0])
        mat_data = _load_mat(mat_path)
        labels   = label_map[sess]

        # pick keys like de_LDS1, de_LDS2, … de_LDS24  (sorted numerically)
        trial_keys = sorted(
            [k for k in mat_data if k.startswith(feature)],
            key=lambda k: int(k.replace(feature, '') or 0)
        )

        for trial_idx, key in enumerate(trial_keys):
            if trial_idx >= len(labels):
                break

            arr = np.array(mat_data[key], dtype=np.float32)  # (62, T, 5)
            if arr.ndim != 3 or arr.shape[0] != 62 or arr.shape[2] != 5:
                continue                     # skip unexpected shapes

            # (62, T, 5) → T samples each of shape (62, 5)
            T = arr.shape[1]
            for t in range(T):
                all_X.append(arr[:, t, :])  # (62, 5)
                all_y.append(labels[trial_idx])

    if not all_X:
        raise RuntimeError(
            f"No data loaded for subject {subject_id}. "
            "Check data_root and folder layout."
        )

    X = np.stack(all_X).astype(np.float32)  # (N, 62, 5)
    y = np.array(all_y, dtype=np.int64)      # (N,)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class ChannelNormalizer:
    """Z-score per channel. Fit on train, apply to val/test."""

    def __init__(self):
        self.mean_ = None  # (1, 62, 1)
        self.std_  = None

    def fit(self, X: np.ndarray):
        # X : (N, 62, 5)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_  = X.std(axis=(0, 2),  keepdims=True).clip(min=1e-8)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Augmentations
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNoise:
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std

class FreqBandDropout:
    """Randomly zero out one frequency band."""
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 62, 5)
        if torch.rand(1).item() < self.p:
            band = torch.randint(0, 5, (1,)).item()
            x = x.clone()
            x[..., band] = 0.0
        return x

class TemporalShift:
    """No-op for pre-extracted features (kept for import compatibility)."""
    def __init__(self, max_shift=10):
        self.max_shift = max_shift
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Split strategies
# ─────────────────────────────────────────────────────────────────────────────

def cross_subject_split(
    data_root   : str,
    val_subject : int,
    dataset     : str = 'SEED-IV',
    n_subjects  : int = 15,
    **load_kwargs,
) -> tuple:
    """Hold out one subject for validation; train on all others."""
    train_X, train_y, train_s = [], [], []
    val_X,   val_y,   val_s   = [], [], []

    for sid in range(1, n_subjects + 1):
        print(f"  Loading subject {sid:02d}/{n_subjects} …", end=' ', flush=True)
        X, y = load_subject_data(data_root, sid, dataset=dataset, **load_kwargs)
        s    = np.full(len(y), sid - 1, dtype=np.int64)
        print(f"{len(y)} windows")

        if sid == val_subject:
            val_X.append(X);   val_y.append(y);   val_s.append(s)
        else:
            train_X.append(X); train_y.append(y); train_s.append(s)

    return (
        np.concatenate(train_X), np.concatenate(train_y), np.concatenate(train_s),
        np.concatenate(val_X),   np.concatenate(val_y),   np.concatenate(val_s),
    )


def within_subject_split(
    data_root    : str,
    subject_id   : int,
    val_sessions : list = None,
    dataset      : str  = 'SEED-IV',
    **load_kwargs,
) -> tuple:
    """Train on sessions 1–2, validate on session 3 (default)."""
    val_sessions   = val_sessions or [3]
    train_sessions = [s for s in [1, 2, 3] if s not in val_sessions]

    X_tr, y_tr = load_subject_data(data_root, subject_id, dataset=dataset,
                                    sessions=train_sessions, **load_kwargs)
    X_va, y_va = load_subject_data(data_root, subject_id, dataset=dataset,
                                    sessions=val_sessions, **load_kwargs)

    s_tr = np.full(len(y_tr), subject_id - 1, dtype=np.int64)
    s_va = np.full(len(y_va), subject_id - 1, dtype=np.int64)
    return X_tr, y_tr, s_tr, X_va, y_va, s_va


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    data_root     : str,
    val_subject   : int   = 1,
    subject_id    : int   = 1,
    strategy      : str   = 'cross_subject',
    dataset       : str   = 'SEED-IV',
    n_per_class   : int   = 8,
    num_workers   : int   = 4,
    augment_train : bool  = True,
    val_sessions  : list  = None,
    feature       : str   = 'de_LDS',
    leave_one_out : bool  = True,    # ← new flag
    val_fraction  : float = 0.15,    # ← fraction per subject held out when LOO=False
    n_subjects    : int   = 15,
    **kwargs,
) -> tuple:
    """
    Load SEED-IV DE features and return (train_loader, val_loader).

    leave_one_out=True  : original behaviour — one subject is held out entirely
                          as the validation set (cross-subject generalisation).

    leave_one_out=False : val set is built by drawing `val_fraction` of windows
                          from EVERY subject, so all subjects appear in both
                          train and val (within-distribution evaluation).

    NOTE: samples have shape (62, 5).
    Build the model with:   build_model(nb_classes=4, Chans=62, Samples=5)
    """
    nb_classes = 4 if dataset == 'SEED-IV' else 3
    load_kw    = dict(feature=feature)

    print(f"\nBuilding loaders  [LOO={leave_one_out}]  strategy={strategy}  "
          f"dataset={dataset}  feature={feature}")

    # ── data loading ──────────────────────────────────────────────────────
    if leave_one_out:
        # original behaviour
        if strategy == 'cross_subject':
            X_tr, y_tr, s_tr, X_va, y_va, s_va = cross_subject_split(
                data_root, val_subject=val_subject, dataset=dataset,
                n_subjects=n_subjects, **load_kw
            )
        elif strategy == 'within_subject':
            X_tr, y_tr, s_tr, X_va, y_va, s_va = within_subject_split(
                data_root, subject_id=subject_id, dataset=dataset,
                val_sessions=val_sessions, **load_kw
            )
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

    else:
        # draw val_fraction of windows from every subject
        X_tr, y_tr, s_tr = [], [], []
        X_va, y_va, s_va = [], [], []

        for sid in range(1, n_subjects + 1):
            print(f"  Loading subject {sid:02d}/{n_subjects} …", end=' ', flush=True)
            X, y = load_subject_data(data_root, sid, dataset=dataset, **load_kw)
            s    = np.full(len(y), sid - 1, dtype=np.int64)
            print(f"{len(y)} windows")

            # stratified split — preserve class balance within each subject
            from sklearn.model_selection import train_test_split as _tts
            idx_tr, idx_va = _tts(
                np.arange(len(y)),
                test_size    = val_fraction,
                stratify     = y,
                random_state = 42,
            )

            X_tr.append(X[idx_tr]); y_tr.append(y[idx_tr]); s_tr.append(s[idx_tr])
            X_va.append(X[idx_va]); y_va.append(y[idx_va]); s_va.append(s[idx_va])

        X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr); s_tr = np.concatenate(s_tr)
        X_va = np.concatenate(X_va); y_va = np.concatenate(y_va); s_va = np.concatenate(s_va)

    # ── normalise (fit on train only) ─────────────────────────────────────
    norm = ChannelNormalizer()
    X_tr = norm.fit_transform(X_tr)
    X_va = norm.transform(X_va)

    print(f"\n  Train : {X_tr.shape}  labels {np.bincount(y_tr)}")
    print(f"  Val   : {X_va.shape}  labels {np.bincount(y_va)}")

    # ── augmentations ─────────────────────────────────────────────────────
    train_transform = (
        ComposeTransforms([GaussianNoise(std=0.05), FreqBandDropout(p=0.2)])
        if augment_train else None
    )

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = EEGDataset(X_tr, y_tr, subject_id=s_tr, transform=train_transform)
    val_ds   = EEGDataset(X_va, y_va, subject_id=s_va)

    # ── samplers & loaders ────────────────────────────────────────────────
    train_sampler = BalancedBatchSampler(
        y_tr, n_per_class=n_per_class, nb_classes=nb_classes
    )
    batch_size = n_per_class * nb_classes

    train_loader = DataLoader(
        train_ds,
        batch_sampler = train_sampler,
        num_workers   = num_workers,
        pin_memory    = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    print(f"\n  Batch size    : {batch_size}  ({n_per_class}/class × {nb_classes} classes)")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}\n")

    return train_loader, val_loader
