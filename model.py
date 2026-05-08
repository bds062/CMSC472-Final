"""
model.py 
EEGNet re-implemented in PyTorch with dual-head architecture for joint
classification + contrastive learning (SupCon / Contrastive-Prototype).

Expected input shape : (B, 1, Chans, Samples)
  B       – batch size
  1       – single EEG "image" channel (Conv2d convention)
  Chans   – number of EEG electrodes  (SEED/SEED-IV: 62)
  Samples – time points per segment   (e.g. 200 @ 200 Hz → 1 s window)

Quick-start
-----------
    from model import EEGNetContrastive, build_model
    from losses import ClassificationLoss, ContrastiveLoss, ContrastivePrototype

    model = build_model(nb_classes=4, Chans=62, Samples=200)
    logits, proj = model(batch_eeg)           # forward pass
    loss = cls_fn(logits, labels) + 0.5 * con_fn(proj, labels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class _DepthwiseConv2d(nn.Module):
    """
    Depthwise Conv2d with an optional per-filter max-norm weight constraint.
    Mirrors Keras DepthwiseConv2D + depthwise_constraint=max_norm(1.).
    """
    def __init__(self, in_channels, depth_multiplier, kernel_size,
                 max_norm_val=1.0, bias=False):
        super().__init__()
        out_channels = in_channels * depth_multiplier
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            groups=in_channels, bias=bias
        )
        self.max_norm_val = max_norm_val

    def _apply_max_norm(self):
        """Clamp each filter's L2 norm to at most max_norm_val."""
        with torch.no_grad():
            w = self.conv.weight                           # (C_out, 1, kH, kW)
            norm = w.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            desired = norm.clamp(max=self.max_norm_val)
            self.conv.weight.copy_(w * desired / norm)

    def forward(self, x):
        self._apply_max_norm()
        return self.conv(x)


class _SeparableConv2d(nn.Module):
    """
    Depthwise-then-pointwise (separable) convolution.
    Mirrors Keras SeparableConv2D.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, bias=False):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    padding=padding, groups=in_channels, bias=bias)
        self.pointwise  = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EEGNET ENCODER  (blocks 1 & 2 → flat embedding)
# ─────────────────────────────────────────────────────────────────────────────

class EEGNetEncoder(nn.Module):
    """
    EEGNet backbone that outputs a flat feature vector.
    Does NOT include a classification head — that lives in EEGNetContrastive.

    Architecture
    ────────────
    Block 1 – temporal convolution
      Conv2d(1, F1, (1, kernLength))        # learn F1 temporal filters
      BatchNorm2d
      DepthwiseConv2d((Chans, 1), D×)       # learn D spatial filters per temporal filter
      BatchNorm2d → ELU → AvgPool(1,4) → Dropout

    Block 2 – separable convolution
      SeparableConv2d(F1*D, F2, (1, 16))   # combine spatial filters across time
      BatchNorm2d → ELU → AvgPool(1,8) → Dropout

    Flatten → embedding of dimension `embed_dim`

    Parameters
    ----------
    Chans       : int   – EEG electrode count          (default 62 for SEED)
    Samples     : int   – time points per window       (default 200)
    F1          : int   – number of temporal filters   (default 8)
    D           : int   – spatial depth multiplier     (default 2)
    F2          : int   – pointwise filter count       (default F1*D = 16)
    kernLength  : int   – temporal kernel length       (default Samples//2)
    dropoutRate : float – dropout probability          (default 0.5)
    """

    def __init__(
        self,
        Chans       = 62,
        Samples     = 200,
        F1          = 8,
        D           = 2,
        F2          = None,       # defaults to F1 * D
        kernLength  = None,       # defaults to Samples // 2
        dropoutRate = 0.5,
    ):
        super().__init__()

        F2         = F2         or F1 * D
        kernLength = kernLength or (Samples // 2)

        # ── Block 1 ───────────────────────────────────────────────────────
        self.b1_temporal = nn.Conv2d(
            1, F1, (1, kernLength),
            padding=(0, kernLength // 2), bias=False
        )
        self.b1_bn1      = nn.BatchNorm2d(F1)
        self.b1_spatial  = _DepthwiseConv2d(F1, D, (Chans, 1), max_norm_val=1.0)
        self.b1_bn2      = nn.BatchNorm2d(F1 * D)
        # Pool sizes scale with Samples so tiny inputs (e.g. Samples=5 DE features)
        # don't collapse to zero. kernLength padding keeps width ≈ Samples after conv.
        _after_b1 = max(1, Samples // 4)
        _pool1    = min(4, max(1, Samples // 2))
        _pool2    = min(8, max(1, _after_b1))
        self.b1_pool     = nn.AvgPool2d((1, _pool1))
        self.b1_drop     = nn.Dropout(dropoutRate)

        # ── Block 2 ───────────────────────────────────────────────────────
        self.b2_sep      = _SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8))
        self.b2_bn       = nn.BatchNorm2d(F2)
        self.b2_pool     = nn.AvgPool2d((1, _pool2))
        self.b2_drop     = nn.Dropout(dropoutRate)

        # ── derive flat embedding size with a dry run ─────────────────────
        self.embed_dim   = self._infer_embed_dim(Chans, Samples)
        self._init_weights()

    # ── helpers ───────────────────────────────────────────────────────────

    def _infer_embed_dim(self, Chans, Samples):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            out   = self._forward_blocks(dummy)
        return out.shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── forward ───────────────────────────────────────────────────────────

    def _forward_blocks(self, x):
        # Block 1
        x = self.b1_temporal(x)
        x = self.b1_bn1(x)
        x = self.b1_spatial(x)
        x = self.b1_bn2(x)
        x = F.elu(x)
        x = self.b1_pool(x)
        x = self.b1_drop(x)
        # Block 2
        x = self.b2_sep(x)
        x = self.b2_bn(x)
        x = F.elu(x)
        x = self.b2_pool(x)
        x = self.b2_drop(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        """
        x : (B, 1, Chans, Samples)
        returns : (B, embed_dim)
        """
        return self._forward_blocks(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PROJECTION HEAD  (encoder output → contrastive loss input)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP that maps the encoder embedding into a lower-dimensional
    space suited for contrastive learning (SimCLR / SupCon style).

    Linear → BN → ReLU → Linear

    The final layer has NO activation; the contrastive losses apply
    F.normalize internally.

    Parameters
    ----------
    in_dim     : int – encoder embedding dimension
    hidden_dim : int – intermediate MLP width     (default 128)
    out_dim    : int – contrastive projection dim (default 64)
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, z):
        """
        z   : (B, in_dim)
        out : (B, out_dim)  – NOT yet L2-normalised; losses do that
        """
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLASSIFICATION HEAD  (encoder output → class logits)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Single linear layer: embed_dim → nb_classes.
    A max-norm constraint on the weight matrix mirrors the original Keras
    model and can help prevent the classification loss from dominating.

    Parameters
    ----------
    in_dim     : int   – encoder embedding dimension
    nb_classes : int   – number of emotion classes (3 for SEED, 4 for SEED-IV)
    max_norm   : float – per-row L2 norm ceiling for the weight matrix
    """

    def __init__(self, in_dim, nb_classes, max_norm_val=0.25):
        super().__init__()
        self.fc           = nn.Linear(in_dim, nb_classes)
        self.max_norm_val = max_norm_val
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _apply_max_norm(self):
        with torch.no_grad():
            w    = self.fc.weight                        # (nb_classes, in_dim)
            norm = w.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
            cap  = norm.clamp(max=self.max_norm_val)
            self.fc.weight.copy_(w * cap / norm)

    def forward(self, z):
        """
        z   : (B, in_dim)
        out : (B, nb_classes) – raw logits (no softmax; losses handle that)
        """
        self._apply_max_norm()
        return self.fc(z)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EEGNetContrastive(nn.Module):
    """
    End-to-end model:

        EEGNetEncoder ──┬──► ProjectionHead ──► proj    (→ ContrastiveLoss)
                        └──► ClassificationHead ──► logits (→ ClassificationLoss)

    Both heads are trained jointly:
        loss = ClassificationLoss(logits, labels)
             + λ * ContrastiveLoss(proj, labels)

    Parameters
    ----------
    nb_classes   : int   – emotion classes (3 = SEED, 4 = SEED-IV)
    Chans        : int   – EEG channels
    Samples      : int   – time points per window
    proj_hidden  : int   – projection head hidden width
    proj_out     : int   – projection head output dimension
    cls_max_norm : float – classification head weight constraint
    **enc_kwargs         – passed directly to EEGNetEncoder
                           (F1, D, F2, kernLength, dropoutRate)
    """

    def __init__(
        self,
        nb_classes   = 4,
        Chans        = 62,
        Samples      = 200,
        proj_hidden  = 128,
        proj_out     = 64,
        cls_max_norm = 0.25,
        **enc_kwargs,
    ):
        super().__init__()
        self.encoder  = EEGNetEncoder(Chans=Chans, Samples=Samples, **enc_kwargs)
        D             = enc_kwargs.get('embed_dim', self.encoder.embed_dim)
        self.proj     = ProjectionHead(self.encoder.embed_dim, proj_hidden, proj_out)
        self.cls_head = ClassificationHead(self.encoder.embed_dim, nb_classes,
                                           max_norm_val=cls_max_norm)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, Chans, Samples)

        Returns
        -------
        logits : (B, nb_classes) – for ClassificationLoss / accuracy
        proj   : (B, proj_out)   – for ContrastiveLoss / ContrastivePrototype
        """
        z      = self.encoder(x)          # (B, embed_dim)
        logits = self.cls_head(z)         # (B, nb_classes)
        proj   = self.proj(z)             # (B, proj_out)
        return logits, proj

    def encode(self, x):
        """Return raw encoder embedding (useful for t-SNE / evaluation)."""
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    For each mini-batch, draw `n_per_class` samples from every class so
    the contrastive losses always have sufficient positive pairs.

    Parameters
    ----------
    labels       : 1-D tensor or list of integer class labels
    n_per_class  : int – samples per class per batch   (≥ 2 recommended)
    nb_classes   : int – total number of classes
    """

    def __init__(self, labels, n_per_class=8, nb_classes=4):
        super().__init__()
        self.labels      = torch.as_tensor(labels)
        self.n_per_class = n_per_class
        self.nb_classes  = nb_classes
        self.batch_size  = n_per_class * nb_classes

        # pre-compute per-class index lists
        self.class_idx = [
            (self.labels == c).nonzero(as_tuple=True)[0].tolist()
            for c in range(nb_classes)
        ]
        # number of batches per epoch – limited by the smallest class
        self.n_batches = min(len(idx) for idx in self.class_idx) // n_per_class

    def __iter__(self):
        # shuffle within each class at the start of every epoch
        perm = [torch.randperm(len(idx)).tolist() for idx in self.class_idx]
        ptr  = [0] * self.nb_classes

        for _ in range(self.n_batches):
            batch = []
            for c in range(self.nb_classes):
                chosen = perm[c][ptr[c]: ptr[c] + self.n_per_class]
                batch += [self.class_idx[c][i] for i in chosen]
                ptr[c] += self.n_per_class
            yield batch

    def __len__(self):
        return self.n_batches


class EEGDataset(torch.utils.data.Dataset):
    """
    Minimal dataset wrapper for pre-segmented EEG data.

    Parameters
    ----------
    X          : np.ndarray or tensor, shape (N, Chans, Samples)
                 Raw EEG segments. The channel dimension is added automatically.
    y          : np.ndarray or tensor, shape (N,) – integer class labels
    subject_id : np.ndarray or tensor, shape (N,) – subject index 0-14 (optional)
    transform  : callable – optional per-sample augmentation
    """

    def __init__(self, X, y, subject_id=None, transform=None):
        self.X          = torch.as_tensor(X, dtype=torch.float32)
        self.y          = torch.as_tensor(y, dtype=torch.long)
        self.subject_id = (torch.as_tensor(subject_id, dtype=torch.long)
                           if subject_id is not None else None)
        self.transform  = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # (Chans, Samples) → (1, Chans, Samples)
        x = self.X[idx].unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        if self.subject_id is not None:
            return x, self.y[idx], self.subject_id[idx]
        return x, self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Handles the joint classification + contrastive training loop.

    Parameters
    ----------
    model          : EEGNetContrastive
    cls_loss_fn    : ClassificationLoss
    con_loss_fn    : ContrastiveLoss  OR  ContrastivePrototype
    optimizer      : torch.optim.Optimizer
    lambda_con     : float – weight applied to the contrastive term (default 0.5)
    warmup_epochs  : int   – epochs to train with contrastive loss only,
                             before enabling classification loss (default 0)
    device         : str   – 'cuda' or 'cpu'
    scheduler      : optional LR scheduler (step called once per epoch)
    """

    def __init__(
        self,
        model,
        cls_loss_fn,
        con_loss_fn,
        optimizer,
        lambda_con    = 0.5,
        warmup_epochs = 0,
        device        = 'cuda',
        scheduler     = None,
    ):
        self.model         = model.to(device)
        self.cls_loss_fn   = cls_loss_fn
        self.con_loss_fn   = con_loss_fn
        self.optimizer     = optimizer
        self.lambda_con    = lambda_con
        self.warmup_epochs = warmup_epochs
        self.device        = device
        self.scheduler     = scheduler

    # ── single epoch ──────────────────────────────────────────────────────

    def _run_epoch(self, loader, train=True, epoch=0):
        self.model.train(train)
        total_loss = total_cls = total_con = correct = n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                # unpack – subject_id is optional
                if len(batch) == 3:
                    x, labels, _ = batch
                else:
                    x, labels    = batch

                x      = x.to(self.device)
                labels = labels.to(self.device)

                logits, proj = self.model(x)

                # ── losses ────────────────────────────────────────────────
                l_con = self.con_loss_fn(proj, labels)

                if epoch <= self.warmup_epochs:
                    # contrastive warm-up: classification head not yet trained
                    loss  = l_con
                    l_cls = torch.tensor(0.0)
                else:
                    l_cls = self.cls_loss_fn(logits, labels)
                    loss  = l_cls + self.lambda_con * l_con

                # ── backward ──────────────────────────────────────────────
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # gradient clipping – prevents occasional spikes with
                    # contrastive loss when batches are imbalanced
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # ── metrics ───────────────────────────────────────────────
                bs          = x.size(0)
                total_loss += loss.item()  * bs
                total_cls  += l_cls.item() * bs
                total_con  += l_con.item() * bs
                correct    += (logits.argmax(1) == labels).sum().item()
                n          += bs

        return {
            'loss'    : total_loss / n,
            'cls_loss': total_cls  / n,
            'con_loss': total_con  / n,
            'acc'     : correct    / n,
        }

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader=None, epochs=50):
        """
        Train for `epochs` epochs and return a history dict.

        Returns
        -------
        history : dict with keys 'train', 'val' → lists of per-epoch metric dicts
        """
        history = {'train': [], 'val': []}

        for epoch in range(1, epochs + 1):
            tr = self._run_epoch(train_loader, train=True,  epoch=epoch)
            history['train'].append(tr)

            log = (f"Epoch {epoch:03d}/{epochs}  "
                   f"train_loss={tr['loss']:.4f}  "
                   f"cls={tr['cls_loss']:.4f}  "
                   f"con={tr['con_loss']:.4f}  "
                   f"acc={tr['acc']:.3f}")

            if val_loader is not None:
                va = self._run_epoch(val_loader, train=False, epoch=epoch)
                history['val'].append(va)
                log += (f"  |  val_loss={va['loss']:.4f}  "
                        f"val_acc={va['acc']:.3f}")

            print(log)

            if self.scheduler is not None:
                self.scheduler.step()

        return history

    @torch.no_grad()
    def evaluate(self, loader):
        """Return metric dict for a given DataLoader (no gradient)."""
        return self._run_epoch(loader, train=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CONVENIENCE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    nb_classes   = 4,
    Chans        = 62,
    Samples      = 200,
    F1           = 8,
    D            = 2,
    kernLength   = None,
    dropoutRate  = 0.5,
    proj_hidden  = 128,
    proj_out     = 64,
) -> EEGNetContrastive:
    """
    One-liner to instantiate the model with sensible defaults.

    Typical usage
    -------------
        # SEED-IV, 1-second windows @ 200 Hz
        model = build_model(nb_classes=4, Chans=62, Samples=200)

        # SEED, 4-second windows @ 128 Hz
        model = build_model(nb_classes=3, Chans=62, Samples=512)
    """
    return EEGNetContrastive(
        nb_classes  = nb_classes,
        Chans       = Chans,
        Samples     = Samples,
        proj_hidden = proj_hidden,
        proj_out    = proj_out,
        F1          = F1,
        D           = D,
        kernLength  = kernLength,
        dropoutRate = dropoutRate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  QUICK SMOKE-TEST  (python model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running smoke-test on {device}\n")

    # ── build model ───────────────────────────────────────────────────────
    model = build_model(nb_classes=4, Chans=62, Samples=200).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  embed_dim  : {model.encoder.embed_dim}")
    print(f"  total params: {total_params:,}\n")

    # ── fake batch ────────────────────────────────────────────────────────
    B      = 32
    x      = torch.randn(B, 1, 62, 200, device=device)
    labels = torch.randint(0, 4, (B,), device=device)

    logits, proj = model(x)
    print(f"  logits shape : {tuple(logits.shape)}  (expected {B} x 4)")
    print(f"  proj   shape : {tuple(proj.shape)}   (expected {B} x 64)\n")

    # ── loss check ────────────────────────────────────────────────────────
    # Import losses if available next to this file
    try:
        sys.path.insert(0, '.')
        from losses import ClassificationLoss, ContrastiveLoss, ContrastivePrototype

        cls_fn = ClassificationLoss()
        con_fn = ContrastiveLoss(temperature=0.1)
        cpr_fn = ContrastivePrototype(num_classes=4, temperature=0.1)

        l_cls = cls_fn(logits, labels)
        l_con = con_fn(proj, labels)
        l_cpr = cpr_fn(proj, labels)
        loss  = l_cls + 0.5 * l_con

        print(f"  ClassificationLoss    : {l_cls.item():.4f}")
        print(f"  ContrastiveLoss       : {l_con.item():.4f}")
        print(f"  ContrastivePrototype  : {l_cpr.item():.4f}")
        print(f"  Combined loss         : {loss.item():.4f}")

        loss.backward()
        print("\n  Backward pass OK ✓")

    except ImportError:
        print("  (losses.py not found – skipping loss check)")

    # ── dataset + balanced sampler demo ───────────────────────────────────
    import numpy as np
    X_np    = np.random.randn(400, 62, 200).astype('float32')
    y_np    = np.repeat(np.arange(4), 100)                   # 100 per class
    subj_np = np.random.randint(0, 15, 400)

    dataset = EEGDataset(X_np, y_np, subject_id=subj_np)
    sampler = BalancedBatchSampler(y_np, n_per_class=8, nb_classes=4)
    loader  = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    xb, yb, sb = next(iter(loader))
    print(f"\n  BalancedBatch – x:{tuple(xb.shape)}  "
          f"y:{tuple(yb.shape)}  subj:{tuple(sb.shape)}")
    for c in range(4):
        print(f"    class {c} count: {(yb == c).sum().item()}")

    print("\nSmoke-test complete ✓")
