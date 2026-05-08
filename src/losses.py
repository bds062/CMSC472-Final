"""
losses.py

Loss functions for EEG emotion recognition on pre-extracted DE features.

This file contains four losses:
  1. ClassificationLoss: cross-entropy over class logits.
  2. ContrastiveLoss: supervised contrastive loss over projection embeddings.
  3. PrototypeLoss: EMA class-prototype loss using per-subject-class prototypes
     averaged over all training subjects (global prototype).
  4. SEPCLoss: subject-excluded prototype contrastive loss using per-subject-class
     EMA prototypes averaged over all subjects except the anchor's subject.

Both PrototypeLoss and SEPCLoss maintain per-subject-class prototypes via
exponential moving average. The only difference is whether the anchor subject
is included (PrototypeLoss) or excluded (SEPCLoss) when forming the class
prototype target. This makes the PrototypeLoss → SEPCLoss comparison a clean
ablation of subject exclusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Standard cross-entropy loss.

    Args:
        logits: Tensor of shape (B, C).
        labels: Tensor of shape (B,).
    """
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, labels)


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss.

    For each anchor, positives are other samples in the batch with the same
    class label. The anchor itself is excluded from both positives and the
    denominator.

    Args:
        features: Tensor of shape (B, D), usually projection-head outputs.
        labels: Tensor of shape (B,).
        subject_ids: Tensor of shape (B,). Accepted for API uniformity but
            not used by this loss.
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, D), got {tuple(features.shape)}")

        device = features.device
        batch_size = features.size(0)

        z = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = torch.eq(labels, labels.T) & (~self_mask)

        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Denominator: all non-self samples.
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0

        if valid.sum() == 0:
            return features.sum() * 0.0

        loss_per_anchor = -(
            log_prob[valid] * positive_mask[valid].float()
        ).sum(dim=1) / positive_counts[valid].float()

        return loss_per_anchor.mean()


# ---------------------------------------------------------------------------
# Shared EMA prototype infrastructure
# ---------------------------------------------------------------------------


class _EMAPrototypeBase(nn.Module):
    """
    Base class for losses that maintain per-subject-class EMA prototypes.

    Stores a prototype tensor of shape (num_subjects, num_classes, D) and a
    boolean mask tracking which (subject, class) pairs have been initialized.
    Prototypes are lazily allocated on first forward so that embed_dim need
    not be specified at construction time.
    """

    def __init__(
        self,
        num_classes: int,
        num_subjects: int,
        temperature: float = 0.1,
        ema_alpha: float = 0.9,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if num_subjects <= 1:
            raise ValueError("num_subjects must be at least 2.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 < ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in (0, 1).")

        self.num_classes = num_classes
        self.num_subjects = num_subjects
        self.temperature = temperature
        self.ema_alpha = ema_alpha
        self.ce = nn.CrossEntropyLoss()

        # Lazily initialized on first forward call.
        self.prototypes: torch.Tensor | None = None
        self.initialized: torch.Tensor | None = None

    # -- buffer management --------------------------------------------------

    def _ensure_buffers(self, dim: int, device: torch.device) -> None:
        """Create or migrate prototype storage to the correct device."""
        if self.prototypes is None:
            self.prototypes = torch.zeros(
                self.num_subjects, self.num_classes, dim, device=device,
            )
            self.initialized = torch.zeros(
                self.num_subjects, self.num_classes, dtype=torch.bool, device=device,
            )
        elif self.prototypes.device != device:
            self.prototypes = self.prototypes.to(device)
            self.initialized = self.initialized.to(device)

    def reset_prototypes(self) -> None:
        """Clear all stored prototypes. Useful between training runs."""
        self.prototypes = None
        self.initialized = None

    # -- EMA update ---------------------------------------------------------

    @torch.no_grad()
    def _update_prototypes(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> None:
        """
        Update per-subject-class EMA prototypes from the current batch.

        z should already be L2-normalized.
        """
        for s in subject_ids.unique():
            s_idx = s.item()
            if s_idx >= self.num_subjects:
                raise ValueError(
                    f"subject_id {s_idx} >= num_subjects {self.num_subjects}."
                )
            for c in range(self.num_classes):
                mask = (subject_ids == s) & (labels == c)
                if not mask.any():
                    continue
                batch_mean = z[mask].mean(dim=0)
                if self.initialized[s_idx, c]:
                    self.prototypes[s_idx, c] = (
                        self.ema_alpha * self.prototypes[s_idx, c]
                        + (1.0 - self.ema_alpha) * batch_mean
                    )
                else:
                    self.prototypes[s_idx, c] = batch_mean
                    self.initialized[s_idx, c] = True
                self.prototypes[s_idx, c] = F.normalize(
                    self.prototypes[s_idx, c], dim=0,
                )


class PrototypeLoss(_EMAPrototypeBase):
    """
    Global EMA prototype loss.

    Maintains per-subject-class prototypes via EMA. The class prototype for
    any anchor is the mean of *all* subject prototypes for that class (no
    exclusion). This serves as the ablation baseline for SEPCLoss.

    Args:
        features: Tensor of shape (B, D).
        labels: Tensor of shape (B,).
        subject_ids: Tensor of shape (B,).
    """

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if subject_ids is None:
            raise ValueError("PrototypeLoss requires subject_ids for EMA prototypes.")
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, D), got {tuple(features.shape)}")

        device = features.device
        z = F.normalize(features, dim=1)

        self._ensure_buffers(z.size(1), device)
        self._update_prototypes(z.detach(), labels, subject_ids)

        # Global class prototypes: average over all initialized subjects.
        protos = []
        for c in range(self.num_classes):
            mask_c = self.initialized[:, c]
            if mask_c.any():
                p = self.prototypes[mask_c, c].mean(dim=0)
                p = F.normalize(p, dim=0)
            else:
                p = torch.zeros(z.size(1), device=device, dtype=z.dtype)
            protos.append(p)
        protos = torch.stack(protos, dim=0)  # (C, D)

        logits = (z @ protos.T) / self.temperature  # (B, C)

        # Mask out classes with no prototype.
        present = torch.stack(
            [self.initialized[:, c].any() for c in range(self.num_classes)],
        ).to(device)
        logits[:, ~present] = -1e9

        return self.ce(logits, labels)


class SEPCLoss(_EMAPrototypeBase):
    """
    Subject-Excluded Prototype Contrastive loss.

    For anchor i with subject s_i, the class c prototype is the mean of
    per-subject-class EMA prototypes from subjects s != s_i. This removes
    the current subject's direct contribution to the contrastive target.

    Args:
        features: Tensor of shape (B, D).
        labels: Tensor of shape (B,).
        subject_ids: Tensor of shape (B,).
    """

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if subject_ids is None:
            raise ValueError("SEPCLoss requires subject_ids.")
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, D), got {tuple(features.shape)}")

        device = features.device
        z = F.normalize(features, dim=1)
        batch_size = z.size(0)

        self._ensure_buffers(z.size(1), device)
        self._update_prototypes(z.detach(), labels, subject_ids)

        logits = torch.full(
            (batch_size, self.num_classes), fill_value=-1e9,
            dtype=z.dtype, device=device,
        )

        # For each unique subject in the batch, compute excluded prototypes
        # and assign logits to all anchors from that subject.  This iterates
        # over subjects (≤14 in SEED-IV training) rather than individual
        # samples, which is much faster than the naive per-sample loop.
        for s in subject_ids.unique():
            s_idx = s.item()
            anchor_mask = subject_ids == s

            for c in range(self.num_classes):
                excl_mask = self.initialized[:, c].clone()
                excl_mask[s_idx] = False
                if not excl_mask.any():
                    continue

                proto = self.prototypes[excl_mask, c].mean(dim=0)
                proto = F.normalize(proto, dim=0)
                logits[anchor_mask, c] = (z[anchor_mask] @ proto) / self.temperature

        # Keep only anchors whose own class has a valid excluded prototype.
        valid = logits[torch.arange(batch_size, device=device), labels] > -1e8

        if valid.sum() == 0:
            return features.sum() * 0.0

        return self.ce(logits[valid], labels[valid])
