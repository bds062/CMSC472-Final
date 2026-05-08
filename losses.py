"""
losses.py

Loss functions for EEG emotion recognition on pre-extracted DE features.

This file intentionally contains four losses:
  1. ClassificationLoss: cross-entropy over class logits.
  2. ContrastiveLoss: supervised contrastive loss over projection embeddings.
  3. PrototypeLoss: class-prototype loss using batch class prototypes.
  4. SEPCLoss: subject-excluded prototype contrastive loss.

SEPCLoss requires subject IDs. It is not equivalent to ordinary leave-one-out
sample contrastive learning; it excludes all samples from the anchor subject
when constructing prototypes.
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
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
            # Differentiable zero, useful for rare degenerate batches.
            return features.sum() * 0.0

        loss_per_anchor = -(
            log_prob[valid] * positive_mask[valid].float()
        ).sum(dim=1) / positive_counts[valid].float()

        return loss_per_anchor.mean()


class PrototypeLoss(nn.Module):
    """
    Batch class-prototype loss.

    A prototype is computed for each class from the mean of normalized embeddings
    in that class. Each sample is then classified by similarity to all available
    class prototypes.

    Args:
        features: Tensor of shape (B, D).
        labels: Tensor of shape (B,).
    """
    def __init__(self, num_classes: int, temperature: float = 0.1) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_classes = num_classes
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, D), got {tuple(features.shape)}")

        device = features.device
        z = F.normalize(features, dim=1)

        prototypes = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                p = z[mask].mean(dim=0)
                p = F.normalize(p, dim=0)
            else:
                # Missing classes are given a very low logit below.
                p = torch.zeros(z.size(1), device=device, dtype=z.dtype)
            prototypes.append(p)

        prototypes = torch.stack(prototypes, dim=0)
        logits = torch.matmul(z, prototypes.T) / self.temperature

        # Prevent absent-class zero prototypes from becoming accidental attractors.
        present = torch.stack([(labels == c).any() for c in range(self.num_classes)]).to(device)
        logits[:, ~present] = -1e9

        return self.loss_fn(logits, labels)


class SEPCLoss(nn.Module):
    """
    Subject-Excluded Prototype Contrastive loss.

    For anchor i with subject s_i, class c prototype p_c^(-s_i) is formed using
    only samples whose subject_id != s_i and label == c. The anchor is classified
    by similarity to these subject-excluded class prototypes.

    This is the loss that matches the SEPC idea. It requires subject_ids.

    Args:
        features: Tensor of shape (B, D).
        labels: Tensor of shape (B,).
        subject_ids: Tensor of shape (B,).
    """
    def __init__(self, num_classes: int, temperature: float = 0.1) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_classes = num_classes
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        if subject_ids is None:
            raise ValueError("SEPCLoss requires subject_ids.")
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, D), got {tuple(features.shape)}")

        device = features.device
        z = F.normalize(features, dim=1)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)

        batch_size = z.size(0)
        logits = torch.full(
            (batch_size, self.num_classes),
            fill_value=-1e9,
            dtype=z.dtype,
            device=device,
        )

        valid_anchor = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(batch_size):
            anchor_subject = subject_ids[i]

            for c in range(self.num_classes):
                mask = (labels == c) & (subject_ids != anchor_subject)
                if not mask.any():
                    continue

                prototype = z[mask].mean(dim=0)
                prototype = F.normalize(prototype, dim=0)
                logits[i, c] = torch.dot(z[i], prototype) / self.temperature

            # Keep only anchors whose own class has an excluded-subject prototype.
            valid_anchor[i] = logits[i, labels[i]] > -1e8

        if valid_anchor.sum() == 0:
            return features.sum() * 0.0

        return self.loss_fn(logits[valid_anchor], labels[valid_anchor])
