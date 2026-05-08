"""
model.py

EEGNet-style spatial-spectral network for SEED / SEED-IV pre-extracted DE
features.

Expected input shape:
    (B, 1, Chans, Samples) = (B, 1, 62, 5)

Here Samples=5 refers to the DE frequency-band axis, not raw EEG time samples.
This model is therefore best described as an EEGNet-style spatial-spectral
encoder, not a raw temporal EEGNet.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import SEPCLoss


class _DepthwiseConv2d(nn.Module):
    """Depthwise Conv2d with optional per-filter max-norm constraint."""
    def __init__(
        self,
        in_channels: int,
        depth_multiplier: int,
        kernel_size: tuple[int, int],
        max_norm_val: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels * depth_multiplier
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
        )
        self.max_norm_val = max_norm_val

    def _apply_max_norm(self) -> None:
        with torch.no_grad():
            w = self.conv.weight
            norm = w.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            desired = norm.clamp(max=self.max_norm_val)
            self.conv.weight.copy_(w * desired / norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._apply_max_norm()
        return self.conv(x)


class _SeparableConv2d(nn.Module):
    """Depthwise spectral convolution followed by pointwise mixing."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int] = (0, 0),
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EEGNetEncoder(nn.Module):
    """
    EEGNet-style encoder for DE feature matrices.

    The first convolution runs along the 5-band spectral axis. The depthwise
    spatial convolution then mixes electrodes for each learned spectral filter.
    """
    def __init__(
        self,
        Chans: int = 62,
        Samples: int = 5,
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = None,
        spectral_kernel: int = 3,
        dropoutRate: float = 0.5,
    ) -> None:
        super().__init__()

        if Samples <= 0:
            raise ValueError("Samples must be positive.")
        if Chans <= 0:
            raise ValueError("Chans must be positive.")

        F2 = F2 or F1 * D
        spectral_kernel = min(spectral_kernel, Samples)
        spectral_padding = spectral_kernel // 2

        self.b1_spectral = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, spectral_kernel),
            padding=(0, spectral_padding),
            bias=False,
        )
        self.b1_bn1 = nn.BatchNorm2d(F1)
        self.b1_spatial = _DepthwiseConv2d(
            F1,
            D,
            kernel_size=(Chans, 1),
            max_norm_val=1.0,
            bias=False,
        )
        self.b1_bn2 = nn.BatchNorm2d(F1 * D)
        self.b1_drop = nn.Dropout(dropoutRate)

        self.b2_sep = _SeparableConv2d(
            F1 * D,
            F2,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=False,
        )
        self.b2_bn = nn.BatchNorm2d(F2)
        self.b2_drop = nn.Dropout(dropoutRate)

        self.embed_dim = self._infer_embed_dim(Chans, Samples)
        self._init_weights()

    def _infer_embed_dim(self, Chans: int, Samples: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self._forward_blocks(dummy)
        return out.shape[1]

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1_spectral(x)
        x = self.b1_bn1(x)
        x = self.b1_spatial(x)
        x = self.b1_bn2(x)
        x = F.elu(x)
        x = self.b1_drop(x)

        x = self.b2_sep(x)
        x = self.b2_bn(x)
        x = F.elu(x)
        x = self.b2_drop(x)

        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_blocks(x)


class ProjectionHead(nn.Module):
    """Projection MLP for contrastive losses."""
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ClassificationHead(nn.Module):
    """Linear classification head with max-norm constraint."""
    def __init__(self, in_dim: int, nb_classes: int, max_norm_val: float = 0.25) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, nb_classes)
        self.max_norm_val = max_norm_val
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _apply_max_norm(self) -> None:
        with torch.no_grad():
            w = self.fc.weight
            norm = w.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
            desired = norm.clamp(max=self.max_norm_val)
            self.fc.weight.copy_(w * desired / norm)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self._apply_max_norm()
        return self.fc(z)


class EEGNetContrastive(nn.Module):
    """
    Encoder with both classification and projection heads.

    forward(x) returns:
        logits: (B, nb_classes)
        proj:   (B, proj_out)
    """
    def __init__(
        self,
        nb_classes: int = 4,
        Chans: int = 62,
        Samples: int = 5,
        proj_hidden: int = 128,
        proj_out: int = 64,
        cls_max_norm: float = 0.25,
        **enc_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = EEGNetEncoder(Chans=Chans, Samples=Samples, **enc_kwargs)
        self.proj = ProjectionHead(self.encoder.embed_dim, proj_hidden, proj_out)
        self.cls_head = ClassificationHead(
            self.encoder.embed_dim,
            nb_classes,
            max_norm_val=cls_max_norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.cls_head(z)
        proj = self.proj(z)
        return logits, proj

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Trainer:
    """
    Joint classification + optional contrastive training loop.

    If con_loss_fn is SEPCLoss, batches must include subject_ids and the trainer
    passes subject_ids to the loss.
    """
    def __init__(
        self,
        model: nn.Module,
        cls_loss_fn: nn.Module,
        con_loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lambda_con: float = 0.5,
        warmup_epochs: int = 0,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.cls_loss_fn = cls_loss_fn
        self.con_loss_fn = con_loss_fn
        self.optimizer = optimizer
        self.lambda_con = lambda_con
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            x, labels, subject_ids = batch
            subject_ids = subject_ids.to(self.device)
        elif len(batch) == 2:
            x, labels = batch
            subject_ids = None
        else:
            raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")

        return x.to(self.device), labels.to(self.device), subject_ids

    def _contrastive_loss(
        self,
        proj: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.con_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        if isinstance(self.con_loss_fn, SEPCLoss):
            if subject_ids is None:
                raise ValueError("SEPCLoss requires subject_ids in the batch.")
            return self.con_loss_fn(proj, labels, subject_ids)

        return self.con_loss_fn(proj, labels)

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict[str, float]:
        self.model.train(train)

        total_loss = 0.0
        total_cls = 0.0
        total_con = 0.0
        correct = 0
        n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                x, labels, subject_ids = self._unpack_batch(batch)
                logits, proj = self.model(x)

                l_cls = self.cls_loss_fn(logits, labels)
                l_con = self._contrastive_loss(proj, labels, subject_ids)

                if self.con_loss_fn is not None and epoch <= self.warmup_epochs:
                    loss = l_con
                    l_cls_for_log = torch.tensor(0.0, device=self.device)
                elif self.con_loss_fn is not None:
                    loss = l_cls + self.lambda_con * l_con
                    l_cls_for_log = l_cls
                else:
                    loss = l_cls
                    l_cls_for_log = l_cls

                if train:
                    if self.optimizer is None:
                        raise RuntimeError("optimizer is required for training.")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_cls += l_cls_for_log.item() * bs
                total_con += l_con.item() * bs
                correct += (logits.argmax(dim=1) == labels).sum().item()
                n += bs

        if n == 0:
            raise RuntimeError("Empty loader.")

        return {
            "loss": total_loss / n,
            "cls_loss": total_cls / n,
            "con_loss": total_con / n,
            "acc": correct / n,
        }

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        start_epoch: int = 0,
    ) -> dict[str, list[dict[str, float]]]:
        """
        Train for epochs. start_epoch is the number of epochs already completed,
        so resumed runs preserve the warmup schedule.
        """
        history = {"train": [], "val": []}

        for local_epoch in range(1, epochs + 1):
            epoch = start_epoch + local_epoch

            tr = self._run_epoch(train_loader, train=True, epoch=epoch)
            history["train"].append(tr)

            log = (
                f"Epoch {epoch:03d} | "
                f"train_loss={tr['loss']:.4f} cls={tr['cls_loss']:.4f} "
                f"con={tr['con_loss']:.4f} acc={tr['acc']:.3f}"
            )

            if val_loader is not None:
                va = self._run_epoch(val_loader, train=False, epoch=epoch)
                history["val"].append(va)
                log += (
                    f" | val_loss={va['loss']:.4f} val_cls={va['cls_loss']:.4f} "
                    f"val_con={va['con_loss']:.4f} val_acc={va['acc']:.3f}"
                )

            print(log)

            if self.scheduler is not None:
                self.scheduler.step()

        return history

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        return self._run_epoch(loader, train=False, epoch=self.warmup_epochs + 1)


def build_model(
    nb_classes: int = 4,
    Chans: int = 62,
    Samples: int = 5,
    F1: int = 8,
    D: int = 2,
    spectral_kernel: int = 3,
    dropoutRate: float = 0.5,
    proj_hidden: int = 128,
    proj_out: int = 64,
) -> EEGNetContrastive:
    return EEGNetContrastive(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        F1=F1,
        D=D,
        spectral_kernel=spectral_kernel,
        dropoutRate=dropoutRate,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(nb_classes=4, Chans=62, Samples=5).to(device)
    x = torch.randn(32, 1, 62, 5, device=device)
    logits, proj = model(x)
    print("logits:", tuple(logits.shape))
    print("proj:  ", tuple(proj.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
