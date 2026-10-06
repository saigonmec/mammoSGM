"""Loss functions (CE / Focal / Focal+label-smoothing / LDAM) and a small factory."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss. `alpha`: None | float | per-class weights [C]."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha = self.alpha.to(loss.device, loss.dtype)
            loss = loss * (alpha if alpha.ndim == 0 else alpha[targets])
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss2(nn.Module):
    """Focal loss + label smoothing (gamma=0 -> smoothed CE, smoothing=0 -> plain focal)."""

    def __init__(self, gamma=2.0, smoothing=0.1, alpha=None, reduction="mean", eps=1e-6):
        super().__init__()
        assert gamma >= 0.0 and 0.0 <= smoothing < 1.0
        assert reduction in ("mean", "sum", "none")
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self.reduction = reduction
        self.eps = float(eps)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        B, C = logits.shape
        assert C > 1, "num_classes must be >= 2 for label smoothing"
        targets = targets.long()
        with torch.no_grad():
            true_dist = torch.full(
                (B, C), self.smoothing / (C - 1), device=logits.device, dtype=logits.dtype
            )
            true_dist.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp().clamp(min=self.eps, max=1.0 - self.eps)
        focal = (1.0 - p) ** self.gamma if self.gamma > 0 else torch.ones_like(p)
        per_class = -true_dist * focal * logp
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device, logits.dtype)
            per_class = per_class * (alpha if alpha.ndim == 0 else alpha.unsqueeze(0))
        loss = per_class.sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """LDAM loss (Cao et al. 2019). Margins m_j ~ n_j^{-1/4}, rescaled so max margin = max_m.

    Note: the paper applies the margin to *normalised* (cosine) logits with scale s=30.
    With a plain linear head the logits are unnormalised, so `s` acts as a very sharp
    temperature; lower `s` (e.g. 1-5) if training is unstable.
    """

    def __init__(self, cls_num_list, max_m: float = 0.5, weight=None, s: float = 30.0):
        super().__init__()
        assert len(cls_num_list) > 0 and all(n > 0 for n in cls_num_list)
        self.s = float(s)
        n = torch.tensor(cls_num_list, dtype=torch.float32)
        m = 1.0 / torch.sqrt(torch.sqrt(n))
        m = m * (float(max_m) / m.max().clamp_min(1e-6))
        self.register_buffer("m_list", m)
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, targets):
        B, C = logits.shape
        assert C == self.m_list.numel(), "logits/class-count mismatch"
        targets = targets.long()
        batch_m = self.m_list[targets].to(logits.dtype)
        logits_m = logits.clone()
        logits_m[torch.arange(B, device=logits.device), targets] -= batch_m
        weight = None if self.weight is None else self.weight.to(logits.device, logits.dtype)
        return F.cross_entropy(self.s * logits_m, targets, weight=weight)


def build_criterion(loss_type: str, class_weights=None, class_counts=None) -> nn.Module:
    """class_weights: tensor [C] or None (only pass it when balancing through the loss)."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights)
    if loss_type == "focal2":
        return FocalLoss2(alpha=class_weights)
    if loss_type == "ldam":
        if class_counts is None:
            raise ValueError("ldam needs class_counts")
        return LDAMLoss(cls_num_list=[max(int(c), 1) for c in class_counts], weight=class_weights)
    raise ValueError(f"Unknown loss_type: {loss_type}")
