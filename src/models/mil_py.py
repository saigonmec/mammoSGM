"""MILClassifierV4 — attention-MIL over local patches + a global (whole-image) view.

Input  : (B, N+1, C, H, W); items [0..N-1] are local patches, item N is the global image.
Output : logits (B, num_classes); with `return_aux=True` also a dict with the attention
         weights / fusion weights (no state is stored on the module, so it is safe with
         DataParallel and with several concurrent forwards).

Differences vs the original implementation (parameter names are unchanged, so checkpoints
trained with fusion="fuse" load as-is):

* ONE backbone call for all N+1 images. The original ran local and global through the shared
  backbone in two separate calls, so BatchNorm saw two different distributions per step and
  its running statistics were a blend that matched neither (train/eval mismatch). It also
  made the Grad-CAM hook fire twice (see gradcam/cam.py).
* Padding mask is NaN-safe (an all-masked bag gives a zero vector instead of NaN).
* `cross_attention` honours the mask, returns its attention as `aux["attn"]`, and no longer
  allocates an unused attention-pool.
* Non-4D backbone outputs are handled explicitly and the feature size is checked.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GatedAttnPool(nn.Module):
    """Gated attention pooling (Ilse et al., 2018). mask: bool (B, N), True = ignore."""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(d_model, hidden)
        self.U = nn.Linear(d_model, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        for m in (self.V, self.U, self.w):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, temperature: float = 1.0):
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        scores = self.w(self.drop(v * u)).squeeze(-1).float() / max(temperature, 1e-6)
        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)  # all-masked -> zeros
        pooled = torch.bmm(attn.unsqueeze(1).to(x.dtype), x).squeeze(1)
        return pooled, attn


class _MILBase(nn.Module):
    """Shared by V4 / V4a: ONE backbone call for all N+1 images of every bag -> (B, N+1, feature_dim)."""

    supports_aux = True  # forward(..., return_aux=True) returns (logits, aux dict)

    def _vectorize(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 4:  # (B, C, h, w) feature map -> global average pool
            feats = feats.mean(dim=(2, 3))
        elif feats.dim() == 3:  # (B, tokens, C) -> mean token
            feats = feats.mean(dim=1)
        if feats.dim() != 2 or feats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"backbone produced features of shape {tuple(feats.shape)}, expected (*, {self.feature_dim})"
            )
        return feats

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.shape[1] < 2:
            raise ValueError(f"expected (B, N+1, C, H, W) with N >= 1, got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        return self._vectorize(self.base_model(x.reshape(B * T, C, H, W))).view(B, T, -1)


class MILClassifierV4(_MILBase):
    FUSIONS = ("fuse", "concat", "cross_attention")

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        num_classes: int = 2,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        cross_attn_heads: int = 4,
        fusion: str = "fuse",
    ):
        super().__init__()
        if fusion not in self.FUSIONS:
            raise ValueError(f"fusion must be one of {self.FUSIONS}, got '{fusion}'")
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion = fusion

        if fusion == "cross_attention":
            # global feature = query, local patches = keys/values
            self.cross_attn = nn.MultiheadAttention(feature_dim, cross_attn_heads, batch_first=True)
            self.global_proj = nn.Linear(feature_dim, feature_dim)
        else:
            self.mil_pool = GatedAttnPool(feature_dim, attn_hidden, attn_dropout)
            if fusion == "concat":
                self.fusion_proj = nn.Sequential(
                    nn.LayerNorm(feature_dim * 2),
                    nn.Dropout(head_dropout),
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.ReLU(inplace=True),
                )
            else:  # "fuse": learned convex combination of local / global feature
                self.fusion_gate = nn.Sequential(
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim, 2),
                    nn.Softmax(dim=-1),
                )

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

        linears = [m for m in self.head if isinstance(m, nn.Linear)]
        if fusion == "cross_attention":
            linears.append(self.global_proj)
        elif fusion == "concat":
            linears += [m for m in self.fusion_proj if isinstance(m, nn.Linear)]
        else:
            linears += [m for m in self.fusion_gate if isinstance(m, nn.Linear)]
        for m in linears:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_aux: bool = False):
        feats = self._encode(x)  # (B, N+1, D)
        N = feats.shape[1] - 1  # item N is the global image
        feats_l, feats_g = feats[:, :N], feats[:, N]

        fusion_w = None
        if self.fusion == "cross_attention":
            kpm = None
            if mask is not None:
                kpm = mask.bool().clone()
                kpm[kpm.all(dim=1)] = False  # an entirely masked bag would give NaN
            q = self.global_proj(feats_g).unsqueeze(1)
            fused, w = self.cross_attn(q, feats_l, feats_l, key_padding_mask=kpm, need_weights=True)
            fused = fused.squeeze(1)
            attn = w.squeeze(1)  # (B, N), averaged over heads
            pooled_l = None
        else:
            pooled_l, attn = self.mil_pool(feats_l, mask=mask)
            if self.fusion == "concat":
                fused = self.fusion_proj(torch.cat([pooled_l, feats_g], dim=1))
            else:
                fusion_w = self.fusion_gate(torch.cat([pooled_l, feats_g], dim=-1))
                fused = fusion_w[:, 0:1] * pooled_l + fusion_w[:, 1:2] * feats_g

        logits = self.head(fused)
        if not return_aux:
            return logits
        aux = {
            "attn": attn.detach(),  # (B, N) weight of every local patch
            "fusion_weights": None if fusion_w is None else fusion_w.detach(),  # (B, 2) [local, global]
            "local_feat": None if pooled_l is None else pooled_l.detach(),
            "global_feat": feats_g.detach(),
        }
        return logits, aux


def to_cells(feats: torch.Tensor, channels: int):
    """Backbone output -> (B, M, C) instances ("cells") + grid (h, w) or None.
    (B, C, h, w) CNN map -> M = h*w; (B, h, w, C) channels-last (Swin) -> same; (B, T, C) ViT tokens -> M = T;
    (B, C) pooled vector -> M = 1 (degenerate, kept so pooled backbones still work)."""
    if feats.dim() == 4:
        if feats.shape[1] != channels and feats.shape[-1] == channels:
            feats = feats.permute(0, 3, 1, 2)
        B, C, h, w = feats.shape
        return feats.flatten(2).transpose(1, 2), (h, w)
    if feats.dim() == 3:
        return feats, None
    if feats.dim() == 2:
        return feats.unsqueeze(1), None
    raise ValueError(f"unsupported backbone output shape {tuple(feats.shape)}")


def top_t_pool(scores: torch.Tensor, t: float) -> torch.Tensor:
    """GMIC top-t% pooling: scores (B, M, C) -> (B, C), mean of the ceil(t*M) largest values per class."""
    M = scores.shape[1]
    k = max(1, min(M, int(math.ceil(t * M))))
    return scores.topk(k, dim=1).values.mean(dim=1)


class MILClassifierV4a(_MILBase):
    """V4a — GMIC with fixed strips: every feature-map cell is a MIL instance, deeply supervised.

    Same input contract as V4 ((B, N+1, C, H, W): N local strips + the whole image, ONE shared backbone
    call) but the backbone returns the last feature map, so an image of 448 px gives 14x14 = 196 cells.

        backbone -> cells f (M per image, C channels)
        neck_local / neck_global : Linear(C, d) - LayerNorm - GELU per cell       (same scale for both)
        saliency                 : Linear(d, num_classes) per cell, shared        (GMIC saliency map)
        GLOBAL branch (GMIC "global module"):
            y_global = top-t% pooling of the global saliency map;  z_g = GAP of the global cells
        LOCAL branch (GMIC "local module" / ABMIL):
            gated attention over ALL N*M strip cells -> z_l ;  y_local = head_local(z_l)
            y_local_sal = top-t% pooling of the strip saliency map   (instance-level supervision)
        FUSION : y = Linear([z_l ; z_g])  (one linear layer, as in GMIC); branch dropout while training
        LOSS (engine): CE(y) + aux_weight * [CE(y_global) + CE(y_local) + CE(y_local_sal)]

    Compared with the first V4a (strip-level attention, pooled vectors):
      * instances are cells, so attention has hundreds of candidates instead of 3-4 (which made it ~uniform)
        and gives a fine attention map for free (aux["attn_cells"]);
      * the saliency map is a *trained* localisation map (GMIC), not only a post-hoc Grad-CAM, and top-t
        pooling keeps small lesions from being averaged away (GAP dilutes a 2-cell lesion in 196 cells);
      * both branches get instance-level and bag-level supervision -> no dead branch, no saturating gate.
    Not done on purpose: strip positional embeddings (vertical flips in the augmentation), GMIC's
    saliency-guided ROI retrieval (strips already cover the image at higher resolution), GMIC's L1 sparsity.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        num_classes: int = 2,
        neck_dim: int = 512,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.2,
        branch_dropout: float = 0.2,
        aux_weight: float = 0.5,
        top_t: float = 0.05,
    ):
        super().__init__()
        if not 0.0 <= branch_dropout <= 0.5:
            raise ValueError("branch_dropout must be in [0, 0.5] (each branch is dropped with this probability)")
        if not 0.0 < top_t <= 1.0:
            raise ValueError("top_t must be in (0, 1]")
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.neck_dim = neck_dim
        self.branch_dropout = branch_dropout
        self.top_t = top_t

        def neck():
            return nn.Sequential(nn.Linear(feature_dim, neck_dim), nn.LayerNorm(neck_dim), nn.GELU())

        self.neck_local, self.neck_global = neck(), neck()
        self.mil_pool = GatedAttnPool(neck_dim, attn_hidden, attn_dropout)
        self.saliency = nn.Linear(neck_dim, num_classes)
        self.head_local = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(neck_dim, num_classes))
        self.head_fusion = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(2 * neck_dim, num_classes))
        # the training engine adds aux_weight * CE(aux[key], y) for every key listed here
        self.aux_loss_weights = (
            {"logits_global": aux_weight, "logits_local": aux_weight, "logits_local_sal": aux_weight}
            if aux_weight > 0 else {}
        )
        for m in (self.neck_local[0], self.neck_global[0], self.saliency, self.head_local[1], self.head_fusion[1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_aux: bool = False):
        if x.dim() != 5 or x.shape[1] < 2:
            raise ValueError(f"expected (B, N+1, C, H, W) with N >= 1, got {tuple(x.shape)}")
        B, T = x.shape[:2]
        N = T - 1
        cells, grid = to_cells(self.base_model(x.reshape(B * T, *x.shape[2:])), self.feature_dim)
        if cells.shape[-1] != self.feature_dim:
            raise ValueError(f"backbone produced {cells.shape[-1]} channels, expected {self.feature_dim}")
        M = cells.shape[1]
        cells = cells.view(B, T, M, -1)

        e_l = self.neck_local(cells[:, :N].reshape(B, N * M, -1))  # (B, N*M, d)
        e_g = self.neck_global(cells[:, N])  # (B, M, d)
        cell_mask = None if mask is None else mask.bool().repeat_interleave(M, dim=1)  # strip mask -> cells

        z_l, attn_cells = self.mil_pool(e_l, mask=cell_mask)  # (B, d), (B, N*M)
        s_l, s_g = self.saliency(e_l), self.saliency(e_g)  # per-cell class scores
        if cell_mask is not None:
            s_l = s_l.masked_fill(cell_mask[..., None], -1e4)
        logits_global = top_t_pool(s_g, self.top_t)
        logits_local_sal = top_t_pool(s_l, self.top_t)
        logits_local = self.head_local(z_l)
        z_g = e_g.mean(dim=1)

        zl_in, zg_in = z_l, z_g
        if self.training and self.branch_dropout > 0:  # drop the local branch w.p. p, the global one w.p. p
            r = torch.rand(B, device=z_l.device)
            p = self.branch_dropout
            zl_in = z_l * (r >= p).to(z_l.dtype)[:, None]
            zg_in = z_g * ((r < p) | (r >= 2 * p)).to(z_g.dtype)[:, None]
        logits = self.head_fusion(torch.cat([zl_in, zg_in], dim=1))
        if not return_aux:
            return logits

        w = self.head_fusion[1].weight  # (C, 2d): logits = W_l z_l + W_g z_g + b  (exactly additive)
        attn_strips = attn_cells.view(B, N, M).sum(dim=2)  # attention mass per strip (sums to 1)
        aux = {
            "attn": attn_strips.detach(),  # (B, N)
            "attn_cells": attn_cells.detach().view(B, N, *grid) if grid else attn_cells.detach().view(B, N, M),
            "saliency_global": (s_g.detach().transpose(1, 2).reshape(B, self.num_classes, *grid) if grid
                                else s_g.detach().transpose(1, 2)),  # (B, C, h, w)
            "saliency_local": (s_l.detach().view(B, N, M, self.num_classes).permute(0, 1, 3, 2).reshape(
                B, N, self.num_classes, *grid) if grid else s_l.detach().view(B, N, M, self.num_classes).permute(0, 1, 3, 2)),
            "fusion_weights": None,  # no gate in V4a
            "local_feat": z_l.detach(),
            "global_feat": z_g.detach(),
            "logits_local": logits_local,  # with grad: the engine trains on them
            "logits_global": logits_global,
            "logits_local_sal": logits_local_sal,
            "contrib_local": (z_l @ w[:, : self.neck_dim].T).detach(),  # (B, C) share of the logits per branch
            "contrib_global": (z_g @ w[:, self.neck_dim:].T).detach(),
        }
        return logits, aux
