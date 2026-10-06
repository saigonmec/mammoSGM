"""Grad-CAM / Grad-CAM++ for conv feature maps (works for plain and MIL models).

Why not `register_full_backward_hook` (what the original used)?  When a layer is called more
than once per forward (the original MILv4 ran local patches and the global image through the
shared backbone in two calls) the backward hooks fire in *reverse* call order, so zipping
"activations" with "gradients" paired local activations with global gradients. Here the
gradient is captured with `tensor.register_hook` on the very activation it belongs to, so the
pairing is correct by construction no matter how often or in which order the layer is called.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CAMResult:
    cams: np.ndarray  # (K, h, w) float32, raw (ReLU-ed, un-normalised), one per image fed to the layer
    signed: np.ndarray  # (K, h, w) the same before the ReLU (plain Grad-CAM: sums to the first-order attribution)
    logits: np.ndarray  # (C,)
    probs: np.ndarray  # (C,)
    class_idx: int


def default_target_layer(model: nn.Module) -> str:
    """Name of the last conv stage of the (possibly wrapped) backbone."""
    prefix = "base_model." if hasattr(model, "base_model") else ""
    bb = model.base_model if prefix else model
    if hasattr(bb, "body"):  # TimmFeatureMaps wrapper
        bb, prefix = bb.body, prefix + "body."
    if hasattr(bb, "patch_embed") or hasattr(bb, "cls_token"):
        raise NotImplementedError(
            "Grad-CAM here needs a convolutional feature map; this looks like a ViT. Pass --target_layer "
            "pointing at a layer with (B, C, H, W) output."
        )
    if hasattr(bb, "layer4"):  # torchvision / timm ResNets
        return f"{prefix}layer4"
    for attr in ("stages", "s4", "blocks"):  # ConvNeXt, MaxViT.. | RegNet | EfficientNet
        if hasattr(bb, attr):
            m = getattr(bb, attr)
            if isinstance(m, (nn.Sequential, nn.ModuleList)):
                return f"{prefix}{attr}.{len(m) - 1}"
            return f"{prefix}{attr}"
    raise ValueError("Could not infer a target layer; pass one explicitly (see model.named_modules()).")


class CAMExtractor:
    def __init__(self, model: nn.Module, target_layer: str, method: str = "gradcam"):
        if method not in ("gradcam", "gradcam++"):
            raise ValueError("method must be 'gradcam' or 'gradcam++'")
        modules = dict(model.named_modules())
        if target_layer not in modules:
            near = [n for n in modules if target_layer.split(".")[-1] in n][:8]
            raise KeyError(f"layer '{target_layer}' not found. Similar: {near}")
        self.model, self.method = model, method
        self._acts: list = []
        self._grads: dict = {}
        self._handle = modules[target_layer].register_forward_hook(self._on_forward)

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _on_forward(self, module, inputs, output):
        if not torch.is_tensor(output) or output.dim() != 4:
            raise RuntimeError(f"target layer must output (B, C, H, W); got {getattr(output, 'shape', type(output))}")
        idx = len(self._acts)
        self._acts.append(output)  # still attached to the graph
        if output.requires_grad:
            output.register_hook(lambda g, i=idx: self._grads.__setitem__(i, g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> CAMResult:
        """x: one sample, batch size 1 ((1,3,H,W) or (1,N+1,3,H,W))."""
        if x.shape[0] != 1:
            raise ValueError("CAMExtractor handles one sample at a time")
        self._acts, self._grads = [], {}
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.detach().clone().requires_grad_(True)  # guarantees a graph even with frozen params
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits.detach(), dim=1)[0]
            if class_idx is None:
                class_idx = int(logits[0].argmax())
            logits[0, class_idx].backward()
        if not self._acts or len(self._grads) != len(self._acts):
            raise RuntimeError("target layer is not on the path from input to logits (no gradient captured)")
        acts = torch.cat([a.detach() for a in self._acts], dim=0)  # (K, C, h, w), in call order
        grads = torch.cat([self._grads[i] for i in range(len(self._acts))], dim=0)
        signed = self._gradcam(acts, grads) if self.method == "gradcam" else self._gradcam_pp(acts, grads)
        return CAMResult(torch.relu(signed).cpu().numpy().astype(np.float32), signed.cpu().numpy().astype(np.float32),
                         logits[0].detach().cpu().numpy(), probs.cpu().numpy(), class_idx)

    @staticmethod
    def _gradcam(acts, grads):
        weights = grads.mean(dim=(2, 3), keepdim=True)
        return (weights * acts).sum(dim=1)  # pre-ReLU

    @staticmethod
    def _gradcam_pp(acts, grads):
        g2, g3 = grads**2, grads**3
        denom = 2.0 * g2 + acts.sum(dim=(2, 3), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        return (weights * acts).sum(dim=1)  # pre-ReLU
