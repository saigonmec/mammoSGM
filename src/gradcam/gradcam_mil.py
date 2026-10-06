"""Reference ("standard") Grad-CAM for the MIL model — the one to compare other heat-maps against.

    python -m src.gradcam.gradcam_mil --checkpoint output/<run>/best.pth --image a.png --out_dir out
    python -m src.gradcam.gradcam_mil --checkpoint mil.pth --data_folder DATA --n_samples 5 \\
        --compare_with based.pth other_mil.pth --out_dir out          # + compare.csv

Definition (what makes it a usable reference)
---------------------------------------------
The model sees N local strips and the whole image (the global item) through ONE backbone call, and
fuses them with attention / a gate. For a target class c and the last conv stage A (K channels) of
image i in {strips, global}:

    alpha_k^i = mean_xy dy_c/dA_k^i(x,y)        cell_i(x,y) = ReLU( sum_k alpha_k^i * A_k^i(x,y) )

* the gradient already contains the attention weight and the fusion weight of that branch, so no
  extra weighting is needed and a branch the model ignores gets (near) zero heat;
* with global-average pooling, sum_{i,x,y} sum_k alpha_k A_k (before the ReLU) equals
  sum_i <dy_c/df_i, f_i>, the first-order ("gradient x input") attribution of the logit to the pooled
  features: all N+1 maps live on ONE scale. `signed_total` exposes this number (the identity is tested
  against autograd with a ResNet). Caveat: a backbone that ends with a LayerNorm after the pooling
  (ConvNeXt) is scale-invariant in f, so by Euler's theorem the total is ~0 there (e.g. 2e-7): the
  identity still holds but says nothing; rely on the (ReLU-ed) maps and `share_local` instead;
* maps are converted to attribution per unit image area ("density") before being pasted on the image,
  so a strip (finer cells) and the global image (coarser cells) are directly comparable, and the maps do
  not depend on the display resolution;
* `map_local` = feather-blended strips (no seams), `map_global`, and `map_combined` = their sum.
  Values are raw; use `result.normalized(...)` for [0, 1] versions. Nothing is normalised per branch.

Everything needed to redo a comparison later is saved in the .npz (`MILCAMResult.save/load`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from src.data.imaging import load_rgb, rotate_box_ccw
from src.data.inference import ModelInput, build_input
from src.gradcam.cam import CAMExtractor, default_target_layer
from src.gradcam.cam_metrics import localization, similarity
from src.gradcam.viz import render_mil, resize_map, stitch_local_cams
from src.models.checkpoint import build_model_from_checkpoint
from src.models.factory import is_mil
from src.utils.common import get_device, set_seed

FORMAT_VERSION = 1


@dataclass
class MILCAMResult:
    image: np.ndarray  # uint8 RGB; geometry of every map below (cropped / rotated like the model saw it, <= max_side)
    probs: np.ndarray  # (C,)
    pred: int
    class_idx: int  # class the maps explain
    method: str
    layer: str
    boxes: list  # [(y0, y1)] strip rows in `image` pixels
    attn: np.ndarray  # (N,) MIL attention of the strips
    fusion_weights: np.ndarray | None  # (2,) [local, global] (fuse mode only)
    cells_local: np.ndarray  # (N, h, w) raw post-ReLU cell values
    cells_global: np.ndarray  # (h, w)
    signed_local: np.ndarray  # (N, h, w) pre-ReLU
    signed_global: np.ndarray  # (h, w)
    dens_local: np.ndarray  # (N, h, w) attribution per unit image area
    dens_global: np.ndarray  # (h, w)
    map_local: np.ndarray  # (H, W) feather-blended strips, density
    map_global: np.ndarray  # (H, W)
    map_combined: np.ndarray  # (H, W) map_local + map_global
    crop_offset: tuple  # (x0, y0) of the foreground crop in the original image
    rotated: bool  # image was rotated 90deg CCW (landscape -> portrait) before the model
    src_hw: tuple  # (H, W) of the cropped source image at full resolution, before the rotation
    display_scale: float  # `image` px per full-resolution (rotated) px
    learned: dict = field(default_factory=dict)  # maps the model computes itself (V4a): "saliency_global",
    # "saliency_local" (class_idx channel, pasted into the image), "attention_local" (cell attention, density)

    # ---- derived numbers
    @property
    def share_local(self) -> float:
        """Share of the total positive attribution that comes from the local strips."""
        loc, glo = float(self.cells_local.sum()), float(self.cells_global.sum())
        return loc / (loc + glo) if loc + glo > 0 else float("nan")

    @property
    def signed_total(self) -> float:
        """sum of all pre-ReLU cell values == sum_i <dy/df_i, f_i> for plain Grad-CAM (see module doc;
        ~0 for LayerNorm-terminated backbones such as ConvNeXt)."""
        return float(self.signed_local.sum() + self.signed_global.sum())

    def strip_maps01(self):
        """(local (N,h,w), global (h,w)) in [0,1] with ONE common maximum (what render_mil expects)."""
        hi = max(float(self.dens_local.max()), float(self.dens_global.max()), 1e-12)
        return self.dens_local / hi, self.dens_global / hi

    def normalized(self, which: str = "combined", mode: str = "max") -> np.ndarray:
        """A full-image map in [0,1]. which: combined | local | global. mode: max | minmax | none."""
        m = {"combined": self.map_combined, "local": self.map_local, "global": self.map_global}[which]
        m = np.maximum(m, 0)
        if mode == "none":
            return m
        if mode == "minmax":
            lo, hi = float(m.min()), float(m.max())
            return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
        return m / m.max() if m.max() > 0 else np.zeros_like(m)

    def in_source_orientation(self, m: np.ndarray) -> np.ndarray:
        """Undo the landscape->portrait rotation so the map lines up with the (cropped) source image."""
        return np.ascontiguousarray(np.rot90(m, k=-1)) if self.rotated else m

    def boxes_to_display(self, boxes_xywh, source_orientation: bool = False):
        """Lesion boxes [[x, y, w, h]] in ORIGINAL image pixels -> pixels of `image`
        (or of the un-rotated image if source_orientation)."""
        ox, oy = self.crop_offset
        out = []
        for x, y, w, h in boxes_xywh or []:
            b = [x - ox, y - oy, w, h]
            if self.rotated and not source_orientation:
                b = rotate_box_ccw(b, self.src_hw[1])
            out.append([v * self.display_scale for v in b])
        return out

    # ---- persistence
    def save(self, path: str) -> None:
        meta = dict(version=FORMAT_VERSION, pred=self.pred, class_idx=self.class_idx, method=self.method,
                    layer=self.layer, crop_offset=list(self.crop_offset), rotated=self.rotated,
                    src_hw=list(self.src_hw), display_scale=self.display_scale, share_local=self.share_local,
                    signed_total=self.signed_total)
        arrays = {k: getattr(self, k) for k in (
            "image", "probs", "attn", "cells_local", "cells_global", "signed_local", "signed_global",
            "dens_local", "dens_global", "map_local", "map_global", "map_combined")}
        arrays.update({f"learned_{k}": v for k, v in self.learned.items()})
        if self.fusion_weights is not None:
            arrays["fusion_weights"] = self.fusion_weights
        np.savez_compressed(path, boxes=np.asarray(self.boxes, np.int32), meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path: str) -> "MILCAMResult":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(
            image=z["image"], probs=z["probs"], pred=meta["pred"], class_idx=meta["class_idx"],
            method=meta["method"], layer=meta["layer"], boxes=[tuple(b) for b in z["boxes"].tolist()],
            attn=z["attn"], fusion_weights=z["fusion_weights"] if "fusion_weights" in z.files else None,
            cells_local=z["cells_local"], cells_global=z["cells_global"], signed_local=z["signed_local"],
            signed_global=z["signed_global"], dens_local=z["dens_local"], dens_global=z["dens_global"],
            map_local=z["map_local"], map_global=z["map_global"], map_combined=z["map_combined"],
            crop_offset=tuple(meta["crop_offset"]), rotated=meta["rotated"], src_hw=tuple(meta["src_hw"]),
            display_scale=meta["display_scale"],
            learned={k[len("learned_"):]: z[k] for k in z.files if k.startswith("learned_")},
        )


class MILGradCAM:
    def __init__(self, model, settings: dict, target_layer: str | None = None, method: str = "gradcam", device=None):
        if not is_mil(settings["arch_type"]):
            raise ValueError(f"MILGradCAM needs a MIL model, got arch_type={settings['arch_type']!r}")
        self.model, self.settings, self.method = model, settings, method
        self.layer = target_layer or default_target_layer(model)
        self.device = device if device is not None else next(model.parameters()).device

    def explain(self, img: np.ndarray, class_idx: int | None = None, crop_foreground: bool = False,
                max_side: int = 1024) -> MILCAMResult:
        """img: uint8 RGB (original resolution)."""
        return self.explain_input(build_input(img, self.settings, crop_foreground), class_idx, max_side)

    def explain_input(self, inp: ModelInput, class_idx: int | None = None, max_side: int = 1024) -> MILCAMResult:
        ps = inp.ps
        if ps is None:
            raise ValueError("input was not built for a MIL model")
        x = inp.x.to(self.device)
        n = len(ps.boxes)

        shown = ps.image
        scale = min(1.0, max_side / max(shown.shape[:2]))
        image = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else shown
        Hd, Wd = image.shape[:2]
        sy = Hd / ps.work_hw[0]
        boxes = [(int(round(y0 * sy)), int(round(y1 * sy))) for y0, y1 in ps.boxes]

        self.model.eval()
        with torch.no_grad():
            logits, aux = self.model(x, return_aux=True)
        pred = int(logits[0].argmax())
        with CAMExtractor(self.model, self.layer, self.method) as ex:
            res = ex(x, class_idx=pred if class_idx is None else class_idx)

        hc, wc = res.cams.shape[-2:]
        frac = np.array([max(y1 - y0, 1) / Hd for y0, y1 in boxes], np.float32)  # image-area fraction of a strip
        cells = res.cams
        dens_local = cells[:n] * (hc * wc) / frac[:, None, None]  # attribution per unit image area
        dens_global = cells[n] * (hc * wc)
        map_local = stitch_local_cams(dens_local, boxes, (Hd, Wd))
        map_global = resize_map(dens_global, Hd, Wd)

        learned = {}  # V4a: the model's own maps for the explained class, in the same geometry as the CAMs
        sg = aux.get("saliency_global")
        if sg is not None and sg.dim() == 4:
            learned["saliency_global"] = resize_map(sg[0, res.class_idx].float().cpu().numpy(), Hd, Wd)
        sl = aux.get("saliency_local")
        if sl is not None and sl.dim() == 5:
            learned["saliency_local"] = stitch_local_cams(sl[0, :, res.class_idx].float().cpu().numpy(), boxes, (Hd, Wd))
        ac = aux.get("attn_cells")
        if ac is not None and ac.dim() == 4:
            a = ac[0].float().cpu().numpy()  # (N, h, w), sums to 1 -> attention per unit image area
            learned["attention_local"] = stitch_local_cams(a * a.shape[1] * a.shape[2] / frac[:, None, None], boxes, (Hd, Wd))
        return MILCAMResult(
            image=image, probs=res.probs, pred=pred, class_idx=res.class_idx, method=self.method, layer=self.layer,
            boxes=boxes, attn=aux["attn"][0].cpu().numpy(),
            fusion_weights=None if aux["fusion_weights"] is None else aux["fusion_weights"][0].cpu().numpy(),
            cells_local=cells[:n], cells_global=cells[n], signed_local=res.signed[:n], signed_global=res.signed[n],
            dens_local=dens_local.astype(np.float32), dens_global=dens_global.astype(np.float32),
            map_local=map_local, map_global=map_global.astype(np.float32),
            map_combined=(map_local + map_global).astype(np.float32),
            crop_offset=inp.crop_offset, rotated=bool(ps.rotated),
            src_hw=(shown.shape[1], shown.shape[0]) if ps.rotated else tuple(shown.shape[:2]),
            display_scale=float(scale),
            learned=learned,
        )


def render(result: MILCAMResult, save_path: str | None = None, show: bool = False, bbx_display=None,
           class_names=None, gt=None) -> None:
    """The standard MIL figure (all maps on one scale, fusion weights in the titles)."""
    loc01, glo01 = result.strip_maps01()
    name = (lambda i: str(class_names[i])) if class_names else str
    render_mil(result.image, result.boxes, loc01, glo01, result.attn, save_path, show, 0.5, bbx_display,
               name(result.pred), float(result.probs[result.pred]), gt, result.fusion_weights,
               result.normalized("combined"))


# ----------------------------------------------------------------------------- CLI
def _plain_map(model, settings, path, device, method, crop, max_side, shape):
    """Combined/plain heat-map of any model (MIL or plain) in the un-rotated, cropped source orientation."""
    from src.gradcam.gradcam_run import explain_image

    if is_mil(settings["arch_type"]):
        r = MILGradCAM(model, settings, method=method, device=device).explain(load_rgb(path), None, crop, max_side)
        m = r.in_source_orientation(r.normalized("combined"))
    else:
        r = explain_image(model, settings, path, device, method=method, max_side=max_side, crop_foreground=crop)
        m = r["cam"]
    return cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def main(argv=None):
    from src.data.metadata import load_labeled_frame
    from src.gradcam.gradcam_run import load_boxes

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="MIL checkpoint (new format, or legacy + the flags below)")
    p.add_argument("--image", nargs="+")
    p.add_argument("--data_folder")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, nargs="+")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_column")
    p.add_argument("--out_dir", default="output/gradcam_mil")
    p.add_argument("--method", choices=["gradcam", "gradcam++"], default="gradcam")
    p.add_argument("--target_layer")
    p.add_argument("--class_idx", type=int, help="explain this class (default: the predicted one)")
    p.add_argument("--crop_foreground", action="store_true")
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--compare_with", nargs="+", default=[], help="other checkpoints (plain or MIL) to compare against")
    p.add_argument("--no_png", action="store_true")
    p.add_argument("--device")
    p.add_argument("--arch_type")
    p.add_argument("--model_type")
    p.add_argument("--num_classes", type=int)
    p.add_argument("--img_size")
    p.add_argument("--num_patches", type=int)
    p.add_argument("--overlap_ratio", type=float)
    p.add_argument("--local_scale", type=float)
    p.add_argument("--fusion")
    p.add_argument("--legacy_preprocess", action="store_true", default=None)
    a = p.parse_args(argv)

    from src.utils.common import parse_img_size

    set_seed(a.seed)
    device = get_device(a.device)
    overrides = dict(arch_type=a.arch_type or "mil_v4", model_type=a.model_type, num_classes=a.num_classes,
                     img_size=parse_img_size(a.img_size), num_patches=a.num_patches, fusion=a.fusion,
                     overlap_ratio=0.2 if a.overlap_ratio is None else a.overlap_ratio,
                     local_scale=1.0 if a.local_scale is None else a.local_scale, rotate_landscape=True,
                     legacy_preprocess=a.legacy_preprocess)
    model, s = build_model_from_checkpoint(a.checkpoint, overrides, device)
    names = s.get("class_names")
    others = [(os.path.basename(c), *build_model_from_checkpoint(c, None, device)) for c in a.compare_with]
    os.makedirs(a.out_dir, exist_ok=True)
    cam = MILGradCAM(model, s, a.target_layer, a.method, device)

    samples = []  # (path, gt, boxes in original pixels)
    if a.image:
        samples = [(q, None, None) for q in a.image]
    else:
        if not a.data_folder:
            raise SystemExit("--data_folder is required unless --image is given")
        df, data_names = load_labeled_frame(a.data_folder, a.target_column or s.get("target_column") or "cancer")
        names = names or data_names
        df = df[df["split"] == a.split].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"no rows with split == '{a.split}'")
        idx = a.index or sorted(np.random.default_rng(a.seed).choice(len(df), min(a.n_samples, len(df)), replace=False).tolist())
        boxes = load_boxes(a.data_folder)
        samples = [(os.path.join(a.data_folder, df.iloc[i]["link"]), int(df.iloc[i]["label"]), boxes.get(df.iloc[i]["link"])) for i in idx]

    rows = []
    for k, (path, gt, bxs) in enumerate(samples):
        stem = os.path.splitext(os.path.basename(path))[0]
        r = cam.explain(load_rgb(path), a.class_idx, a.crop_foreground, a.max_side)
        r.save(os.path.join(a.out_dir, f"{stem}.npz"))
        if not a.no_png:
            render(r, os.path.join(a.out_dir, f"{stem}.png"), bbx_display=r.boxes_to_display(bxs), class_names=names,
                   gt=None if gt is None else (names[gt] if names else gt))
        fw = "n/a" if r.fusion_weights is None else np.round(r.fusion_weights, 3).tolist()
        print(f"[{k + 1}/{len(samples)}] {stem}: pred={r.pred} p={r.probs[r.pred]:.3f} explains class {r.class_idx} | "
              f"fusion[local,global]={fw} | local share of attribution={r.share_local:.3f} | attn={np.round(r.attn, 2).tolist()}")
        mine = r.in_source_orientation(r.normalized("combined"))
        src_boxes = r.boxes_to_display(bxs, source_orientation=True)
        loc = localization(mine, src_boxes)
        if loc:
            rows.append(dict(image=stem, model=os.path.basename(a.checkpoint), vs="", **loc))
        for lname, lmap in r.learned.items():  # the model's own maps vs. the Grad-CAM reference
            lm = r.in_source_orientation(lmap)
            row = dict(image=stem, model=os.path.basename(a.checkpoint), vs=f"learned:{lname}", **similarity(mine, lm))
            lo = localization(lm, src_boxes)
            if lo:
                row.update({f"other_{k2}": v for k2, v in lo.items()})
            rows.append(row)
            print(f"      learned {lname:16s} vs Grad-CAM: pearson {row['pearson'] if row['pearson'] is None else round(row['pearson'], 2)}"
                  f"  iou_top10 {row['iou_top10']:.2f}")
        for name, om, os_ in others:
            other = _plain_map(om, os_, path, device, a.method, a.crop_foreground, a.max_side, mine.shape)
            row = dict(image=stem, model=os.path.basename(a.checkpoint), vs=name, **similarity(mine, other))
            lo = localization(other, src_boxes)
            if lo:
                row.update({f"other_{k2}": v for k2, v in lo.items()})
            rows.append(row)
    if rows:
        keys = sorted({k2 for r_ in rows for k2 in r_}, key=lambda k2: (k2 not in ("image", "model", "vs"), k2))
        with open(os.path.join(a.out_dir, "compare.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", os.path.join(a.out_dir, "compare.csv"))


if __name__ == "__main__":
    main()
