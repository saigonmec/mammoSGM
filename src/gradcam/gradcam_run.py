"""Grad-CAM from a checkpoint (plain or MIL model).

  python -m src.gradcam.gradcam_run --checkpoint output/<run>/best.pth \
      --data_folder /path/to/data --split test --n_samples 5 --out_dir output/gradcam

  python -m src.gradcam.gradcam_run --checkpoint best.pth --image a.png b.png --out_dir out

The model, preprocessing (patch geometry, sizes, normalisation) and class names all come
from the checkpoint, and images go through exactly the code path of the test dataset.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.imaging import load_rgb
from src.data.inference import build_input
from src.data.metadata import load_labeled_frame
from src.gradcam.cam import CAMExtractor, default_target_layer
from src.gradcam.gradcam_mil import MILGradCAM
from src.gradcam.viz import normalize_cams, render_based, render_mil
from src.models.checkpoint import build_model_from_checkpoint
from src.utils.common import get_device, parse_img_size, set_seed


def load_boxes(data_folder: str) -> dict:
    """link -> [[x, y, w, h], ...] from metadata2.csv / metadata.csv (columns x, y, width, height)."""
    path = os.path.join(data_folder, "metadata2.csv")
    if not os.path.exists(path):
        path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(path)
    if not {"x", "y", "width", "height", "link"} <= set(df.columns):
        return {}
    df = df.dropna(subset=["x", "y", "width", "height"])
    return {link: g[["x", "y", "width", "height"]].values.tolist() for link, g in df.groupby("link")}


def explain_image(model, settings, image_path, device, method="gradcam", target_layer=None,
                  class_idx=None, max_side=1400, bbx_list=None, crop_foreground=False):
    """Run the model + Grad-CAM on one image. Returns everything the renderers need.
    MIL models go through gradcam_mil.MILGradCAM (the reference implementation).
    crop_foreground: first crop the breast's bounding box (full-field mammograms with a black background)."""
    inp = build_input(load_rgb(image_path), settings, crop_foreground)

    if inp.ps is not None:  # MIL
        r = MILGradCAM(model, settings, target_layer, method, device).explain_input(inp, class_idx, max_side)
        loc, glo = r.strip_maps01()
        out = {"mil": True, "display": r.image, "pred": r.pred, "class_idx": r.class_idx, "probs": r.probs,
               "target_layer": r.layer, "boxes_disp": r.boxes, "local_cams": loc, "global_cam": glo,
               "attn": r.attn, "result": r}
        if r.fusion_weights is not None:
            out["fusion_weights"] = r.fusion_weights
        if bbx_list:
            out["bbx"] = r.boxes_to_display(bbx_list)
        return out

    x, shown = inp.x.to(device), inp.shown
    scale = min(1.0, max_side / max(shown.shape[:2]))
    display = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else shown
    model.eval()
    with torch.no_grad():
        logits = model(x)
    pred = int(logits[0].argmax())
    layer = target_layer or default_target_layer(model)
    with CAMExtractor(model, layer, method) as ex:
        res = ex(x, class_idx=pred if class_idx is None else class_idx)
    out = {"mil": False, "display": display, "pred": pred, "class_idx": res.class_idx, "probs": res.probs,
           "target_layer": layer, "cam": normalize_cams(res.cams, per_map=True)[0]}
    if bbx_list:  # boxes are in original-image pixels: undo the crop, then the display scale
        ox, oy = inp.crop_offset
        f = display.shape[1] / shown.shape[1]
        out["bbx"] = [[(b[0] - ox) * f, (b[1] - oy) * f, b[2] * f, b[3] * f] for b in bbx_list]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_folder")
    p.add_argument("--image", nargs="+", help="explicit image path(s) instead of sampling from metadata")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, nargs="+", help="row indices inside --split")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_column")
    p.add_argument("--out_dir", default="output/gradcam")
    p.add_argument("--method", choices=["gradcam", "gradcam++"], default="gradcam")
    p.add_argument("--target_layer", help="override the auto-detected layer, e.g. base_model.layer4")
    p.add_argument("--class_idx", type=int, help="explain this class instead of the predicted one")
    p.add_argument("--option", type=int, default=5, choices=[1, 2, 3, 4, 5], help="panel layout (plain models)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max_side", type=int, default=1400, help="downscale the figure image to this size")
    p.add_argument("--show", action="store_true")
    p.add_argument("--device")
    # only needed for legacy checkpoints without embedded settings
    p.add_argument("--arch_type")
    p.add_argument("--model_type")
    p.add_argument("--num_classes", type=int)
    p.add_argument("--img_size")
    p.add_argument("--num_patches", type=int)
    p.add_argument("--overlap_ratio", type=float)
    p.add_argument("--local_scale", type=float)
    p.add_argument("--fusion")
    p.add_argument("--legacy_preprocess", action="store_true", default=None)
    p.add_argument("--crop_foreground", action="store_true",
                   help="crop the breast bounding box (black-background full-field images) before inference")
    a = p.parse_args(argv)

    set_seed(a.seed)
    device = get_device(a.device)
    overrides = dict(arch_type=a.arch_type, model_type=a.model_type, num_classes=a.num_classes,
                     img_size=parse_img_size(a.img_size), num_patches=a.num_patches, fusion=a.fusion,
                     overlap_ratio=0.2 if a.overlap_ratio is None else a.overlap_ratio,
                     local_scale=1.0 if a.local_scale is None else a.local_scale, rotate_landscape=True,
                     legacy_preprocess=a.legacy_preprocess)
    model, s = build_model_from_checkpoint(a.checkpoint, overrides, device)
    os.makedirs(a.out_dir, exist_ok=True)
    names = s.get("class_names")

    samples = []  # (path, gt label, boxes)
    if a.image:
        samples = [(pth, None, None) for pth in a.image]
    else:
        if not a.data_folder:
            raise SystemExit("--data_folder is required unless --image is given")
        df, data_names = load_labeled_frame(a.data_folder, a.target_column or s.get("target_column") or "cancer")
        names = names or data_names
        df = df[df["split"] == a.split].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"no rows with split == '{a.split}' in metadata.csv")
        idx = a.index or sorted(np.random.default_rng(a.seed).choice(len(df), min(a.n_samples, len(df)), replace=False).tolist())
        boxes = load_boxes(a.data_folder)
        for i in idx:
            row = df.iloc[i]
            samples.append((os.path.join(a.data_folder, row["link"]), int(row["label"]), boxes.get(row["link"])))

    def cname(i):
        return str(names[i]) if names and i < len(names) else str(i)

    for k, (path, gt, bbx) in enumerate(samples):
        r = explain_image(model, s, path, device, a.method, a.target_layer, a.class_idx, a.max_side, bbx,
                          a.crop_foreground)
        pred, prob = r["pred"], float(r["probs"][r["pred"]])
        gt_s = None if gt is None else cname(gt)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(a.out_dir, f"{k:03d}_{stem}_gt{gt_s}_pred{cname(pred)}.png")
        if r["mil"]:
            render_mil(r["display"], r["boxes_disp"], r["local_cams"], r["global_cam"], r["attn"], out_path,
                       a.show, a.alpha, r.get("bbx"), cname(pred), prob, gt_s, r.get("fusion_weights"),
                       r["result"].normalized("combined"))
        else:
            render_based(r["display"], r["cam"], out_path, a.show, a.option, a.alpha, r.get("bbx"),
                         cname(pred), prob, gt_s)
        extra = f" | attn={np.round(r['attn'], 2).tolist()}" if r["mil"] else ""
        if r.get("fusion_weights") is not None:
            extra += f" | fusion[local,global]={np.round(r['fusion_weights'], 2).tolist()}"
        print(f"[{k + 1}/{len(samples)}] {os.path.basename(path)} gt={gt_s} pred={cname(pred)} "
              f"p={prob:.3f} layer={r['target_layer']}{extra} -> {out_path}")


if __name__ == "__main__":
    main()
