"""Command line for MammoModel.

    python -m deploy.predict --weights model.pth --info
    python -m deploy.predict --weights model.pth --images a.png b.png --out results.csv
    python -m deploy.predict --weights model.pth --input_dir crops/ --out results.csv --heatmaps heatmaps/
    python -m deploy.predict --weights model.pth --images full.png --roi_csv rois.csv   # rois: image,x0,y0,x1,y1
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

from .core.gradcam.viz import overlay
from .predictor import MammoModel

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--images", nargs="+", default=[])
    p.add_argument("--input_dir", help="all images in this folder (recursively)")
    p.add_argument("--roi_csv", help="csv with columns image,x0,y0,x1,y1 (image = file name) from the breast detector")
    p.add_argument("--crop", default="none", choices=["none", "foreground"],
                   help="images are breast crops already (none) or full-field with a black background (foreground)")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--heatmaps", help="folder for heat-map overlays (Grad-CAM of the positive class)")
    p.add_argument("--device")
    p.add_argument("--info", action="store_true", help="print the settings stored in the weight and exit")
    a = p.parse_args(argv)

    model = MammoModel(a.weights, a.device)
    if a.info:
        print(json.dumps(model.info(), indent=2, default=str))
        return 0

    paths = list(a.images)
    if a.input_dir:
        paths += sorted(f for f in glob.glob(os.path.join(a.input_dir, "**", "*"), recursive=True) if f.lower().endswith(EXTS))
    if not paths:
        p.error("no images: use --images and/or --input_dir")
    rois = {}
    if a.roi_csv:
        for r in csv.DictReader(open(a.roi_csv)):
            rois[os.path.basename(r["image"])] = tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1"))
    if a.heatmaps:
        os.makedirs(a.heatmaps, exist_ok=True)

    rows = []
    for i, path in enumerate(paths):
        name = os.path.basename(path)
        try:
            r = model.predict(path, roi=rois.get(name), crop=a.crop, explain=bool(a.heatmaps))
        except Exception as e:  # one bad file must not stop a batch; it is reported in the csv
            print(f"[{i + 1}/{len(paths)}] {name}: ERROR {e}", file=sys.stderr)
            rows.append({"image": path, "error": str(e)})
            continue
        row = {"image": path, "label": r["label"], "pred": r["pred"], "positive_prob": r["positive_prob"],
               "threshold": r["threshold"], "roi": " ".join(map(str, r["roi"])),
               **{f"prob_{k}": v for k, v in r["probs"].items()}, "error": ""}
        rows.append(row)
        msg = f"p={r['positive_prob']:.4f}" if r["positive_prob"] is not None else ""
        print(f"[{i + 1}/{len(paths)}] {name}: {r['label']} {msg}")
        if a.heatmaps:
            from .core.data.imaging import load_rgb

            img = load_rgb(path)
            stem = os.path.splitext(name)[0]
            cv2.imwrite(os.path.join(a.heatmaps, f"{stem}_heatmap.png"),
                        cv2.cvtColor(overlay(img, r["heatmap"], 0.45), cv2.COLOR_RGB2BGR))
            if "attention" in r:
                cv2.imwrite(os.path.join(a.heatmaps, f"{stem}_attention.png"),
                            cv2.cvtColor(overlay(img, r["attention"], 0.45), cv2.COLOR_RGB2BGR))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)
    return 1 if any(r.get("error") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
