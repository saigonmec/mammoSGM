# deploy — run a trained mammoSGM model

This folder + one weight file (`.pth`) is everything needed. All settings (architecture, backbone, input
size, strips, preprocessing, decision threshold, class names, library versions) are stored inside the
weight; there is no config to keep in sync.

## Install

```bash
pip install -r deploy/requirements.txt     # torch, torchvision, timm, numpy, opencv-python, Pillow
```

## Input contract (important)

The models are trained on **breast ROI crops** (images cropped to the breast by the YOLO breast detector).
Give the model the same kind of image:

* already-cropped images: pass them as they are;
* full-field mammograms: pass the detector box as `roi=(x0, y0, x1, y1)` (or `--roi_csv`).
  `crop="foreground"` (bounding box of non-black pixels) exists only as an approximation.

A warning is printed if an image looks un-cropped (mostly black background). 8- and 16-bit, grayscale or RGB
PNG/TIFF/JPEG and numpy arrays are accepted (16-bit is rescaled correctly).

## Python

```python
from deploy import MammoModel

model = MammoModel("model.pth")                 # cuda if available, else cpu; device="cpu" to force
r = model.predict("crop.png")
r["positive_prob"]   # probability of the positive class (e.g. cancer)
r["pred"], r["label"]   # decision with the threshold stored in the weight (r["threshold"])
r["probs"]           # {class name: probability}

r = model.predict(full_image, roi=(x0, y0, x1, y1), explain=True)
r["heatmap"]         # float32 (H, W) in [0, 1], same pixel grid as the input image, 0 outside the roi
r["attention"]       # MILv4a only: the model's own cell-attention map
r["details"]         # MIL only: strip attention, fusion weights, share of heat from the strips

model.info()         # everything stored in the weight
```

`heatmap` is Grad-CAM of the positive class (`explain_class=` to change it), computed exactly like the
reference implementation used in development (`src/gradcam/gradcam_mil.py`).

## Command line

```bash
python -m deploy.predict --weights model.pth --info
```

```bash
python -m deploy.predict --weights model.pth --input_dir crops/ --out predictions.csv --heatmaps heatmaps/
```

```bash
python -m deploy.predict --weights model.pth --images full1.png full2.png --roi_csv rois.csv --out predictions.csv
```

`rois.csv`: columns `image,x0,y0,x1,y1` (`image` = file name). The csv output has one row per image
(`label, pred, positive_prob, threshold, prob_<class>..., roi, error`); a failing image is reported in
`error` and does not stop the batch.

## Where the weight comes from

* `best.pth` written by training (`src/trainer`) is deployable as is (threshold 0.5 = the training metrics).
* `python -m src.export --checkpoint best.pth --out model.pth [--threshold youden|sens:0.9|0.4]
  [--label_names normal,cancer] [--data_folder DATA]` sets another threshold / readable class names, and
  verifies that this folder reproduces the development code on real images.
* Old weights (`*_full.pth` from `src_legacy`): `python -m src.models.legacy` then `python -m src.export`.

## For developers

`deploy/core/` is a byte-for-byte copy of the model / preprocessing / Grad-CAM modules of `src/`
(do not edit it). After changing those files in `src/`, run `python tools/sync_deploy.py`;
`tests/test_deploy.py` fails if the copy is stale or if deploy and src disagree.
