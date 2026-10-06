# src_legacy — the original mammoSGM code (archived)

This is the code the project started with, kept unchanged for reference and to reproduce old results.
It was moved here on 2026-10-06; the only edit is the package name in its imports (`src.` → `src_legacy.`).
It is **not maintained** — new work uses `src/` (see the main [README](../README.md)).

What is only available here: MIL versions other than V4a (`mil`, v2, v3, v4, v5–v10), the `patch_*`,
`token_mixer`, `global_local*` architectures, the DINOv3 / MambaVision / FasterViT backbones, the old
plots and the old Grad-CAM scripts.

## Running it

```bash
pip install -r src_legacy/requirements.txt   # + scikit-learn, seaborn, scikit-image, albumentations 2.x
```

```bash
python -m src_legacy.trainer.train_based --mode test --config config.yaml --data_folder /content/SGM815_cropbbx --model_type resnet50 --batch_size 16 --num_epochs 10 --output /content/runs --img_size 224x224 --pretrained_model_path /content/runs/models/SGM815_cropbbx_224x224_based_resnet50_7255.pth
```

```bash
python -m src_legacy.trainer.train_patch --mode train --batch_size 32 --config config.yaml --data_folder /data/.../SGM1k --model_type resnet50 --num_epochs 200 --output /data/.../Run_SGM1k --img_size 448x448 --arch_type mil_v4 --num_patches 4
```

Weights were saved as `<output>/models/<dataset>_<HxW>_<arch>_<backbone>[_p<N>]_<acc*10000>.pth` (bare
state_dicts) and, by `--mode test` / `--mode gradcam`, as `*_full.pth` (pickled whole models).

## Known problems of this code (fixed in `src/`)

| # | problem | in `src/` |
|---|---|---|
| 1 | every training step ran the model **twice** (the first time outside AMP); BatchNorm statistics updated twice | one forward per step |
| 2 | early stopping, LR schedule and the best checkpoint were chosen **on the test set**, by accuracy -> reported test numbers are optimistic | a patient-grouped validation split selects the model; the test set is evaluated once |
| 3 | strip splitting left part of every image uncovered (6.8% of the height with 3 strips, 10.2% with 4) and 2 strips did not overlap | strips cover the whole image with the requested overlap |
| 4 | class imbalance corrected twice (weighted sampler **and** class-weighted / focal loss) | `--balance sampler|loss|none` |
| 5 | a failed checkpoint load printed a warning and evaluated random weights | strict loading, raises |
| 6 | MIL Grad-CAM paired the strips' activations with the global image's gradients; each strip normalised on its own; `has_global` hard-coded to "v4" | `src/gradcam/gradcam_mil.py` (reference implementation) |
| 7 | MILv4 ran strips and global image through the shared backbone in two calls (BatchNorm saw two distributions) | one backbone call |
| 8 | the MILv4 fusion gate saturates: in all 4 trained V4 checkpoints one branch contributes ~nothing (3 global-only, 1 strip-only) and the strip attention is ~uniform, so "MIL" behaves like the plain model | replaced by MILv4a |
| 9 | multi-class "AUC" from hard predictions; 16-bit PNGs saturated by `convert("RGB")`; DINOv2 backbones silently random; worker processes shared one augmentation RNG; `requirements.txt` allowed albumentations 1.3 | fixed |

## Using old weights with the new code

1. Convert the pickled `*_full.pth` (read with an allow-list, checked numerically against the original):

   ```bash
   python -m src.models.legacy --src /path/to/weights --out /path/to/converted
   ```

   The converted checkpoint carries `legacy_preprocess=True`, so the new code reproduces the old
   preprocessing (bilinear resize, the old strip geometry with its gap) the weights were trained with.
   Bare state_dicts (`<...>_<acc>.pth`) can be evaluated directly by passing their settings, e.g.
   `python -m src.trainer.train_patch --mode test --arch_type mil_v4 --model_type resnet50 --img_size 448x448 --num_patches 4 --legacy_preprocess --pretrained_model_path X.pth --data_folder DATA`.

2. For deployment, export it (adds threshold / class names / normalisation and verifies deploy == src):

   ```bash
   python -m src.export --checkpoint converted/X.pth --out X_deploy.pth --label_names normal,cancer
   ```

`--legacy_preprocess` is for evaluating old weights only; the new code refuses to train MIL models with it.

## The old batch-testing notebook loop

`src.trainer.train_patch_with_metrics` is kept as an alias of `src.trainer.train_patch`, and `--setting` /
`--backbone_name` are still accepted, so this loop (as it was in the original README) still runs against
the new code. Old bare weights need `--legacy_preprocess` to be evaluated with the preprocessing they were
trained with.

```python
import os, shutil
from pathlib import Path

PROJECT_DIR = os.getcwd()
TARGET_COLUMN = "cancer"
DATA_FOLDER = "/mnt/data/SGM_PROJECT/OPTIMAM/optimam_v3"  # Update this path to your data folder
RESULTS_FOLDER = "/mnt/data/SGM_PROJECT/OPTIMAM/results"
SETTING = "mil"

for weights_folder_name in os.listdir(Path(f"{RESULTS_FOLDER}/{SETTING}")):
    weights_folder = os.path.join(RESULTS_FOLDER, SETTING, weights_folder_name)
    if TARGET_COLUMN.lower() in str(weights_folder).lower():
        if weights_folder.endswith("resnet34") or weights_folder.endswith("resnet50"):
            BACKBONE = weights_folder.split("/")[-1].split("_")[-1]
        else:
            BACKBONE = "convnextv2_tiny"
        output_dir = os.path.join(PROJECT_DIR, "final_results")
        os.makedirs(output_dir, exist_ok=True)
        paths = [os.path.join(r, f) for r, _, fs in os.walk(weights_folder) for f in fs
                 if f.endswith(".pth") and not f.endswith("_full.pth")]  # *_full.pth: convert first (see above)
        %cd mammoSGM
        for PRETRAINED_MODEL_PATH in paths:
            print(f"Testing with pretrained model: {PRETRAINED_MODEL_PATH}")
            !python -m src.trainer.train_patch_with_metrics \
                --mode test \
                --config config.yaml \
                --data_folder "{DATA_FOLDER}" \
                --model_type "{BACKBONE}" \
                --batch_size 16 \
                --output "{output_dir}" \
                --img_size 224x224 \
                --arch_type "mil_v4" \
                --legacy_preprocess \
                --target_column "{TARGET_COLUMN}" \
                --pretrained_model_path "{PRETRAINED_MODEL_PATH}" \
                --setting "{SETTING}" \
                --backbone_name "{BACKBONE}"
```
