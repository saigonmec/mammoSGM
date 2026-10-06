# mammoSGM

Breast-cancer classification on mammogram **breast crops** (images cropped to the breast by the YOLO breast
detector), with two models at **448 px**:

| model | `--arch_type` | input | what it is |
|---|---|---|---|
| **based** | (train_based) | the whole crop, 448×448 | one CNN (ConvNeXt-V2 tiny / ResNet50) + linear head; the baseline |
| **MILv4a** | `mil_v4a` | 3 strips + the whole crop, each 448×448 | multiple-instance model: every feature-map cell is an instance (GMIC-style, see [How MILv4a works](#how-milv4a-works)) |

Every checkpoint carries all its settings, and `deploy/` runs it with nothing else (see [Deploy](#deploy)).

## Repository

| folder | content |
|---|---|
| `src/` | training, evaluation, Grad-CAM |
| `deploy/` | self-contained inference package: this folder + a weight file is all a deployment needs ([deploy/README.md](deploy/README.md)) |
| `tests/` | `python tests/test_core.py`, `python tests/test_deploy.py` (no pytest needed) |
| `tools/` | `sync_deploy.py`: copies the core modules of `src/` into `deploy/core/` |
| `src_legacy/` | the original code, its README, the old MILv4 and how to use old weights ([src_legacy/README.md](src_legacy/README.md)) |

## Install

```bash
pip install -r src/requirements.txt
```

`albumentations >= 2.0` is required. Train on a GPU; AMP is used automatically, and several GPUs are used
with DataParallel unless `--single_gpu` is given.

## Data

`<data_folder>/metadata.csv`:

| column | meaning |
|---|---|
| `link` | image path relative to `data_folder` (a YOLO breast crop; 8- or 16-bit PNG is fine) |
| `split` | `train` / `test` (an optional `val` split is used as-is) |
| `cancer` (or `--target_column`) | label; numeric or text, mapped to classes 0..C-1 in sorted order |
| `patient_id` (strongly recommended) | the validation split is taken **by patient** and patient overlap between splits is reported |

Without a `val` split, `--val_ratio` of the training *patients* is held out for model selection; the test
split is evaluated once, with the selected checkpoint.

## Recommended settings (448)

These are the defaults in `src/config/config.yaml`, so the commands below only repeat the important ones.

| | based | MILv4a | why |
|---|---|---|---|
| `--model_type` | `convnextv2_tiny` (or `resnet50`) | same as based | compare like with like |
| `--img_size` | `448x448` | `448x448` | every network input (each strip and the global view for MILv4a) |
| `--num_patches` | – | `3` | 3 strips are closest to square for a typical crop (height ≈ 2.4 × width) |
| `--local_scale` | – | `2` | strips are cut from a copy `448 × local_scale` wide: use ≈ **crop width / 448** (crops ~900 px wide → 2) |
| `--batch_size` | 16–32 | 8–16 | a MILv4a sample is 4 images of 448² |
| `--num_epochs` | 300 | 300 | upper bound, early stopping decides |
| `--patience` | 40 | 40 | epochs without validation improvement |
| `--lr` / `--warmup_epochs` / `--lr_patience` | 1e-4 / 3 / 10 | same | LR halves after 10 epochs without val-loss improvement (resets the patience) |
| `--val_ratio` | 0.15 | 0.15 | larger validation set = steadier model selection over long runs |
| `--loss_type` / `--balance` | `ce` / `sampler` | same | one imbalance correction only |
| `--seed` | 1, 2, 3 | 1, 2, 3 | compare models over seeds, not single runs |

Nothing in the backbone is frozen; all layers are fine-tuned.

## Train

In a Jupyter notebook, prefix the commands with `!` (and `cd` to the folder that contains `src/`).

**based**

```bash
python -m src.trainer.train_based --data_folder /data/.../SGM1k --output /data/.../Run_SGM1k --model_type convnextv2_tiny --img_size 448x448 --batch_size 16 --num_epochs 300 --patience 40 --seed 1 --run_name based_448_s1
```

**MILv4a**

```bash
python -m src.trainer.train_patch --arch_type mil_v4a --data_folder /data/.../SGM1k --output /data/.../Run_SGM1k --model_type convnextv2_tiny --img_size 448x448 --num_patches 3 --local_scale 2 --batch_size 8 --num_epochs 300 --patience 40 --seed 1 --run_name v4a_448_s1
```

Run `--num_epochs 1` first on a new machine to check memory and speed. `--help` lists every option;
`--config other.yaml` changes the defaults.

### What to watch during training

Each epoch prints and logs (`log.csv`) the training / validation loss, accuracy and AUC. For MILv4a it also
shows the health of its two branches:

* `branch auc g:… l:…` — validation AUC of the global head and of the strip (MIL) head alone. If `l` stays
  ≈ 0.5 after ~10 epochs, the strips do not help.
* `local reliance` — share of the prediction coming from the strips (0.2–0.8 = both branches used).

### Outputs: `<output>/<run_name>/`

| file | content |
|---|---|
| `best.pth` | best validation checkpoint, **with all settings inside** (directly usable by `deploy/`) |
| `last.pth` | last epoch |
| `metrics.json` | best epoch, validation and test metrics (+ per-branch AUC for MILv4a) |
| `log.csv`, `curves.png` | per-epoch history |
| `{val,test}_predictions.csv` | per-image probabilities |
| `{val,test}_roc.png`, `{val,test}_confusion_matrix.png` | plots |
| `config.json` | the resolved settings |

### Comparing based and MILv4a

Compare **validation AUC** (mean ± spread over seeds) and run the test set only for the chosen setting.
Prefer MILv4a only if it beats based by more than the seed-to-seed spread. To tell whether a gain comes from
MIL or simply from resolution, also train based with `--img_size 896x448` (keeps the crop's aspect ratio).

## Evaluate a checkpoint

Only the checkpoint is needed; architecture, input size and strips are read from it.

```bash
python -m src.trainer.train_patch --mode test --data_folder /data/.../SGM1k --pretrained_model_path /data/.../Run_SGM1k/v4a_448_s1/best.pth --output /data/.../Run_SGM1k/eval
```

(`train_based --mode test` works the same way.)

## Heat-maps

**MILv4a**: the reference Grad-CAM (strips and global view on one scale, mapped back onto the crop) plus the
model's own cell-attention map, optionally compared with another model:

```bash
python -m src.gradcam.gradcam_mil --checkpoint /data/.../v4a_448_s1/best.pth --data_folder /data/.../SGM1k --n_samples 10 --compare_with /data/.../based_448_s1/best.pth --out_dir /data/.../cam_v4a
```

Writes one figure + `.npz` (all maps, for later comparisons) per image and `compare.csv` (agreement between
maps; overlap with lesion boxes if `metadata.csv` has `x,y,width,height`).

**based**: Grad-CAM / Grad-CAM++:

```bash
python -m src.gradcam.gradcam_run --checkpoint /data/.../based_448_s1/best.pth --data_folder /data/.../SGM1k --n_samples 10 --out_dir /data/.../cam_based
```

Heat-maps explain the predicted class unless `--class_idx 1` is given; on negative images the map of the
"cancer" class is weak and not meaningful.

## Deploy

`best.pth` from training is deployable as it is (decision threshold 0.5 = the reported metrics). To set a
validation-based threshold and readable class names, export it; the export also verifies that `deploy/`
reproduces `src/` on real validation images:

```bash
python -m src.export --checkpoint /data/.../v4a_448_s1/best.pth --out v4a_448.pth --threshold youden --label_names normal,cancer --data_folder /data/.../SGM1k
```

Then, with only `deploy/` and the weight:

```bash
python -m deploy.predict --weights v4a_448.pth --input_dir crops/ --out predictions.csv --heatmaps heatmaps/
```

Python API, input contract (breast crops, or full images + the YOLO box as `roi`) and CLI options:
[deploy/README.md](deploy/README.md).

## How MILv4a works

```
crop ──► 3 overlapping strips (cut at local_scale × resolution) + the whole crop ──► 4 × 448² ──► ONE backbone call
          │                                                                                       │
          │      every cell of the last feature map (14×14 per image) is an instance              ▼
          ├─ global view : saliency per cell → top-t% pooling → y_global ;  mean of cells → z_g
          ├─ strips      : gated attention over all 3×14×14 cells → z_l → y_local
          │                saliency per cell → top-t% pooling → y_local_sal
          └─ prediction  : Linear([z_l ; z_g])     (branch dropout while training)
loss = CE(prediction) + 0.5 × [CE(y_global) + CE(y_local) + CE(y_local_sal)]
```

* Built from known parts: GMIC (Shen et al. 2021) global/local modules with top-t pooling and per-branch
  losses, gated attention MIL (Ilse et al. 2018), branch (modality) dropout.
* Each branch has its own loss, so neither can "die"; the old MILv4 gate did exactly that (one branch got
  ~0 weight in every trained V4 checkpoint, see `src_legacy/README.md`).
* Strips cover the whole crop with 20 % overlap; they add **vertical** detail (the global view squeezes the
  tall crop), not horizontal detail.
* The prediction splits exactly into a strip part and a global part, which is what `local reliance` reports.

Options: `--aux_weight 0.5`, `--branch_dropout 0.2`, `--top_t 0.05`, `--neck_dim 512`, `--backbone_lr_mult 1.0`
(a smaller value trains the pretrained backbone more slowly than the new heads).

## Tests

```bash
python tests/test_core.py
```

```bash
python tests/test_deploy.py
```

After editing `src/models/`, `src/data/imaging.py`, `src/data/patches.py` or `src/gradcam/{cam,viz}.py`, run
`python tools/sync_deploy.py` (the deploy tests fail if `deploy/core` is out of date).

## Status

MILv4a has been validated on synthetic data and the sample images only; it has not yet been trained on the
real dataset, so no accuracy claim is made. The code has been run on CPU; check the first GPU run with
`--num_epochs 1`.

## Contact

If you encounter any issues running the code, please contact the development team.
