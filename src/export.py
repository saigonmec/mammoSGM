"""Make a deploy-ready weight file (everything in `meta`) and verify deploy/ reproduces src/.

    python -m src.export --checkpoint RUN/best.pth --out model_deploy.pth
    python -m src.export --checkpoint RUN/best.pth --out m.pth --threshold youden          # from RUN/val_predictions.csv
    python -m src.export --checkpoint RUN/best.pth --out m.pth --threshold sens:0.90 --label_names normal,cancer
    python -m src.export --checkpoint converted_old.pth --out m.pth --data_folder DATA      # + check on real images

Weights written by the training code are already deployable (they carry a `deploy` block with threshold
0.5); use this to set another decision threshold, readable class names, or to upgrade a converted old weight.
--threshold: a number | youden | sens:X (largest threshold with val sensitivity >= X) | spec:X.
Verification: deploy/ must give the same probabilities as src/ (random + real images) and, when the run's
val_predictions.csv and --data_folder are available, the probabilities stored there.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

from src.data.imaging import MEAN, STD, as_rgb_uint8, load_rgb
from src.data.inference import predict_probs
from src.models.checkpoint import build_model_from_checkpoint, deploy_info, plain, read_checkpoint, validate_meta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_val_predictions(path: str, positive_class: int = 1):
    """-> (links, labels, positive-class probabilities) from a *_predictions.csv written by training."""
    rows = list(csv.DictReader(open(path)))
    prob_cols = [c for c in rows[0] if c.startswith("prob_")]
    col = prob_cols[positive_class]
    return [r["link"] for r in rows], np.array([int(r["label"]) for r in rows]), np.array([float(r[col]) for r in rows])


def choose_threshold(labels: np.ndarray, probs: np.ndarray, rule: str):
    """-> (threshold, sens, spec). Positive iff prob > threshold (the deploy rule)."""
    y = labels.astype(bool)
    if y.all() or (~y).all():
        raise ValueError("validation predictions contain a single class; cannot choose a threshold")
    u = np.unique(probs)
    cands = np.r_[u[0] - 1e-6, (u[:-1] + u[1:]) / 2, u[-1]]
    sens = np.array([(probs[y] > t).mean() for t in cands])
    spec = np.array([(probs[~y] <= t).mean() for t in cands])
    if rule == "youden":
        j = sens + spec - 1
        best = np.flatnonzero(np.isclose(j, j.max()))
        i = best[np.argmin(np.abs(cands[best] - 0.5))]  # ties: closest to 0.5
    elif rule.startswith("sens:"):
        ok = np.flatnonzero(sens >= float(rule[5:]))
        i = ok[np.argmax(cands[ok])]
    elif rule.startswith("spec:"):
        ok = np.flatnonzero(spec >= float(rule[5:]))
        i = ok[np.argmin(cands[ok])]
    else:
        raise ValueError(f"unknown threshold rule {rule!r}")
    return float(cands[i]), float(sens[i]), float(spec[i])


def verify(out_path: str, images, expected=None, atol: float = 1e-5) -> None:
    """deploy.MammoModel must reproduce src/ (and the stored val probabilities, if given)."""
    sys.path.insert(0, ROOT)
    from deploy import MammoModel

    dm = MammoModel(out_path, "cpu")
    sm, s = build_model_from_checkpoint(out_path, None, "cpu")
    pc = dm.positive_class if dm.positive_class is not None else 0
    worst = 0.0
    for k, img in enumerate(images):
        p_src = predict_probs(sm, s, as_rgb_uint8(img), torch.device("cpu"))  # deploy gets the raw array
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = dm.predict(img)
        p_dep = np.array(list(r["probs"].values()))
        worst = max(worst, float(np.abs(p_src - p_dep).max()))
        if expected is not None and expected[k] is not None and abs(p_dep[pc] - expected[k]) > 1e-4:
            raise RuntimeError(f"deploy prob {p_dep[pc]:.6f} != stored val prediction {expected[k]:.6f} (image {k})")
    if worst > atol:
        raise RuntimeError(f"deploy/ and src/ disagree (max |diff| = {worst:.2e}); run tools/sync_deploy.py")
    print(f"verified on {len(images)} images: deploy == src (max |diff| {worst:.1e})"
          + (" and == stored val predictions" if expected is not None else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", default="0.5")
    p.add_argument("--val_predictions", help="default: val_predictions.csv next to the checkpoint")
    p.add_argument("--label_names", help="comma separated, in class-index order, e.g. normal,cancer")
    p.add_argument("--data_folder", help="also verify on real validation images (needs val_predictions.csv)")
    p.add_argument("--n_verify", type=int, default=8)
    a = p.parse_args(argv)

    state, meta = read_checkpoint(a.checkpoint)
    if not meta:
        raise SystemExit("bare state_dict without settings: convert it first with `python -m src.models.legacy`")
    meta = dict(meta)
    for k, v in (("mean", list(MEAN)), ("std", list(STD))):
        if meta.get(k) is None:
            print(f"[INFO] '{k}' missing in the checkpoint, set to the training default {v}")
            meta[k] = v
    if meta.get("legacy_preprocess") is None:
        print("[INFO] 'legacy_preprocess' missing, set to False (the src/ default)")
        meta["legacy_preprocess"] = False
    validate_meta(meta)

    class_names = meta.get("class_names") or list(range(int(meta["num_classes"])))
    labels = a.label_names.split(",") if a.label_names else None
    if labels and len(labels) != int(meta["num_classes"]):
        raise SystemExit(f"--label_names has {len(labels)} names for {meta['num_classes']} classes")

    run_dir = os.path.dirname(os.path.abspath(a.checkpoint))
    vp = a.val_predictions or os.path.join(run_dir, "val_predictions.csv")
    rule_text = "argmax (p > 0.5)"
    try:
        thr = float(a.threshold)
        if thr != 0.5:
            rule_text = f"fixed {thr}"
    except ValueError:
        if not os.path.exists(vp):
            raise SystemExit(f"--threshold {a.threshold} needs validation predictions: {vp} not found")
        _, y, pr = read_val_predictions(vp)
        thr, se, sp = choose_threshold(y, pr, a.threshold)
        rule_text = f"{a.threshold} on validation (n={len(y)}): sens {se:.3f}, spec {sp:.3f}"
        print(f"threshold {thr:.4f} ({rule_text})")
    meta["deploy"] = deploy_info(class_names, thr, rule_text, labels, source=os.path.basename(a.checkpoint),
                                 source_epoch=meta.get("epoch"))
    validate_meta(meta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"state_dict": state, "meta": plain(meta)}, a.out)
    print("wrote", a.out)

    rng = np.random.default_rng(0)
    images = [(rng.random((380, 200, 3)) * 255).astype(np.uint8), (rng.random((150, 260)) * 65535).astype(np.uint16)]
    expected = [None, None]
    if a.data_folder and os.path.exists(vp):
        links, _, pr = read_val_predictions(vp)
        for link, prob in list(zip(links, pr))[: a.n_verify]:
            images.append(load_rgb(os.path.join(a.data_folder, link)))
            expected.append(float(prob))
    verify(a.out, images, expected if a.data_folder else None)


if __name__ == "__main__":
    main()
