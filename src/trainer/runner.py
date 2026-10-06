"""Shared CLI / orchestration behind train_based.py and train_patch.py.

Precedence for every setting: command line > config yaml > DEFAULTS below.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.data.imaging import MEAN, STD
from src.data.loaders import build_loaders
from src.data.metadata import LABEL_COL, balanced_class_weights, class_counts, load_metadata
from src.models.checkpoint import build_model_from_checkpoint, deploy_info, load_weights, read_checkpoint
from src.models.factory import ARCH_TYPES, get_model, is_mil
from src.trainer.engines import evaluate, report, save_predictions, train_model
from src.utils.common import (
    clear_cuda_memory, get_device, load_config, num_workers_default, parse_img_size, set_seed,
)
from src.utils.loss import build_criterion
from src.utils.plot import plot_confusion_matrix, plot_roc

DEFAULTS = dict(
    mode="train", data_folder=None, model_type="resnet50", arch_type=None,
    batch_size=16, num_epochs=50, lr=1e-4, weight_decay=1e-2, output="output", img_size=448,
    target_column="cancer", pretrained_model_path=None, patience=50, loss_type="ce",
    balance="sampler", val_ratio=0.1, group_column=None, monitor="auc", seed=42,
    num_workers=None, amp=True, pretrained=True, multi_gpu=True, run_name=None, overwrite=False,
    warmup_epochs=5, lr_patience=20, grad_clip=None, device=None, legacy_preprocess=False,
    # patch / MIL only
    num_patches=3, overlap_ratio=0.2, local_scale=1.0, rotate_landscape=True, fusion="fuse",
    # mil_v4a only
    aux_weight=0.5, branch_dropout=0.2, neck_dim=512, top_t=0.05,
    backbone_lr_mult=1.0,
)
ALIASES = {"dataset_folder": "data_folder", "outputs": "output", "image_size": "img_size"}  # old config keys


# ----------------------------------------------------------------------------- CLI
def build_parser(kind: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plain-image classifier" if kind == "based" else "MIL (local patches + global image) classifier"
    )
    p.add_argument("--config", default="config.yaml", help="yaml in src/config or a path")
    p.add_argument("--mode", choices=["train", "test"])
    p.add_argument("--data_folder", help="folder with metadata.csv and the images")
    p.add_argument("--model_type", help="backbone, e.g. resnet50, convnextv2_tiny")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--output", help="output root; each run gets its own sub-folder")
    p.add_argument("--img_size", help="448 or HxW, e.g. 224x224 (size of every network input)")
    p.add_argument("--target_column", help="label column in metadata.csv")
    p.add_argument("--pretrained_model_path", help="test: checkpoint to evaluate; train: init weights")
    p.add_argument("--patience", type=int, help="early-stopping patience (epochs)")
    p.add_argument("--loss_type", choices=["ce", "focal", "focal2", "ldam"])
    p.add_argument("--balance", choices=["sampler", "loss", "none"],
                   help="class imbalance: balanced sampler OR class-weighted loss (never both)")
    p.add_argument("--val_ratio", type=float, help="fraction of train patients held out as validation")
    p.add_argument("--group_column", help="patient-id column used to split / check leakage")
    p.add_argument("--monitor", choices=["auc", "acc", "loss"], help="validation metric for model selection")
    p.add_argument("--seed", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--warmup_epochs", type=int)
    p.add_argument("--lr_patience", type=int)
    p.add_argument("--grad_clip", type=float)
    p.add_argument("--backbone_lr_mult", type=float, help="LR of the pretrained backbone = lr * this (MIL models)")
    p.add_argument("--device", help="cuda | cpu | cuda:1 ...")
    p.add_argument("--run_name")
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument("--backbone_name", help="alias of --model_type (kept for the old notebook workflow)")
    p.add_argument("--setting", help="free label (e.g. 'mil') added to the test output folder name")
    p.add_argument("--legacy_preprocess", action="store_true", default=None,
                   help="src_legacy resizing / patch geometry (needed to evaluate weights trained with src_legacy)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_pretrained", action="store_true", help="do not load ImageNet weights")
    p.add_argument("--single_gpu", action="store_true", help="disable DataParallel")
    if kind == "patch":
        p.add_argument("--arch_type", choices=[a for a in ARCH_TYPES if is_mil(a)], help="MIL variant")
        p.add_argument("--num_patches", type=int, help="number of local strips (the global image is extra)")
        p.add_argument("--overlap_ratio", type=float, help="overlap between neighbouring strips")
        p.add_argument("--local_scale", type=float,
                       help="local patches are cut from an image this many times wider than --img_size")
        p.add_argument("--fusion", choices=["fuse", "concat", "cross_attention"])
        p.add_argument("--no_rotate_landscape", action="store_true")
        p.add_argument("--aux_weight", type=float, help="mil_v4a: weight of the per-branch auxiliary losses (0 = off)")
        p.add_argument("--branch_dropout", type=float, help="mil_v4a: prob. of dropping each branch for the fusion head (<=0.5)")
        p.add_argument("--neck_dim", type=int, help="mil_v4a: width of the per-branch projection")
        p.add_argument("--top_t", type=float, help="mil_v4a: top-t fraction of cells pooled from the saliency maps (GMIC)")
    return p


def resolve_config(args: argparse.Namespace, kind: str) -> dict:
    cfg = dict(DEFAULTS)
    cfg["arch_type"] = "based" if kind == "based" else "mil_v4"
    ignored = []
    for k, v in load_config(args.config).items():
        k = ALIASES.get(k, k)
        if k in DEFAULTS:
            if v is not None:
                cfg[k] = v
        else:
            ignored.append(k)
    if ignored:
        print(f"[INFO] ignoring config keys not used by this code: {ignored}")
    for k, v in vars(args).items():
        if k in DEFAULTS and v is not None:
            cfg[k] = v
    if args.backbone_name and not args.model_type:
        cfg["model_type"] = args.backbone_name
    cfg["setting"] = args.setting
    if args.no_amp:
        cfg["amp"] = False
    if args.no_pretrained:
        cfg["pretrained"] = False
    if args.single_gpu:
        cfg["multi_gpu"] = False
    if getattr(args, "no_rotate_landscape", False):
        cfg["rotate_landscape"] = False
    if kind == "based":
        cfg["arch_type"] = "based"
    elif not is_mil(str(cfg["arch_type"])) or cfg["arch_type"] not in ARCH_TYPES:
        raise SystemExit(f"arch_type '{cfg['arch_type']}' is not available in this code base "
                         f"(supported: {[a for a in ARCH_TYPES if is_mil(a)]}; older variants live in src_legacy/)")
    cfg["img_size"] = parse_img_size(cfg["img_size"])
    if cfg["num_workers"] is None:
        cfg["num_workers"] = num_workers_default()
    if not cfg["data_folder"]:
        raise SystemExit("--data_folder (or data_folder in the config) is required")
    return cfg


# ----------------------------------------------------------------------------- helpers
def _patch_kwargs(s: dict) -> dict:
    return dict(
        num_patches=int(s["num_patches"]), overlap_ratio=float(s["overlap_ratio"]),
        local_scale=float(s["local_scale"]), rotate_landscape=bool(s["rotate_landscape"]),
    )


def _model_kwargs(s: dict) -> dict:
    return dict(neck_dim=int(s["neck_dim"]), branch_dropout=float(s["branch_dropout"]), aux_weight=float(s["aux_weight"]),
                top_t=float(s["top_t"]))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _evaluate_and_save(model, loader, df, device, num_classes, class_names, out_dir, tag, amp):
    metrics, labels, probs = evaluate(model, loader, device, num_classes, amp)
    report(metrics, class_names, title=f"[{tag}]")
    save_predictions(os.path.join(out_dir, f"{tag}_predictions.csv"), df["link"].tolist(), labels, probs, class_names)
    plot_confusion_matrix(metrics["cm"], [str(c) for c in class_names],
                          os.path.join(out_dir, f"{tag}_confusion_matrix.png"), title=f"{tag} confusion matrix")
    if num_classes == 2:
        plot_roc(labels, probs, os.path.join(out_dir, f"{tag}_roc.png"), title=f"{tag} ROC")
    return metrics


# ----------------------------------------------------------------------------- train
def run_train(cfg: dict, kind: str, device: torch.device) -> dict:
    arch = cfg["arch_type"]
    if cfg["legacy_preprocess"]:
        if kind == "patch":  # the old strip geometry leaves ~7-10% of every image uncovered by the strips
            raise SystemExit("--legacy_preprocess reproduces src_legacy's strip geometry, which leaves 7-10% of the "
                             "image height uncovered by any strip. It exists only to evaluate old weights (--mode test / "
                             "Grad-CAM); do not train MIL models with it.")
        print("[WARN] training with --legacy_preprocess (src_legacy resizing); meant for evaluating old weights only.")
    train_df, val_df, test_df, class_names = load_metadata(
        cfg["data_folder"], cfg["target_column"], cfg["val_ratio"], cfg["seed"], cfg["group_column"]
    )
    num_classes = len(class_names)
    h, w = cfg["img_size"]
    patch = _patch_kwargs(cfg) if kind == "patch" else None
    loaders = build_loaders(
        kind, train_df, val_df, test_df, cfg["data_folder"], (h, w), num_classes,
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], balance=cfg["balance"],
        seed=cfg["seed"], pin_memory=device.type == "cuda", patch_kwargs=patch,
        legacy=bool(cfg["legacy_preprocess"]),
    )

    model_kwargs = _model_kwargs(cfg) if arch == "mil_v4a" else {}
    model = get_model(arch, cfg["model_type"], num_classes, pretrained=cfg["pretrained"], fusion=cfg["fusion"], **model_kwargs)
    if cfg["pretrained_model_path"]:
        state, _ = read_checkpoint(cfg["pretrained_model_path"])
        load_weights(model, state, cfg["pretrained_model_path"])  # raises on mismatch
        print(f"Initialised weights from {cfg['pretrained_model_path']}")
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and cfg["multi_gpu"]:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    labels = train_df[LABEL_COL].values
    weights = balanced_class_weights(labels, num_classes).to(device) if cfg["balance"] == "loss" else None
    criterion = build_criterion(cfg["loss_type"], weights, class_counts(labels, num_classes)).to(device)
    print(f"Loss: {cfg['loss_type']} | class balancing via: {cfg['balance']}")

    dataset = os.path.basename(os.path.normpath(cfg["data_folder"]))
    if cfg["run_name"]:
        run_name = cfg["run_name"]
    else:
        parts = [dataset, arch, cfg["model_type"], f"{h}x{w}"]
        if kind == "patch":
            parts += [f"p{cfg['num_patches']}"] + ([cfg["fusion"]] if arch == "mil_v4" else [])
        parts.append(time.strftime("%Y%m%d-%H%M%S"))
        run_name = "_".join(parts)
    run_dir = os.path.join(cfg["output"], run_name)
    if os.path.exists(os.path.join(run_dir, "best.pth")) and not cfg["overwrite"]:
        raise SystemExit(f"{run_dir} already contains a run; use another --run_name or --overwrite")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    meta = dict(
        version=2, arch_type=arch, model_type=cfg["model_type"], num_classes=num_classes,
        class_names=class_names, target_column=cfg["target_column"], img_size=[h, w],
        mean=list(MEAN), std=list(STD), seed=cfg["seed"],
        fusion=cfg["fusion"] if (kind == "patch" and arch == "mil_v4") else None,
        num_patches=cfg["num_patches"] if kind == "patch" else None,
        overlap_ratio=cfg["overlap_ratio"] if kind == "patch" else None,
        local_scale=cfg["local_scale"] if kind == "patch" else None,
        rotate_landscape=cfg["rotate_landscape"] if kind == "patch" else None,
        legacy_preprocess=bool(cfg["legacy_preprocess"]),
        model_kwargs=model_kwargs or None,
        deploy=deploy_info(class_names, source_run=run_name),  # best.pth / last.pth are deployable as they are
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_jsonable({"cfg": cfg, "meta": meta}), f, indent=2)

    result = train_model(
        model, loaders["train"], loaders["val"], num_classes=num_classes, run_dir=run_dir,
        criterion=criterion, device=device, meta=meta, num_epochs=cfg["num_epochs"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], use_amp=cfg["amp"], patience=cfg["patience"],
        monitor=cfg["monitor"], warmup_epochs=cfg["warmup_epochs"], lr_patience=cfg["lr_patience"],
        grad_clip=cfg["grad_clip"], backbone_lr_mult=cfg["backbone_lr_mult"],
    )
    if not os.path.exists(result["best_path"]):
        raise RuntimeError("training produced no checkpoint (num_epochs=0?)")

    print(f"\nBest epoch {result['best_epoch']} (val {result['monitor']} = {result['best_score']:.4f}); "
          f"evaluating that checkpoint:")
    state, _ = read_checkpoint(result["best_path"])
    load_weights(model, state, result["best_path"])
    val_m = _evaluate_and_save(model, loaders["val"], val_df, device, num_classes, class_names, run_dir, "val", cfg["amp"])
    test_m = _evaluate_and_save(model, loaders["test"], test_df, device, num_classes, class_names, run_dir, "test", cfg["amp"])
    summary = {"best_epoch": result["best_epoch"], "monitor": result["monitor"],
               "best_val_score": result["best_score"], "val": val_m, "test": test_m}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\nArtifacts in {run_dir}")
    return {"run_dir": run_dir, **summary}


# ----------------------------------------------------------------------------- test
def run_test(cfg: dict, kind: str, device: torch.device) -> dict:
    path = cfg["pretrained_model_path"]
    if not path:
        raise SystemExit("--mode test needs --pretrained_model_path")
    _, _, test_df, class_names = load_metadata(
        cfg["data_folder"], cfg["target_column"], make_val=False, seed=cfg["seed"], group_column=cfg["group_column"]
    )
    overrides = {k: cfg[k] for k in ("arch_type", "model_type", "fusion", "img_size", "num_patches",
                                      "overlap_ratio", "local_scale", "rotate_landscape", "legacy_preprocess")}
    overrides["num_classes"] = len(class_names)
    model, s = build_model_from_checkpoint(path, overrides, device)
    if int(s["num_classes"]) != len(class_names):
        raise SystemExit(f"checkpoint has {s['num_classes']} classes but '{cfg['target_column']}' has {len(class_names)}")
    if s.get("class_names") is not None and list(s["class_names"]) != list(class_names):
        print(f"[WARN] class order differs from training: ckpt {s['class_names']} vs data {class_names}")
    kind = "patch" if is_mil(s["arch_type"]) else "based"
    img_size = parse_img_size(s["img_size"])
    print(f"Evaluating {path}\n  arch={s['arch_type']} backbone={s['model_type']} img_size={img_size}"
          + (f" patches={s['num_patches']} overlap={s['overlap_ratio']} fusion={s.get('fusion')}" if kind == "patch" else ""))
    loaders = build_loaders(
        kind, None, None, test_df, cfg["data_folder"], img_size, len(class_names),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], seed=cfg["seed"],
        pin_memory=device.type == "cuda", patch_kwargs=_patch_kwargs(s) if kind == "patch" else None,
        legacy=bool(s.get("legacy_preprocess")),
    )
    stem = os.path.splitext(os.path.basename(path))[0]
    tag = f"{cfg['setting']}_" if cfg.get("setting") else ""
    out_dir = os.path.join(cfg["output"], f"test_{tag}{os.path.basename(os.path.dirname(os.path.abspath(path)))}_{stem}")
    os.makedirs(out_dir, exist_ok=True)
    m = _evaluate_and_save(model, loaders["test"], test_df, device, len(class_names), class_names, out_dir, "test", cfg["amp"])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(_jsonable({"checkpoint": path, "settings": s, "test": m}), f, indent=2)
    print(f"Artifacts in {out_dir}")
    return {"out_dir": out_dir, "test": m}


def main(kind: str, argv=None) -> dict:
    args = build_parser(kind).parse_args(argv)
    cfg = resolve_config(args, kind)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    clear_cuda_memory()
    return run_train(cfg, kind, device) if cfg["mode"] == "train" else run_test(cfg, kind, device)
