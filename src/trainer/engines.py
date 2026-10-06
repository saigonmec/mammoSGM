"""Train / evaluate loops.

Protocol (fixes the original, which selected on the test set):
  * model selection, LR scheduling and early stopping use the VALIDATION loader only;
  * the test set is evaluated once, with the best-validation checkpoint (see runner.py).
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.checkpoint import save_checkpoint
from src.utils.metrics import compute_metrics, format_metrics
from src.utils.plot import plot_training_curves


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False, grad_clip=None, desc=""):
    model.train()
    # models with auxiliary heads (V4a) declare {aux key: weight}; the loss then also trains every branch alone
    aux_w = getattr(unwrap(model), "aux_loss_weights", None) or {}
    running, correct, total, skipped = 0.0, 0, 0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty()):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):  # exactly ONE forward per step
            out = model(images, return_aux=True) if aux_w else model(images)
        logits, aux = out if aux_w else (out, None)
        main_loss = criterion(logits.float(), labels)
        loss = main_loss
        for key, w in aux_w.items():
            loss = loss + w * criterion(aux[key].float(), labels)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = labels.size(0)
        running += main_loss.item() * bs  # logged loss = the main (fusion) loss, comparable with the val loss
        correct += (logits.argmax(1) == labels).sum().item()
        total += bs
    if skipped:
        print(f"[WARN] skipped {skipped} batch(es) with non-finite loss")
    if total == 0:
        raise FloatingPointError("every batch of the epoch had a non-finite loss")
    return running / total, correct / total


@torch.no_grad()
def predict(model, loader, device, use_amp=False, diagnostics=False):
    """-> labels (n,), logits (n, C) in loader order (eval loaders never shuffle), diag dict.
    diagnostics=True (MIL models): per-branch logits, mean attention peak, and how much the model relies
    on its local branch (V4: gate weight, V4a: share of the |logit contributions|)."""
    model.eval()
    want = diagnostics and getattr(unwrap(model), "supports_aux", False)
    all_logits, all_labels = [], []
    branch = {"global": [], "local": []}
    reliance, attn_max = [], []
    for images, labels in tqdm(loader, desc="eval", leave=False, disable=not sys.stderr.isatty()):
        with _autocast(device, use_amp):
            out = model(images.to(device, non_blocking=True), return_aux=True) if want else model(images.to(device, non_blocking=True))
        logits, aux = out if want else (out, None)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
        if want:
            for k in branch:
                if aux.get(f"logits_{k}") is not None:
                    branch[k].append(aux[f"logits_{k}"].float().cpu())
            attn_max.append(aux["attn"].float().max(dim=1).values.cpu())
            if aux.get("fusion_weights") is not None:
                reliance.append(aux["fusion_weights"][:, 0].float().cpu())
            elif aux.get("contrib_local") is not None:
                cl, cg = aux["contrib_local"].float().abs().sum(1), aux["contrib_global"].float().abs().sum(1)
                reliance.append((cl / (cl + cg).clamp_min(1e-12)).cpu())
    diag = {}
    if want:
        diag["branch_logits"] = {k: torch.cat(v) for k, v in branch.items() if v}
        diag["attn_max"] = float(torch.cat(attn_max).mean())
        if reliance:
            diag["local_reliance"] = float(torch.cat(reliance).mean())
    return torch.cat(all_labels).numpy(), torch.cat(all_logits), diag


def evaluate(model, loader, device, num_classes, use_amp=False):
    """-> (metrics dict, labels, probs). Loss is plain (unweighted) cross-entropy.
    For MIL models metrics also holds 'branches' (acc / auc of each auxiliary head), 'local_reliance'
    and 'attn_max' (a peak of 1/N means uniform attention)."""
    labels, logits, diag = predict(model, loader, device, use_amp, diagnostics=True)
    loss = F.cross_entropy(logits, torch.from_numpy(labels)).item()
    probs = torch.softmax(logits, dim=1).numpy()
    metrics = compute_metrics(labels, probs, num_classes, loss=loss)
    for name, bl in diag.get("branch_logits", {}).items():
        bm = compute_metrics(labels, torch.softmax(bl, dim=1).numpy(), num_classes)
        metrics.setdefault("branches", {})[name] = {"acc": bm["acc"], "auc": bm["auc"] if num_classes == 2 else bm["macro_auc"]}
    for k in ("local_reliance", "attn_max"):
        if k in diag:
            metrics[k] = diag[k]
    return metrics, labels, probs


def monitor_score(metrics: dict, monitor: str, num_classes: int):
    """Higher is better. -> (score, name actually used)."""
    if monitor == "loss":
        return -metrics["loss"], "loss"
    if monitor == "auc":
        auc = metrics["auc"] if num_classes == 2 else metrics["macro_auc"]
        if auc is not None:
            return auc, "auc"
        return metrics["acc"], "acc"  # AUC undefined (single class in val)
    return metrics["acc"], "acc"


def is_improvement(score, best_score, loss, best_loss, tol=1e-6) -> bool:
    """Better monitor score; on a tie (e.g. AUC saturated at 1.0 on an easy validation set) the lower
    validation loss wins, so that an early, badly calibrated epoch is not kept just for being first."""
    if score > best_score + tol:
        return True
    return abs(score - best_score) <= tol and loss < best_loss - tol


def _param_groups(model, weight_decay, lr, backbone_lr_mult=1.0):
    """AdamW groups: no weight decay on biases / norm params; the pretrained backbone (`base_model.*` of
    the MIL models) can use a smaller learning rate than the freshly initialised heads."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    has_bb = any(n.startswith(("base_model.", "module.base_model.")) for n in names)
    if backbone_lr_mult != 1.0 and not has_bb:
        print("[INFO] backbone_lr_mult ignored: this model has no separate backbone (plain classifier)")
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        bb = has_bb and n.startswith(("base_model.", "module.base_model."))
        key = (bb, p.ndim <= 1)
        groups.setdefault(key, []).append(p)
    out = []
    for (bb, no_decay), params in sorted(groups.items()):
        base_lr = lr * (backbone_lr_mult if bb else 1.0)
        out.append({"params": params, "weight_decay": 0.0 if no_decay else weight_decay, "lr": base_lr, "base_lr": base_lr})
    return out


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    num_classes,
    run_dir,
    criterion,
    device,
    meta,
    num_epochs=50,
    lr=1e-4,
    weight_decay=1e-2,
    use_amp=True,
    patience=50,
    monitor="auc",
    warmup_epochs=5,
    lr_patience=20,
    grad_clip=None,
    min_lr=1e-6,
    backbone_lr_mult=1.0,
):
    """Trains, keeping `best.pth` (best validation `monitor`) and `last.pth` in `run_dir`.
    Returns {"best_epoch", "best_score", "monitor", "history", "best_path", "last_path"}."""
    os.makedirs(run_dir, exist_ok=True)
    best_path, last_path = os.path.join(run_dir, "best.pth"), os.path.join(run_dir, "last.pth")
    log_path = os.path.join(run_dir, "log.csv")

    optimizer = torch.optim.AdamW(_param_groups(model, weight_decay, lr, backbone_lr_mult), lr=lr)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=min_lr
    )
    use_amp = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    print(f"Device: {device} | AMP: {use_amp} | epochs: {num_epochs} | monitor: val {monitor} | patience: {patience}")

    columns = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc",
               "val_sens", "val_spec", "monitor_score", "is_best", "seconds"]
    header_written = False

    history, best_score, best_loss, best_epoch, bad_epochs, used_monitor = [], -np.inf, np.inf, 0, 0, monitor
    try:
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            if epoch <= warmup_epochs:  # linear warm-up, then ReduceLROnPlateau on val loss
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * epoch / warmup_epochs
            cur_lr = max(g["lr"] for g in optimizer.param_groups)  # the head learning rate

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, use_amp, grad_clip,
                desc=f"epoch {epoch}/{num_epochs}",
            )
            m, _, _ = evaluate(model, val_loader, device, num_classes, use_amp)
            score, used = monitor_score(m, monitor, num_classes)
            if used != monitor and used_monitor == monitor:
                print(f"[WARN] val {monitor} undefined, falling back to val {used} for model selection")
            used_monitor = used

            improved = is_improvement(score, best_score, m["loss"], best_loss)
            if improved:
                best_score, best_loss, best_epoch, bad_epochs = score, m["loss"], epoch, 0
                save_checkpoint(best_path, model, {**meta, "epoch": epoch, "monitor": used, "best_score": float(score)})
            else:
                bad_epochs += 1

            if epoch > warmup_epochs:
                plateau.step(m["loss"])
                if max(g["lr"] for g in optimizer.param_groups) < cur_lr:
                    print(f"  lr reduced -> {max(g['lr'] for g in optimizer.param_groups):.2e} (patience counter reset)")
                    bad_epochs = 0

            row = {
                "epoch": epoch, "lr": cur_lr, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": m["loss"], "val_acc": m["acc"], "val_auc": m["auc"] if num_classes == 2 else m["macro_auc"],
                "val_sens": m["sens"], "val_spec": m["spec"], "monitor_score": float(score),
                "is_best": bool(improved), "seconds": round(time.time() - t0, 1),
            }
            for name, bm in m.get("branches", {}).items():  # health of each branch of a MIL model
                row[f"val_auc_{name}"] = bm["auc"]
            for k in ("local_reliance", "attn_max"):
                if k in m:
                    row[k] = m[k]
            if not header_written:
                columns = columns + [k for k in row if k not in columns]
                with open(log_path, "w", newline="") as f:
                    csv.writer(f).writerow(columns)
                header_written = True
            history.append(row)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([row.get(c) for c in columns])
            auc_s = "n/a" if row["val_auc"] is None else f"{row['val_auc']:.4f}"
            extra = ""
            if m.get("branches"):
                extra += " | branch auc " + " ".join(f"{n[0]}:{'n/a' if b['auc'] is None else format(b['auc'], '.3f')}" for n, b in m["branches"].items())
            if "local_reliance" in m:
                extra += f" | local reliance {m['local_reliance']:.2f}"
            print(
                f"Epoch {epoch:>3}/{num_epochs} lr {cur_lr:.2e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {m['loss']:.4f} acc {m['acc']:.4f} auc {auc_s}{extra}" + ("  * best" if improved else "")
            )
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no val {used_monitor} improvement for {patience} epochs)")
                break
        last_rel = history[-1].get("local_reliance") if history else None
        if last_rel is not None and not 0.05 <= last_rel <= 0.95:
            print(f"[WARN] branch collapse: the model relies on {'local' if last_rel > 0.95 else 'global'} features only "
                  f"(local reliance {last_rel:.3f}); the other branch contributes nothing. See README (MILv4a).")
    finally:
        if history:  # also on Ctrl-C: keep what we have
            save_checkpoint(last_path, model, {**meta, "epoch": history[-1]["epoch"]})
            plot_training_curves(history, os.path.join(run_dir, "curves.png"))

    return {
        "best_epoch": best_epoch, "best_score": float(best_score), "monitor": used_monitor,
        "history": history, "best_path": best_path, "last_path": last_path,
    }


def save_predictions(path, links, labels, probs, class_names) -> None:
    probs = np.asarray(probs)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["link", "label", "pred"] + [f"prob_{c}" for c in class_names])
        for link, y, p in zip(links, labels, probs):
            w.writerow([link, int(y), int(p.argmax())] + [f"{v:.6f}" for v in p])


def report(metrics: dict, class_names, title: str) -> None:
    print(format_metrics(metrics, class_names, title=title))
    if metrics.get("branches"):
        print("   branches (val/test AUC): " + ", ".join(
            f"{n}={'n/a' if b['auc'] is None else format(b['auc'], '.3f')}" for n, b in metrics["branches"].items())
              + (f" | local reliance {metrics['local_reliance']:.2f}" if "local_reliance" in metrics else "")
              + (f" | attention peak {metrics['attn_max']:.2f}" if "attn_max" in metrics else ""))
