"""numpy-only classification metrics (no sklearn needed)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(labels, preds, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels, int), np.asarray(preds, int)), 1)
    return cm


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    _, inv, counts = np.unique(sorted_a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = (cum - counts + 1 + cum) / 2.0
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = avg[inv]
    return ranks


def roc_auc(y_true_bool, score) -> float | None:
    """Binary ROC-AUC via the Mann-Whitney U statistic. None if only one class present."""
    y = np.asarray(y_true_bool).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(y_true_bool, score):
    """Returns fpr, tpr (with the (0,0) start point)."""
    y = np.asarray(y_true_bool).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    tps = np.cumsum(y)
    fps = np.cumsum(~y)
    keep = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]  # last index of each tie group
    tps, fps = tps[keep], fps[keep]
    tpr = np.r_[0.0, tps / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps / max(fps[-1], 1)]
    return fpr, tpr


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def compute_metrics(labels, probs, num_classes: int, loss: float | None = None) -> dict:
    """labels [n], probs [n, C] (softmax). Everything returned is plain python / lists."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    preds = probs.argmax(1)
    n = len(labels)
    cm = confusion_matrix(labels, preds, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    pred_cnt = cm.sum(0).astype(np.float64)
    precision_c = _safe_div(tp, pred_cnt)
    recall_c = _safe_div(tp, support)
    f1_c = _safe_div(2 * precision_c * recall_c, precision_c + recall_c)

    out = {
        "n": int(n),
        "loss": None if loss is None else float(loss),
        "acc": float(tp.sum() / max(n, 1)),
        "macro_f1": float(f1_c.mean()),
        "cm": cm.tolist(),
        "per_class": {
            "precision": precision_c.tolist(),
            "recall": recall_c.tolist(),
            "f1": f1_c.tolist(),
            "support": support.astype(int).tolist(),
        },
        "auc": None,
        "macro_auc": None,
        "weighted_auc": None,
    }
    if num_classes == 2:
        out["auc"] = roc_auc(labels == 1, probs[:, 1])
        out["precision"] = float(precision_c[1])
        out["sens"] = float(recall_c[1])
        out["spec"] = float(recall_c[0])
        out["f1"] = float(f1_c[1])
    else:
        aucs, weights = [], []
        for c in range(num_classes):
            a = roc_auc(labels == c, probs[:, c])
            if a is not None:
                aucs.append(a)
                weights.append(support[c])
        if aucs:
            out["macro_auc"] = float(np.mean(aucs))
            out["weighted_auc"] = float(np.average(aucs, weights=weights))
        out["precision"] = float(precision_c.mean())
        out["sens"] = float(recall_c.mean())
        out["spec"] = None
        out["f1"] = out["macro_f1"]
    return out


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.2f}%"


def format_metrics(m: dict, class_names=None, title: str = "") -> str:
    head = f"{title} " if title else ""
    loss = "n/a" if m.get("loss") is None else f"{m['loss']:.4f}"
    line1 = f"{head}Acc {_pct(m['acc'])} | Loss {loss}"
    if m.get("auc") is not None:
        line1 += f" | AUC {_pct(m['auc'])}"
    if m.get("macro_auc") is not None:
        line1 += f" | macro-AUC {_pct(m['macro_auc'])} | weighted-AUC {_pct(m['weighted_auc'])}"
    line2 = f"{head}Prec {_pct(m['precision'])} | Sens {_pct(m['sens'])}"
    if m.get("spec") is not None:
        line2 += f" | Spec {_pct(m['spec'])}"
    line2 += f" | F1 {_pct(m['f1'])}"
    lines = [line1, line2]
    pc = m["per_class"]
    names = class_names or list(range(len(pc["f1"])))
    for i, name in enumerate(names):
        lines.append(
            f"   class {name}: P {pc['precision'][i]:.3f}  R {pc['recall'][i]:.3f}  "
            f"F1 {pc['f1'][i]:.3f}  n={pc['support'][i]}"
        )
    lines.append(f"   confusion matrix (rows=true, cols=pred): {m['cm']}")
    return "\n".join(lines)
