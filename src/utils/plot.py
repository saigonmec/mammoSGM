"""Training curves / confusion matrix / ROC. matplotlib is imported lazily and optional."""

from __future__ import annotations

import numpy as np

from .metrics import roc_curve


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable, skipping plot ({e})")
        return None


def plot_training_curves(history: list[dict], save_path: str) -> None:
    plt = _plt()
    if plt is None or not history:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    auc = [h["val_auc"] if h.get("val_auc") is not None else np.nan for h in history]
    axes[2].plot(ep, auc, label="val AUC")
    axes[2].set_title("Validation AUC")
    best = [h for h in history if h.get("is_best")]
    if best:
        b = best[-1]["epoch"]
        for ax in axes:
            ax.axvline(b, color="gray", linestyle="--", alpha=0.6, label=f"best epoch {b}")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path: str, title: str = "Confusion matrix") -> None:
    plt = _plt()
    if plt is None:
        return
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.2 * len(class_names) + 2.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_roc(labels, probs, save_path: str, title: str = "ROC") -> None:
    """Binary ROC from the positive-class probability column."""
    plt = _plt()
    if plt is None:
        return
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr = roc_curve(labels == 1, np.asarray(probs)[:, 1])
    auc = np.trapz(tpr, fpr) if hasattr(np, "trapz") else np.trapezoid(tpr, fpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
