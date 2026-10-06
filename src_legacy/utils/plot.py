import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_metrics(train_losses, train_accs, test_losses, test_accs, save_path):
    """
    Plot training and test loss/accuracy with highlighted best epochs in fivethirtyeight style.

    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test/validation losses per epoch
        test_accs: List of test/validation accuracies per epoch
        save_path: Path to save the plot (e.g., 'output/figures/model.png')
    """
    # Set epoch range
    epochs = list(range(1, len(train_losses) + 1))

    # Find best epochs
    index_loss = np.argmin(test_losses)  # Epoch with lowest test loss
    val_lowest = test_losses[index_loss]
    index_acc = np.argmax(test_accs)  # Epoch with highest test accuracy
    acc_highest = test_accs[index_acc]

    # Set plot style
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Set lighter background color
    fig.patch.set_facecolor("#f7f7f7")
    for ax in axes:
        ax.set_facecolor("#f7f7f7")

    # Plot Loss
    axes[0].plot(epochs, train_losses, "r", label="Training Loss")
    axes[0].plot(epochs, test_losses, "g", label="Test Loss")
    axes[0].scatter(
        index_loss + 1,
        val_lowest,
        s=150,
        c="blue",
        label=f"Best Loss: {val_lowest:.4f} (Epoch {index_loss + 1:d})",
    )
    axes[0].set_title("Training and Test Loss")
    axes[0].set_xlabel("Epochs", color="#222831")
    axes[0].set_ylabel("Loss", color="#222831")
    axes[0].tick_params(axis="x", colors="#222831")
    axes[0].tick_params(axis="y", colors="#222831")
    axes[0].grid(True, linestyle="--", alpha=0.5, color="navy")
    legend = axes[0].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#222831")
    # Set legend text color
    for text in legend.get_texts():
        text.set_color("#222831")

    # Plot Accuracy
    axes[1].plot(epochs, train_accs, "r", label="Training Accuracy")
    axes[1].plot(epochs, test_accs, "g", label="Test Accuracy")
    axes[1].scatter(
        index_acc + 1,
        acc_highest,
        s=150,
        c="blue",
        label=f"Best Acc: {acc_highest:.4f} (Epoch {index_acc + 1:d})",
    )
    axes[1].set_title("Training and Test Accuracy")
    axes[1].set_xlabel("Epochs", color="#222831")
    axes[1].set_ylabel("Accuracy", color="#222831")
    axes[1].tick_params(axis="x", colors="#222831")
    axes[1].tick_params(axis="y", colors="#222831")
    axes[1].grid(True, linestyle="--", alpha=0.5, color="navy")
    legend = axes[1].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#222831")
    for text in legend.get_texts():
        text.set_color("#222831")

    # Thêm đường viền xung quanh khu vực plot
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color("#161A1F")

    # Điều chỉnh vị trí và kích thước để hai ảnh cover 95% chiều rộng, subplot phải không sát viền phải
    fig.subplots_adjust(
        left=0.07, right=0.84, wspace=0.15
    )  # giảm right xuống để subplot phải cách viền

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_metrics_orig(train_losses, train_accs, test_losses, test_accs, save_path):
    """
    Plot training and test loss/accuracy with highlighted best epochs in fivethirtyeight style.

    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test/validation losses per epoch
        test_accs: List of test/validation accuracies per epoch
        save_path: Path to save the plot (e.g., 'output/figures/model.png')
    """
    # Set epoch range
    epochs = list(range(1, len(train_losses) + 1))

    # Find best epochs
    index_loss = np.argmin(test_losses)  # Epoch with lowest test loss
    val_lowest = test_losses[index_loss]
    index_acc = np.argmax(test_accs)  # Epoch with highest test accuracy
    acc_highest = test_accs[index_acc]

    # Set plot style
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Plot Loss
    axes[0].plot(epochs, train_losses, "r", label="Training Loss")
    axes[0].plot(epochs, test_losses, "g", label="Test Loss")
    axes[0].scatter(
        index_loss + 1,
        val_lowest,
        s=150,
        c="blue",
        label=f"Best Loss: {val_lowest:.4f} (Epoch {index_loss + 1:d})",
    )
    axes[0].set_title("Training and Test Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5, color="white")  # grid màu trắng
    legend = axes[0].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    # Plot Accuracy
    axes[1].plot(epochs, train_accs, "r", label="Training Accuracy")
    axes[1].plot(epochs, test_accs, "g", label="Test Accuracy")
    axes[1].scatter(
        index_acc + 1,
        acc_highest,
        s=150,
        c="blue",
        label=f"Best Acc: {acc_highest:.4f} (Epoch {index_acc + 1:d})",
    )
    axes[1].set_title("Training and Test Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5, color="white")  # grid màu trắng
    legend = axes[1].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    # print(f"Saved training plot to {save_path}")


def plot_metrics2(train_losses, train_accs, test_losses, test_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "-o", label="Train Loss")
    plt.plot(epochs, test_losses, "-o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "-o", label="Train Acc")
    plt.plot(epochs, test_accs, "-o", label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy per Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title=None):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create an annotation matrix with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            annot[i, j] = f"{percent:.1f}%\n({count})"

    # Plot the heatmap
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"fontsize": 14},
    )
    # Customize the color bar
    ticks = np.linspace(0, 1, 5)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])

    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)

    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix_orig(
    y_true, y_pred, class_names, save_path=None, title="Confusion Matrix", cmap="Blues"
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.grid(False)  # Tắt grid nếu có
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_cm_roc(
    y_true,
    y_pred,
    y_pred_proba,
    class_names=None,
    title="Confusion Matrix, Metrics & ROC Curve",
):
    """
    Plots confusion matrix, metrics, and ROC curve for binary classification, all side by side.
    Args:
        y_true: Ground truth labels (1D array, shape [n_samples])
        y_pred: Predicted labels (1D array, shape [n_samples])
        y_pred_proba: Predicted probabilities for class 1 (1D array, shape [n_samples])
        class_names: List of class names (default: None)
        title: Title for confusion matrix plot
    """
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create an annotation matrix with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            annot[i, j] = f"{percent:.1f}%\n({count})"

    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    true_negatives = np.sum(cm) - (true_positives + false_positives + false_negatives)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

    auc = roc_auc_score(y_true, y_pred_proba)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Plot Confusion Matrix, Metrics, and ROC Curve side by side
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.4, 1.1, 0.8]
    )  # Make metrics column narrower
    plt.subplots_adjust(wspace=0.01)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # Confusion Matrix
    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"fontsize": 14},
        ax=ax0,
    )
    # Customize the color bar
    ticks = np.linspace(0, 1, 5)
    cbar = ax0.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])

    ax0.set_xlabel("Predicted", fontsize=15)
    ax0.set_ylabel("Actual", fontsize=15)
    ax0.set_title("Confusion Matrix", fontsize=16)

    # Metrics text (percentages with 2 decimal places)
    metrics_text = ""
    for i, name in enumerate(
        class_names if class_names is not None else range(len(cm))
    ):
        metrics_text += f"Class {name}:\n"
        metrics_text += f"  Precision: {precision[i] * 100:.2f}%\n"
        metrics_text += f"  Sensitivity (Recall): {sensitivity[i] * 100:.2f}%\n"
        metrics_text += f"  Specificity: {specificity[i] * 100:.2f}%\n"
        metrics_text += f"  F1 Score: {f1[i] * 100:.2f}%\n\n"
    metrics_text += f"AUC: {auc * 100:.2f}%\n"

    ax2.axis("off")
    ax2.text(
        0,
        0.5,
        metrics_text,
        fontsize=12,
        va="center",
        ha="left",
        family="monospace",
        wrap=True,
    )
    ax2.set_title("Metrics", fontsize=12)

    # ROC Curve (make square)
    ax1.plot(
        fpr, tpr, color="green", linewidth=2.5, label=f"ROC curve (AUC = {auc:.2f})"
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("False Positive Rate", fontsize=13)
    ax1.set_ylabel("True Positive Rate", fontsize=13)
    ax1.set_title("ROC Curve", fontsize=16)
    ax1.legend(loc="lower right", prop={"size": 12})

    plt.suptitle(title, fontsize=20, y=1.05)
    plt.tight_layout()
    plt.show()


def plot_cm_roc_multiclass(
    y_true,
    y_pred,
    y_pred_softmax,
    class_names=None,
    title="Confusion Matrix, Metrics & ROC Curve",
):
    """
    Plots confusion matrix, metrics, and ROC curve for multi-class classification.
    Args:
        y_true: Ground truth labels (1D array, shape [n_samples])
        y_pred: Predicted labels (1D array, shape [n_samples])
        y_pred_softmax: Predicted probabilities for each class (2D array, shape [n_samples, n_classes])
        class_names: List of class names (default: None)
        title: Title for confusion matrix plot
    """
    n_classes = y_pred_softmax.shape[1]
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Annotation matrix
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            annot[i, j] = f"{percent:.1f}%\n({count})"

    # Metrics
    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    true_negatives = np.sum(cm) - (true_positives + false_positives + false_negatives)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

    # ROC & AUC for each class
    aucs = []
    fpr_dict = {}
    tpr_dict = {}
    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        y_score = y_pred_softmax[:, i]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        auc = roc_auc_score(y_true_bin, y_score)
        aucs.append(auc)
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr

    # Plot
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.1, 0.8])
    plt.subplots_adjust(wspace=0.01)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # Confusion Matrix
    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"fontsize": 14},
        ax=ax0,
    )
    ticks = np.linspace(0, 1, 5)
    cbar = ax0.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])
    ax0.set_xlabel("Predicted", fontsize=15)
    ax0.set_ylabel("Actual", fontsize=15)
    ax0.set_title("Confusion Matrix", fontsize=16)

    # Metrics text
    metrics_text = ""
    for i, name in enumerate(
        class_names if class_names is not None else range(n_classes)
    ):
        metrics_text += f"Class {name}:\n"
        metrics_text += f"  Precision: {precision[i] * 100:.2f}%\n"
        metrics_text += f"  Sensitivity (Recall): {sensitivity[i] * 100:.2f}%\n"
        metrics_text += f"  Specificity: {specificity[i] * 100:.2f}%\n"
        metrics_text += f"  F1 Score: {f1[i] * 100:.2f}%\n"
        metrics_text += f"  AUC: {aucs[i] * 100:.2f}%\n\n"
        # Accuracy
        accuracy = np.mean(y_true == y_pred)
        metrics_text += f"Overall Accuracy: {accuracy * 100:.2f}%\n"

    ax2.axis("off")
    ax2.text(
        0,
        0.5,
        metrics_text,
        fontsize=12,
        va="center",
        ha="left",
        family="monospace",
        wrap=True,
    )
    ax2.set_title("Metrics", fontsize=12)

    # ROC Curve for each class
    for i in range(n_classes):
        label = class_names[i] if class_names is not None else f"Class {i}"
        ax1.plot(
            fpr_dict[i],
            tpr_dict[i],
            linewidth=2.5,
            label=f"{label} (AUC={aucs[i]:.2f})",
        )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("False Positive Rate", fontsize=13)
    ax1.set_ylabel("True Positive Rate", fontsize=13)
    ax1.set_title("ROC Curve", fontsize=16)
    ax1.legend(loc="lower right", prop={"size": 12})

    plt.suptitle(title, fontsize=20, y=1.05)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def plot_pr_curve_full(y_true, y_prob, title="PR Curve"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    i_max = np.argmax(f1_scores)
    precision_max, recall_max = precision[i_max], recall[i_max]
    threshold_max = thresholds[i_max] if i_max < len(thresholds) else 1.0
    f1_max = f1_scores[i_max]

    _, ax = plt.subplots(figsize=(8, 6))

    # Đường đẳng F1
    f_scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for f in f_scores:
        x = np.linspace(0.01, 1)
        y = (f * x) / (2 * x - f)
        ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate(f"f1={f:0.1f}", xy=(0.9, y[45] + 0.02))

    # PR curve + threshold color
    s = ax.scatter(recall[:-1], precision[:-1], c=thresholds, cmap="hsv")
    ax.plot(recall, precision, color="blue", alpha=0.7)

    # Chấm đen tại điểm F1 cao nhất
    ax.scatter(recall_max, precision_max, s=30, c="k", label="Max F1")

    # 🔗 Đường kết nối từ trục x
    ax.plot([recall_max, recall_max], [0, precision_max], "k--", linewidth=1, alpha=0.6)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.colorbar(s, label="threshold")
    # plt.title(f"MAX F1 {f1_max:.3f} @ th={threshold_max:.3f}\nPR AUC={pr_auc:.3f}")
    plt.title(
        f"MAX F1 {f1_max:.3f} @ th={threshold_max:.3f}\n prec={precision_max:0.3f}, recall={recall_max:0.3f}, PR AUC={pr_auc:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
