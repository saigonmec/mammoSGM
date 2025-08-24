import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import csv
from datetime import datetime

from src.utils.loss import FocalLoss
from src.utils.plot import plot_metrics, plot_confusion_matrix


def evaluate_model(model, data_loader, device="cpu", mode="Test", return_loss=False):
    # Multi-GPU unwrap if needed
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(
                probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy()
            )

    acc = correct / total
    avg_loss = total_loss / total
    # Tính các chỉ số
    try:
        auc = (
            roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
        )
    except Exception:
        auc = None
    # Calculate precision and recall safely for binary/multiclass and degenerate cases
    try:
        precision = precision_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        precision = 0.0
        pass
    try:
        recall = recall_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        recall = 0.0
        pass
    cm = confusion_matrix(all_labels, all_preds)
    # Sensitivity (Recall) và Specificity
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sen = recall
        spec = None

    # Gọn lại phần in ra
    acc_loss_str = f"{mode} Accuracy : {acc * 100:.2f}% | Loss: {avg_loss:.4f}"
    if auc is not None:
        acc_loss_str += f" | AUC: {auc * 100:.2f}%"
    print(acc_loss_str)
    print(
        f"{mode} Precision: {precision * 100:.2f}% | Sens: {recall * 100:.2f}%"
        + (f" | Spec: {spec * 100:.2f}%" if spec is not None else "")
    )

    # In thông số tốt nhất nếu có
    if hasattr(evaluate_model, "best_acc"):
        best_str = f"Best Accuracy : {evaluate_model.best_acc * 100:.2f}%"
        if hasattr(evaluate_model, "best_loss"):
            best_str += f" | Loss: {evaluate_model.best_loss:.4f}"
        if hasattr(evaluate_model, "best_auc") and evaluate_model.best_auc is not None:
            best_str += f" | AUC: {evaluate_model.best_auc * 100:.2f}%"
        print(best_str)
    # Cập nhật best nếu tốt hơn
    if not hasattr(evaluate_model, "best_acc") or acc > evaluate_model.best_acc:
        evaluate_model.best_acc = acc
        evaluate_model.best_loss = avg_loss
        evaluate_model.best_auc = auc

    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    return (avg_loss, acc) if return_loss else acc


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    lr=1e-4,
    device="cpu",
    model_name="model",
    output="output",
    dataset_folder="None",
    train_df=None,
    patience=50,
    loss_type="ce",
    arch_type=None,
    num_patches=None,
):
    # Multi-GPU support
    if device != "cpu" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        print("Multi-GPU is available and will be used for training.")
        model = nn.DataParallel(model)
    else:
        if device != "cpu" and torch.cuda.device_count() == 1:
            print("Only one GPU detected. Multi-GPU is not available.")
        elif device == "cpu":
            print("Running on CPU. Multi-GPU is not available.")
    model = model.to(device)
    if train_df is not None:
        class_counts = train_df["cancer"].value_counts()
        total_samples = len(train_df)
        num_classes = len(class_counts)
        weights = torch.tensor(
            [
                total_samples / (num_classes * class_counts[i])
                for i in range(num_classes)
            ],
            dtype=torch.float,
        ).to(device)
    else:
        weights = None

    if loss_type == "focal":
        criterion = FocalLoss(alpha=weights)
        print("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(
            "Using CrossEntropyLoss (with weight)"
            if weights is not None
            else "Using CrossEntropyLoss"
        )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler() if device != "cpu" else None
    print(f"Using AMP: {scaler is not None}")

    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "figures")
    log_dir = os.path.join(output, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    dataset = dataset_folder.split("/")[-1]
    img_size = None
    try:
        sample = next(iter(train_loader))
        images, _ = sample
        if images.ndim == 4:
            img_size = (images.shape[2], images.shape[3])
        elif images.ndim == 5:
            img_size = (images.shape[3], images.shape[4])
    except Exception:
        img_size = (448, 448)
    imgsize_str = f"{img_size[0]}x{img_size[1]}"
    # Thêm p{num_patches} vào tên model nếu có
    patch_str = f"_p{num_patches}" if num_patches is not None else ""
    model_key = f"{dataset}_{imgsize_str}_{arch_type}_{model_name}{patch_str}"

    print(f"Checking for existing weights in {model_dir} with model_key: {model_key}")
    existing_weights = []
    for fname in os.listdir(model_dir):
        if fname.startswith(model_key) and fname.endswith(".pth"):
            try:
                acc_part = fname.replace(".pth", "").split("_")[-1]
                acc_val = float(acc_part) / 10000
                existing_weights.append((fname, acc_val))
            except Exception:
                print(f"Skipping invalid model file: {fname}")
    if existing_weights:
        print("Found existing weights:")
        for fname, acc_val in existing_weights:
            print(f"  - {fname} (accuracy: {acc_val:.4f})")
    else:
        print("No existing weights found.")

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    best_acc = 0.0
    patience_counter = 0
    last_lr = lr

    log_file = os.path.join(log_dir, f"{model_key}.csv")
    # Ghi header nếu file chưa tồn tại
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as logf:
            writer = csv.writer(logf)
            writer.writerow(
                [
                    "datetime",
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "test_loss",
                    "test_acc",
                    "lr",
                    "patience_counter",
                    "best_acc",
                ]
            )

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for batch in loop:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            # print("Labels:", labels)
            outputs = model(images)
            # print("outputs.shape:", outputs.shape)
            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        test_loss, test_acc = evaluate_model(
            model, test_loader, device=device, mode="Test", return_loss=True
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Ghi log csv mỗi epoch
        with open(log_file, "a", newline="") as logf:
            writer = csv.writer(logf)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    epoch + 1,
                    round(epoch_loss, 6),
                    round(epoch_acc, 6),
                    round(test_loss, 6),
                    round(test_acc, 6),
                    optimizer.param_groups[0]["lr"],
                    patience_counter,
                    round(best_acc, 6),
                ]
            )

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < last_lr:
            print(
                f"Learning rate reduced to {current_lr:.6f}, resetting patience counter"
            )
            patience_counter = 0
            last_lr = current_lr
        else:
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    with open(log_file, "a", newline="") as logf:
                        writer = csv.writer(logf)
                        writer.writerow(
                            [
                                datetime.now().isoformat(),
                                f"early_stop_epoch_{epoch + 1}",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                    break

        # Save/load model: handle DataParallel
        acc4 = int(round(test_acc * 10000))
        weight_name = f"{model_key}_{acc4}.pth"
        weight_path = os.path.join(model_dir, weight_name)

        related_weights = []
        for fname in os.listdir(model_dir):
            if fname.startswith(model_key) and fname.endswith(".pth"):
                try:
                    acc_part = fname.replace(".pth", "").split("_")[-1]
                    acc_val = float(acc_part) / 10000
                    related_weights.append((acc_val, os.path.join(model_dir, fname)))
                except Exception:
                    print(f"Skipping invalid model file: {fname}")
                    continue

        related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
        top2_accs = set(acc for acc, _ in related_weights[:2])

        if test_acc in top2_accs:
            print(
                f"⏩ Skipped saving {weight_name} (accuracy {test_acc:.6f} already in top 2)"
            )
        else:
            related_weights.append((test_acc, weight_path))
            related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
            top2_paths = set(path for _, path in related_weights[:2])
            if weight_path in top2_paths:
                # Save only model.module if DataParallel
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), weight_path)
                else:
                    torch.save(model.state_dict(), weight_path)
                print(f"✅ Saved new model: {weight_name} (acc = {test_acc:.6f})")
            for _, fname_path in related_weights:
                if fname_path not in top2_paths and os.path.exists(fname_path):
                    try:
                        os.remove(fname_path)
                        print(f"🗑️ Deleted model: {fname_path}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {fname_path}: {e}")

        plot_path = os.path.join(plot_dir, f"{model_key}.png")
        plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)

        all_labels = []
        all_preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        cm_path = os.path.join(plot_dir, f"{model_key}_confusion_matrix.png")
        class_names = [str(i) for i in sorted(set(all_labels))]
        # plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)

    with open(log_file, "a", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(
            [datetime.now().isoformat(), "finished", "", "", "", "", "", "", ""]
        )

    # Evaluate best weight trong top2_accs
    if top2_accs:
        best_acc = max(top2_accs)
        best_weight_name = f"{model_key}_{int(round(best_acc * 10000))}.pth"
        best_weight_path = os.path.join(model_dir, best_weight_name)
        if os.path.exists(best_weight_path):
            print(
                f"\n🔎 Evaluating best model: {best_weight_name} (acc={best_acc:.4f})"
            )
            # Load weights into model.module if DataParallel
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(
                    torch.load(best_weight_path, map_location=device)
                )
            else:
                model.load_state_dict(torch.load(best_weight_path, map_location=device))
            evaluate_model(model, test_loader, device=device, mode="Best Test")
        else:
            print(f"Best weight file {best_weight_name} not found.")
    else:
        print("No best weight found for evaluation.")

    print(f"Training log saved to {log_file}")
    print(f"{model_name} training finished.")
    return model
