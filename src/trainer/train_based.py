import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import torch
from src.data.based_data import get_dataloaders
from src.data.dataloader import load_metadata

from src.data.dataloader import load_metadata
from src.trainer.engines import train_model, evaluate_model
from src.utils.common import load_config, get_arg_or_config, clear_cuda_memory


def parse_img_size(val):
    if val is None:
        return None
    if "x" in val:
        h, w = val.lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def prepare_data_and_model(
    data_folder,
    model,
    batch_size,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    mode="train",  # thêm mode để truyền vào get_dataloaders
    target_column=None,  # thêm target_column
):
    clear_cuda_memory()
    train_df, test_df, _ = load_metadata(
        data_folder, config_path, target_column=target_column
    )
    train_loader, test_loader = get_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode=mode,  # Đặt mode là test để không sử dụng nhiều worker
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained_model_path:
        try:
            model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device)
            )
            print(f"Loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"⚠️ Error loading pretrained model: {e}. Training from scratch.")
    model = model.to(device)
    return train_df, test_df, train_loader, test_loader, model, device


def run_train(
    data_folder,
    model,
    batch_size,
    num_epochs,
    lr,
    output,
    config_path="config/config.yaml",
    img_size=None,
    patience=50,
    loss_type="ce",
    model_type=None,  # thêm model_type để truyền vào
    pretrained_model_path=None,  # thêm tham số này
    target_column=None,  # thêm target_column
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_data_and_model(
            data_folder,
            model,
            batch_size,
            config_path=config_path,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,  # truyền vào đây
            target_column=target_column,
        )
    )
    model_name = f"{model_type}" if model_type else "based"
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_name,  # tên model + kiến trúc
        output=output,
        dataset_folder=data_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
        arch_type="based",
    )
    return trained_model


def run_test(
    data_folder,
    model,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,  # thêm target_column
):
    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        data_folder,
        model,
        batch_size,
        config_path=config_path,
        img_size=img_size,
        pretrained_model_path=pretrained_model_path,
        mode="test",  # Đặt mode là test để không sử dụng nhiều worker
        target_column=target_column,
    )
    print("\nEvaluation on Test Set:")
    test_loss, test_acc = evaluate_model(
        model, test_loader, device=device, mode="Test", return_loss=True
    )
    # Save the full model (architecture + weights) after test

    if pretrained_model_path:
        full_model_path = (
            pretrained_model_path.replace(".pth", "_full.pth")
            if pretrained_model_path.endswith(".pth")
            else pretrained_model_path + "_full"
        )
        try:
            torch.save(model, full_model_path)
            print(f"Saved full model (architecture + weights) to {full_model_path}")
        except Exception as e:
            print(f"⚠️ Error saving full model: {e}")
    return test_loss, test_acc


def run_gradcam(
    data_folder,
    model,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,  # thêm target_column
):
    import time

    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        data_folder,
        model,
        batch_size,
        config_path=config_path,
        img_size=img_size,
        pretrained_model_path=pretrained_model_path,
        mode="test",
        target_column=target_column,
    )
    # Determine save path based on pretrained_model_path
    if pretrained_model_path:
        gradcam_model_path = (
            pretrained_model_path.replace(".pth", "_full.pth")
            if pretrained_model_path.endswith(".pth")
            else pretrained_model_path + "_full"
        )
    else:
        gradcam_model_path = os.path.join(
            output, f"gradcam_{type(model).__name__}_full.pth"
        )
    # Use get_gradcam_layer to determine the correct gradcam layer
    model_name = type(model).__name__.lower()
    gradcam_layer = get_gradcam_layer(model, model_name)
    # Default normalization (same as A.Normalize([0.5]*3, [0.5]*3))
    normalize = {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    # Calculate real inference time using a dummy input
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    model.eval()
    with torch.no_grad():
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        inference_time = end - start  # seconds for one forward pass
    # Save dict with model, input_size, gradcam_layer, model_name, normalize, inference_time
    model_info = {
        "model": model,
        "input_size": img_size,
        "gradcam_layer": gradcam_layer,
        "model_name": type(model).__name__,
        "normalize": normalize,
        "inference_time": inference_time,
    }
    try:
        torch.save(model_info, gradcam_model_path)
        print(f"Saved full model and info for GradCAM to {gradcam_model_path}")
        print("Model info:")
        for k, v in model_info.items():
            if k != "model":
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"⚠️ Error saving GradCAM model: {e}")
    return gradcam_model_path


def get_gradcam_layer(model, model_name):
    """
    Return the name of the appropriate layer for GradCAM based on model_name.
    If not matched, return the last layer's name.
    """
    # ResNet, ResNeXt, ResNeSt
    if "resnet" in model_name or "resnext" in model_name or "resnest" in model_name:
        return "layer4"
    # ConvNeXtV2 Tiny
    elif "convnextv2_tiny" in model_name:
        return "stages.3.blocks.2.conv_dw"
    # ConvNeXt, ConvNeXtV2
    elif "convnext" in model_name:
        return "stages.3.blocks.2.conv_dw"
    # MaxViT Tiny
    elif "maxvit_tiny" in model_name:
        return "stages.-1.blocks.-1.conv.norm2"
    # RegNetY
    elif "regnety" in model_name:
        return "s4.b1.conv3.conv"
    # EfficientNet, EfficientNetV2
    elif "efficientnet" in model_name:
        return "blocks.5.-1.conv_pwl"
    # If not matched, return the last layer's name
    else:
        # Try to get the last layer's name
        children = list(model.named_children())
        if children:
            return children[-1][0]
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file name in config folder",
    )
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "gradcam"], default="train"
    )
    parser.add_argument("--patience", type=int)
    parser.add_argument(
        "--loss_type", type=str, choices=["ce", "focal", "focal2", "ldam"]
    )
    parser.add_argument(
        "--img_size", type=str, default=None, help="Image size, e.g. 448 or 448x448"
    )
    parser.add_argument(
        "--target_column", type=str, default=None, help="Name of target column"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    data_folder = get_arg_or_config(args.data_folder, config.get("data_folder"), None)
    model_type = get_arg_or_config(args.model_type, config.get("model_type"), None)
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    num_epochs = get_arg_or_config(args.num_epochs, config.get("num_epochs"), 10)
    lr = get_arg_or_config(args.lr, config.get("lr"), 1e-4)
    pretrained_model_path = get_arg_or_config(
        args.pretrained_model_path, config.get("pretrained_model_path"), None
    )
    output = get_arg_or_config(args.output, config.get("output"), "output")
    patience = get_arg_or_config(args.patience, config.get("patience"), 50)
    loss_type = get_arg_or_config(args.loss_type, config.get("loss_type"), "ce")
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    target_column = get_arg_or_config(
        args.target_column, config.get("target_column"), None
    )
    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    from src.models.based_model import get_based_model

    train_df, test_df, class_names = load_metadata(
        data_folder, args.config, print_stats=False, target_column=target_column
    )
    model = get_based_model(model_type=model_type, num_classes=len(class_names))

    if args.mode == "train":
        run_train(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            output=output,
            config_path=args.config,
            img_size=img_size,
            patience=patience,
            loss_type=loss_type,
            model_type=model_type,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
        )
    elif args.mode == "test":
        run_test(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
        )
    elif args.mode == "gradcam":
        run_gradcam(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
        )
