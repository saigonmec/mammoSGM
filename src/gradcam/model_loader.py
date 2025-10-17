"""Model loading utilities for GradCAM visualization."""

import os
import torch
from torch import nn


def load_full_model(
    model_path: str,
) -> tuple[nn.Module, tuple[int, int], str | None, str | None, dict[str, list[float]] | None]:
    """
    Load a complete model with metadata from checkpoint file.

    This function loads a PyTorch model checkpoint and extracts the model
    along with associated metadata including input size, model name,
    GradCAM target layer, and normalization parameters.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint file (.pth or .pt format).

    Returns
    -------
    model : torch.nn.Module
        The loaded PyTorch model in evaluation mode.
    input_size : tuple of int
        Expected input image size as (height, width). Defaults to (448, 448)
        if not specified in checkpoint.
    model_name : str or None
        Name of the model architecture (e.g., 'resnet50', 'efficientnet').
        None if not stored in checkpoint.
    gradcam_layer : str or None
        Name of the target layer for GradCAM visualization (e.g., 'layer4').
        None if not stored in checkpoint.
    normalize : dict or None
        Normalization parameters with keys 'mean' and 'std', each containing
        a list of float values for RGB channels. None if not stored in checkpoint.
        Example: {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    Raises
    ------
    FileNotFoundError
        If the model checkpoint file does not exist at the specified path.

    Notes
    -----
    - The model is automatically set to evaluation mode (model.eval()).
    - Uses 'cpu' as the default map_location for compatibility.
    - The checkpoint is expected to contain a dictionary with key 'model'
      storing the actual model object.

    Examples
    --------
    >>> model, input_size, model_name, gradcam_layer, normalize = load_full_model('checkpoint.pth')
    >>> print(f"Model: {model_name}, Input size: {input_size}")
    Model: resnet50, Input size: (448, 448)

    >>> # Use with GradCAM
    >>> model_tuple = load_full_model('trained_model.pth')
    >>> model, size, name, layer, norm = model_tuple
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print("Checkpoint:", checkpoint.keys())
    model = checkpoint["model"]
    # model = checkpoint
    model.eval()

    input_size = checkpoint.get("input_size", (448, 448))
    model_name = checkpoint.get("model_name", None)
    gradcam_layer = checkpoint.get("gradcam_layer", None)
    normalize = checkpoint.get("normalize", None)

    return model, input_size, model_name, gradcam_layer, normalize

"""Model loading utilities for GradCAM visualization."""

import os
import torch
from torch import nn


def load_full_mil_model(
    model_path: str
) -> tuple[nn.Module, tuple[int, int], str | None, str | None, dict[str, list[float]] | None, int | None, str | None]:
    """
    Load a complete model with metadata from checkpoint file.

    This function loads a PyTorch model checkpoint and extracts the model
    along with associated metadata including input size, model name,
    GradCAM target layer, normalization parameters, number of patches,
    and architecture type.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint file (.pth or .pt format).

    Returns
    -------
    model : torch.nn.Module
        The loaded PyTorch model in evaluation mode.
    input_size : tuple of int
        Expected input image size as (height, width). Defaults to (448, 448)
        if not specified in checkpoint.
    model_name : str or None
        Name of the model architecture (e.g., 'resnet50', 'efficientnet').
        None if not stored in checkpoint.
    gradcam_layer : str or None
        Name of the target layer for GradCAM visualization (e.g., 'layer4').
        None if not stored in checkpoint.
    normalize : dict or None
        Normalization parameters with keys 'mean' and 'std', each containing
        a list of float values for RGB channels. None if not stored in checkpoint.
        Example: {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    num_patches : int or None
        Number of patches for MIL/patch-based models. None for standard models.
    arch_type : str or None
        Architecture type (e.g., 'mil', 'mil_v2', 'patch_resnet'). None for standard models.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint file does not exist at the specified path.

    Notes
    -----
    - The model is automatically set to evaluation mode (model.eval()).
    - Uses 'cpu' as the default map_location for compatibility.
    - The checkpoint is expected to contain a dictionary with key 'model'
      storing the actual model object.
    - For MIL/patch-based models, num_patches and arch_type are required for
      proper input preprocessing.

    Examples
    --------
    >>> model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = load_full_model('checkpoint.pth')
    >>> print(f"Model: {model_name}, Input size: {input_size}, Patches: {num_patches}")
    Model: resnet50, Input size: (448, 448), Patches: None
    
    >>> # For MIL model
    >>> model_tuple = load_full_model('mil_model.pth')
    >>> model, size, name, layer, norm, patches, arch = model_tuple
    >>> print(f"MIL model with {patches} patches, arch: {arch}")
    MIL model with 4 patches, arch: mil_v4
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = checkpoint['model']
    model.eval()
    
    input_size = checkpoint.get('input_size', (448, 448))
    model_name = checkpoint.get('model_name', None)
    gradcam_layer = checkpoint.get('gradcam_layer', None)
    normalize = checkpoint.get('normalize', None)
    num_patches = checkpoint.get('num_patches', None)
    arch_type = checkpoint.get('arch_type', None)
    
    return model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type