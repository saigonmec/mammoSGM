"""Deployment package: this folder + a weight file are all that is needed (see deploy/README.md)."""

from .predictor import MammoModel

__all__ = ["MammoModel"]
__version__ = "1.0"
