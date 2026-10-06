"""Compatibility entry point for the old notebook workflow (`--mode test --setting ... --backbone_name ...`).
Identical to train_patch: the test mode already reports metrics and writes metrics.json / predictions."""

from src.trainer.runner import main

if __name__ == "__main__":
    main("patch")
