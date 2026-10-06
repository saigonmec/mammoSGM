"""MIL classifier (local patches + global image, MILv4).   python -m src.trainer.train_patch --help"""

from src.trainer.runner import main

if __name__ == "__main__":
    main("patch")
