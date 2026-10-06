"""Copy the dependency-free core modules of src/ verbatim into deploy/core/.

    python tools/sync_deploy.py           # copy (run after changing any of the files below)
    python tools/sync_deploy.py --check   # exit 1 if deploy/core is out of date (tests run this)

deploy/ must work with only itself + a weight file, so it cannot import src/. Re-implementing the model
or the preprocessing there would let the two drift apart; instead these files are copied byte for byte
and only the composition (deploy/predictor.py) is deploy-specific (its outputs are tested against src/).
"""

from __future__ import annotations

import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED = [  # (src path, deploy/core path) -- these files may only import each other (relative) + pip packages
    ("src/models/backbone.py", "models/backbone.py"),
    ("src/models/mil_py.py", "models/mil_py.py"),
    ("src/models/factory.py", "models/factory.py"),
    ("src/models/checkpoint.py", "models/checkpoint.py"),
    ("src/data/imaging.py", "data/imaging.py"),
    ("src/data/patches.py", "data/patches.py"),
    ("src/gradcam/cam.py", "gradcam/cam.py"),
    ("src/gradcam/viz.py", "gradcam/viz.py"),
]
INIT_NOTE = '"""Copied verbatim from src/ by tools/sync_deploy.py -- do not edit here."""\n'


def main() -> int:
    check = "--check" in sys.argv
    core = os.path.join(ROOT, "deploy", "core")
    stale = []
    for src, dst in VENDORED:
        s, d = os.path.join(ROOT, src), os.path.join(core, dst)
        if not (os.path.exists(d) and filecmp.cmp(s, d, shallow=False)):
            stale.append(dst)
            if not check:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(s, d)
    if not check:
        for pkg in ("", "models", "data", "gradcam"):
            init = os.path.join(core, pkg, "__init__.py")
            if not os.path.exists(init):
                open(init, "w").write(INIT_NOTE)
    if check:
        if stale:
            print("deploy/core is out of date for:", ", ".join(stale), "-> run: python tools/sync_deploy.py")
            return 1
        print("deploy/core is in sync with src/")
        return 0
    print("copied:", ", ".join(stale) if stale else "nothing (already in sync)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
