"""
Put the repo root and notebooks/ on sys.path so the notebooks can import the
project packages from any working directory. Every notebook starts with
`from nblib.bootstrap import REPO_ROOT`.
"""

import sys
import pathlib

_parents = pathlib.Path(__file__).resolve().parents
REPO_ROOT = next((p for p in _parents if (p / "baseline_code.py").exists()), None)


for _p in (str(REPO_ROOT), str(REPO_ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
