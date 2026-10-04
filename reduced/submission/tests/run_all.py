"""
Runs the tests/test_*.py files in dependency order (core, physics,
control, simulation), one subprocess each, stopping at the first
failure. Each file is a standalone script that shares tests/_helpers.py;
this just calls them in sequence. Run with `python3 tests/run_all.py`.
"""

import subprocess
import sys
from pathlib import Path

ORDER = [
    "test_core.py",       # config + naming + geometry + track
    "test_physics.py",    # physics + data_io
    "test_control.py",
    "test_simulation.py",
]

here = Path(__file__).resolve().parent

for name in ORDER:
    path = here / name
    print(f"\n=== {name} ===")
    result = subprocess.run([sys.executable, str(path)])
    if result.returncode != 0:
        print(f"\nFAILED: {name}")
        sys.exit(result.returncode)

print("\nAll test files passed.")
