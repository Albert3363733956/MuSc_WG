"""Compatibility launcher for MRAD's Windows test script."""

from pathlib import Path
import runpy
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "MRAD-main" / "scripts" / "test_mrad_windows.py"

if not SCRIPT_PATH.exists():
    raise FileNotFoundError(f"MRAD Windows test script not found: {SCRIPT_PATH}")

sys.argv[0] = str(SCRIPT_PATH)
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
