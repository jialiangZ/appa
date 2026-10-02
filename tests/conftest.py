r"""Shared test configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ensure the src layout and the repository root are importable regardless of
# how pytest is invoked.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
