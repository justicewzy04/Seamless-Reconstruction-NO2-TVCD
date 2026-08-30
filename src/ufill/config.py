"""Portable project paths.

Override data and output locations with ``UFILL_DATA_ROOT`` and
``UFILL_OUTPUT_ROOT`` instead of editing source files.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("UFILL_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("UFILL_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser()
CHECKPOINT_ROOT = Path(
    os.environ.get("UFILL_CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")
).expanduser()
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def env_path(name: str, default: Path) -> Path:
    """Return a path from an environment variable, falling back to *default*."""

    return Path(os.environ.get(name, default)).expanduser()
