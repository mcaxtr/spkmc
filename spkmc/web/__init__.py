"""
SPKMC Web Interface.

This package provides a Streamlit-based web interface for managing and running
SPKMC epidemic simulations through a browser.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def atomic_json_write(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write a JSON file atomically via a temp-file + os.replace().

    Prevents partial/corrupt files when the process is interrupted mid-write.
    """
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(str(tmp), str(path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


__all__ = ["app", "config", "state", "plotting", "components", "runner"]
