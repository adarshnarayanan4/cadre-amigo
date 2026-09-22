"""Resolve files from the external CADRE checkout."""

import os
from pathlib import Path


def cadre_path(relative_path, legacy_root=None):
    """Return a CADRE file path, honoring ``CADRE_ROOT`` when set."""
    configured_root = os.environ.get("CADRE_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser()
    elif legacy_root is not None:
        root = Path(legacy_root)
    else:
        root = Path(__file__).resolve().parent.parent / "CADRE"

    return root / "src" / "CADRE" / relative_path
