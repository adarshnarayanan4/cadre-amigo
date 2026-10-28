"""Locate the OpenMDAO/CADRE reference source and record its identity."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

SOURCE_REPOSITORY = "https://github.com/OpenMDAO/CADRE"
REFERENCE_MANIFEST = Path(__file__).with_name("cadre_source.json")


def cadre_root(legacy_root=None):
    """Find the repository root; reject the old OpenMDAO-Plugins layout."""
    configured_root = os.environ.get("CADRE_ROOT")
    project = Path(__file__).resolve().parent

    if configured_root:
        root = Path(configured_root).expanduser()
    elif legacy_root is not None:
        root = Path(legacy_root)
    elif (project.parent / "CADRE-OpenMDAO").is_dir():
        root = project.parent / "CADRE-OpenMDAO"
    else:
        root = project.parent / "CADRE"

    root = root.resolve()

    if not (root / "CADRE" / "CADRE_group.py").is_file():
        raise FileNotFoundError(
            f"Expected an OpenMDAO/CADRE checkout at {root}. "
            "The old src/CADRE layout from OpenMDAO-Plugins is not supported. "
            "Clone https://github.com/OpenMDAO/CADRE.git into ../CADRE-OpenMDAO "
            "from the cadre-amigo folder, or set CADRE_ROOT to that checkout."
        )

    return root


def cadre_path(relative_path, legacy_root=None):
    """Return a file inside the requested repository's CADRE package."""
    path = cadre_root(legacy_root) / "CADRE" / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"OpenMDAO/CADRE reference file is missing: {path}")

    return path


def source_provenance():
    """Verify the pinned source files and data before recording a new result."""
    root = cadre_root()
    manifest = json.loads(REFERENCE_MANIFEST.read_text())

    for relative, expected in manifest["sha256"].items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()

        if actual != expected:
            raise ValueError(
                f"CADRE source differs from the audited revision: {relative}. "
                f"Use OpenMDAO/CADRE revision {manifest['revision']} or audit the change."
            )

    revision = None

    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        revision = result.stdout.strip()

    return {
        "repository": SOURCE_REPOSITORY,
        "reference_revision": manifest["revision"],
        "checkout_revision": revision,
        "reference_files_verified": True,
        "benchmark_sha256": manifest["sha256"]["CADRE/test/data1346.pkl"],
    }
