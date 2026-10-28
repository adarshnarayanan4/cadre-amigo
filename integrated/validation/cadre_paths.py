"""Check reference selection and rejection of missing or changed CADRE files."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cadre_paths


class CadrePathChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.package = self.root / "CADRE"
        self.package.mkdir()
        (self.package / "CADRE_group.py").write_text("# Reference marker\n")
        self.environment = patch.dict("os.environ", {"CADRE_ROOT": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_explicit_checkout_overrides_default(self):
        self.assertEqual(cadre_paths.cadre_root(), self.root)
        self.assertEqual(
            cadre_paths.cadre_path("CADRE_group.py"), self.package / "CADRE_group.py"
        )

    def test_old_layout_is_rejected(self):
        (self.package / "CADRE_group.py").unlink()
        old = self.root / "src" / "CADRE"
        old.mkdir(parents=True)
        (old / "CADRE_assembly.py").touch()

        with self.assertRaisesRegex(FileNotFoundError, "OpenMDAO-Plugins"):
            cadre_paths.cadre_root()

    def test_missing_file_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "reference file is missing"):
            cadre_paths.cadre_path("test/data1346.pkl")

    def test_changed_reference_is_rejected(self):
        relative = "CADRE/CADRE_group.py"
        expected = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
        manifest = self.root / "source.json"
        manifest.write_text(
            json.dumps({"revision": "reference-commit", "sha256": {relative: expected}})
        )
        (self.root / relative).write_text("# Changed source\n")

        with patch.object(cadre_paths, "REFERENCE_MANIFEST", manifest):
            with self.assertRaisesRegex(
                ValueError, "differs from the audited revision"
            ):
                cadre_paths.source_provenance()


if __name__ == "__main__":
    unittest.main()
