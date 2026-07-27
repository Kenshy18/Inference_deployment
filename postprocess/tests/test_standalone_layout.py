from __future__ import annotations

import unittest
from pathlib import Path


class StandaloneLayoutTests(unittest.TestCase):
    def test_repository_has_only_the_runtime_layers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "run_pipeline.py").is_file())
        for package in ("common", "contracts"):
            self.assertTrue((root / package / "__init__.py").is_file())
        for feature in (
            "preprocessing",
            "nms",
            "cut_detection",
            "tracking",
            "approximation/ellipse",
            "approximation/polygon",
            "keyframes/ellipse",
            "keyframes/polygon",
            "gap_fill",
            "gap_fill/ellipse",
            "gap_fill/polygon",
            "evaluation",
            "artifacts",
            "visualization",
            "tentative",
        ):
            self.assertTrue((root / feature / "__init__.py").is_file(), feature)
        self.assertFalse((root / "atosyori_postprocess").exists())
        self.assertFalse((root / "workflow").exists())
        self.assertFalse((root / "scripts").exists())
        self.assertFalse((root / "tools").exists())

    def test_polygon_responsibilities_are_separate_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "approximation/polygon/rdp.py",
            "keyframes/polygon/interval.py",
            "gap_fill/polygon/interpolate.py",
        ):
            self.assertTrue((root / relative).is_file(), relative)
        self.assertFalse((root / "keyframes/polygon/advanced.py").exists())
        self.assertFalse((root / "approximation/polygon/optimizer.py").exists())

    def test_runtime_does_not_modify_python_import_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime_roots = (
            "common",
            "contracts",
            "preprocessing",
            "nms",
            "cut_detection",
            "tracking",
            "approximation",
            "keyframes",
            "gap_fill",
            "evaluation",
            "artifacts",
            "visualization",
        )
        offenders = [
            str(path.relative_to(root))
            for relative in runtime_roots
            for path in (root / relative).rglob("*.py")
            if "sys.path" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
