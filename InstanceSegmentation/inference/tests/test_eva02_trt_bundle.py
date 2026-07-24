from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eva02_cascade.trt.bundle import (
    MANIFEST_SCHEMA,
    PROFILE,
    file_record,
    load_trt_bundle,
)


class Eva02TrtBundleTest(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, dict[str, Path]]:
        engine = root / "engines" / "eva02.engine"
        engine.parent.mkdir(parents=True)
        engine.write_bytes(b"engine-data")
        source = {
            "checkpoint": root / "source" / "model.pth",
            "classifier_checkpoint": root / "source" / "classifier.pt",
            "config": root / "source" / "config.py",
        }
        for name, path in source.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode("utf-8"))
        payload = {
            "schema": MANIFEST_SCHEMA,
            "profile": PROFILE,
            "status": "complete",
            "precision": "fp16",
            "shape_profile": {
                "target_size": 1280,
                "min_batch": 1,
                "opt_batch": 12,
                "max_batch": 20,
            },
            "io": {"input_name": "images", "output_name": "last_feat"},
            "engine": file_record(
                engine, stored_path="engines/eva02.engine"
            ),
            "source": {
                name: file_record(path) for name, path in source.items()
            },
            "validation": {"status": "pass"},
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest, source

    def test_full_verification_accepts_bound_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, source = self._bundle(Path(directory))
            bundle = load_trt_bundle(
                manifest,
                verify="full",
                checkpoint_path=source["checkpoint"],
                classifier_checkpoint=source["classifier_checkpoint"],
                config_path=source["config"],
            )
            self.assertEqual(bundle.max_batch, 20)
            self.assertEqual(bundle.output_name, "last_feat")

    def test_supplied_source_can_move_when_hash_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, source = self._bundle(root)
            moved = root / "moved-checkpoint.pth"
            moved.write_bytes(source["checkpoint"].read_bytes())
            source["checkpoint"].unlink()
            load_trt_bundle(
                manifest,
                verify="full",
                checkpoint_path=moved,
                classifier_checkpoint=source["classifier_checkpoint"],
                config_path=source["config"],
            )

    def test_engine_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _source = self._bundle(Path(directory))
            engine = manifest.parent / "engines" / "eva02.engine"
            engine.write_bytes(b"tampered-engine")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_trt_bundle(manifest, verify="engine")


if __name__ == "__main__":
    unittest.main()
