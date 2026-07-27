from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from face_dino_v2.trt.bundle import (
    BATCH_SIZE,
    ENGINE_FILES,
    INPUT_SHAPE,
    PLUGIN_FILES,
    PROFILE,
    SCHEMA,
    load_engine_bundle,
    sha256_file,
)
from registry import get_model


def _record(path: Path) -> dict[str, object]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


class FaceDinoV2BundleTest(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        records = {}
        for name, relative in {**ENGINE_FILES, **PLUGIN_FILES}.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            records[name] = _record(path)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "profile": PROFILE,
                    "status": "complete",
                    "batch_size": BATCH_SIZE,
                    "input_shape": list(INPUT_SHAPE),
                    "checkpoint": {"sha256": "checkpoint-hash"},
                    "artifacts": records,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_model_is_registered_as_second_face_backend(self) -> None:
        registration = get_model("face_dino_v2")
        self.assertEqual(registration.default_backend, "tensorrt-fast")
        self.assertEqual(registration.backends, ("tensorrt-fast",))

    def test_bundle_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = load_engine_bundle(
                self._bundle(Path(directory)),
                verify="engines",
            )
            self.assertEqual(bundle.batch_size, 8)
            self.assertEqual(bundle.input_shape, INPUT_SHAPE)

    def test_tampered_engine_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._bundle(root)
            (root / ENGINE_FILES["decoder"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                load_engine_bundle(manifest, verify="engines")


if __name__ == "__main__":
    unittest.main()
