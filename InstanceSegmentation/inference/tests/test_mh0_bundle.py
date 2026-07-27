from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dinov3_codino_mh0.trt.bundle import (
    ENGINE_FILES,
    PLUGIN_FILE,
    PROFILE,
    SCHEMA,
    load_engine_bundle,
    sha256_file,
)
from registry import get_model


def _record(path: Path) -> dict[str, object]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


class Mh0BundleTest(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        artifacts = {}
        for name, relative in ENGINE_FILES.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}-engine".encode())
            artifacts[name] = _record(path)
        plugin = root / PLUGIN_FILE
        plugin.parent.mkdir(parents=True, exist_ok=True)
        plugin.write_bytes(b"plugin")
        artifacts["plugin"] = _record(plugin)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "profile": PROFILE,
                    "status": "complete",
                    "batch_size": 16,
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_registered_model_uses_fast_backend_by_default(self) -> None:
        registration = get_model("dinov3_codino_mh0")
        self.assertEqual(registration.default_backend, "tensorrt-fast")
        self.assertEqual(
            registration.backends,
            ("tensorrt-fast", "pytorch"),
        )

    def test_complete_bundle_passes_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = load_engine_bundle(
                self._bundle(Path(directory)),
                verify="engines",
            )
            self.assertEqual(bundle.batch_size, 16)
            self.assertEqual(set(bundle.engines), set(ENGINE_FILES))

    def test_tampered_engine_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._bundle(root)
            (root / ENGINE_FILES["decoder"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                load_engine_bundle(manifest, verify="engines")


if __name__ == "__main__":
    unittest.main()
