from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dinov3_codino.trt.bundle import (
    ENGINE_FILENAMES,
    FAST_ENGINE_FILENAMES,
    FAST_PLUGIN_FILENAME,
    FAST_PRECISION_POLICY,
    FAST_PRECISION_POLICY_CLEAN,
    IMAGE_SIZE,
    INPUT_SIZE,
    MANIFEST_SCHEMA,
    PRECISION_POLICY,
    PROFILE,
    PROFILE_FAST,
    QUERY_SHAPES,
    RUNTIME_CHECKPOINT_FILE,
    load_engine_bundle,
    sha256_file,
)


def _record(path: Path, *, stored_path: str | None = None) -> dict[str, object]:
    return {
        "path": stored_path or str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class CoDinoBundleTest(unittest.TestCase):
    def _create_bundle(self, root: Path) -> tuple[Path, dict[str, Path]]:
        engines_dir = root / "engines"
        engines_dir.mkdir()
        engines: dict[str, Path] = {}
        records: dict[str, dict[str, object]] = {}
        for name, filename in ENGINE_FILENAMES.items():
            path = engines_dir / filename
            path.write_bytes(f"fake-{name}-engine".encode())
            engines[name] = path
            records[name] = _record(path, stored_path=f"engines/{filename}")

        config = root / "resolved_config.py"
        checkpoint = root / "epoch_2.pth"
        classifier = root / "best.pt"
        runtime = root / "python"
        config.write_text("model = {}\n", encoding="utf-8")
        checkpoint.write_bytes(b"detector checkpoint")
        classifier.write_bytes(b"classifier checkpoint")
        runtime.write_bytes(b"runtime python")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "profile": PROFILE,
            "status": "complete",
            "fixed_batch": True,
            "batch_size": 2,
            "input_tensor_size": list(INPUT_SIZE),
            "runtime_image_size": list(IMAGE_SIZE),
            "query_encoder_shapes": QUERY_SHAPES,
            "precision_policy": PRECISION_POLICY,
            "engines": records,
            "source": {
                "config": _record(config),
                "checkpoint": _record(checkpoint),
                "classifier_checkpoint": _record(classifier),
            },
            "runtime_python": _record(runtime),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, {
            "config": config,
            "checkpoint": checkpoint,
            "classifier": classifier,
            "runtime": runtime,
            **engines,
        }

    def _create_fast_bundle(self, root: Path) -> tuple[Path, Path]:
        manifest, files = self._create_bundle(root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        fast_records = {}
        for name, filename in FAST_ENGINE_FILENAMES.items():
            source = files[name]
            target = source.with_name(filename)
            source.rename(target)
            fast_records[name] = _record(
                target,
                stored_path=f"engines/{filename}",
            )
        plugins = root / "plugins"
        plugins.mkdir()
        plugin = plugins / FAST_PLUGIN_FILENAME
        plugin.write_bytes(b"fake SM120 plugin")
        runtime_checkpoint = root / RUNTIME_CHECKPOINT_FILE
        runtime_checkpoint.parent.mkdir()
        runtime_checkpoint.write_bytes(b"fake runtime checkpoint")
        payload.update(
            {
                "profile": PROFILE_FAST,
                "precision_policy": FAST_PRECISION_POLICY,
                "execution_policy": {"runtime_profile": "fast-b2"},
                "engines": fast_records,
                "query_plugin_extension": _record(
                    plugin,
                    stored_path=f"plugins/{FAST_PLUGIN_FILENAME}",
                ),
                "runtime_checkpoint": _record(
                    runtime_checkpoint,
                    stored_path=RUNTIME_CHECKPOINT_FILE,
                ),
            }
        )
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest, plugin

    def test_complete_bundle_passes_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, files = self._create_bundle(Path(directory))
            bundle = load_engine_bundle(
                manifest,
                verify="full",
                config_path=files["config"],
                checkpoint_path=files["checkpoint"],
                classifier_checkpoint=files["classifier"],
                runtime_python=files["runtime"],
            )

            self.assertEqual(bundle.profile, PROFILE)
            self.assertEqual(bundle.batch_size, 2)
            self.assertEqual(bundle.engines["decoder"], files["decoder"].resolve())

    def test_engine_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, files = self._create_bundle(Path(directory))
            files["decoder"].write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "decoder engine size mismatch"):
                load_engine_bundle(manifest, verify="engines")

    def test_engine_records_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._create_bundle(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["engines"]["unexpected"] = payload["engines"]["decoder"]
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly four engine"):
                load_engine_bundle(manifest, verify="metadata")

    def test_fast_bundle_requires_verified_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, plugin = self._create_fast_bundle(Path(directory))
            bundle = load_engine_bundle(manifest, verify="engines")
            self.assertEqual(bundle.profile, PROFILE_FAST)
            self.assertEqual(bundle.runtime_profile, "fast-b2")
            self.assertIsNotNone(bundle.runtime_checkpoint)

            plugin.write_bytes(b"tampered plugin")
            with self.assertRaisesRegex(ValueError, "plugin size mismatch"):
                load_engine_bundle(manifest, verify="engines")

    def test_fast_runtime_checkpoint_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._create_fast_bundle(root)
            runtime_checkpoint = root / RUNTIME_CHECKPOINT_FILE
            runtime_checkpoint.write_bytes(b"tampered runtime checkpoint")

            with self.assertRaisesRegex(
                ValueError,
                "runtime checkpoint size mismatch",
            ):
                load_engine_bundle(manifest, verify="engines")

    def test_clean_rebuild_precision_policy_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._create_fast_bundle(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["precision_policy"] = FAST_PRECISION_POLICY_CLEAN
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            bundle = load_engine_bundle(manifest, verify="engines")

            self.assertEqual(bundle.precision_policy, FAST_PRECISION_POLICY_CLEAN)


if __name__ == "__main__":
    unittest.main()
