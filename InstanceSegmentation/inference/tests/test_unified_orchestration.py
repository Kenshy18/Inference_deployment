from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contracts import (
    BoundingBox,
    Classification,
    Detection,
    DetectionFrame,
    FrameReference,
    ModelDescriptor,
    Segmentation,
    SegmentationFrame,
    SegmentationInstance,
    TaskType,
)
from orchestration.config import InferenceMode, OrchestrationRequest
from orchestration.model_process import build_invocation
from orchestration.pipeline import run_orchestrated_inference
from persistence import SqliteWriter
from registry import get_model


def _segmentation_result(model_id: str) -> SegmentationFrame:
    detection = Detection(
        class_id=1,
        class_name="segment",
        score=0.9,
        bbox=BoundingBox(1, 2, 9, 10),
        classification=Classification(
            class_id=2,
            class_name="classified",
            score=0.8,
            probabilities=(0.1, 0.2, 0.7),
        ),
    )
    return SegmentationFrame(
        model=ModelDescriptor(
            model_id=model_id,
            task=TaskType.INSTANCE_SEGMENTATION,
            implementation="tests.segmenter",
        ),
        frame=FrameReference(0, 0.0, 16, 12),
        instances=(
            SegmentationInstance(
                detection,
                Segmentation(((1, 2, 9, 2, 9, 10),)),
            ),
        ),
    )


def _face_result(model_id: str) -> DetectionFrame:
    return DetectionFrame(
        model=ModelDescriptor(
            model_id=model_id,
            task=TaskType.OBJECT_DETECTION,
            implementation="tests.face_detector",
        ),
        frame=FrameReference(0, 0.0, 16, 12),
        detections=(
            Detection(
                class_id=1,
                class_name="Face",
                score=0.75,
                bbox=BoundingBox(2, 3, 8, 9),
            ),
        ),
    )


def _fake_execute(invocation) -> None:
    writer = SqliteWriter(invocation.output_path)
    result = (
        _segmentation_result(invocation.registration.model_id)
        if invocation.role == "instance_segmentation"
        else _face_result(invocation.registration.model_id)
    )
    writer.set_metadata(
        {
            "input": "input.mp4",
            "model_id": result.model.model_id,
            "task": result.model.task.value,
            "video": {
                "frames": 1,
                "fps": 30.0,
                "width": 16,
                "height": 12,
            },
        }
    )
    writer.write(result)
    writer.close()


def _schema(path: Path) -> tuple[tuple[str, str], ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        )


class UnifiedOrchestrationTest(unittest.TestCase):
    def test_all_modes_publish_the_identical_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.write_bytes(b"test")
            outputs: dict[InferenceMode, Path] = {}
            with patch(
                "orchestration.pipeline.execute_invocation",
                side_effect=_fake_execute,
            ):
                for mode in InferenceMode:
                    output = root / f"{mode.value}.sqlite"
                    run_orchestrated_inference(
                        OrchestrationRequest(
                            input_path=input_path,
                            output_path=output,
                            mode=mode,
                            segmentation_model=(
                                "dinov3_codino"
                                if mode.uses_segmentation
                                else None
                            ),
                            runtime_python=Path(sys.executable),
                        )
                    )
                    outputs[mode] = output

            schemas = [_schema(path) for path in outputs.values()]
            self.assertTrue(all(schema == schemas[0] for schema in schemas))
            with sqlite3.connect(
                outputs[InferenceMode.SEGMENTATION_FACE]
            ) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM model_executions"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM frames"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM detections"
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM classifications"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentations"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                )

    def test_model_command_resolves_supported_backend_and_face_classes(self) -> None:
        request = OrchestrationRequest(
            input_path=Path("input.mp4"),
            output_path=Path("output.sqlite"),
            mode=InferenceMode.SEGMENTATION_FACE,
            segmentation_model="dinov3_codino",
            segmentation_backend="pytorch",
            face_classes=("Face", "Head"),
            runtime_python=Path(sys.executable),
        )
        segmentation = build_invocation(
            get_model("dinov3_codino"),
            role="instance_segmentation",
            output_path=Path("seg.sqlite"),
            request=request,
        )
        face = build_invocation(
            get_model("rtdetr_head_face"),
            role="face_detection",
            output_path=Path("face.sqlite"),
            request=request,
        )
        self.assertIn("--backend", segmentation.command)
        self.assertIn("pytorch", segmentation.command)
        self.assertEqual(face.command[-3:], ("--classes", "Face", "Head"))

    def test_unsupported_backend_is_rejected_before_model_execution(self) -> None:
        request = OrchestrationRequest(
            input_path=Path("input.mp4"),
            output_path=Path("output.sqlite"),
            mode=InferenceMode.SEGMENTATION,
            segmentation_model="eva02_cascade",
            segmentation_backend="tensorrt-fast",
            runtime_python=Path(sys.executable),
        )
        with self.assertRaisesRegex(ValueError, "does not support backend"):
            build_invocation(
                get_model("eva02_cascade"),
                role="instance_segmentation",
                output_path=Path("eva.sqlite"),
                request=request,
            )

    def test_failed_model_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            output_path = root / "existing.sqlite"
            input_path.write_bytes(b"test")
            output_path.write_bytes(b"previous-result")
            request = OrchestrationRequest(
                input_path=input_path,
                output_path=output_path,
                mode=InferenceMode.FACE,
                runtime_python=Path(sys.executable),
                overwrite=True,
            )
            with patch(
                "orchestration.pipeline.execute_invocation",
                side_effect=RuntimeError("model failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    run_orchestrated_inference(request)
            self.assertEqual(output_path.read_bytes(), b"previous-result")


if __name__ == "__main__":
    unittest.main()
