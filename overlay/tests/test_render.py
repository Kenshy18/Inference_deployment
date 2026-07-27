from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from overlay_renderer.cli import (
    _fast_command,
    _selected_codec,
    _validate_mode_args,
    build_parser,
    main,
    NATIVE_FFMPEG,
    NATIVE_RENDERER,
)
from overlay_renderer.render import RenderOptions

from helpers import (
    create_mask_sqlite,
    create_rich_face_sqlite,
    create_unified_sqlite,
    create_video,
)


def _read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_index}: {path}")
    return frame


class RenderTests(unittest.TestCase):
    def test_execution_modes_and_overlay_type_alias_are_resolved(self) -> None:
        parser = build_parser()
        common = [
            "--overlay-type",
            "faces",
            "--video",
            "input.mp4",
            "--sqlite",
            "input.sqlite",
            "--output",
            "output.mp4",
        ]
        cpu = parser.parse_args(["--execution-mode", "cpu", *common])
        _validate_mode_args(cpu)
        self.assertEqual("faces", cpu.mode)
        self.assertEqual("h264", _selected_codec(cpu))

        nvenc = parser.parse_args(["--execution-mode", "nvenc", *common])
        _validate_mode_args(nvenc)
        self.assertEqual("h264_nvenc", _selected_codec(nvenc))

        fast = parser.parse_args(
            [
                "--execution-mode",
                "fast",
                "--target-bitrate-mbps",
                "8",
                *common,
            ]
        )
        _validate_mode_args(fast)
        command = _fast_command(fast, output_dir=Path("/tmp/overlay-fast"))
        self.assertIn("/native/segmented.py", " ".join(command))
        self.assertIn("/native/build/overlay_native", " ".join(command))
        self.assertEqual("6", command[command.index("--workers") + 1])
        self.assertEqual("0", command[command.index("--cpu-workers") + 1])

    def test_fast_mode_rejects_missing_bitrate(self) -> None:
        args = build_parser().parse_args(
            [
                "--execution-mode",
                "fast",
                "--mode",
                "final",
                "--video",
                "input.mp4",
                "--sqlite",
                "input.sqlite",
                "--output",
                "output.mp4",
            ]
        )
        with self.assertRaisesRegex(ValueError, "target-bitrate"):
            _validate_mode_args(args)

    @unittest.skipUnless(
        NATIVE_RENDERER.is_file() and NATIVE_FFMPEG.is_file(),
        "native overlay dependencies are not built",
    )
    def test_fast_mode_runs_through_unified_cli_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_h264 = root / "input.mp4"
            subprocess.run(
                [
                    str(NATIVE_FFMPEG),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=0x303030:s=64x48:r=10:d=0.4",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    "-y",
                    str(source_h264),
                ],
                check=True,
            )
            masks = create_mask_sqlite(root / "masks.sqlite")
            output = root / "fast.mp4"
            manifest = root / "fast.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--execution-mode",
                        "fast",
                        "--overlay-type",
                        "final",
                        "--video",
                        str(source_h264),
                        "--sqlite",
                        str(masks),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--target-bitrate-mbps",
                        "1",
                        "--workers",
                        "1",
                        "--cpu-workers",
                        "1",
                        "--ffmpeg-bin",
                        str(NATIVE_FFMPEG),
                    ]
                )
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(4, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual("fast", payload["execution_mode"])
            self.assertEqual("final", payload["overlay_type"])
            self.assertEqual(
                ["libx264"],
                payload["encoding"]["segment_encoders"],
            )

    def test_all_four_modes_create_readable_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            inference = create_unified_sqlite(root / "inference.sqlite")
            masks = create_mask_sqlite(root / "masks.sqlite")
            outputs = {
                "raw": root / "raw.mp4",
                "tracked": root / "tracked.mp4",
                "final": root / "final.mp4",
                "faces": root / "faces.mp4",
            }

            invocations = (
                [
                    "--mode",
                    "raw",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(inference),
                    "--output",
                    str(outputs["raw"]),
                ],
                [
                    "--mode",
                    "tracked",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(masks),
                    "--output",
                    str(outputs["tracked"]),
                ],
                [
                    "--mode",
                    "final",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(masks),
                    "--include-faces",
                    "--face-sqlite",
                    str(inference),
                    "--output",
                    str(outputs["final"]),
                ],
                [
                    "--mode",
                    "faces",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(inference),
                    "--output",
                    str(outputs["faces"]),
                ],
            )
            for invocation in invocations:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(invocation + ["--progress-every", "0"])

            for output in outputs.values():
                self.assertTrue(output.is_file())
                capture = cv2.VideoCapture(str(output))
                self.assertTrue(capture.isOpened())
                self.assertEqual(4, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()

            original_mask_frame = _read_frame(video, 0)
            raw_mask_frame = _read_frame(outputs["raw"], 0)
            tracked_mask_frame = _read_frame(outputs["tracked"], 0)
            original_face_frame = _read_frame(video, 1)
            face_frame = _read_frame(outputs["faces"], 1)
            final_face_frame = _read_frame(outputs["final"], 1)
            self.assertGreater(
                float(np.mean(cv2.absdiff(original_mask_frame, raw_mask_frame))),
                2.0,
            )
            self.assertGreater(
                float(np.mean(cv2.absdiff(original_mask_frame, tracked_mask_frame))),
                2.0,
            )
            self.assertGreater(
                float(np.mean(cv2.absdiff(original_face_frame, face_frame))),
                0.5,
            )
            self.assertGreater(
                float(np.mean(cv2.absdiff(original_face_frame, final_face_frame))),
                0.5,
            )

    def test_final_faces_requires_face_sqlite(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --face-sqlite"):
            main(
                [
                    "--mode",
                    "final",
                    "--video",
                    "input.mp4",
                    "--sqlite",
                    "final.sqlite",
                    "--include-faces",
                    "--output",
                    "output.mp4",
                ]
            )

    def test_rich_face_ellipse_mask_and_keypoints_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            sqlite = create_rich_face_sqlite(root / "rich.sqlite")
            output = root / "faces.mp4"

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--mode",
                        "faces",
                        "--video",
                        str(video),
                        "--sqlite",
                        str(sqlite),
                        "--output",
                        str(output),
                        "--progress-every",
                        "0",
                    ]
                )

            original = _read_frame(video, 1)
            rendered = _read_frame(output, 1)
            self.assertGreater(
                float(np.mean(cv2.absdiff(original, rendered))),
                0.5,
            )

    def test_h264_mode_creates_readable_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            masks = create_mask_sqlite(root / "masks.sqlite")
            output = root / "final_h264.mp4"
            manifest = root / "final_h264.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--mode",
                        "final",
                        "--video",
                        str(video),
                        "--sqlite",
                        str(masks),
                        "--codec",
                        "h264",
                        "--h264-preset",
                        "ultrafast",
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--progress-every",
                        "0",
                    ]
                )
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(4, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()
            self.assertIn('"codec": "h264"', manifest.read_text())

    def test_h264_crf_zero_does_not_force_incompatible_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi")
            masks = create_mask_sqlite(root / "masks.sqlite")
            output = root / "lossless.mp4"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--mode",
                        "final",
                        "--video",
                        str(video),
                        "--sqlite",
                        str(masks),
                        "--codec",
                        "h264",
                        "--h264-crf",
                        "0",
                        "--h264-preset",
                        "ultrafast",
                        "--output",
                        str(output),
                        "--progress-every",
                        "0",
                    ]
                )
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(4, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()

    def test_frame_range_seeks_to_requested_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = create_video(root / "input.avi", frames=8)
            masks = create_mask_sqlite(root / "masks.sqlite")
            output = root / "range.mp4"
            manifest = root / "range.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--mode",
                        "final",
                        "--video",
                        str(video),
                        "--sqlite",
                        str(masks),
                        "--start-frame",
                        "3",
                        "--end-frame",
                        "5",
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--progress-every",
                        "0",
                    ]
                )
            summary = json.loads(manifest.read_text())["summary"]
            self.assertEqual(3, summary["frames_written"])
            self.assertEqual(3, summary["first_frame"])
            self.assertEqual(5, summary["last_frame"])
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(3, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()

    def test_h264_options_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "h264_crf"):
            RenderOptions(mode="final", codec="h264", h264_crf=52).validate()
        with self.assertRaisesRegex(ValueError, "h264_preset"):
            RenderOptions(
                mode="final",
                codec="h264",
                h264_preset="not-a-preset",
            ).validate()
        with self.assertRaisesRegex(ValueError, "nvenc_cq"):
            RenderOptions(
                mode="final",
                codec="h264_nvenc",
                nvenc_cq=52,
            ).validate()
        with self.assertRaisesRegex(ValueError, "nvenc_preset"):
            RenderOptions(
                mode="final",
                codec="h264_nvenc",
                nvenc_preset="p8",
            ).validate()
        with self.assertRaisesRegex(ValueError, "target_bitrate_mbps"):
            RenderOptions(
                mode="final",
                codec="h264_nvenc",
                target_bitrate_mbps=0,
            ).validate()

    def test_nvenc_codec_is_normalized(self) -> None:
        options = RenderOptions(mode="final", codec="nvenc")
        options.validate()
        self.assertTrue(options.uses_nvenc)
        self.assertEqual("h264_nvenc", options.normalized_codec)


if __name__ == "__main__":
    unittest.main()
