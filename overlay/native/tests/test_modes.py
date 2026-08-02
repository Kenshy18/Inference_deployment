from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "build" / "overlay_native"
FFMPEG = ROOT.parent / ".runtime" / "ffmpeg-nvenc-btbn-8.1" / "bin" / "ffmpeg"
FFPROBE = FFMPEG.with_name("ffprobe")
SEGMENTED = ROOT / "segmented.py"


def load_segmented_module():
    module_name = "_overlay_segmented_test"
    specification = importlib.util.spec_from_file_location(module_name, SEGMENTED)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load segmented runner: {SEGMENTED}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def create_video(path: Path) -> None:
    subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x303030:s=64x48:r=10:d=0.8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:d=0.8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(path),
        ],
        check=True,
    )


def create_pts_gap_video(path: Path) -> None:
    """Create eight frames with a three-frame timestamp gap before frame 4."""
    subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=10:d=0.8",
            "-vf",
            "setpts='PTS+if(gte(N,4),3,0)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "4",
            "-keyint_min",
            "4",
            "-sc_threshold",
            "0",
            "-y",
            str(path),
        ],
        check=True,
    )


def create_inference_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE videos(
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                reported_frame_count INTEGER,
                fps REAL,
                width INTEGER,
                height INTEGER
            );
            CREATE TABLE model_executions(
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL
            );
            CREATE TABLE frames(
                id INTEGER PRIMARY KEY,
                frame_index INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            CREATE TABLE detections(
                id INTEGER PRIMARY KEY,
                frame_id INTEGER NOT NULL,
                model_execution_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL
            );
            CREATE TABLE classifications(
                detection_id INTEGER PRIMARY KEY,
                class_name TEXT NOT NULL,
                score REAL NOT NULL
            );
            CREATE TABLE segmentations(
                detection_id INTEGER PRIMARY KEY,
                encoding TEXT NOT NULL
            );
            CREATE TABLE segmentation_polygons(
                id INTEGER PRIMARY KEY,
                detection_id INTEGER NOT NULL,
                polygon_index INTEGER NOT NULL
            );
            CREATE TABLE segmentation_points(
                polygon_id INTEGER NOT NULL,
                point_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO schema_info(key, value) VALUES (?, ?)",
            (
                (
                    "schema_name",
                    "instance-segmentation-unified-inference",
                ),
                ("schema_version", "3"),
            ),
        )
        connection.execute("INSERT INTO videos VALUES (1, 'input.mp4', 8, 10, 64, 48)")
        connection.executemany(
            "INSERT INTO model_executions(id, role) VALUES (?, ?)",
            ((1, "instance_segmentation"), (2, "face_detection")),
        )
        connection.executemany(
            "INSERT INTO frames VALUES (?, ?, 64, 48)",
            ((1, 0), (2, 1)),
        )
        connection.execute(
            "INSERT INTO detections VALUES "
            "(10, 1, 1, 'foreground', 0.91, 8, 8, 32, 32)"
        )
        connection.execute("INSERT INTO classifications VALUES (10, 'sample', 0.82)")
        connection.execute("INSERT INTO segmentations VALUES (10, 'polygon')")
        connection.execute("INSERT INTO segmentation_polygons VALUES (100, 10, 0)")
        connection.executemany(
            "INSERT INTO segmentation_points VALUES (100, ?, ?, ?)",
            (
                (0, 8.0, 8.0),
                (1, 32.0, 8.0),
                (2, 32.0, 32.0),
                (3, 8.0, 32.0),
            ),
        )
        connection.execute(
            "INSERT INTO detections VALUES " "(20, 2, 2, 'Face', 0.95, 18, 10, 42, 34)"
        )


def create_mask_sqlite(path: Path) -> None:
    polygons = json.dumps([[[10.0, 10.0], [36.0, 10.0], [36.0, 36.0], [10.0, 36.0]]])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE masks("
            "frame INTEGER, track_id TEXT, polygons TEXT, label TEXT)"
        )
        connection.execute(
            "INSERT INTO masks VALUES (0, '7', ?, '男性器')",
            (polygons,),
        )


def create_keyframe_mask_sqlite(path: Path) -> None:
    polygons = json.dumps(
        [[[10.0, 10.0], [36.0, 10.0], [36.0, 36.0], [10.0, 36.0]]]
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE masks("
            "frame INTEGER, track_id TEXT, polygons TEXT, label TEXT, "
            "is_keyframe INTEGER)"
        )
        connection.executemany(
            "INSERT INTO masks VALUES (?, '7', ?, '男性器', ?)",
            ((frame, polygons, int(frame % 2 == 0)) for frame in range(8)),
        )


def create_typed_mask_pair(
    expanded_path: Path,
    compact_path: Path,
) -> None:
    values = (28.0, 23.0, 12.0, 7.0, 0.25)
    cx, cy, radius_x, radius_y, theta = values
    polygons = [
        [
            [
                cx
                + radius_x * math.cos(phase) * math.cos(theta)
                - radius_y * math.sin(phase) * math.sin(theta),
                cy
                + radius_x * math.cos(phase) * math.sin(theta)
                + radius_y * math.sin(phase) * math.cos(theta),
            ]
            for phase in (2.0 * math.pi * index / 64 for index in range(64))
        ]
    ]
    rectangle_values = (30.0, 24.0, 14.0, 8.0, -0.2)
    rect_cx, rect_cy, half_width, half_height, rect_theta = rectangle_values
    rectangle = [
        [
            rect_cx + x * math.cos(rect_theta) - y * math.sin(rect_theta),
            rect_cy + x * math.sin(rect_theta) + y * math.cos(rect_theta),
        ]
        for x, y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]
    with sqlite3.connect(expanded_path) as connection:
        connection.execute(
            "CREATE TABLE masks("
            "frame INTEGER, track_id TEXT, polygons TEXT, label TEXT)"
        )
        connection.execute(
            "INSERT INTO masks VALUES (0, 'face:1', ?, 'Eyes')",
            (json.dumps(polygons, separators=(",", ":")),),
        )
        connection.execute(
            "INSERT INTO masks VALUES (1, 'face:2', ?, 'Eyes')",
            (json.dumps([rectangle], separators=(",", ":")),),
        )
    with sqlite3.connect(compact_path) as connection:
        connection.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER,
                track_id TEXT,
                polygons TEXT,
                label TEXT
            );
            CREATE TABLE mask_ellipses(
                frame INTEGER,
                track_id TEXT,
                slot_index INTEGER,
                cx REAL,
                cy REAL,
                radius_x REAL,
                radius_y REAL,
                theta_radians REAL,
                point_count INTEGER,
                label TEXT
            );
            CREATE TABLE mask_rectangles(
                frame INTEGER,
                track_id TEXT,
                slot_index INTEGER,
                cx REAL,
                cy REAL,
                half_width REAL,
                half_height REAL,
                theta_radians REAL,
                label TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO mask_ellipses
            VALUES (0, 'face:1', 0, ?, ?, ?, ?, ?, 64, 'Eyes')
            """,
            values,
        )
        connection.execute(
            """
            INSERT INTO mask_rectangles
            VALUES (1, 'face:2', 0, ?, ?, ?, ?, ?, 'Eyes')
            """,
            rectangle_values,
        )


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,nb_read_frames,start_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class PacketIndexTests(unittest.TestCase):
    def test_hidden_negative_pts_preroll_does_not_shift_frame_ordinals(self) -> None:
        segmented = load_segmented_module()
        packet_output = "-2|K_\n0|__\n2|__\n1|K_\n"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=packet_output, stderr=""
        )
        with mock.patch.object(
            segmented.subprocess, "check_output", return_value="0\n"
        ), mock.patch.object(
            segmented.subprocess, "run", return_value=completed
        ):
            index = segmented.probe_video_frame_index(
                Path("/runtime/ffmpeg"), Path("input.mp4")
            )

        self.assertEqual([0, 1, 2], [frame.timestamp for frame in index.frames])
        self.assertEqual(4, index.total_packets)
        self.assertEqual(1, index.hidden_preroll_packets)

    def test_seek_uses_first_visible_pts_when_edit_list_hides_keyframe(self) -> None:
        segmented = load_segmented_module()
        frames = [
            segmented.VideoFrame(timestamp=0, keyframe=False),
            segmented.VideoFrame(timestamp=1, keyframe=False),
            segmented.VideoFrame(timestamp=2, keyframe=True),
        ]

        anchor_index, anchor = segmented.seek_anchor(frames, 0)

        self.assertEqual(0, anchor_index)
        self.assertEqual(0, anchor.timestamp)
        self.assertFalse(anchor.keyframe)

    def test_reported_count_accepts_only_visible_or_total_packet_count(self) -> None:
        segmented = load_segmented_module()
        index = segmented.VideoFrameIndex(
            frames=(
                segmented.VideoFrame(timestamp=0, keyframe=True),
                segmented.VideoFrame(timestamp=1, keyframe=False),
                segmented.VideoFrame(timestamp=2, keyframe=False),
            ),
            total_packets=4,
            hidden_preroll_packets=1,
        )
        segmented.validate_reported_frame_count(index, None)
        segmented.validate_reported_frame_count(index, 3)
        segmented.validate_reported_frame_count(index, 4)
        with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
            segmented.validate_reported_frame_count(index, 5)


@unittest.skipUnless(
    RENDERER.is_file() and FFMPEG.is_file(),
    "build the native renderer and FFmpeg runtime first",
)
class LowLevelModeTests(unittest.TestCase):
    def test_detailed_mode_draws_cut_indicator_only_on_cut_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            masks = root / "masks.sqlite"
            output = root / "cuts.mp4"
            create_video(video)
            create_mask_sqlite(masks)
            with sqlite3.connect(masks) as connection:
                connection.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
                connection.execute("INSERT INTO cuts VALUES (3)")

            subprocess.run(
                [
                    str(RENDERER),
                    "--mode",
                    "final",
                    "--display-style",
                    "detailed",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(masks),
                    "--output",
                    str(output),
                    "--encoder",
                    "libx264",
                    "--preset",
                    "ultrafast",
                    "--crf",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gray = subprocess.run(
                [
                    str(FFMPEG),
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            frame_size = 64 * 48
            before_cut = gray[2 * frame_size : 3 * frame_size]
            cut_frame = gray[3 * frame_size : 4 * frame_size]
            after_cut = gray[4 * frame_size : 5 * frame_size]
            self.assertEqual(before_cut, after_cut)
            self.assertNotEqual(before_cut, cut_frame)

    def test_detailed_keyframe_changes_outline_not_mask_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            masks = root / "keyframes.sqlite"
            output = root / "detailed.mp4"
            create_video(video)
            create_keyframe_mask_sqlite(masks)

            subprocess.run(
                [
                    str(RENDERER),
                    "--mode",
                    "final",
                    "--display-style",
                    "detailed",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(masks),
                    "--output",
                    str(output),
                    "--encoder",
                    "libx264",
                    "--preset",
                    "ultrafast",
                    "--crf",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gray = subprocess.run(
                [
                    str(FFMPEG),
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            frame_size = 64 * 48
            self.assertEqual(frame_size * 8, len(gray))
            center = [
                gray[frame * frame_size + 24 * 64 + 24]
                for frame in range(8)
            ]
            decoded_frames = [
                gray[frame * frame_size : (frame + 1) * frame_size]
                for frame in range(8)
            ]
            self.assertEqual(1, len(set(center)))
            self.assertEqual(2, len(set(decoded_frames)))

    def test_compact_typed_cache_matches_expanded_polygon_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            expanded = root / "expanded.sqlite"
            compact = root / "compact.sqlite"
            create_video(video)
            create_typed_mask_pair(expanded, compact)

            outputs: list[Path] = []
            for name, source in (
                ("expanded", expanded),
                ("compact", compact),
            ):
                output = root / f"{name}.mp4"
                subprocess.run(
                    [
                        str(RENDERER),
                        "--mode",
                        "final",
                        "--video",
                        str(video),
                        "--sqlite",
                        str(source),
                        "--output",
                        str(output),
                        "--encoder",
                        "libx264",
                        "--preset",
                        "ultrafast",
                        "--crf",
                        "18",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                outputs.append(output)

            decoded = [
                subprocess.run(
                    [
                        str(FFMPEG),
                        "-v",
                        "error",
                        "-i",
                        str(output),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "yuv420p",
                        "-",
                    ],
                    check=True,
                    capture_output=True,
                ).stdout
                for output in outputs
            ]
            self.assertEqual(decoded[0], decoded[1])

    def test_all_modes_manifest_atomic_output_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            inference = root / "inference.sqlite"
            masks = root / "masks.sqlite"
            create_video(video)
            create_inference_sqlite(inference)
            create_mask_sqlite(masks)

            modes = {
                "raw": (inference, []),
                "tracked": (masks, []),
                "final": (
                    masks,
                    [
                        "--include-faces",
                        "--face-sqlite",
                        str(inference),
                    ],
                ),
                "faces": (inference, []),
            }
            for mode, (sqlite, extra) in modes.items():
                output = root / f"{mode}.mp4"
                manifest = root / f"{mode}.json"
                result = subprocess.run(
                    [
                        str(RENDERER),
                        "--mode",
                        mode,
                        "--video",
                        str(video),
                        "--sqlite",
                        str(sqlite),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--codec",
                        "h264",
                        "--h264-preset",
                        "ultrafast",
                        "--h264-crf",
                        "18",
                        "--copy-audio",
                        "--overwrite",
                        *extra,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                stdout = json.loads(result.stdout)
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(stdout, saved)
                self.assertEqual(8, stdout["frames_written"])
                self.assertTrue(stdout["audio_copied"])
                streams = probe(output)["streams"]
                self.assertEqual(
                    {"video", "audio"},
                    {stream["codec_type"] for stream in streams},
                )
                self.assertFalse(
                    list(root.glob(f".{output.stem}.*.tmp{output.suffix}"))
                )

    def test_integrated_sqlite_keeps_tracked_and_final_modes_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            result_sqlite = root / "result.sqlite"
            create_video(video)
            create_mask_sqlite(result_sqlite)
            with sqlite3.connect(result_sqlite) as connection:
                connection.execute(
                    """
                    CREATE TABLE tracked_masks(
                        frame INTEGER,
                        track_id TEXT,
                        polygons TEXT,
                        label TEXT
                    )
                    """
                )

            counts: dict[str, int] = {}
            for mode in ("tracked", "final"):
                output = root / f"{mode}.mp4"
                result = subprocess.run(
                    [
                        str(RENDERER),
                        "--mode",
                        mode,
                        "--video",
                        str(video),
                        "--sqlite",
                        str(result_sqlite),
                        "--output",
                        str(output),
                        "--encoder",
                        "libx264",
                        "--codec",
                        "h264",
                        "--preset",
                        "ultrafast",
                        "--crf",
                        "18",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                counts[mode] = int(json.loads(result.stdout)["mask_rows_drawn"])

            self.assertEqual(0, counts["tracked"])
            self.assertEqual(1, counts["final"])

    def test_decode_ordinal_is_not_derived_from_gapped_pts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "pts-gap.mp4"
            masks = root / "masks.sqlite"
            output = root / "direct.mp4"
            create_pts_gap_video(video)
            create_mask_sqlite(masks)

            result = subprocess.run(
                [
                    str(RENDERER),
                    "--mode",
                    "final",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(masks),
                    "--output",
                    str(output),
                    "--encoder",
                    "libx264",
                    "--preset",
                    "ultrafast",
                    "--crf",
                    "30",
                    "--start-frame",
                    "0",
                    "--end-frame",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(8, summary["frames_written"])
            self.assertEqual(
                "8",
                probe(output)["streams"][0]["nb_read_frames"],
            )

    def test_segmented_runner_seeks_by_packet_index_across_pts_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "pts-gap.mp4"
            masks = root / "masks.sqlite"
            output_dir = root / "segments"
            create_pts_gap_video(video)
            create_mask_sqlite(masks)

            subprocess.run(
                [
                    "python3",
                    str(SEGMENTED),
                    "--mode",
                    "final",
                    "--video",
                    str(video),
                    "--sqlite",
                    str(masks),
                    "--output-dir",
                    str(output_dir),
                    "--renderer",
                    str(RENDERER),
                    "--ffmpeg-bin",
                    str(FFMPEG),
                    "--workers",
                    "2",
                    "--cpu-workers",
                    "2",
                    "--start-frame",
                    "0",
                    "--end-frame",
                    "7",
                    "--bitrate-mbps",
                    "1",
                    "--cpu-preset",
                    "ultrafast",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (output_dir / "benchmark_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(8, summary["frames"])
            self.assertEqual(
                8,
                summary["source_frame_index"]["indexed_frames"],
            )
            self.assertEqual(
                8,
                summary["source_frame_index"][
                    "container_reported_frames"
                ],
            )
            self.assertEqual(
                1,
                summary["source_frame_index"][
                    "non_uniform_timestamp_deltas"
                ],
            )
            self.assertEqual(
                [4, 4],
                [
                    worker["renderer_summary"]["frames_written"]
                    for worker in summary["workers_detail"]
                ],
            )
            self.assertEqual(
                "8",
                probe(output_dir / "final.mp4")["streams"][0][
                    "nb_read_frames"
                ],
            )


if __name__ == "__main__":
    unittest.main()
