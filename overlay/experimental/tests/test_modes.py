from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "build" / "overlay_lowlevel"
FFMPEG = (
    ROOT.parent / ".runtime" / "ffmpeg-nvenc-btbn-8.1" / "bin" / "ffmpeg"
)
FFPROBE = FFMPEG.with_name("ffprobe")


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
                ("schema_version", "2"),
            ),
        )
        connection.execute(
            "INSERT INTO videos VALUES (1, 'input.mp4', 8, 10, 64, 48)"
        )
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
        connection.execute(
            "INSERT INTO classifications VALUES (10, 'sample', 0.82)"
        )
        connection.execute(
            "INSERT INTO segmentations VALUES (10, 'polygon')"
        )
        connection.execute(
            "INSERT INTO segmentation_polygons VALUES (100, 10, 0)"
        )
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
            "INSERT INTO detections VALUES "
            "(20, 2, 2, 'Face', 0.95, 18, 10, 42, 34)"
        )


def create_mask_sqlite(path: Path) -> None:
    polygons = json.dumps(
        [[[10.0, 10.0], [36.0, 10.0], [36.0, 36.0], [10.0, 36.0]]]
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE masks("
            "frame INTEGER, track_id TEXT, polygons TEXT, label TEXT)"
        )
        connection.execute(
            "INSERT INTO masks VALUES (0, '7', ?, 'target')",
            (polygons,),
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


@unittest.skipUnless(
    RENDERER.is_file() and FFMPEG.is_file(),
    "build the experimental renderer and FFmpeg runtime first",
)
class LowLevelModeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
