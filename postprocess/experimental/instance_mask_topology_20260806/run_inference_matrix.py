#!/usr/bin/env python3
"""Prepare and execute the bounded V3/V3-lite topology inference matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "新しいフォルダー"
OUTPUT = REPO / "output" / "instance_mask_topology_20260806"
RUNTIME_PYTHON = Path(
    "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10"
)

VIDEOS = [
    ("kpi_2025_12", "12月KPI動画.mp4"),
    ("white_2025_03_0210", "3月以降解析白カン動画-0210.mp4"),
    (
        "heyzo_3545_full",
        "HEYZO-3545 乙葉いおり おとはいおり 性の悩みはホクかトヒュっと解決しますおしゃふりは浮気し.mp4",
    ),
    ("heyzo_3545_30_45_duplicate", "HEYZO-3545_30分-45分.mp4"),
    (
        "heyzo_3549_full",
        "HEYZO-3549 浜田希 はまたのそみ 激しめイラマか好き - 無修正アタルト動画 HEYZO -.mp4",
    ),
    (
        "heyzo_3554_full",
        "HEYZO-3554 小野寺まり おのてらまり 熟女泡姫のテクてイカせてアケル美女コレクションVol .mp4",
    ),
    (
        "heyzo_3560_full",
        "HEYZO-3560 夏目りんか なつめりんか 欲求不満な私を好きにしてくたさい - 無修正アタルト動画.mp4",
    ),
    ("sdam_151_long", "SDAM-151_AIモザイク_アクセル様.mp4"),
    ("white_axel_0126", "アクセル様２月解析用白カン01.26.mp4"),
    ("joined_long", "連結済み_長時間動画.mp4"),
]

# Ten minutes at each source rate.  These are conservative limits; the
# orchestrator clamps an unavailable last frame to the decoded/SQLite range.
V3_LIMITS = {
    "sdam_151_long": 18_000,
    "joined_long": 14_400,
}
V3_SKIP = {"heyzo_3545_30_45_duplicate"}


def _entry(model_key: str, model: str, slug: str, filename: str) -> dict[str, object]:
    max_frames = V3_LIMITS.get(slug) if model_key == "v3" else None
    return {
        "run_key": f"{model_key}__{slug}",
        "model_key": model_key,
        "segmentation_model": model,
        "video_slug": slug,
        "input_video": str((DATA / filename).resolve()),
        "max_frames": max_frames,
        "output_root": str((OUTPUT / "inference" / model_key / slug).resolve()),
    }


def build_matrix() -> list[dict[str, object]]:
    matrix = [
        _entry("v3lite", "dinov3_codino_mh0", slug, filename)
        for slug, filename in VIDEOS
    ]
    matrix.extend(
        _entry("v3", "dinov3_codino", slug, filename)
        for slug, filename in VIDEOS
        if slug not in V3_SKIP
    )
    return matrix


def prepare() -> list[dict[str, object]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs = OUTPUT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    for item in matrix:
        input_video = Path(str(item["input_video"]))
        if not input_video.is_file():
            raise FileNotFoundError(input_video)
        config = {
            "schema_version": 1,
            "input_video": str(input_video),
            "output_root": item["output_root"],
            "execution": {
                "runtime_python": str(RUNTIME_PYTHON),
                "resume": False,
            },
            "inference": {
                "enabled": True,
                "mode": "segmentation",
                "segmentation_model": item["segmentation_model"],
                "segmentation_backend": "tensorrt-fast",
                "device": "cuda:0",
                "warmup_frames": 0,
                "fast_sqlite": True,
            },
            "postprocess": {"enabled": False},
            "overlay": {"enabled": False},
        }
        if item["max_frames"] is not None:
            config["inference"]["max_frames"] = item["max_frames"]
        config_path = configs / f'{item["run_key"]}.json'
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        item["config"] = str(config_path.resolve())
        item["inference_sqlite"] = str(
            (
                Path(str(item["output_root"]))
                / f"{Path(str(item['input_video'])).stem}.sqlite"
            ).resolve()
        )
    (OUTPUT / "matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return matrix


def _published_result(item: dict[str, object]) -> Path:
    manifest = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            published = payload.get("artifacts", {}).get("result_sqlite")
            if published:
                return Path(str(published))
        except (OSError, ValueError, TypeError):
            pass
    return Path(str(item["inference_sqlite"]))


def _result_ok(item: dict[str, object]) -> bool:
    sqlite_path = _published_result(item)
    manifest = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
    if sqlite_path.is_file() and sqlite_path.stat().st_size > 0 and manifest.is_file():
        item["inference_sqlite"] = str(sqlite_path.resolve())
        return True
    return False


def run(matrix: list[dict[str, object]], deadline_hours: float) -> int:
    started = time.monotonic()
    deadline = started + deadline_hours * 3600.0
    results_path = OUTPUT / "batch_results.json"
    results: list[dict[str, object]] = []
    # V3-lite first gives full-dataset coverage.  V3 then follows from short
    # and diverse inputs to long inputs, so a hard deadline still leaves a
    # useful paired comparison.
    order = {"v3lite": 0, "v3": 1}
    matrix = sorted(matrix, key=lambda row: (order[str(row["model_key"])], VIDEOS.index(next(v for v in VIDEOS if v[0] == row["video_slug"]))))
    for item in matrix:
        if _result_ok(item):
            results.append({"run_key": item["run_key"], "status": "reused"})
            continue
        if time.monotonic() >= deadline:
            results.append({"run_key": item["run_key"], "status": "deadline_skipped"})
            continue
        run_started = time.monotonic()
        log_path = OUTPUT / "batch_logs" / f'{item["run_key"]}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[matrix] START {item["run_key"]}', flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [str(RUNTIME_PYTHON), "-m", "orchestration", "--config", str(item["config"])],
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if "[progress]" in line or "processed " in line or "saved SQLite" in line:
                    print(f'[{item["run_key"]}] {line.rstrip()}', flush=True)
            code = proc.wait()
        elapsed = time.monotonic() - run_started
        status = "complete" if code == 0 and _result_ok(item) else "failed"
        result = {
            "run_key": item["run_key"],
            "status": status,
            "exit_code": code,
            "elapsed_seconds": elapsed,
            "log": str(log_path.resolve()),
        }
        results.append(result)
        results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f'[matrix] END {item["run_key"]} status={status} elapsed={elapsed:.1f}s', flush=True)
        if status == "failed":
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--deadline-hours", type=float, default=8.75)
    args = parser.parse_args()
    if not args.prepare and not args.run:
        parser.error("select --prepare and/or --run")
    matrix = prepare()
    print(f"prepared {len(matrix)} runs at {OUTPUT}")
    return run(matrix, args.deadline_hours) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
