#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def probe(ffprobe: Path, source: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_packets",
            "-show_entries",
            "format=format_name,duration,size,start_time:stream=index,codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_read_packets",
            "-of",
            "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload: dict[str, object]
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "probe_exit_code": result.returncode,
        "probe_stderr": result.stderr.strip(),
        "probe": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_root = Path(
        os.environ.get(
            "MASK_RUNTIME_ROOT", "/home/kenshin/.local/share/video-mask-runtime"
        )
    )
    ffprobe = runtime_root / "tools/ffmpeg/bin/ffprobe"
    rows = [probe(ffprobe, path) for path in sorted(args.fixtures.iterdir()) if path.is_file()]
    report = {
        "schema_version": 1,
        "fixture_root": str(args.fixtures),
        "fixtures": rows,
        "issues": [
            f"{Path(row['path']).name}: probe failed"
            for row in rows
            if row["probe_exit_code"] != 0
            and Path(str(row["path"])).name != "invalid_truncated.mp4"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixtures": len(rows), "issues": report["issues"]}, ensure_ascii=False))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
