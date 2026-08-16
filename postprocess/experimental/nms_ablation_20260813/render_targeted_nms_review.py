#!/usr/bin/env python3
"""Render a bounded, local-only NMS review sequence.

This is a thin range-oriented companion to
``render_component_candidate_v2_review_gallery.py``.  It deliberately reuses
that audit's colours and panel renderer, but skips category sampling so every
requested frame is emitted exactly once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from render_component_candidate_v2_review_gallery import (
    CLASS_NAMES,
    DEFAULT_ABLATION,
    DEFAULT_TOPOLOGY,
    _candidate_events,
    _detection_id,
    _legacy_events,
    _open_ro,
    _polygon_signature,
    _put,
    _render_panel,
    seek_frame,
)

from contracts.detections import iter_detection_records


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_KEY = "v3__kpi_2025_12"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "output/nms_review_kpi_f4275_4280_20260813"


def _load_selected_records(
    run_dir: Path,
    scored_jsonl: Path,
    wanted: set[int],
) -> dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Load either the v2 or virtual-component v3 candidate arm.

    The bounded renderer predates the v3 experiment and originally hard-coded
    ``component_candidate_v2.jsonl``.  Prefer v3 when present so a corrected
    virtual-component run can be used for explicit regression frames, while
    retaining compatibility with the older v2 review layout.
    """
    candidate_path = run_dir / "arm_outputs/virtual_component_v3.jsonl"
    if not candidate_path.is_file():
        candidate_path = run_dir / "arm_outputs/component_candidate_v2.jsonl"
    paths = (scored_jsonl, run_dir / "arm_outputs/legacy.jsonl", candidate_path)
    result: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    iterators = [iter_detection_records(path) for path in paths]
    for rows in itertools.zip_longest(*iterators):
        if any(row is None for row in rows):
            raise RuntimeError(f"arm JSONL length mismatch: {run_dir}")
        frames = {int(row["frame_index"]) for row in rows if row is not None}
        if len(frames) != 1:
            raise RuntimeError(f"arm JSONL frame mismatch: {run_dir}: {frames}")
        frame = next(iter(frames))
        if frame in wanted:
            result[frame] = rows  # type: ignore[assignment]
        if len(result) == len(wanted):
            break
    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(
            f"selected frames absent from arm JSONL: {run_dir}: {missing}"
        )
    return result


def _ids(record: dict[str, Any]) -> list[int | str]:
    frame = int(record["frame_index"])
    return [
        _detection_id(detection, frame, index)
        for index, detection in enumerate(record["detections"])
    ]


def _raw_candidate_equal(
    raw_record: dict[str, Any], candidate_record: dict[str, Any]
) -> bool:
    frame = int(raw_record["frame_index"])
    raw = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(raw_record["detections"])
    }
    candidate = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(candidate_record["detections"])
    }
    if set(raw) != set(candidate):
        return False
    fields = ("bbox_xyxy", "score", "class_name")
    return all(
        all(
            raw[detection_id].get(field) == candidate[detection_id].get(field)
            for field in fields
        )
        and _polygon_signature(raw[detection_id])
        == _polygon_signature(candidate[detection_id])
        for detection_id in raw
    )


def _frame_metadata(
    records: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> dict[str, Any]:
    raw, legacy, candidate = records
    frame = int(raw["frame_index"])
    frames = {int(record["frame_index"]) for record in records}
    if frames != {frame}:
        raise RuntimeError(f"arm frame mismatch: {sorted(frames)}")
    raw_detections = list(raw["detections"])
    return {
        "frame": frame,
        "raw": [
            {
                "id": _detection_id(detection, frame, index),
                "score": float(detection.get("score") or 0.0),
                "class": str(detection.get("class_name", "")),
            }
            for index, detection in enumerate(raw_detections)
        ],
        "raw_ids": _ids(raw),
        "legacy_ids": _ids(legacy),
        "candidate_ids": _ids(candidate),
        "candidate_equals_raw_geometry_and_metadata": _raw_candidate_equal(
            raw, candidate
        ),
        "legacy_events": _legacy_events(raw_detections, frame),
        "candidate_events": _candidate_events(raw_detections, frame),
    }


def _header(
    width: int,
    run_key: str,
    metadata: dict[str, Any],
) -> np.ndarray:
    legacy = set(metadata["legacy_ids"])
    candidate = set(metadata["candidate_ids"])
    lines = [
        f"run={run_key} frame={metadata['frame']}",
    ]
    for detection in metadata["raw"]:
        detection_id = detection["id"]
        class_name = CLASS_NAMES.get(
            detection["class"], detection["class"] or "unknown"
        )
        lines.append(
            f"D{detection_id} score={detection['score']:.3f} class={class_name} "
            f"raw=K legacy={'K' if detection_id in legacy else 'X'} "
            f"candidate={'K' if detection_id in candidate else 'X'}"
        )
    event_text = "; ".join(
        f"legacy D{event['winner_id']} -> D{event['loser_id']} "
        f"{event['reason']} bboxIoU={event['bbox_iou']:.3f} "
        f"maskIoU={event['mask_iou']:.3f}"
        for event in metadata["legacy_events"]
    )
    lines.append(event_text or "legacy suppression: none")
    lines.append(
        "same D-ID = same colour across panels; solid+fill=kept; "
        "dashed=raw suppressed/changed"
    )
    header = np.full((20 + 28 * len(lines), width, 3), 12, np.uint8)
    for index, line in enumerate(lines):
        _put(header, line, (14, 24 + 28 * index), scale=0.52)
    return header


def _resize_panel(panel: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or panel.shape[1] <= width:
        return panel
    height = max(1, int(round(panel.shape[0] * width / panel.shape[1])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def _render(
    *,
    run_key: str,
    video: Path,
    records: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    metadata: dict[str, Any],
    layout: str,
    panel_width: int,
    output_path: Path,
) -> None:
    raw, legacy, candidate = records
    image = seek_frame(video, int(metadata["frame"]))
    panels = []
    if layout == "raw-legacy-candidate":
        panels.append(_render_panel(image, raw, raw, "RAW: scored pre-NMS input"))
    panels.extend(
        [
            _render_panel(image, raw, legacy, "LEGACY: Production NMS"),
            _render_panel(image, raw, candidate, "CANDIDATE: component mask v2"),
        ]
    )
    panels = [_resize_panel(panel, panel_width) for panel in panels]
    combined_panels = np.concatenate(panels, axis=1)
    combined = np.vstack(
        [_header(combined_panels.shape[1], run_key, metadata), combined_panels]
    )
    if output_path.exists():
        raise FileExistsError(output_path)
    if not cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--topology-sqlite", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--run-key", default=DEFAULT_RUN_KEY)
    parser.add_argument("--frame-start", type=int, default=4275)
    parser.add_argument("--frame-end", type=int, default=4280)
    parser.add_argument(
        "--layout",
        choices=("legacy-candidate", "raw-legacy-candidate"),
        default="legacy-candidate",
    )
    parser.add_argument("--panel-width", type=int, default=1280)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (
        not args.run_key
        or args.run_key in {".", ".."}
        or "/" in args.run_key
        or "\\" in args.run_key
    ):
        raise ValueError("run-key must be one run-directory basename")
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise ValueError("require 0 <= frame-start <= frame-end")
    if args.frame_end - args.frame_start + 1 > 120:
        raise ValueError("refusing to render more than 120 frames in a bounded review")
    if args.panel_width != 0 and args.panel_width < 640:
        raise ValueError("panel-width must be 0 or at least 640")

    ablation_root = args.ablation_root.expanduser().resolve()
    topology_path = args.topology_sqlite.expanduser().resolve()
    output = args.output.expanduser().resolve()
    run_dir = ablation_root / "runs" / args.run_key
    summary_path = run_dir / "summary.json"
    required = [topology_path, summary_path]
    if not ablation_root.is_dir() or not all(path.is_file() for path in required):
        raise FileNotFoundError(
            {
                "ablation_root": str(ablation_root),
                "required": [str(path) for path in required],
            }
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing review: {output}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("run_key")) != args.run_key:
        raise RuntimeError(
            f"summary run_key mismatch: expected {args.run_key!r}, "
            f"found {summary.get('run_key')!r}"
        )
    source_jsonl = summary.get("source_jsonl")
    if source_jsonl is None:
        source_jsonl = summary.get("input", {}).get("jsonl")
    if source_jsonl is None:
        raise KeyError("summary must provide source_jsonl or input.jsonl")
    scored_jsonl = Path(str(source_jsonl)).expanduser().resolve()
    if not scored_jsonl.is_file():
        raise FileNotFoundError(scored_jsonl)
    topology = _open_ro(topology_path)
    try:
        row = topology.execute(
            "SELECT input_video FROM audit_runs WHERE run_key=?", (args.run_key,)
        ).fetchone()
    finally:
        topology.close()
    if row is None:
        raise KeyError(f"run missing from topology SQLite: {args.run_key}")
    video = Path(str(row["input_video"])).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    wanted = set(range(args.frame_start, args.frame_end + 1))
    records = _load_selected_records(run_dir, scored_jsonl, wanted)
    frames = [_frame_metadata(records[frame]) for frame in sorted(wanted)]
    plan = {
        "privacy": (
            "local files and local OpenCV decoding only; "
            "no network or image-view tool"
        ),
        "run_key": args.run_key,
        "video": str(video),
        "scored_jsonl": str(scored_jsonl),
        "layout": args.layout,
        "panel_width": args.panel_width,
        "output": str(output),
        "frames": frames,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    manifest: list[dict[str, Any]] = []
    try:
        for metadata in frames:
            frame = int(metadata["frame"])
            filename = f"{args.run_key}_f{frame:06d}.jpg"
            _render(
                run_key=args.run_key,
                video=video,
                records=records[frame],
                metadata=metadata,
                layout=args.layout,
                panel_width=args.panel_width,
                output_path=staging / filename,
            )
            manifest.append({**metadata, "image": str(output / filename)})
        (staging / "manifest.json").write_text(
            json.dumps({**plan, "frames": manifest}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(output)
        os.replace(staging, output)
    except BaseException:
        # Preserve the staging directory for diagnosis; never publish a partial review.
        raise

    print(json.dumps({"output": str(output), "images": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
