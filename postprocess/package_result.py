"""Package inference and optional postprocess data into the stable result SQLite."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Sequence

from artifacts.unified_sqlite import build_integrated_result, record_processing_run
from face_privacy.sqlite import export_face_masks, merge_face_masks
from face_privacy.tracking import FaceTrackingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postprocess-package-result",
        description=(
            "Create one stable result SQLite. Components that were not run are "
            "represented by empty canonical tables and result_capabilities rows."
        ),
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--tracked-sqlite", type=Path)
    parser.add_argument("--final-sqlite", type=Path)
    parser.add_argument("--polygon-keyframes-sqlite", type=Path)
    parser.add_argument("--ellipse-keyframes-json", type=Path)
    parser.add_argument("--classwise-manifest", type=Path)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--orchestration-config-json", type=Path)
    parser.add_argument(
        "--face-mask-target",
        choices=("none", "face", "eyes"),
        default="none",
    )
    parser.add_argument(
        "--eye-mask-shape",
        choices=("ellipse", "rectangle"),
        default="ellipse",
    )
    parser.add_argument("--minimum-eye-confidence", type=float, default=0.35)
    parser.add_argument("--face-tracking-max-gap-frames", type=int, default=5)
    parser.add_argument(
        "--face-tracking-high-score-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--face-tracking-low-score-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--face-short-track-max-hits", type=int, default=2)
    parser.add_argument(
        "--face-short-track-keep-score",
        type=float,
        default=0.90,
    )
    parser.add_argument("--face-interpolation-max-gap", type=int, default=3)
    parser.add_argument("--precomputed-cuts-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_sqlite.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    face_summary: dict[str, object] | None = None
    final_source = args.final_sqlite
    try:
        if args.face_mask_target != "none":
            face_masks = output.parent / (
                f".{output.name}.{uuid.uuid4().hex}.face-masks.sqlite"
            )
            temporary_paths.append(face_masks)
            face_summary = export_face_masks(
                args.input_sqlite,
                face_masks,
                target=args.face_mask_target,
                eye_shape=args.eye_mask_shape,
                minimum_eye_confidence=args.minimum_eye_confidence,
                tracking_config=FaceTrackingConfig(
                    max_gap_frames=args.face_tracking_max_gap_frames,
                    high_score_threshold=(args.face_tracking_high_score_threshold),
                    low_score_threshold=args.face_tracking_low_score_threshold,
                    short_track_max_hits=args.face_short_track_max_hits,
                    short_track_keep_score=args.face_short_track_keep_score,
                ),
                interpolation_max_gap=args.face_interpolation_max_gap,
                cuts_json=args.precomputed_cuts_json,
            )
            if final_source is None:
                final_source = face_masks
            else:
                merged = output.parent / (
                    f".{output.name}.{uuid.uuid4().hex}.with-faces.sqlite"
                )
                temporary_paths.append(merged)
                merge_summary = merge_face_masks(final_source, face_masks, merged)
                face_summary = {
                    **face_summary,
                    "merge": merge_summary,
                }
                final_source = merged
        summary = build_integrated_result(
            args.input_sqlite,
            args.tracked_sqlite,
            final_source,
            output,
            polygon_keyframes_sqlite=args.polygon_keyframes_sqlite,
            ellipse_keyframes_json=args.ellipse_keyframes_json,
            classwise_manifest=args.classwise_manifest,
        )
        record_processing_run(
            output,
            kind="result_packaging",
            name="postprocess-package-result",
            resolved_config=vars(args),
            stages=[
                {
                    "id": "result_packaging",
                    "implementation": "artifacts.unified_sqlite",
                    "options": vars(args),
                    "status": "complete",
                }
            ],
        )
        if args.orchestration_config_json is not None:
            record_processing_run(
                output,
                kind="orchestration",
                name="inference-postprocess-overlay",
                resolved_config=json.loads(
                    args.orchestration_config_json.read_text(encoding="utf-8")
                ),
                stages=[],
            )
    finally:
        for path in temporary_paths:
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
                if candidate.exists():
                    candidate.unlink()
    if face_summary is not None:
        summary["face_privacy"] = face_summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
