"""Classwise preparation plus polygon optimizer stage."""

from __future__ import annotations

import json
from dataclasses import dataclass

from contracts.stages import StageContext, StageResult

from ..config import CANDIDATE
from .engine import run_polygon_optimizer
from .preparation import prepare_classwise_source


@dataclass(frozen=True)
class CandidatePolygonStage:
    width: int = 0
    height: int = 0
    max_tracks: int = 0
    force: bool = False
    name: str = "production_candidate_20260814_polygon"
    requires: frozenset[str] = frozenset({"tracked_sqlite", "input_video"})
    provides: frozenset[str] = frozenset({"polygon_candidate_manifest"})

    def run(self, context: StageContext) -> StageResult:
        width = int(self.width)
        height = int(self.height)
        if width <= 0 or height <= 0:
            raise ValueError("candidate polygon stage requires positive width/height")
        if int(self.max_tracks) < 0:
            raise ValueError("max_tracks must be non-negative")
        video = context.artifacts["input_video"]
        preparation_root = context.stage_dir / "preparation"
        source_root, preparation = prepare_classwise_source(
            context.artifacts["tracked_sqlite"],
            preparation_root,
            width=width,
            height=height,
            input_video=video,
        )
        optimizer = run_polygon_optimizer(
            source_root,
            context.stage_dir / "optimizer",
            labels=tuple(preparation["active_labels"]),
            max_tracks=int(self.max_tracks),
            force=bool(self.force),
        )
        manifest_path = context.stage_dir / "polygon_candidate_manifest.json"
        payload = {
            "schema_version": 1,
            "candidate": CANDIDATE.to_dict(),
            "preparation": preparation,
            "optimizer": optimizer,
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return StageResult(
            {"polygon_candidate_manifest": manifest_path},
            payload,
        )
