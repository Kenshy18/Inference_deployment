"""Canonical JSONL and cut-list artifact I/O.

The canonical frame schema is::

    {"frame_index": int, "detections": [canonical detection, ...]}

Intermediate stages preserve this schema.  A stage may only filter or enrich
the detections documented in ``CONTRACT.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


def iter_detection_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield already-normalized frame records from a canonical JSONL file."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: frame must be an object")
            if not isinstance(value.get("frame_index"), int):
                raise ValueError(
                    f"{source}:{line_number}: canonical frame_index must be int"
                )
            if not isinstance(value.get("detections"), list):
                raise ValueError(
                    f"{source}:{line_number}: canonical detections must be list"
                )
            yield value


def transform_detection_jsonl(
    input_path: Path,
    output_path: Path,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, int]:
    """Stream a canonical JSONL artifact through one feature transformation."""

    source = Path(input_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("a stage output must differ from its input")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = detections_in = detections_out = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_detection_records(source):
            detections_in += len(record["detections"])
            transformed = transform(record)
            if not isinstance(transformed, dict):
                raise TypeError("detection transform must return a frame object")
            detections = transformed.get("detections")
            if not isinstance(detections, list):
                raise ValueError("transformed detections must be a list")
            handle.write(
                json.dumps(
                    transformed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            frames += 1
            detections_out += len(detections)
    return {
        "frames": frames,
        "detections_in": detections_in,
        "detections_out": detections_out,
    }


@dataclass(frozen=True)
class CutList:
    frames: tuple[int, ...]
    method: str
    elapsed_seconds: float = 0.0

    @classmethod
    def read(cls, path: Path) -> "CutList":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported cut-list contract")
        frames = value.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"{path}: cut-list frames must be a list")
        return cls(
            frames=tuple(sorted({int(frame) for frame in frames})),
            method=str(value.get("method", "unknown")),
            elapsed_seconds=float(value.get("elapsed_seconds", 0.0)),
        )


def write_cut_list(path: Path, cuts: CutList) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        **asdict(cuts),
        "frames": list(cuts.frames),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
