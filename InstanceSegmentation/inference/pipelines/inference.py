"""Task-neutral decode → adapter → persistence video pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from contracts import (
    DetectionFrame,
    InferenceFrame,
    SegmentationFrame,
    VisionAdapter,
)
from video import AsyncVideoDecoder


class ResultWriter(Protocol):
    def set_metadata(self, values: Mapping[str, object]) -> None:
        ...

    def write(self, result: InferenceFrame) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class InferenceRunSummary:
    input: str
    model_id: str
    task: str
    processed_frames: int
    result_items: int
    wall_elapsed_sec: float
    wall_fps: float
    warmup_frames: int
    measured_frames: int
    measured_time_sec: float
    compute_fps: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ProgressCallback = Callable[[InferenceRunSummary], None]


def _result_count(results: Sequence[InferenceFrame]) -> int:
    total = 0
    for result in results:
        if isinstance(result, DetectionFrame):
            total += len(result.detections)
        elif isinstance(result, SegmentationFrame):
            total += len(result.instances)
        else:
            raise TypeError(f"unsupported inference result: {type(result)!r}")
    return total


def run_video_inference(
    *,
    input_path: Path,
    adapter: VisionAdapter,
    writer: ResultWriter,
    batch_size: int,
    max_frames: int | None,
    warmup_frames: int,
    prefetch_batches: int = 2,
    metadata: Mapping[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> InferenceRunSummary:
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    decoder = AsyncVideoDecoder(
        input_path,
        batch_size=batch_size,
        max_frames=max_frames,
        prefetch_batches=prefetch_batches,
    )
    writer.set_metadata(
        {
            "input": str(input_path.expanduser().resolve()),
            "model_id": adapter.descriptor.model_id,
            "task": adapter.descriptor.task.value,
            "contract_version": adapter.descriptor.contract_version,
            "video": asdict(decoder.metadata),
            **({} if metadata is None else dict(metadata)),
        }
    )
    processed = 0
    items = 0
    measured_frames = 0
    measured_time = 0.0
    started = time.perf_counter()
    try:
        for batch in decoder:
            adapter.synchronize()
            batch_started = time.perf_counter()
            results = tuple(adapter.predict(batch))
            adapter.synchronize()
            elapsed = time.perf_counter() - batch_started
            expected_indices = tuple(frame.index for frame in batch.frames)
            actual_indices = tuple(result.frame.index for result in results)
            if actual_indices != expected_indices:
                raise RuntimeError(
                    "adapter result frame order mismatch: "
                    f"expected={expected_indices}, actual={actual_indices}"
                )
            if any(result.model != adapter.descriptor for result in results):
                raise RuntimeError(
                    "adapter result model descriptor does not match its registration"
                )
            measured = sum(
                frame.index >= warmup_frames for frame in batch.frames
            )
            if measured:
                measured_time += elapsed * (measured / len(batch))
                measured_frames += measured
            for result in results:
                writer.write(result)
            processed += len(batch)
            items += _result_count(results)
            if progress is not None:
                wall_elapsed = time.perf_counter() - started
                progress(
                    InferenceRunSummary(
                        input=str(input_path),
                        model_id=adapter.descriptor.model_id,
                        task=adapter.descriptor.task.value,
                        processed_frames=processed,
                        result_items=items,
                        wall_elapsed_sec=wall_elapsed,
                        wall_fps=processed / max(wall_elapsed, 1e-9),
                        warmup_frames=warmup_frames,
                        measured_frames=measured_frames,
                        measured_time_sec=measured_time,
                        compute_fps=(
                            measured_frames / measured_time
                            if measured_time > 0
                            else 0.0
                        ),
                    )
                )
    finally:
        try:
            decoder.close()
        finally:
            try:
                writer.close()
            finally:
                adapter.close()
    wall_elapsed = time.perf_counter() - started
    return InferenceRunSummary(
        input=str(input_path),
        model_id=adapter.descriptor.model_id,
        task=adapter.descriptor.task.value,
        processed_frames=processed,
        result_items=items,
        wall_elapsed_sec=wall_elapsed,
        wall_fps=processed / max(wall_elapsed, 1e-9),
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
        measured_time_sec=measured_time,
        compute_fps=(
            measured_frames / measured_time if measured_time > 0 else 0.0
        ),
    )


__all__ = ["InferenceRunSummary", "ProgressCallback", "run_video_inference"]
