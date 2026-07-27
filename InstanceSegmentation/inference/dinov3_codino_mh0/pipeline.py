"""Bounded MH0 pipeline overlapping CPU contract conversion with GPU inference."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from pipelines import InferenceRunSummary
from video import AsyncVideoDecoder

try:
    from .adapter import Mh0Adapter
except ImportError:
    from adapter import Mh0Adapter


ProgressCallback = Callable[[InferenceRunSummary], None]


def run_mh0_video_inference(
    *,
    input_path: Path,
    adapter: Mh0Adapter,
    writer,
    batch_size: int,
    max_frames: int | None,
    warmup_frames: int,
    prefetch_batches: int = 2,
    output_queue_batches: int = 2,
    metadata: Mapping[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> InferenceRunSummary:
    """Run MH0 with source-ordered, bounded CPU/GPU overlap."""

    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    if output_queue_batches < 1:
        raise ValueError("output_queue_batches must be positive")
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
    output_pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="mh0-contract-output",
    )
    pending: deque[Future[tuple[tuple[object, ...], int]]] = deque()

    def convert_batch(batch, raw_results) -> tuple[tuple[object, ...], int]:
        results = adapter.convert_raw(batch, raw_results)
        expected = tuple(frame.index for frame in batch.frames)
        actual = tuple(result.frame.index for result in results)
        if actual != expected:
            raise RuntimeError(
                "adapter result frame order mismatch: "
                f"expected={expected}, actual={actual}"
            )
        if any(result.model != adapter.descriptor for result in results):
            raise RuntimeError("adapter returned a mismatched model descriptor")
        return results, sum(len(result.instances) for result in results)

    def persist_completed(
        future: Future[tuple[tuple[object, ...], int]],
    ) -> int:
        results, count = future.result()
        for result in results:
            writer.write(result)
        return count

    try:
        for batch in decoder:
            batch_started = time.perf_counter()
            raw_results = adapter.infer_raw(batch)
            elapsed = time.perf_counter() - batch_started
            measured = sum(
                frame.index >= warmup_frames for frame in batch.frames
            )
            if measured:
                measured_time += elapsed * (measured / len(batch))
                measured_frames += measured
            pending.append(
                output_pool.submit(convert_batch, batch, raw_results)
            )
            if len(pending) >= output_queue_batches:
                items += persist_completed(pending.popleft())
            processed += len(batch)
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
        while pending:
            items += persist_completed(pending.popleft())
    finally:
        decoder.close()
        output_pool.shutdown(wait=True, cancel_futures=True)
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


__all__ = ["run_mh0_video_inference"]
