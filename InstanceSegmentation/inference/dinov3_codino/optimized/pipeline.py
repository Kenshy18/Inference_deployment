"""Optimized fixed-B2 Co-DINO video pipeline with SQLite contracts."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from contracts import (
    ModelDescriptor,
    SegmentationFrame,
    TaskType,
    segmentation_frame_from_rows,
)
from pipelines import InferenceRunSummary
from video import read_video_metadata

try:
    from ..model import CoDinoRuntime, VideoInferenceSettings
    from ..postprocessing import (
        detections_to_rows,
        normalize_result,
        restore_boxes,
        restore_segmentations,
    )
except ImportError:
    from model import CoDinoRuntime, VideoInferenceSettings
    from postprocessing import (
        detections_to_rows,
        normalize_result,
        restore_boxes,
        restore_segmentations,
    )

from .core import FixedB2DetectorGraph
from .executor import FastB2Executor
from .prepared import iter_prepared_video_batches


FAST_DESCRIPTOR = ModelDescriptor(
    model_id="dinov3_codino_trt_fast",
    task=TaskType.INSTANCE_SEGMENTATION,
    implementation="sm120_t090_t140_cuda_graph_double_buffer",
)
ProgressCallback = Callable[[InferenceRunSummary], None]


def _to_contract_result(
    *,
    runtime: CoDinoRuntime,
    frame,
    raw_result,
    settings: VideoInferenceSettings,
) -> SegmentationFrame:
    box_results, segmentation_results = normalize_result(raw_result)
    original_shape = frame.image.shape[:2]
    restored_boxes = restore_boxes(
        box_results,
        original_shape,
        runtime.target_size,
    )
    restored_masks = restore_segmentations(
        segmentation_results,
        original_shape,
        runtime.target_size,
    )
    rows = detections_to_rows(
        restored_boxes,
        restored_masks,
        class_names=list(runtime.class_names),
        class_ids=list(runtime.class_ids),
        score_threshold=settings.score_threshold,
    )
    return segmentation_frame_from_rows(
        model=FAST_DESCRIPTOR,
        frame=frame,
        rows=rows,
    )


def run_fast_video_inference(
    *,
    input_path: Path,
    runtime: CoDinoRuntime,
    writer,
    settings: VideoInferenceSettings,
    max_frames: int | None,
    warmup_frames: int,
    metadata: Mapping[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> InferenceRunSummary:
    """Run the validated fast profile without leaving shared contracts."""

    if runtime.fixed_batch_size != 2 or settings.batch_size != 2:
        raise ValueError("optimized Co-DINO requires fixed batch size 2")
    if runtime.classifier is None:
        raise RuntimeError("optimized Co-DINO requires its bundled classifier")
    if not next(runtime.model.parameters()).is_cuda:
        raise RuntimeError("optimized Co-DINO requires CUDA")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")

    video_metadata = read_video_metadata(input_path)
    writer.set_metadata(
        {
            "input": str(input_path.expanduser().resolve()),
            "model_id": FAST_DESCRIPTOR.model_id,
            "task": FAST_DESCRIPTOR.task.value,
            "contract_version": FAST_DESCRIPTOR.contract_version,
            "backend": "tensorrt-fast",
            "video": {
                "frames": video_metadata.frames,
                "fps": video_metadata.fps,
                "width": video_metadata.width,
                "height": video_metadata.height,
            },
            **({} if metadata is None else dict(metadata)),
        }
    )

    detector_graph = FixedB2DetectorGraph(
        runtime.model,
        amp=settings.amp,
    )
    prepared_source = (
        (prepared.decoded, prepared.model_data)
        for prepared in iter_prepared_video_batches(
            input_path,
            batch_size=2,
            max_frames=max_frames,
            target_size=runtime.target_size,
        )
    )
    executor = FastB2Executor(
        model=runtime.model,
        classifier=runtime.classifier,
        detector_graph=detector_graph,
        target_size=runtime.target_size,
        num_classifier_classes=len(runtime.class_names),
    )

    processed = 0
    items = 0
    measured_frames = 0
    measured_time = 0.0
    started = time.perf_counter()
    output_pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="codino-contract-output",
    )
    pending_output: deque[Future] = deque()

    def persist_completed(completed) -> int:
        written = 0
        batch = completed.decoded
        valid_count = len(batch)
        for frame, raw_result in zip(
            batch.frames,
            completed.results[:valid_count],
        ):
            result = _to_contract_result(
                runtime=runtime,
                frame=frame,
                raw_result=raw_result,
                settings=settings,
            )
            writer.write(result)
            written += len(result.instances)
        return written

    try:
        for completed in executor.iter_results(prepared_source):
            batch = completed.decoded
            elapsed = completed.completion_interval_sec
            valid_count = len(batch)
            measured = sum(
                frame.index >= warmup_frames for frame in batch.frames
            )
            if measured:
                measured_time += elapsed * (measured / valid_count)
                measured_frames += measured
            pending_output.append(output_pool.submit(persist_completed, completed))
            if len(pending_output) >= 2:
                items += int(pending_output.popleft().result())
            processed += valid_count
            if progress is not None:
                wall_elapsed = time.perf_counter() - started
                progress(
                    InferenceRunSummary(
                        input=str(input_path),
                        model_id=FAST_DESCRIPTOR.model_id,
                        task=FAST_DESCRIPTOR.task.value,
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
        while pending_output:
            items += int(pending_output.popleft().result())
    finally:
        output_pool.shutdown(wait=True, cancel_futures=True)
        writer.close()
    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - started
    return InferenceRunSummary(
        input=str(input_path),
        model_id=FAST_DESCRIPTOR.model_id,
        task=FAST_DESCRIPTOR.task.value,
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


__all__ = ["FAST_DESCRIPTOR", "run_fast_video_inference"]
