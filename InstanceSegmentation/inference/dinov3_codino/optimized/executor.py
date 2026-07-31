"""Double-buffered core/tail scheduler for the optimized fixed-B2 profile."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
from contracts import FrameBatch

from .tail import (
    FastCorePayload,
    capture_fast_core,
    infer_fast_b2_tail,
)


@dataclass(slots=True)
class CompletedFastBatch:
    decoded: FrameBatch
    results: Any
    completion_interval_sec: float


@dataclass(slots=True)
class _PendingBatch:
    decoded: FrameBatch
    future: Future


class FastB2Executor:
    """Overlap the previous tail with the next stable-pointer core replay."""

    def __init__(
        self,
        *,
        model,
        classifier,
        detector_graph,
        target_size: tuple[int, int],
        num_classifier_classes: int,
    ) -> None:
        self.model = model
        self.classifier = classifier
        self.detector_graph = detector_graph
        self.target_size = target_size
        self.num_classifier_classes = num_classifier_classes
        self.worker_stream = torch.cuda.Stream(priority=-1)
        self.pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="codino-tail",
        )
        self.slots: list[FastCorePayload | None] = [None, None]
        self.slot_index = 0

    def _submit(self, decoded: FrameBatch, prepared_data) -> _PendingBatch:
        slot = self.slot_index
        self.slot_index = (self.slot_index + 1) % len(self.slots)
        payload = capture_fast_core(
            self.model,
            detector_graph=self.detector_graph,
            prepared_data=prepared_data,
            destination=self.slots[slot],
        )
        self.slots[slot] = payload
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())

        def run_tail():
            with torch.cuda.stream(self.worker_stream), torch.inference_mode():
                self.worker_stream.wait_event(ready)
                return infer_fast_b2_tail(
                    self.model,
                    payload=payload,
                    target_size=self.target_size,
                    classifier=self.classifier,
                    num_classifier_classes=self.num_classifier_classes,
                )

        return _PendingBatch(decoded, self.pool.submit(run_tail))

    def iter_results(
        self,
        source: Iterable[tuple[FrameBatch, Any]],
    ) -> Iterator[CompletedFastBatch]:
        pending: _PendingBatch | None = None
        last_completion = time.perf_counter()
        try:
            for decoded, prepared_data in source:
                current = self._submit(decoded, prepared_data)
                if pending is not None:
                    results = pending.future.result()
                    completed = time.perf_counter()
                    yield CompletedFastBatch(
                        pending.decoded,
                        results,
                        completed - last_completion,
                    )
                    last_completion = completed
                pending = current
            if pending is not None:
                results = pending.future.result()
                completed = time.perf_counter()
                yield CompletedFastBatch(
                    pending.decoded,
                    results,
                    completed - last_completion,
                )
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)


__all__ = ["CompletedFastBatch", "FastB2Executor"]
