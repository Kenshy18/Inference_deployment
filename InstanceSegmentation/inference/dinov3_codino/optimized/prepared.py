"""Bounded decode and preprocessing lookahead for optimized Co-DINO."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from contracts import FrameBatch
from video import OpenCvVideoDecoder

try:
    from ..preprocessing import prepare_batch_host
except ImportError:
    from preprocessing import prepare_batch_host


@dataclass(slots=True)
class PreparedVideoBatch:
    decoded: FrameBatch
    model_data: dict[str, list[Any]]


@dataclass(slots=True)
class _ProducerFailure:
    error: BaseException


_END = object()


def iter_prepared_video_batches(
    video_path: Path,
    *,
    batch_size: int,
    max_frames: int | None,
    target_size: tuple[int, int],
    queue_size: int = 2,
) -> Iterator[PreparedVideoBatch]:
    """Overlap bounded decode/OpenCV work with the current GPU batch."""

    items: queue.Queue[object] = queue.Queue(maxsize=max(1, queue_size))
    stopped = threading.Event()

    def offer(value: object) -> bool:
        while not stopped.is_set():
            try:
                items.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            decoder = OpenCvVideoDecoder(
                video_path,
                batch_size=batch_size,
                max_frames=max_frames,
            )
            for decoded in decoder:
                inference_frames = decoded.images
                if len(inference_frames) < batch_size:
                    inference_frames = [
                        *inference_frames,
                        *[inference_frames[-1]]
                        * (batch_size - len(inference_frames)),
                    ]
                prepared = prepare_batch_host(inference_frames, target_size)
                prepared["img"][0] = prepared["img"][0].pin_memory()
                if not offer(PreparedVideoBatch(decoded, prepared)):
                    return
        except BaseException as error:
            offer(_ProducerFailure(error))
        finally:
            offer(_END)

    worker = threading.Thread(
        target=produce,
        name="codino-decode-preprocess",
        daemon=True,
    )
    worker.start()
    try:
        while True:
            item = items.get()
            if item is _END:
                return
            if isinstance(item, _ProducerFailure):
                raise item.error
            assert isinstance(item, PreparedVideoBatch)
            yield item
    finally:
        stopped.set()
        worker.join(timeout=5.0)
        if worker.is_alive():
            raise RuntimeError("Co-DINO preprocess producer did not stop")


__all__ = ["PreparedVideoBatch", "iter_prepared_video_batches"]
