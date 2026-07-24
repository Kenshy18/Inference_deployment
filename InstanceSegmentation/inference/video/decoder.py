"""Bounded source-order OpenCV decoder used by inference pipelines."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path

from contracts import ColorSpace, Frame, FrameBatch

from .metadata import VideoMetadata, read_video_metadata


class OpenCvVideoDecoder:
    def __init__(
        self,
        path: Path,
        *,
        batch_size: int,
        max_frames: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self.path = path.expanduser().resolve()
        self.batch_size = int(batch_size)
        self.max_frames = max_frames
        self.metadata: VideoMetadata = read_video_metadata(self.path)

    def __iter__(self) -> Iterator[FrameBatch]:
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")
        index = 0
        try:
            while self.max_frames is None or index < self.max_frames:
                frames: list[Frame] = []
                while len(frames) < self.batch_size:
                    if self.max_frames is not None and index >= self.max_frames:
                        break
                    ok, image = capture.read()
                    if not ok:
                        break
                    frames.append(
                        Frame(
                            index=index,
                            timestamp_sec=index / self.metadata.fps,
                            image=image,
                            color_space=ColorSpace.BGR,
                        )
                    )
                    index += 1
                if not frames:
                    break
                yield FrameBatch.from_sequence(frames)
                if len(frames) < self.batch_size:
                    break
        finally:
            capture.release()


class AsyncVideoDecoder:
    """Prefetch decoded batches without moving preprocessing into shared code."""

    def __init__(
        self,
        path: Path,
        *,
        batch_size: int,
        max_frames: int | None = None,
        prefetch_batches: int = 2,
    ) -> None:
        self.decoder = OpenCvVideoDecoder(
            path, batch_size=batch_size, max_frames=max_frames
        )
        self.metadata = self.decoder.metadata
        self.queue: queue.Queue[FrameBatch | BaseException | None] = queue.Queue(
            maxsize=max(1, int(prefetch_batches))
        )
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._worker,
            name=f"decode:{self.decoder.path.name}",
            daemon=True,
        )

    def _put(self, value: FrameBatch | BaseException | None) -> bool:
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        try:
            for batch in self.decoder:
                if not self._put(batch):
                    return
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(None)

    def __iter__(self) -> Iterator[FrameBatch]:
        self.thread.start()
        try:
            while True:
                value = self.queue.get()
                if value is None:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield value
        finally:
            self.close()

    def close(self) -> None:
        self.stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


__all__ = ["AsyncVideoDecoder", "OpenCvVideoDecoder"]
