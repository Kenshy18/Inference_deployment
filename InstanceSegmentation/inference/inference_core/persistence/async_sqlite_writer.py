"""Bounded asynchronous SQLite persistence for GPU-overlapped pipelines."""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from pathlib import Path

from .sqlite_writer import SqliteWriter


_END = object()


class AsyncSqliteWriter:
    """Own a SqliteWriter on one worker thread and preserve write order."""

    def __init__(
        self,
        path: Path,
        *,
        overwrite: bool = False,
        safe: bool = True,
        commit_interval: int = 256,
        queue_size: int = 32,
    ) -> None:
        self.path = path
        self.overwrite = overwrite
        self.safe = safe
        self.commit_interval = commit_interval
        self.items: queue.Queue[object] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self.ready = threading.Event()
        self.failure: BaseException | None = None
        self.closed = False
        self.worker = threading.Thread(
            target=self._run,
            name=f"sqlite:{Path(path).name}",
            daemon=True,
        )
        self.worker.start()
        self.ready.wait()
        self._raise_if_failed()

    def _run(self) -> None:
        writer: SqliteWriter | None = None
        try:
            writer = SqliteWriter(
                self.path,
                overwrite=self.overwrite,
                safe=self.safe,
                commit_interval=self.commit_interval,
            )
            self.ready.set()
            while True:
                item = self.items.get()
                if item is _END:
                    break
                operation, value = item
                if operation == "metadata":
                    writer.set_metadata(value)
                elif operation == "write":
                    writer.write(value)
                else:
                    raise RuntimeError(
                        f"unknown asynchronous SQLite operation: {operation}"
                    )
        except BaseException as exc:
            self.failure = exc
        finally:
            self.ready.set()
            if writer is not None:
                try:
                    writer.close()
                except BaseException as exc:
                    if self.failure is None:
                        self.failure = exc

    def _raise_if_failed(self) -> None:
        if self.failure is not None:
            raise RuntimeError("asynchronous SQLite writer failed") from self.failure

    def _put(self, value: object) -> None:
        if self.closed:
            raise RuntimeError("asynchronous SQLite writer is closed")
        while True:
            self._raise_if_failed()
            try:
                self.items.put(value, timeout=0.1)
                return
            except queue.Full:
                if not self.worker.is_alive():
                    self._raise_if_failed()
                    raise RuntimeError("asynchronous SQLite worker stopped")

    def set_metadata(self, values: Mapping[str, object]) -> None:
        self._put(("metadata", dict(values)))

    def write(self, result) -> None:
        self._put(("write", result))

    def close(self) -> None:
        if self.closed:
            return
        self._put(_END)
        self.closed = True
        self.worker.join()
        self._raise_if_failed()


__all__ = ["AsyncSqliteWriter"]
