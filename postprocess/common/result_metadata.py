"""Late-bound access to result metadata owned by the artifact feature."""

from __future__ import annotations

from pathlib import Path


def record_result_processing_run(
    result_sqlite: Path,
    *,
    kind: str,
    name: str,
    resolved_config: object,
    stages: list[dict[str, object]],
) -> dict[str, object]:
    from artifacts.unified_sqlite import record_processing_run

    return record_processing_run(
        result_sqlite,
        kind=kind,
        name=name,
        resolved_config=resolved_config,
        stages=stages,
    )


__all__ = ["record_result_processing_run"]
