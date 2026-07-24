"""Postprocess artifact writers."""

from .contract import OutputContractError, OutputStats, validate_mask_sqlite
from contracts.mask_sqlite import (
    MaskRow,
    read_mask_rows,
    track_sort_key,
    write_mask_sqlite,
)


def union2sqlite_main() -> None:
    """Convert a union artifact without an approximation-runtime dependency."""

    from .sqlite import union2sqlite_main as _main

    _main()


__all__ = [
    "OutputContractError",
    "OutputStats",
    "MaskRow",
    "read_mask_rows",
    "track_sort_key",
    "union2sqlite_main",
    "validate_mask_sqlite",
    "write_mask_sqlite",
]
