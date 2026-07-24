"""Public, implementation-neutral postprocess contracts.

Feature packages may depend on this package.  They must not import another
feature package's private implementation.
"""

from .detections import (
    CutList,
    iter_detection_records,
    transform_detection_jsonl,
    write_cut_list,
)
from .detector_sqlite import (
    MaskSqliteKind,
    detect_mask_sqlite_kind,
    validate_raw_detection_sqlite,
)
from .artifacts import (
    ArtifactContractError,
    artifact_contract_names,
    register_artifact_contract,
    validate_artifact,
)
from .mask_sqlite import (
    MaskRow,
    read_mask_rows,
    track_sort_key,
    write_mask_sqlite,
)
from .stages import PostprocessStage, StageContext, StageResult

__all__ = [
    "CutList",
    "MaskSqliteKind",
    "ArtifactContractError",
    "MaskRow",
    "PostprocessStage",
    "StageContext",
    "StageResult",
    "iter_detection_records",
    "detect_mask_sqlite_kind",
    "artifact_contract_names",
    "read_mask_rows",
    "register_artifact_contract",
    "track_sort_key",
    "transform_detection_jsonl",
    "validate_artifact",
    "write_cut_list",
    "validate_raw_detection_sqlite",
    "write_mask_sqlite",
]
