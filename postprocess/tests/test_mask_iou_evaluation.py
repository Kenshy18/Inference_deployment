from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contracts.mask_sqlite import MaskRow, write_mask_sqlite
from evaluation.mask_iou import _decode, _metrics, evaluate_mask_sqlites


def test_decode_preserves_sub_float32_half_pixel_geometry() -> None:
    points = [
        [1000.50000001, 500.0],
        [1000.49999999, 501.0],
        [1001.0, 502.0],
    ]
    decoded = _decode(json.dumps([points]))[0]
    assert decoded.dtype == np.float64
    assert decoded[0, 0] > 1000.5
    assert decoded[1, 0] < 1000.5
    assert np.float32(decoded[0, 0]) == np.float32(decoded[1, 0])


def test_streaming_merge_uses_the_sqlite_text_track_order(tmp_path: Path) -> None:
    square = json.dumps([[[0, 0], [4, 0], [4, 4], [0, 4]]])
    reference = write_mask_sqlite(
        tmp_path / "reference.sqlite",
        (
            MaskRow(frame=0, track_id="10", polygons=square),
            MaskRow(frame=0, track_id="2", polygons=square),
        ),
    )
    prediction = write_mask_sqlite(
        tmp_path / "prediction.sqlite",
        (MaskRow(frame=0, track_id="2", polygons=square),),
    )
    summary = evaluate_mask_sqlites(
        reference,
        prediction,
        tmp_path / "metrics.json",
    )
    assert summary["row_count_reference"] == 2
    assert summary["row_count_prediction"] == 1
    assert summary["reference_area"] == 50
    assert summary["prediction_area"] == 25
    assert summary["intersection"] == 25
    assert summary["union"] == 50
    assert summary["iou"] == 0.5


def test_raster_origin_matches_production_even_padding_parity() -> None:
    reference = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        dtype=np.float64,
    )
    prediction = reference + np.asarray([0.5, 0.0], dtype=np.float64)

    # With Production's ties-to-even origin convention, the half-pixel shift
    # rounds onto the same 3x3 raster.  An odd one-pixel padding would instead
    # flip the tie and incorrectly report only a 6/9 intersection.
    assert _metrics([reference], [prediction]) == (9, 9, 9, 9)
