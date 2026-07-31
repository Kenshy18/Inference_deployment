#!/usr/bin/env python3
"""Create a temporary MH0 classifier checkpoint for integration testing.

This does not claim classification accuracy.  It preserves the trained
Spatial-GAP blocks and class head from the large Co-DINO classifier, while
deterministically resampling only its 256-channel input projection to MH0's
192-channel ROI feature space.  Replace the generated checkpoint in-place
with a properly trained MH0 checkpoint later.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional


FAMILY_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = FAMILY_ROOT.parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from dinov3_codino.classifier import RoiSpatialGapClassifier
from dinov3_codino_mh0.classifier import (
    CLASS_IDS,
    CLASS_NAMES,
    INPUT_DIM,
    POOLER_CHANNELS,
    POOLER_SIZE,
)


DEFAULT_SOURCE = (
    INFERENCE_ROOT / "dinov3_codino" / "artifacts" / "classifier" / "best.pt"
)
DEFAULT_OUTPUT = FAMILY_ROOT / "artifacts" / "classifier" / "best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("source classifier checkpoint root must be a mapping")
    return value


def build_payload(
    source: dict[str, object],
    *,
    source_sha256: str,
) -> dict[str, object]:
    source_cfg = source.get("model_cfg")
    source_state = source.get("model_state")
    if not isinstance(source_cfg, dict) or not isinstance(source_state, dict):
        raise ValueError("source checkpoint must contain model_cfg and model_state")
    if int(source_cfg.get("pooler_channels", 0)) != 256:
        raise ValueError("bootstrap source must be the 256-channel Co-DINO classifier")
    if int(source_cfg.get("pooler_size", 0)) != POOLER_SIZE:
        raise ValueError("bootstrap source pooler_size must be 14")

    cfg = dict(source_cfg)
    cfg.update(
        {
            "input_dim": INPUT_DIM,
            "pooler_channels": POOLER_CHANNELS,
            "pooler_size": POOLER_SIZE,
            "num_classes": len(CLASS_NAMES),
        }
    )
    model = RoiSpatialGapClassifier(
        input_dim=INPUT_DIM,
        num_classes=len(CLASS_NAMES),
        use_meta=bool(cfg.get("use_meta", True)),
        pooler_channels=POOLER_CHANNELS,
        pooler_size=POOLER_SIZE,
        stem_channels=int(cfg.get("gap_stem_channels", 96)),
        mid_channels=int(cfg.get("gap_mid_channels", 96)),
        dw_kernel=int(cfg.get("gap_dw_kernel", 3)),
        dropout=float(cfg.get("gap_dropout", cfg.get("dropout", 0.0))),
    )
    target_state = model.state_dict()
    for name, target in tuple(target_state.items()):
        source_value = source_state.get(name)
        if not isinstance(source_value, torch.Tensor):
            raise KeyError(f"source classifier is missing tensor {name!r}")
        if tuple(source_value.shape) == tuple(target.shape):
            target_state[name] = source_value.detach().to(dtype=target.dtype)
            continue
        if name != "stem.0.weight":
            raise ValueError(
                f"unexpected bootstrap tensor mismatch for {name}: "
                f"source={tuple(source_value.shape)} target={tuple(target.shape)}"
            )
        # [out, in, 1, 1] -> interpolate the input-channel axis only.
        projected = functional.interpolate(
            source_value.detach().squeeze(-1).squeeze(-1).unsqueeze(0),
            size=POOLER_CHANNELS,
            mode="linear",
            align_corners=True,
        ).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        target_state[name] = projected.to(dtype=target.dtype)
    model.load_state_dict(target_state, strict=True)

    return {
        "artifact_status": "bootstrap_only_not_accuracy_validated",
        "class_ids": list(CLASS_IDS),
        "class_names": list(CLASS_NAMES),
        "epoch": None,
        "model_cfg": cfg,
        "model_state": model.state_dict(),
        "provenance": {
            "source_classifier_sha256": source_sha256,
            "source_pooler_channels": 256,
            "target_pooler_channels": POOLER_CHANNELS,
            "transform": "linear_resample_stem_input_axis",
            "warning": (
                "Temporary integration artifact. Replace with an MH0-trained "
                "checkpoint before evaluating classification accuracy."
            ),
        },
        "val_metrics": {},
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--overwrite", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        load_checkpoint(source_path),
        source_sha256=sha256(source_path),
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"saved bootstrap MH0 classifier: {output_path}")
    print(f"sha256: {sha256(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
