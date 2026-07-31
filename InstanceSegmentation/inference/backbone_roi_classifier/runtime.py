"""Manifest-driven backbone ROI classification with a fixed SQLite class map."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.ops import roi_align

from .models import build_model


INTERNAL_CLASS_NAMES = ("male", "female", "junction")
INTERNAL_CLASS_IDS = (1, 2, 3)
CANONICAL_CLASS_NAMES = ("女性器", "男性器", "結合部分")
CANONICAL_CLASS_IDS = (1, 2, 3)
# Delivered logits are male/female/junction; SQLite is female/male/junction.
CANONICAL_FROM_INTERNAL = (1, 0, 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"classifier checkpoint root must be a mapping: {path}")
    if tuple(payload.get("class_names", ())) != INTERNAL_CLASS_NAMES:
        raise ValueError(f"unexpected classifier class_names in {path}")
    if tuple(int(value) for value in payload.get("class_ids", ())) != INTERNAL_CLASS_IDS:
        raise ValueError(f"unexpected classifier class_ids in {path}")
    return payload


def load_classifier_manifest(
    manifest_path: Path,
    *,
    mode: str = "fast",
) -> tuple[list[Path], list[float] | None, dict[str, object]]:
    manifest = manifest_path.expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if tuple(payload.get("class_names", ())) != INTERNAL_CLASS_NAMES:
        raise ValueError("classifier manifest has an unexpected class order")
    base = manifest.parent
    if "classifier" in payload:
        specification = payload["classifier"]
        paths = [base / specification["checkpoint"]]
        expected = [specification.get("checkpoint_sha256")]
        weights = None
        resolved_mode = "single"
    elif mode == "fast":
        specification = payload["fast"]
        paths = [base / specification["checkpoint"]]
        expected = [specification.get("checkpoint_sha256")]
        weights = None
        resolved_mode = "fast"
    elif mode == "accuracy":
        specification = payload["accuracy"]
        paths = [base / value for value in specification["checkpoints"]]
        expected = list(specification.get("checkpoint_sha256", ()))
        weights = [float(value) for value in specification["weights"]]
        resolved_mode = "accuracy"
    else:
        raise ValueError(f"unsupported classifier mode: {mode!r}")
    if len(expected) != len(paths):
        raise ValueError("classifier manifest hash count does not match checkpoints")
    for path, digest in zip(paths, expected):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"classifier checkpoint not found: {path}")
        if not digest or _sha256(path) != digest:
            raise ValueError(f"classifier checkpoint SHA-256 mismatch: {path}")
    provenance = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "mode": resolved_mode,
        "checkpoints": [str(path.resolve()) for path in paths],
        "detector_minimum_score": float(payload.get("detector_minimum_score", 0.0)),
        "input": payload.get("input"),
    }
    return [path.resolve() for path in paths], weights, provenance


class BackboneRoiClassifier(nn.Module):
    """One head or weighted ensemble over a shared stride-16 ROIAlign tensor."""

    def __init__(
        self,
        checkpoints: Sequence[Path],
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if not checkpoints:
            raise ValueError("at least one classifier checkpoint is required")
        loaded = [_load_checkpoint(path) for path in checkpoints]
        configs = [value.get("model_cfg") for value in loaded]
        if any(not isinstance(value, Mapping) for value in configs):
            raise ValueError("classifier checkpoint is missing model_cfg")
        first = configs[0]
        assert isinstance(first, Mapping)
        self.pooler_channels = int(first["pooler_channels"])
        self.pooler_size = int(first["pooler_size"])
        for index, config in enumerate(configs):
            assert isinstance(config, Mapping)
            if (
                int(config["pooler_channels"]) != self.pooler_channels
                or int(config["pooler_size"]) != self.pooler_size
            ):
                raise ValueError(f"classifier member {index} has incompatible ROI input")
        self.models = nn.ModuleList()
        for payload, config in zip(loaded, configs):
            assert isinstance(config, Mapping)
            model = build_model(config)
            state = payload.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError("classifier checkpoint is missing model_state")
            model.load_state_dict(state, strict=True)
            self.models.append(model.eval())
        values = (
            torch.ones(len(self.models), dtype=torch.float32)
            if weights is None
            else torch.as_tensor(weights, dtype=torch.float32)
        )
        if (
            len(values) != len(self.models)
            or not torch.isfinite(values).all()
            or float(values.sum()) <= 0.0
        ):
            raise ValueError("invalid classifier ensemble weights")
        self.register_buffer("ensemble_weights", values / values.sum())
        self.class_names = CANONICAL_CLASS_NAMES
        self.class_ids = CANONICAL_CLASS_IDS

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        mode: str = "fast",
    ) -> tuple["BackboneRoiClassifier", dict[str, object]]:
        paths, weights, provenance = load_classifier_manifest(
            manifest_path, mode=mode
        )
        classifier = cls(paths, weights=weights)
        input_contract = provenance.get("input")
        if not isinstance(input_contract, Mapping):
            raise ValueError("classifier manifest is missing its input contract")
        expected_shape = (
            classifier.pooler_channels,
            classifier.pooler_size,
            classifier.pooler_size,
        )
        manifest_shape = tuple(
            int(value) for value in input_contract.get("shape", ())
        )
        if manifest_shape != expected_shape:
            raise ValueError("classifier manifest input shape does not match checkpoint")
        if int(input_contract.get("stride", 0)) != 16:
            raise ValueError("classifier manifest requires unsupported feature stride")
        if input_contract.get("metadata") != "geo_v2":
            raise ValueError("classifier manifest requires unsupported metadata")
        return classifier, provenance

    def forward(self, roi_features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters())
        roi = roi_features
        if parameter.device.type == "cpu" and roi.dtype != parameter.dtype:
            roi = roi.to(parameter.dtype)
        metadata = metadata.to(device=parameter.device, dtype=torch.float32)
        combined: torch.Tensor | None = None
        with torch.autocast(
            device_type=parameter.device.type,
            dtype=torch.float16,
            enabled=parameter.device.type == "cuda",
        ):
            for index, model in enumerate(self.models):
                logits = model(roi, metadata).float()
                weighted = logits * self.ensemble_weights[index]
                combined = weighted if combined is None else combined + weighted
        assert combined is not None
        order = torch.as_tensor(
            CANONICAL_FROM_INTERNAL, device=combined.device, dtype=torch.long
        )
        return combined.index_select(1, order)

    def classify_backbone(
        self,
        backbone_feature: torch.Tensor,
        boxes_model_xyxy: torch.Tensor,
        metadata: torch.Tensor,
        *,
        batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = backbone_feature
        if feature.ndim != 4 or int(feature.shape[1]) != self.pooler_channels:
            raise ValueError(
                f"backbone feature must be [B,{self.pooler_channels},H,W], got {tuple(feature.shape)}"
            )
        boxes = boxes_model_xyxy.to(device=feature.device)
        if batch_indices is None:
            batch = torch.zeros(len(boxes), device=feature.device, dtype=boxes.dtype)
        else:
            batch = batch_indices.to(device=feature.device, dtype=boxes.dtype)
        rois = torch.cat((batch.reshape(-1, 1), boxes), dim=1).to(feature.dtype)
        roi = roi_align(
            feature,
            rois,
            output_size=(self.pooler_size, self.pooler_size),
            spatial_scale=1.0 / 16.0,
            sampling_ratio=0,
            aligned=True,
        )
        logits = self(roi, metadata)
        probabilities = functional.softmax(logits.float(), dim=1)
        scores, classes = probabilities.max(dim=1)
        return classes, scores, probabilities


__all__ = [
    "CANONICAL_CLASS_IDS",
    "CANONICAL_CLASS_NAMES",
    "BackboneRoiClassifier",
    "load_classifier_manifest",
]
