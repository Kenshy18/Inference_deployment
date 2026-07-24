"""EVA-02 model execution settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoInferenceSettings:
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "fp16"
    gpu_preprocess_float: bool = True
    model_half: bool = False
    pin_inputs: bool = False
    pack_inputs: bool = True
    score_threshold: float = 0.1
    use_inplace_boxes: bool = False
    use_raw_to_orig_masks: bool = True
    classifier_batch_size: int = 2048

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError(f"unsupported amp_dtype: {self.amp_dtype!r}")
        if self.classifier_batch_size <= 0:
            raise ValueError("classifier_batch_size must be positive")


__all__ = ["VideoInferenceSettings"]
