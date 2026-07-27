"""Construction and execution of the supported fixed-batch MH0 backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batched_bbox import install_batched_bbox_test
from .batched_mask import install_batched_mask_test
from .gpu_preprocessing import GPUPreprocessor
from .preprocessing import prepare_fixed_b2
from .trt_backbone import TensorRTBackboneNeck
from .trt_mask_head import TensorRTMaskHead
from .trt_transformer import install_trt_transformer


SUPPORTED_BACKENDS = frozenset({"tensorrt-fast", "pytorch"})
TRT_ARTIFACT_NAMES = frozenset(
    {
        "backbone_neck",
        "query_encoder",
        "decoder",
        "mask_head",
        "plugin",
        "preprocess_plugin",
    }
)


def _validate_trt_artifacts(
    artifacts: dict[str, Path] | None,
) -> dict[str, Path]:
    if artifacts is None:
        raise ValueError("tensorrt-fast requires a verified engine bundle")
    observed = set(artifacts)
    if observed != TRT_ARTIFACT_NAMES:
        raise ValueError(
            "TensorRT artifact set mismatch: "
            f"missing={sorted(TRT_ARTIFACT_NAMES - observed)}, "
            f"extra={sorted(observed - TRT_ARTIFACT_NAMES)}"
        )
    return artifacts


def build_backend_model(
    *,
    config,
    checkpoint,
    backend: str,
    device: str,
    model_score_threshold: float,
    artifacts: dict[str, Path] | None = None,
):
    """Build the public fixed-B16 TensorRT or fixed-B2 PyTorch backend."""

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if not 0.0 <= model_score_threshold <= 1.0:
        raise ValueError("model_score_threshold must be in [0, 1]")
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("MH0 inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    trt_artifacts = None
    if backend == "tensorrt-fast":
        trt_artifacts = _validate_trt_artifacts(artifacts)
        capability = tuple(torch.cuda.get_device_capability(target_device))
        if capability != (12, 0):
            raise RuntimeError(
                f"the bundled TensorRT engine requires SM120, got {capability}"
            )
    elif artifacts is not None:
        raise ValueError("the PyTorch backend does not accept TensorRT artifacts")

    try:
        from ..bootstrap import build_model
    except ImportError:
        from bootstrap import build_model
    model = build_model(
        config=config,
        checkpoint=checkpoint,
        device=device,
    )
    query_test_config = model.query_head.test_cfg
    query_test_config["score_thr"] = float(model_score_threshold)

    if trt_artifacts is not None:
        model.backbone = TensorRTBackboneNeck(
            trt_artifacts["backbone_neck"]
        ).to(device).eval()
        model.neck = torch.nn.Identity()
        install_trt_transformer(
            model,
            query_engine=trt_artifacts["query_encoder"],
            decoder_engine=trt_artifacts["decoder"],
            plugin=trt_artifacts["plugin"],
        )
        install_batched_bbox_test(model)
        model.mask_head = TensorRTMaskHead(
            model.mask_head,
            trt_artifacts["mask_head"],
        ).eval()
        install_batched_mask_test(model)
        # This auxiliary head only emits mask-quality scores; it neither
        # changes boxes nor masks and is not consumed by the local renderer.
        if hasattr(model, "mask_iou_head"):
            delattr(model, "mask_iou_head")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model._mh0_inference_stream = torch.cuda.Stream(device=device)
    if trt_artifacts is None:
        model._mh0_batch_size = 2
        model._mh0_gpu_preprocessor = None
    else:
        model._mh0_batch_size = int(model.backbone.input_shape[0])
        if model._mh0_batch_size != 16:
            raise RuntimeError(
                "tensorrt-fast requires a fixed-B16 backbone engine, "
                f"got B{model._mh0_batch_size}"
            )
        model._mh0_gpu_preprocessor = GPUPreprocessor(
            device,
            model._mh0_batch_size,
            fused_extension=trt_artifacts["preprocess_plugin"],
        )
    return model


def infer_fixed_batch(
    model,
    frames: list[np.ndarray],
    *,
    device: str,
) -> tuple[Any, int]:
    """Run one partial or full batch using the backend's fixed capacity."""

    if not frames:
        raise ValueError("frames must not be empty")
    caller_stream = torch.cuda.current_stream(device=device)
    inference_stream = model._mh0_inference_stream
    inference_stream.wait_stream(caller_stream)
    if model._mh0_gpu_preprocessor is None:
        prepared, valid_count = prepare_fixed_b2(frames, device)
    else:
        prepared, valid_count = model._mh0_gpu_preprocessor.prepare(
            frames, inference_stream
        )
    with torch.inference_mode():
        with torch.cuda.stream(inference_stream):
            results = model(return_loss=False, rescale=True, **prepared)
    caller_stream.wait_stream(inference_stream)
    return results[:valid_count], valid_count


__all__ = [
    "SUPPORTED_BACKENDS",
    "TRT_ARTIFACT_NAMES",
    "build_backend_model",
    "infer_fixed_batch",
]
