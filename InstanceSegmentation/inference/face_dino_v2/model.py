"""Build and execute a fixed-batch full-TensorRT Face DINO runtime."""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .cudagraph import build_zero_copy_detector_backend
from .deployment import (
    deployment_import_stubs,
    prepare_deployment_shell,
    validate_deployment_state,
)
from .optimized_predict import install_prefiltered_predict
from .preprocessing import (
    FusedVideoPreprocessor,
    LetterboxTransform,
    restore_result,
)
from .trt.bundle import FaceDinoEngineBundle, load_engine_bundle, sha256_file


FAMILY_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = FAMILY_ROOT / ".runtime" / "src" / "face_detection"
DEFAULT_CHECKPOINT = FAMILY_ROOT / "artifacts" / "detector" / "model_residual_v2.pth"
DEFAULT_TRT_BUNDLE = (
    FAMILY_ROOT / "artifacts" / "trt" / "fast-sm120-fixed-b8-v1" / "manifest.json"
)
DEFAULT_TRT_BUNDLE_B16 = (
    FAMILY_ROOT / "artifacts" / "trt" / "fast-sm120-fixed-b16-v1" / "manifest.json"
)


def configure_source_root(source_root: Path) -> Path:
    root = source_root.expanduser().resolve()
    required = (
        root / "face_dino_v1",
        root / "codino_face_detection" / "models",
        root / "codino_face_detection" / ".runtime" / "external" / "codino",
        root / "codino_face_detection" / ".runtime" / "external" / "dinov3",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Face DINO runtime source is incomplete: " + ", ".join(missing)
        )
    paths = (
        root,
        root / "codino_face_detection" / ".runtime" / "external" / "codino",
        root / "codino_face_detection" / ".runtime" / "external" / "dinov3",
    )
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


@dataclass(slots=True)
class FaceDinoRuntime:
    model: torch.nn.Module
    device: torch.device
    bundle: FaceDinoEngineBundle
    score_threshold: float
    warmup_iterations: int
    inference_stream: torch.cuda.Stream
    preprocessor: FusedVideoPreprocessor | None = None
    warmed_up: bool = False

    @property
    def fixed_batch_size(self) -> int:
        return self.bundle.batch_size

    def _ensure_preprocessor(self, frame: np.ndarray) -> FusedVideoPreprocessor:
        height, width = frame.shape[:2]
        if self.preprocessor is None:
            self.preprocessor = FusedVideoPreprocessor(
                batch_size=self.fixed_batch_size,
                frame_height=height,
                frame_width=width,
                device=self.device,
                plugin=self.bundle.plugins["preprocess_plugin"],
            )
        elif (
            self.preprocessor.frame_height != height
            or self.preprocessor.frame_width != width
        ):
            raise ValueError("video resolution changed during Face DINO inference")
        return self.preprocessor

    def _execute(
        self,
        images: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        sizes = [(images.shape[-2], images.shape[-1])] * self.fixed_batch_size
        caller = torch.cuda.current_stream(self.device)
        self.inference_stream.wait_stream(caller)
        with torch.cuda.stream(self.inference_stream):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                results = self.model.predict(
                    images,
                    sizes,
                    self.score_threshold,
                    return_ellipse_masks=True,
                )
        caller.wait_stream(self.inference_stream)
        images.record_stream(self.inference_stream)
        return results

    def predict(
        self,
        frames: list[np.ndarray],
    ) -> list[dict[str, torch.Tensor]]:
        results, transform = self.predict_raw(frames)
        return [restore_result(result, transform) for result in results]

    def predict_raw(
        self,
        frames: list[np.ndarray],
    ) -> tuple[list[dict[str, torch.Tensor]], LetterboxTransform]:
        """Return network-space tensors so an adapter can restore them in bulk."""

        if not 1 <= len(frames) <= self.fixed_batch_size:
            raise ValueError(f"Face DINO expects 1..{self.fixed_batch_size} frames")
        valid_count = len(frames)
        padded = [
            *frames,
            *[frames[-1]] * (self.fixed_batch_size - valid_count),
        ]
        preprocessor = self._ensure_preprocessor(frames[0])
        images, transforms = preprocessor.prepare(padded)
        if not self.warmed_up:
            for _ in range(self.warmup_iterations):
                self._execute(images)
            self.warmed_up = True
        results = self._execute(images)
        return results[:valid_count], transforms[0]

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        self.preprocessor = None
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def build_runtime(
    *,
    source_root: Path,
    checkpoint: Path,
    trt_bundle: Path,
    device: str,
    score_threshold: float,
    warmup_iterations: int,
    verify: str = "engines",
    cuda_graph: bool = False,
) -> FaceDinoRuntime:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("face_dino_v2 requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if tuple(torch.cuda.get_device_capability(target)) != (12, 0):
        raise RuntimeError("the bundled Face DINO engines require an SM120 GPU")
    source_root = configure_source_root(source_root)
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    bundle = load_engine_bundle(trt_bundle, verify=verify)
    observed_checkpoint_hash = sha256_file(checkpoint)
    if observed_checkpoint_hash != bundle.checkpoint_sha256:
        raise ValueError(
            "checkpoint does not match the TensorRT bundle: "
            f"expected={bundle.checkpoint_sha256}, "
            f"observed={observed_checkpoint_hash}"
        )

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with deployment_import_stubs():
        # Import only after the isolated runtime source paths are configured.
        from codino_face_detection.models import compatibility as _compatibility
        from face_dino_v1.inference import (
            TensorRTAttributeBackend,
            TensorRTBackboneNeck,
            install_batched_bbox,
            install_tensorrt_transformer,
        )
        from face_dino_v1.models.build import build_face_dino_explored

        del _compatibility
        model = build_face_dino_explored(
            attribute_config=payload.get("attribute_config")
        ).eval()
    prepare_deployment_shell(model)
    model = model.to(target)
    incompatible = model.load_state_dict(
        payload["model"],
        strict=False,
    )
    validate_deployment_state(incompatible)
    calibration = payload.get("inference_calibration", {})
    model.face_threshold = float(calibration.get("face_threshold", 0.55))
    model.point_threshold = float(calibration.get("point_threshold", 0.75))
    model = model.to(memory_format=torch.channels_last)
    model.detector.backbone = TensorRTBackboneNeck(bundle.engines["backbone_neck"])
    model.detector.neck = torch.nn.Identity()
    install_tensorrt_transformer(
        model.detector,
        query_engine=bundle.engines["query_encoder"],
        decoder_engine=bundle.engines["decoder"],
        plugin=bundle.plugins["msda_plugin"],
    )
    install_batched_bbox(
        model.detector,
        score_threshold=score_threshold,
    )
    install_prefiltered_predict(
        model,
        score_threshold=score_threshold,
    )
    model.set_attribute_backend(
        TensorRTAttributeBackend(
            bundle.engines["attribute"],
            ellipse_moment_power=model.attribute_model.ellipse_moment_power,
        )
    )
    if cuda_graph:
        model.set_detector_backend(
            build_zero_copy_detector_backend(
                model.detector,
                warmup_iterations=1,
            )
        )
    del payload
    gc.collect()
    return FaceDinoRuntime(
        model=model,
        device=target,
        bundle=bundle,
        score_threshold=float(score_threshold),
        warmup_iterations=int(warmup_iterations),
        inference_stream=torch.cuda.Stream(device=target),
    )


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SOURCE_ROOT",
    "DEFAULT_TRT_BUNDLE",
    "DEFAULT_TRT_BUNDLE_B16",
    "FaceDinoRuntime",
    "build_runtime",
    "configure_source_root",
]
