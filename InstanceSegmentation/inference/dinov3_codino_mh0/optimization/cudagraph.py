"""Zero-copy CUDA Graph for the fixed-shape MH0 detector core."""

from __future__ import annotations

from typing import Any

import torch


class Mh0CudaGraphCore:
    """Capture backbone, query encoder, and decoder against a stable input.

    The fused video preprocessor reuses one CUDA output tensor for every B16
    batch.  Capturing directly against that address avoids an additional
    67 MiB device-to-device copy while removing the per-batch Python/kernel
    launch overhead from the fixed-shape detector core.  Variable-length bbox
    and mask postprocessing deliberately remains eager.
    """

    def __init__(self, model: torch.nn.Module, *, warmup_iterations: int = 1):
        if warmup_iterations < 1:
            raise ValueError("warmup_iterations must be positive")
        self.model = model
        self.warmup_iterations = int(warmup_iterations)
        self.signature: tuple[Any, ...] | None = None
        self.static_input: torch.Tensor | None = None
        self.static_pyramid: Any = None
        self.static_outputs: Any = None
        self.graph: torch.cuda.CUDAGraph | None = None

    @staticmethod
    def _signature(
        image: torch.Tensor, img_metas: list[dict[str, Any]]
    ) -> tuple[Any, ...]:
        geometry = tuple(
            (
                tuple(meta["ori_shape"]),
                tuple(meta["img_shape"]),
                tuple(float(value) for value in meta["scale_factor"]),
            )
            for meta in img_metas
        )
        return (
            tuple(image.shape),
            image.dtype,
            image.device,
            geometry,
        )

    def _core(
        self, image_list: list[torch.Tensor], img_metas: list[dict[str, Any]]
    ):
        pyramid = self.model.extract_feat(image_list, img_metas)
        query_features = self.model._query_features(pyramid)
        outputs = self.model.query_head.forward(query_features, img_metas)
        return pyramid, outputs

    def _capture(
        self, image_list: list[torch.Tensor], img_metas: list[dict[str, Any]]
    ) -> None:
        image = image_list[0]
        self.signature = self._signature(image, img_metas)
        self.static_input = image
        current_stream = torch.cuda.current_stream(image.device)
        side_stream = torch.cuda.Stream(device=image.device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(self.warmup_iterations):
                self.static_pyramid, self.static_outputs = self._core(
                    image_list, img_metas
                )
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(image.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_pyramid, self.static_outputs = self._core(
                image_list, img_metas
            )

    def __call__(
        self, image_list: list[torch.Tensor], img_metas: list[dict[str, Any]]
    ):
        image = image_list[0]
        signature = self._signature(image, img_metas)
        if self.signature is None:
            self._capture(image_list, img_metas)
        elif (
            signature != self.signature
            or self.static_input is None
            or image.data_ptr() != self.static_input.data_ptr()
        ):
            # Video resolution changes are unusual but valid.  Preserve
            # correctness instead of replaying a graph against stale storage.
            return self._core(image_list, img_metas)
        assert self.graph is not None
        self.graph.replay()
        return self.static_pyramid, self.static_outputs


def install_mh0_cuda_graph(
    model: torch.nn.Module, *, warmup_iterations: int = 1
) -> torch.nn.Module:
    if getattr(model, "_mh0_gpu_preprocessor", None) is None:
        raise ValueError("MH0 CUDA Graph requires the fixed TensorRT backend")
    model._mh0_cuda_graph_core = Mh0CudaGraphCore(
        model, warmup_iterations=warmup_iterations
    )
    return model


__all__ = ["Mh0CudaGraphCore", "install_mh0_cuda_graph"]
