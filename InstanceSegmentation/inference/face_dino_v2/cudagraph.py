"""Zero-copy CUDA Graph adapter for the fixed Face DINO video buffer."""

from __future__ import annotations

import torch


def build_zero_copy_detector_backend(
    detector,
    *,
    warmup_iterations: int = 1,
):
    """Capture the detector against the preprocessor's reusable CUDA tensor.

    The upstream graph helper copies every batch into a private static tensor.
    Video inference already owns a fixed-address output buffer, so retaining
    that tensor removes a full Bx3x736x1280 device copy per batch.
    """

    if int(detector.query_head.num_classes) != 1:
        raise ValueError(
            "zero-copy Face DINO CUDA Graph requires one detector class"
        )
    from face_dino_v1.inference import CudaGraphDetectorBackend

    class ZeroCopyCudaGraphDetectorBackend(CudaGraphDetectorBackend):
        def _capture(self, images, image_sizes) -> None:
            self.signature = self._signature(images, image_sizes)
            self.static_input = images
            metas = self.detector._metas(images, image_sizes)
            side_stream = torch.cuda.Stream(device=images.device)
            side_stream.wait_stream(torch.cuda.current_stream(images.device))
            with torch.cuda.stream(side_stream):
                for _ in range(self.warmup_iterations):
                    self.static_pyramid, self.static_outputs = self._core(
                        images,
                        metas,
                    )
            torch.cuda.current_stream(images.device).wait_stream(side_stream)
            torch.cuda.synchronize(images.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_pyramid, self.static_outputs = self._core(
                    images,
                    metas,
                )

        def __call__(self, images, image_sizes):
            signature = self._signature(images, image_sizes)
            if self.signature is None:
                self._capture(images, image_sizes)
            elif (
                signature != self.signature
                or self.static_input is None
                or images.data_ptr() != self.static_input.data_ptr()
            ):
                if self.fallback_on_mismatch:
                    return self._eager(images, image_sizes)
                raise ValueError(
                    "zero-copy CUDA Graph input address or signature changed"
                )
            assert self.static_pyramid is not None
            assert self.static_outputs is not None
            assert self.graph is not None
            self.graph.replay()
            metas = self.detector._metas(images, image_sizes)
            return self.static_pyramid, self._postprocess(
                self.static_outputs,
                metas,
            )

        def _postprocess(self, outputs, metas):
            head = self.detector.query_head
            results = head.get_bboxes(
                *outputs,
                metas,
                rescale=False,
                with_nms=head.test_cfg.get("nms", None) is not None,
            )
            return [
                {
                    "boxes": boxes_with_scores[:, :4],
                    "scores": boxes_with_scores[:, 4],
                }
                for boxes_with_scores, _labels in results
            ]

    return ZeroCopyCudaGraphDetectorBackend(
        detector,
        warmup_iterations=warmup_iterations,
    )


__all__ = ["build_zero_copy_detector_backend"]
