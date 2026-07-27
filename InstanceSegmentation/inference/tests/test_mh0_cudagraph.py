from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from dinov3_codino_mh0.optimization.cudagraph import (  # noqa: E402
    Mh0CudaGraphCore,
    install_mh0_cuda_graph,
)


class Mh0CudaGraphTest(unittest.TestCase):
    def test_signature_includes_input_and_video_geometry(self):
        image = torch.empty((16, 3, 736, 1280))
        meta = {
            "ori_shape": (1080, 1920, 3),
            "img_shape": (736, 1280, 3),
            "scale_factor": np.asarray(
                (2 / 3, 2 / 3, 2 / 3, 2 / 3), dtype=np.float32
            ),
        }
        signature = Mh0CudaGraphCore._signature(image, [meta] * 16)
        self.assertEqual(signature[0], (16, 3, 736, 1280))
        self.assertEqual(signature[1], torch.float32)
        self.assertEqual(signature[3][0][0], (1080, 1920, 3))

    def test_install_requires_fixed_tensorrt_preprocessor(self):
        model = torch.nn.Module()
        model._mh0_gpu_preprocessor = None
        with self.assertRaisesRegex(ValueError, "TensorRT"):
            install_mh0_cuda_graph(model)

    def test_install_is_lazy_and_does_not_require_cuda(self):
        model = torch.nn.Module()
        model._mh0_gpu_preprocessor = object()
        returned = install_mh0_cuda_graph(model)
        self.assertIs(returned, model)
        self.assertIsInstance(model._mh0_cuda_graph_core, Mh0CudaGraphCore)


if __name__ == "__main__":
    unittest.main()
