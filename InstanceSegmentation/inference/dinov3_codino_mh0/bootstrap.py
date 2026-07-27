"""Import and construct the exact MH0 PyTorch graph."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def prepare_imports() -> None:
    codino = ROOT / ".runtime" / "src" / "codino"
    dinov3 = ROOT / ".runtime" / "src" / "dinov3_root"
    for path in (ROOT, codino, dinov3):
        if not path.is_dir():
            raise FileNotFoundError(
                f"MH0 runtime source missing: {path}; "
                "run setup_environment.py first"
            )
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("CODINO_ROOT", str(codino))
    os.environ.setdefault("UNIFIED_DINOV3_PACKAGE_ROOT", str(dinov3))
    os.environ.setdefault("DINOV3_USE_XFORMERS", "0")

    import projects.models  # noqa: F401
    import mmdet.datasets.pipelines.unified_letterbox_aug  # noqa: F401
    import deployment.mh0_models  # noqa: F401

    import mmcv.parallel._functions as parallel_functions
    import torch

    original = parallel_functions._get_stream
    if not getattr(original, "_mh0_torch_compatible", False):
        def compatible_get_stream(device):
            if isinstance(device, int):
                device = torch.device("cuda", device)
            return original(device)

        compatible_get_stream._mh0_torch_compatible = True
        parallel_functions._get_stream = compatible_get_stream


def build_model(*, config: Path, checkpoint: Path, device: str):
    prepare_imports()
    import torch
    from mmcv import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(str(config))
    cfg.model.backbone.pretrained = False
    cfg.model.backbone.weights = None
    cfg.model.train_cfg = None
    cfg.load_from = None
    cfg.resume_from = None
    cfg.work_dir = str(ROOT / ".runtime" / "work")
    cfg.gpu_ids = range(1)
    cfg.pop("fp16", None)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = init_detector(cfg, str(checkpoint), device=device)
    model.CLASSES = ("foreground",)
    return model.eval()


__all__ = ["build_model", "prepare_imports"]
