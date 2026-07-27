#!/usr/bin/env python3
"""Export a fixed-N MH0 mask refinement core to ONNX."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def prepare_model(config: Path, checkpoint: Path, device: str):
    if str(BUNDLE_ROOT) not in sys.path:
        sys.path.insert(0, str(BUNDLE_ROOT))
    from bootstrap import build_model
    return build_model(
        config=config,
        checkpoint=checkpoint,
        device=device,
    )


class MaskHeadCoreExport(torch.nn.Module):
    """The convolutional refinement core after semantic RoI extraction."""

    def __init__(self, mask_head: torch.nn.Module) -> None:
        super().__init__()
        self.mask_head = mask_head
        if len(mask_head.stages) != 2:
            raise RuntimeError(
                f"MH0 export expects two refinement stages, got {len(mask_head.stages)}"
            )

    def forward(
        self,
        instance_feats: torch.Tensor,
        sem0: torch.Tensor,
        sem1: torch.Tensor,
    ):
        semantic_rois = (sem0, sem1)
        for conv in self.mask_head.instance_convs:
            instance_feats = conv(instance_feats)

        predictions = []
        for index, stage in enumerate(self.mask_head.stages):
            logits = self.mask_head.stage_instance_logits[index](
                instance_feats
            )[:, :1]
            upsample = (
                self.mask_head.pre_upsample_last_stage
                or index < len(self.mask_head.stages) - 1
            )
            fused = torch.cat(
                [instance_feats, semantic_rois[index], logits.sigmoid()],
                dim=1,
            )
            for conv in stage.fuse_conv:
                fused = stage.relu(conv(fused))
            fused = stage.relu(stage.fuse_transform_out(fused))
            fused = torch.cat([fused, logits.sigmoid()], dim=1)
            instance_feats = stage.upsample(fused) if upsample else fused
            predictions.append(logits)

        final = self.mask_head.stage_instance_logits[-1](instance_feats)[:, :1]
        if not self.mask_head.pre_upsample_last_stage:
            final = functional.interpolate(
                final,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        predictions.append(final)
        predictions.append(
            functional.interpolate(
                final,
                size=(112, 112),
                mode="bilinear",
                align_corners=True,
            )
        )
        return tuple(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BUNDLE_ROOT / "configs" / "source_resolved_config.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BUNDLE_ROOT
        / "checkpoints"
        / "video_pseudo_mh0_epoch6_ema_deploy.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-rois", type=int, default=1)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=BUNDLE_ROOT
        / "artifacts"
        / "trt"
        / "mh0_mask_head_core_n1_fp32.onnx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_rois < 1:
        raise ValueError("--num-rois must be positive")
    model = prepare_model(args.config, args.checkpoint, args.device)
    wrapper = MaskHeadCoreExport(model.mask_head).to(args.device).eval()
    dummies = (
        torch.randn(args.num_rois, 192, 14, 14, device=args.device),
        torch.randn(args.num_rois, 192, 14, 14, device=args.device),
        torch.randn(args.num_rois, 96, 28, 28, device=args.device),
    )
    with torch.inference_mode():
        samples = wrapper(*dummies)
    expected = (
        (args.num_rois, 1, 14, 14),
        (args.num_rois, 1, 28, 28),
        (args.num_rois, 1, 56, 56),
        (args.num_rois, 1, 112, 112),
    )
    actual = tuple(tuple(value.shape) for value in samples)
    if actual != expected:
        raise RuntimeError(f"mask output shape drift: {actual}")
    print(f"[sample] outputs={actual}")
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummies,
            str(args.onnx),
            input_names=["instance_feats", "sem0", "sem1"],
            output_names=["stage0", "stage1", "stage2", "stage3"],
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
        )
    torch.cuda.synchronize()
    print(
        f"[export] {args.onnx} size={args.onnx.stat().st_size} "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
