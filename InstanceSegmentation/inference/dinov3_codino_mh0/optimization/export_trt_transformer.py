#!/usr/bin/env python3
"""Export fixed-batch MH0 query encoder and decoder ONNX graphs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DINOV3_USE_XFORMERS", "0")

import torch


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SHAPES = ((92, 160), (46, 80), (23, 40))
INPUT_HEIGHT = 736
INPUT_WIDTH = 1280
NUM_QUERY = 100


def prepare_model(config: Path, checkpoint: Path, device: str):
    if str(BUNDLE_ROOT) not in sys.path:
        sys.path.insert(0, str(BUNDLE_ROOT))
    from bootstrap import build_model
    return build_model(
        config=config,
        checkpoint=checkpoint,
        device=device,
    )


def install_msda_trt_symbolic() -> None:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

    def symbolic(
        graph,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        del im2col_step
        output = graph.op(
            "trt::MultiscaleDeformableAttnPlugin_TRT",
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        shape = graph.op(
            "Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.long)
        )
        return graph.op("Reshape", output, shape)

    MultiScaleDeformableAttnFunction.symbolic = staticmethod(symbolic)


def fixed_transformer_constants(model, device: str, batch_size: int):
    query_head = model.query_head
    transformer = query_head.transformer
    with torch.inference_mode():
        image_masks = torch.zeros(
            (batch_size, INPUT_HEIGHT, INPUT_WIDTH), device=device
        )
        level_masks = []
        level_positions = []
        for height, width in FEATURE_SHAPES:
            mask = (
                torch.nn.functional.interpolate(
                    image_masks[None], size=(height, width)
                )
                .to(torch.bool)
                .squeeze(0)
            )
            level_masks.append(mask)
            level_positions.append(query_head.positional_encoding(mask))
        spatial_shapes = torch.as_tensor(
            FEATURE_SHAPES, dtype=torch.long, device=device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios = torch.stack(
            [transformer.get_valid_ratio(mask) for mask in level_masks], 1
        )
        reference_points = transformer.get_reference_points(
            spatial_shapes, valid_ratios, device=device
        )
        positions = []
        for level, position in enumerate(level_positions):
            positions.append(
                position.flatten(2).transpose(1, 2)
                + transformer.level_embeds[level].view(1, 1, -1)
            )
        position_flatten = (
            torch.cat(positions, 1).permute(1, 0, 2).contiguous()
        )
        mask_flatten = torch.cat(
            [mask.flatten(1) for mask in level_masks], 1
        )
    return (
        mask_flatten,
        position_flatten,
        spatial_shapes,
        level_start_index,
        reference_points,
        valid_ratios,
    )


class QueryEncoderExport(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, constants) -> None:
        super().__init__()
        self.encoder = model.query_head.transformer.encoder
        (
            mask_flatten,
            position_flatten,
            spatial_shapes,
            level_start_index,
            reference_points,
            valid_ratios,
        ) = constants
        self.register_buffer("mask_flatten", mask_flatten)
        self.register_buffer("position_flatten", position_flatten)
        self.register_buffer("spatial_shapes", spatial_shapes)
        self.register_buffer("level_start_index", level_start_index)
        self.register_buffer("reference_points", reference_points)
        self.register_buffer("valid_ratios", valid_ratios)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.encoder(
            query=query.contiguous(),
            key=None,
            value=None,
            query_pos=self.position_flatten,
            query_key_padding_mask=self.mask_flatten,
            spatial_shapes=self.spatial_shapes,
            reference_points=self.reference_points,
            level_start_index=self.level_start_index,
            valid_ratios=self.valid_ratios,
        ).contiguous()


class DecoderExport(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, constants) -> None:
        super().__init__()
        transformer = model.query_head.transformer
        self.decoder = transformer.decoder
        self.reg_branches = model.query_head.reg_branches
        mask_flatten, _, spatial_shapes, level_start_index, _, valid_ratios = (
            constants
        )
        self.register_buffer("mask_flatten", mask_flatten)
        self.register_buffer("spatial_shapes", spatial_shapes)
        self.register_buffer("level_start_index", level_start_index)
        self.register_buffer("valid_ratios", valid_ratios)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
    ):
        return self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=self.mask_flatten,
            reference_points=reference_points,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            valid_ratios=self.valid_ratios,
            reg_branches=self.reg_branches,
        )


def export_one(
    wrapper: torch.nn.Module,
    dummies: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        samples = wrapper(*dummies)
    if isinstance(samples, torch.Tensor):
        samples = (samples,)
    print(
        f"[sample] {destination.name}: "
        f"{[(tuple(value.shape), str(value.dtype)) for value in samples]}"
    )
    started = time.perf_counter()
    with torch.inference_mode():
        try:
            torch.onnx.export(
                wrapper,
                dummies,
                str(destination),
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=False,
                custom_opsets={"trt": 1},
                dynamo=False,
            )
        except Exception as exc:
            if (
                exc.__class__.__name__ != "CheckerError"
                or not destination.is_file()
                or destination.stat().st_size == 0
            ):
                raise
            print(f"[warn] ONNX checker rejected custom TRT op: {exc}")
    torch.cuda.synchronize()
    print(
        f"[export] {destination} size={destination.stat().st_size} "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )


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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--query-onnx",
        type=Path,
        default=BUNDLE_ROOT
        / "artifacts"
        / "trt"
        / "mh0_query_encoder_b2_736x1280_fp16.onnx",
    )
    parser.add_argument(
        "--decoder-onnx",
        type=Path,
        default=BUNDLE_ROOT
        / "artifacts"
        / "trt"
        / "mh0_decoder_b2_736x1280_fp32.onnx",
    )
    parser.add_argument(
        "--component",
        choices=("query_encoder", "decoder", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_msda_trt_symbolic()
    model = prepare_model(args.config, args.checkpoint, args.device)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    constants = fixed_transformer_constants(model, args.device, args.batch_size)
    total_tokens = sum(height * width for height, width in FEATURE_SHAPES)
    if args.component in {"query_encoder", "all"}:
        query_wrapper = QueryEncoderExport(model, constants).to(args.device).half().eval()
        export_one(
            query_wrapper,
            (
                torch.randn(
                    total_tokens,
                    args.batch_size,
                    256,
                    device=args.device,
                    dtype=torch.float16,
                ),
            ),
            ["query"],
            ["memory"],
            args.query_onnx,
        )
    if args.component in {"decoder", "all"}:
        decoder_wrapper = DecoderExport(model, constants).to(args.device).eval()
        export_one(
            decoder_wrapper,
            (
                torch.randn(
                    NUM_QUERY,
                    args.batch_size,
                    256,
                    device=args.device,
                ),
                torch.randn(
                    total_tokens,
                    args.batch_size,
                    256,
                    device=args.device,
                ),
                torch.rand(
                    args.batch_size,
                    NUM_QUERY,
                    4,
                    device=args.device,
                ),
            ),
            ["query", "memory", "reference_points"],
            ["inter_states", "inter_references"],
            args.decoder_onnx,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
