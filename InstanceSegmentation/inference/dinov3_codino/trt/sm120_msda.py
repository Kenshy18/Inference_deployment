"""Register the fixed-B2 SM120 MSDA TensorRT Python plugin."""

import importlib.util
from pathlib import Path
from typing import Tuple

import torch


PLUGIN_ID = "codino::MSDA_SM120"
EXTENSION_MODULE_NAME = "codino_msda_direct_t140"

_extension = None
_extension_path: Path | None = None
_registered = False


def _load_extension(path: Path):
    spec = importlib.util.spec_from_file_location(EXTENSION_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Co-DINO SM120 MSDA extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "forward_out"):
        raise RuntimeError(
            "Co-DINO SM120 MSDA extension does not export forward_out"
        )
    return module


def register_sm120_msda_plugin(extension_path: Path) -> None:
    """Load one verified extension and register its TensorRT plugin."""

    global _extension, _extension_path, _registered
    resolved = extension_path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(
            f"Co-DINO SM120 MSDA extension not found: {resolved}"
        )
    if _registered:
        if resolved != _extension_path:
            raise RuntimeError(
                "Co-DINO SM120 MSDA plugin is already registered from "
                f"{_extension_path}, refusing {resolved}"
            )
        return

    import tensorrt_bindings.plugin as trtp

    extension = _load_extension(resolved)

    @trtp.register(PLUGIN_ID)
    def msda_description(
        value: trtp.TensorDesc,
        spatial_shapes: trtp.TensorDesc,
        level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        del spatial_shapes, level_start_index, attention_weights
        return trtp.from_shape_expr(
            (
                sampling_locations.shape_expr[0],
                sampling_locations.shape_expr[1],
                value.shape_expr[2] * value.shape_expr[3],
            ),
            value.dtype,
        )

    @trtp.impl(PLUGIN_ID)
    def msda_implementation(
        value: trtp.Tensor,
        spatial_shapes: trtp.Tensor,
        level_start_index: trtp.Tensor,
        sampling_locations: trtp.Tensor,
        attention_weights: trtp.Tensor,
        outputs: Tuple[trtp.Tensor],
        stream: int,
    ) -> None:
        extension.forward_out(
            torch.as_tensor(value, device="cuda"),
            torch.as_tensor(spatial_shapes, device="cuda"),
            torch.as_tensor(level_start_index, device="cuda"),
            torch.as_tensor(sampling_locations, device="cuda"),
            torch.as_tensor(attention_weights, device="cuda"),
            torch.as_tensor(outputs[0], device="cuda"),
            int(stream),
        )

    @trtp.autotune(PLUGIN_ID)
    def msda_autotune(
        value: trtp.TensorDesc,
        spatial_shapes: trtp.TensorDesc,
        level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
    ) -> list[trtp.AutoTuneCombination]:
        del (
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            outputs,
        )
        return [
            trtp.AutoTuneCombination(
                "FP16,INT64,INT64,FP16,FP16,FP16",
                "LINEAR",
            )
        ]

    _extension = extension
    _extension_path = resolved
    _registered = True


__all__ = ["PLUGIN_ID", "register_sm120_msda_plugin"]
