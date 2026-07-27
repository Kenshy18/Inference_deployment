"""Register the fixed-B2 SM120 MSDA TensorRT Python plugin."""

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import torch


PLUGIN_ID = "codino::MSDA_SM120"
EXTENSION_MODULE_NAME = "codino_msda_direct_t140"
_extension = None
_extension_path: Path | None = None
_registered = False


def register_sm120_msda_plugin(extension_path: Path) -> None:
    global _extension, _extension_path, _registered
    resolved = extension_path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"SM120 MSDA extension not found: {resolved}")
    if _registered:
        if resolved != _extension_path:
            raise RuntimeError(
                f"SM120 plugin already registered from {_extension_path}"
            )
        return

    import tensorrt_bindings.plugin as trtp

    spec = importlib.util.spec_from_file_location(EXTENSION_MODULE_NAME, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SM120 MSDA extension: {resolved}")
    extension = importlib.util.module_from_spec(spec)
    sys.modules[EXTENSION_MODULE_NAME] = extension
    spec.loader.exec_module(extension)
    if not hasattr(extension, "forward_out"):
        raise RuntimeError("SM120 extension does not export forward_out")

    @trtp.register(PLUGIN_ID)
    def description(
        value: trtp.TensorDesc,
        spatial_shapes: trtp.TensorDesc,
        level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
    ) -> Tuple[trtp.TensorDesc]:
        del spatial_shapes, level_start_index, sampling_locations, attention_weights
        return (value.like(),)

    @trtp.impl(PLUGIN_ID)
    def implementation(
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
    def autotune(
        value: trtp.TensorDesc,
        spatial_shapes: trtp.TensorDesc,
        level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
    ) -> list[trtp.AutoTuneCombination]:
        del value, spatial_shapes, level_start_index
        del sampling_locations, attention_weights, outputs
        return [
            trtp.AutoTuneCombination(
                "FP16,INT64,INT64,FP16,FP16,FP16", "LINEAR"
            )
        ]

    _extension = extension
    _extension_path = resolved
    _registered = True


__all__ = ["PLUGIN_ID", "register_sm120_msda_plugin"]
