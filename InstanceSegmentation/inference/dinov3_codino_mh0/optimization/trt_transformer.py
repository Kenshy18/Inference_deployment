"""Fixed-shape TensorRT adapters for the MH0 query encoder and decoder."""

from __future__ import annotations

from pathlib import Path

import torch

from .sm120_msda import register_sm120_msda_plugin
from .trt_backbone import prepare_tensorrt_libraries


FEATURE_SHAPES = ((92, 160), (46, 80), (23, 40))


def _torch_dtype(value, trt) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise TypeError(f"unsupported TensorRT dtype: {value}") from exc


class _FixedEngine:
    def __init__(
        self,
        engine_path: Path,
        *,
        expected_inputs: tuple[str, ...],
        expected_outputs: tuple[str, ...],
    ) -> None:
        prepare_tensorrt_libraries()
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")
        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"could not deserialize: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"could not create execution context: {self.engine_path}")
        names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        )
        self.input_names = tuple(
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names = tuple(
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        if self.input_names != expected_inputs or self.output_names != expected_outputs:
            raise RuntimeError(
                f"unexpected engine IO: inputs={self.input_names}, "
                f"outputs={self.output_names}"
            )
        self.input_shapes = {
            name: tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            for name in self.input_names
        }
        self.output_shapes = {
            name: tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            for name in self.output_names
        }
        self.input_dtypes = {
            name: _torch_dtype(self.engine.get_tensor_dtype(name), trt)
            for name in self.input_names
        }
        self.output_dtypes = {
            name: _torch_dtype(self.engine.get_tensor_dtype(name), trt)
            for name in self.output_names
        }
        self.outputs: dict[str, torch.Tensor] = {}
        self.retained_inputs: tuple[torch.Tensor, ...] = ()

    def execute(self, raw_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prepared = []
        for name in self.input_names:
            value = raw_inputs[name]
            if not value.is_cuda:
                raise RuntimeError(f"TensorRT input {name} must be CUDA")
            expected_shape = self.input_shapes[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"TensorRT input {name} requires {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )
            value = value.to(dtype=self.input_dtypes[name]).contiguous()
            prepared.append(value)
            self.context.set_tensor_address(name, int(value.data_ptr()))
        device = prepared[0].device
        for name in self.output_names:
            output = self.outputs.get(name)
            if (
                output is None
                or output.device != device
                or output.dtype != self.output_dtypes[name]
            ):
                output = torch.empty(
                    self.output_shapes[name],
                    device=device,
                    dtype=self.output_dtypes[name],
                )
                self.outputs[name] = output
            self.context.set_tensor_address(name, int(output.data_ptr()))
        self.retained_inputs = tuple(prepared)
        stream = torch.cuda.current_stream(device=device)
        for value in prepared:
            value.record_stream(stream)
        for output in self.outputs.values():
            output.record_stream(stream)
        if not self.context.execute_async_v3(
            stream_handle=int(stream.cuda_stream)
        ):
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
        return self.outputs


class TensorRTQueryEncoder(torch.nn.Module):
    def __init__(self, engine_path: Path, plugin_path: Path) -> None:
        super().__init__()
        prepare_tensorrt_libraries()
        register_sm120_msda_plugin(plugin_path)
        self.fixed = _FixedEngine(
            engine_path,
            expected_inputs=("query",),
            expected_outputs=("memory",),
        )
        expected_tokens = sum(height * width for height, width in FEATURE_SHAPES)
        query_shape = self.fixed.input_shapes["query"]
        if (
            len(query_shape) != 3
            or query_shape[0] != expected_tokens
            or query_shape[1] < 1
            or query_shape[2] != 256
        ):
            raise RuntimeError(
                f"query encoder shape drift: {query_shape}"
            )
        self.batch_size = query_shape[1]

    def forward(self, query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return self.fixed.execute({"query": query})["memory"]


class TensorRTDecoder(torch.nn.Module):
    def __init__(self, engine_path: Path) -> None:
        super().__init__()
        self.fixed = _FixedEngine(
            engine_path,
            expected_inputs=("query", "memory", "reference_points"),
            expected_outputs=("inter_states", "inter_references"),
        )
        self.num_layers = self.fixed.output_shapes["inter_states"][0]

    def forward(
        self,
        query: torch.Tensor,
        *args,
        reference_points: torch.Tensor | None = None,
        **kwargs,
    ):
        del args
        memory = kwargs.get("value")
        if memory is None or reference_points is None:
            raise RuntimeError(
                "TensorRT decoder requires value and reference_points"
            )
        outputs = self.fixed.execute(
            {
                "query": query,
                "memory": memory,
                "reference_points": reference_points,
            }
        )
        return outputs["inter_states"], outputs["inter_references"]


def install_trt_transformer(
    model: torch.nn.Module,
    *,
    query_engine: Path,
    decoder_engine: Path,
    plugin: Path,
) -> torch.nn.Module:
    transformer = model.query_head.transformer
    encoder = TensorRTQueryEncoder(query_engine, plugin).eval()
    transformer.encoder = encoder
    transformer.decoder = TensorRTDecoder(decoder_engine).eval()
    device = next(model.parameters()).device
    spatial_shapes = torch.tensor(
        FEATURE_SHAPES, dtype=torch.long, device=device
    )
    level_start_index = torch.cat(
        (
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        )
    )
    valid_ratios = torch.ones(
        (encoder.batch_size, len(FEATURE_SHAPES), 2),
        dtype=torch.float32,
        device=device,
    )
    # The fixed TensorRT encoder owns its reference points internally and
    # ignores this argument. Keep a stable CUDA tensor to avoid a per-batch
    # CPU-to-GPU construction and synchronization in the outer transformer.
    reference_points = torch.empty(
        (0,), dtype=torch.float32, device=device
    )
    transformer._mh0_fixed_geometry = (
        spatial_shapes,
        level_start_index,
        valid_ratios,
        reference_points,
    )
    token_count = sum(height * width for height, width in FEATURE_SHAPES)
    transformer._mh0_fixed_mask_flatten = torch.zeros(
        (encoder.batch_size, token_count),
        dtype=torch.bool,
        device=device,
    )

    proposals = []
    for level, (height, width) in enumerate(FEATURE_SHAPES):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0, height - 1, height, dtype=torch.float32, device=device),
            torch.linspace(
                0, width - 1, width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)
        scale = torch.tensor(
            (width, height), dtype=torch.float32, device=device
        )
        grid = (
            grid.unsqueeze(0).expand(encoder.batch_size, -1, -1, -1) + 0.5
        ) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**level)
        proposals.append(
            torch.cat((grid, wh), dim=-1).reshape(
                encoder.batch_size, -1, 4
            )
        )
    output_proposals = torch.cat(proposals, dim=1)
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(dim=-1, keepdim=True)
    output_proposals = torch.log(
        output_proposals / (1.0 - output_proposals)
    ).masked_fill(~output_proposals_valid, float("inf"))
    transformer._mh0_fixed_encoder_proposals = (
        output_proposals,
        output_proposals_valid,
    )
    model.query_head._mh0_fixed_query_inputs = True
    return model


__all__ = [
    "FEATURE_SHAPES",
    "TensorRTDecoder",
    "TensorRTQueryEncoder",
    "install_trt_transformer",
]
