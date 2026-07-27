#include <torch/extension.h>

void msda_cuda_out(
    torch::Tensor value, torch::Tensor shapes, torch::Tensor starts,
    torch::Tensor locations, torch::Tensor weights, torch::Tensor output,
    uint64_t stream_handle);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward_out", &msda_cuda_out,
      "MH0 TensorRT-exact fixed-batch SM120 MSDA direct output");
}
