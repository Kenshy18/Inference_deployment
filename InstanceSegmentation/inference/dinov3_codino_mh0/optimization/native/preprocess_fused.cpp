#include <torch/extension.h>

void preprocess_cuda_out(
    torch::Tensor input, torch::Tensor output, int resized_height,
    int resized_width, int pad_top, int pad_left, uint64_t stream_handle);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward_out", &preprocess_cuda_out,
      "Fused BGR letterbox and ImageNet normalization output");
}
