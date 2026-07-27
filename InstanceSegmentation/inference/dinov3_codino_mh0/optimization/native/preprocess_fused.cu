#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

constexpr int OUTPUT_HEIGHT = 736;
constexpr int OUTPUT_WIDTH = 1280;

__device__ __forceinline__ float sample_channel(
    const unsigned char* input, int height, int width, float source_y,
    float source_x, int channel) {
  int y0 = int(floorf(source_y));
  int x0 = int(floorf(source_x));
  int y1 = y0 + 1;
  int x1 = x0 + 1;
  const float wy = source_y - float(y0);
  const float wx = source_x - float(x0);
  y0 = max(y0, 0);
  x0 = max(x0, 0);
  y1 = min(y1, height - 1);
  x1 = min(x1, width - 1);
  const int row0 = y0 * width * 3;
  const int row1 = y1 * width * 3;
  const float v00 = float(input[row0 + x0 * 3 + channel]);
  const float v01 = float(input[row0 + x1 * 3 + channel]);
  const float v10 = float(input[row1 + x0 * 3 + channel]);
  const float v11 = float(input[row1 + x1 * 3 + channel]);
  const float top = fmaf(wx, v01 - v00, v00);
  const float bottom = fmaf(wx, v11 - v10, v10);
  return fmaf(wy, bottom - top, top);
}

__global__ void preprocess_fused(
    const unsigned char* input, float* output, int batch, int input_height,
    int input_width, int resized_height, int resized_width, int pad_top,
    int pad_left) {
  const int64_t logical =
      int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pixels =
      int64_t(batch) * OUTPUT_HEIGHT * OUTPUT_WIDTH;
  if (logical >= pixels) return;
  int64_t value = logical;
  const int x = int(value % OUTPUT_WIDTH);
  value /= OUTPUT_WIDTH;
  const int y = int(value % OUTPUT_HEIGHT);
  const int image = int(value / OUTPUT_HEIGHT);
  const int resized_y = y - pad_top;
  const int resized_x = x - pad_left;
  float red = 128.0f;
  float green = 128.0f;
  float blue = 128.0f;
  if (resized_y >= 0 && resized_y < resized_height &&
      resized_x >= 0 && resized_x < resized_width) {
    const float source_y =
        (float(resized_y) + 0.5f) * float(input_height) /
            float(resized_height) -
        0.5f;
    const float source_x =
        (float(resized_x) + 0.5f) * float(input_width) /
            float(resized_width) -
        0.5f;
    const unsigned char* image_input =
        input + int64_t(image) * input_height * input_width * 3;
    blue = sample_channel(
        image_input, input_height, input_width, source_y, source_x, 0);
    green = sample_channel(
        image_input, input_height, input_width, source_y, source_x, 1);
    red = sample_channel(
        image_input, input_height, input_width, source_y, source_x, 2);
  }
  const int64_t plane = int64_t(OUTPUT_HEIGHT) * OUTPUT_WIDTH;
  float* image_output = output + int64_t(image) * 3 * plane;
  const int64_t offset = int64_t(y) * OUTPUT_WIDTH + x;
  image_output[offset] = (red - 123.675f) / 58.395f;
  image_output[plane + offset] = (green - 116.28f) / 57.12f;
  image_output[2 * plane + offset] = (blue - 103.53f) / 57.375f;
}

void preprocess_cuda_out(
    torch::Tensor input, torch::Tensor output, int resized_height,
    int resized_width, int pad_top, int pad_left, uint64_t stream_handle) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(
      input.device() == output.device(),
      "input and output must use the same CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kByte, "input must be uint8");
  TORCH_CHECK(output.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(
      input.dim() == 4 && input.size(3) == 3, "input must be NHWC BGR");
  TORCH_CHECK(
      output.sizes() ==
          at::IntArrayRef({input.size(0), 3, OUTPUT_HEIGHT, OUTPUT_WIDTH}),
      "output shape drift");
  TORCH_CHECK(
      input.is_contiguous() && output.is_contiguous(),
      "contiguous tensors required");
  TORCH_CHECK(
      resized_height > 0 && resized_width > 0 &&
          resized_height <= OUTPUT_HEIGHT && resized_width <= OUTPUT_WIDTH,
      "invalid resized shape");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t pixels =
      input.size(0) * int64_t(OUTPUT_HEIGHT) * OUTPUT_WIDTH;
  const int threads = 256;
  const int blocks = int((pixels + threads - 1) / threads);
  preprocess_fused<<<
      blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream_handle)>>>(
      input.data_ptr<unsigned char>(), output.data_ptr<float>(),
      int(input.size(0)), int(input.size(1)), int(input.size(2)),
      resized_height, resized_width, pad_top, pad_left);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
