#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int B = 2, Q = 78440, H = 8, C = 32, L = 5, P = 4, CP = C / 4;
constexpr int64_t N = int64_t(B) * Q * H * CP;

__device__ __forceinline__ __half bilinear_trt(
    const __half*& bottom, int height, int width, __half h, __half w,
    int head, int channel) {
  const int h_low = __half2int_rd(h), w_low = __half2int_rd(w);
  const int h_high = h_low + 1, w_high = w_low + 1;
  const __half zero = __int2half_rz(0), one = __int2half_rz(1);
  const __half lh = __hsub(h, __int2half_rd(h_low));
  const __half lw = __hsub(w, __int2half_rd(w_low));
  const __half hh = __hsub(one, lh), hw = __hsub(one, lw);
  const int w_stride = H * C, h_stride = width * w_stride;
  const int h_low_offset = h_low * h_stride;
  const int h_high_offset = h_low_offset + h_stride;
  const int w_low_offset = w_low * w_stride;
  const int w_high_offset = w_low_offset + w_stride;
  const int base = head * C + channel;
  __half v1 = zero, v2 = zero, v3 = zero, v4 = zero;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom[h_low_offset + w_low_offset + base];
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom[h_low_offset + w_high_offset + base];
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom[h_high_offset + w_low_offset + base];
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom[h_high_offset + w_high_offset + base];
  __half pair0 = __hmul(__hmul(hh, hw), v1);
  __half pair1 = __hmul(__hmul(hh, lw), v2);
  __half pair2 = __hmul(__hmul(lh, hw), v3);
  __half pair3 = __hmul(__hmul(lh, lw), v4);
  pair0 = __hadd(pair0, pair1);
  pair2 = __hadd(pair2, pair3);
  return __hadd(pair0, pair2);
}

__global__ void msda_trt_exact(
    const __half* value, const int64_t* shapes, const int64_t* starts,
    const __half* locations, const __half* weights, __half* output) {
  int64_t logical = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= N) return;
  int t = int(logical);
  const int lane = t % CP;
  t /= CP;
  const int head = t % H;
  t /= H;
  const int query = t % Q, batch = t / Q;
  const int c0 = lane * 4, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
  int weight_index = ((batch * Q + query) * H + head) * L * P;
  int location_index = weight_index * 2;
  const int stride = H * C, base_batch = batch * Q * stride;
  const __half zero = __int2half_rz(0), half = __float2half(0.5f);
  const __half minus_one = __float2half(-1.0f);
  __half col0 = zero, col1 = zero, col2 = zero, col3 = zero;
  for (int level = 0; level < L; ++level) {
    const int height = int(shapes[level * 2]);
    const int width = int(shapes[level * 2 + 1]);
    const __half height_half = __int2half_rd(height);
    const __half width_half = __int2half_rd(width);
    const __half* bottom = value + base_batch + int(starts[level]) * stride;
    for (int point = 0; point < P; ++point) {
      const __half loc_w = locations[location_index];
      const __half loc_h = locations[location_index + 1];
      const __half weight = weights[weight_index];
      const __half h_im = __hsub(__hmul(loc_h, height_half), half);
      const __half w_im = __hsub(__hmul(loc_w, width_half), half);
      if (__hgt(h_im, minus_one) && __hgt(w_im, minus_one)
          && __hlt(h_im, height_half) && __hlt(w_im, width_half)) {
        col0 = __hadd(
            col0, __hmul(
                bilinear_trt(bottom, height, width, h_im, w_im, head, c0),
                weight));
        col1 = __hadd(
            col1, __hmul(
                bilinear_trt(bottom, height, width, h_im, w_im, head, c1),
                weight));
        col2 = __hadd(
            col2, __hmul(
                bilinear_trt(bottom, height, width, h_im, w_im, head, c2),
                weight));
        col3 = __hadd(
            col3, __hmul(
                bilinear_trt(bottom, height, width, h_im, w_im, head, c3),
                weight));
      }
      ++weight_index;
      location_index += 2;
    }
  }
  const int64_t index =
      ((int64_t(batch) * Q + query) * H + head) * C + c0;
  output[index] = col0;
  output[index + 1] = col1;
  output[index + 2] = col2;
  output[index + 3] = col3;
}

void msda_cuda_out(
    torch::Tensor value, torch::Tensor shapes, torch::Tensor starts,
    torch::Tensor locations, torch::Tensor weights, torch::Tensor output,
    uint64_t stream_handle) {
  TORCH_CHECK(
      value.is_cuda() && shapes.is_cuda() && starts.is_cuda()
      && locations.is_cuda() && weights.is_cuda() && output.is_cuda(),
      "CUDA tensors required");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf
      && locations.scalar_type() == at::kHalf
      && weights.scalar_type() == at::kHalf
      && output.scalar_type() == at::kHalf,
      "FP16 floating tensors required");
  TORCH_CHECK(
      shapes.scalar_type() == at::kLong && starts.scalar_type() == at::kLong,
      "INT64 geometry tensors required");
  TORCH_CHECK(
      value.sizes() == at::IntArrayRef({B, Q, H, C})
      && shapes.sizes() == at::IntArrayRef({L, 2})
      && starts.sizes() == at::IntArrayRef({L})
      && locations.sizes() == at::IntArrayRef({B, Q, H, L, P, 2})
      && weights.sizes() == at::IntArrayRef({B, Q, H, L, P})
      && output.sizes() == at::IntArrayRef({B, Q, H, C}),
      "fixed shape drift");
  TORCH_CHECK(
      value.is_contiguous() && shapes.is_contiguous()
      && starts.is_contiguous() && locations.is_contiguous()
      && weights.is_contiguous() && output.is_contiguous(),
      "contiguous tensors required");
  c10::cuda::CUDAGuard guard(value.device());
  const int threads = 256, blocks = int((N + threads - 1) / threads);
  msda_trt_exact<<<
      blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream_handle)>>>(
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
      shapes.data_ptr<int64_t>(), starts.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(locations.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
