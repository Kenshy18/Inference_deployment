#include "cuda_overlay.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <new>

namespace {

struct Context {
    cudaStream_t producer_stream{};
    cudaEvent_t producer_ready{};
    cudaEvent_t overlay_complete{};
    CudaOverlaySpan* device_fill{};
    std::size_t fill_capacity{};
    CudaOverlaySpan* device_outline{};
    std::size_t outline_capacity{};
    cudaStream_t stream{};
};

void write_error(
    char* output,
    std::size_t output_size,
    const char* operation,
    cudaError_t error
) {
    if (!output || output_size == 0) {
        return;
    }
    std::snprintf(
        output,
        output_size,
        "%s: %s",
        operation,
        cudaGetErrorString(error)
    );
}

cudaError_t ensure_capacity(
    CudaOverlaySpan** device,
    std::size_t* capacity,
    std::size_t required
) {
    if (required <= *capacity) {
        return cudaSuccess;
    }
    if (*device) {
        const cudaError_t free_result = cudaFree(*device);
        if (free_result != cudaSuccess) {
            return free_result;
        }
    }
    std::size_t next_capacity = std::max<std::size_t>(256, *capacity);
    while (next_capacity < required) {
        next_capacity *= 2;
    }
    const cudaError_t allocation_result = cudaMalloc(
        reinterpret_cast<void**>(device),
        next_capacity * sizeof(CudaOverlaySpan)
    );
    if (allocation_result == cudaSuccess) {
        *capacity = next_capacity;
    }
    return allocation_result;
}

__device__ __forceinline__ std::uint8_t blend_fixed(
    std::uint8_t background,
    std::uint8_t foreground,
    int alpha
) {
    return static_cast<std::uint8_t>(
        (
            static_cast<int>(foreground) * alpha +
            static_cast<int>(background) * (255 - alpha) +
            127
        ) /
        255
    );
}

__global__ void apply_spans_nv12(
    std::uint8_t* luma,
    std::uint8_t* chroma,
    int luma_pitch,
    int chroma_pitch,
    int width,
    int height,
    const CudaOverlaySpan* spans,
    std::size_t span_count
) {
    const std::size_t span_index = blockIdx.y;
    if (span_index >= span_count) {
        return;
    }
    const CudaOverlaySpan span = spans[span_index];
    const int x =
        span.first_x + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (
        x > span.last_x || x < 0 || x >= width ||
        span.y < 0 || span.y >= height
    ) {
        return;
    }
    auto& y_value = luma[span.y * luma_pitch + x];
    y_value = blend_fixed(y_value, span.luma, span.alpha);

    if (((x | span.y) & 1) == 0) {
        const int chroma_offset =
            (span.y / 2) * chroma_pitch + (x / 2) * 2;
        auto& u_value = chroma[chroma_offset];
        auto& v_value = chroma[chroma_offset + 1];
        u_value = blend_fixed(u_value, span.u, span.alpha);
        v_value = blend_fixed(v_value, span.v, span.alpha);
    }
}

cudaError_t launch_spans(
    std::uint8_t* luma,
    std::uint8_t* chroma,
    int luma_pitch,
    int chroma_pitch,
    int width,
    int height,
    const CudaOverlaySpan* spans,
    std::size_t count,
    int maximum_span_width,
    cudaStream_t stream
) {
    if (count == 0) {
        return cudaSuccess;
    }
    constexpr unsigned int threads = 256;
    const unsigned int x_blocks =
        static_cast<unsigned int>(
            (maximum_span_width + threads - 1) / threads
        );
    const dim3 grid(x_blocks, static_cast<unsigned int>(count), 1);
    apply_spans_nv12<<<grid, threads, 0, stream>>>(
        luma,
        chroma,
        luma_pitch,
        chroma_pitch,
        width,
        height,
        spans,
        count
    );
    return cudaGetLastError();
}

int maximum_span_width(
    const CudaOverlaySpan* spans,
    std::size_t count
) {
    int result = 0;
    for (std::size_t index = 0; index < count; ++index) {
        result = std::max(
            result,
            spans[index].last_x - spans[index].first_x + 1
        );
    }
    return result;
}

cudaError_t launch_batches(
    std::uint8_t* luma,
    std::uint8_t* chroma,
    int luma_pitch,
    int chroma_pitch,
    int width,
    int height,
    const CudaOverlaySpan* host_spans,
    const CudaOverlaySpan* device_spans,
    std::size_t span_count,
    const std::size_t* batch_ends,
    std::size_t batch_count,
    cudaStream_t stream
) {
    if (span_count == 0) {
        return cudaSuccess;
    }
    if (!batch_ends || batch_count == 0) {
        return cudaErrorInvalidValue;
    }
    std::size_t batch_start = 0;
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        const std::size_t batch_end = batch_ends[batch];
        if (batch_end <= batch_start || batch_end > span_count) {
            return cudaErrorInvalidValue;
        }
        const std::size_t count = batch_end - batch_start;
        const cudaError_t result = launch_spans(
            luma,
            chroma,
            luma_pitch,
            chroma_pitch,
            width,
            height,
            device_spans + batch_start,
            count,
            maximum_span_width(host_spans + batch_start, count),
            stream
        );
        if (result != cudaSuccess) {
            return result;
        }
        batch_start = batch_end;
    }
    return batch_start == span_count ? cudaSuccess : cudaErrorInvalidValue;
}

}  // namespace

extern "C" void* cuda_overlay_create(
    int device_index,
    void* producer_stream,
    char* error,
    std::size_t error_size
) {
    const cudaError_t device_result = cudaSetDevice(device_index);
    if (device_result != cudaSuccess) {
        write_error(error, error_size, "cudaSetDevice", device_result);
        return nullptr;
    }
    auto* context = new (std::nothrow) Context;
    if (!context && error && error_size) {
        std::snprintf(error, error_size, "CUDA context allocation failed");
        return nullptr;
    }
    context->producer_stream = static_cast<cudaStream_t>(producer_stream);
    const cudaError_t stream_result = cudaStreamCreateWithFlags(
        &context->stream,
        cudaStreamNonBlocking
    );
    if (stream_result != cudaSuccess) {
        write_error(
            error,
            error_size,
            "create overlay stream",
            stream_result
        );
        delete context;
        return nullptr;
    }
    const cudaError_t event_result = cudaEventCreateWithFlags(
        &context->producer_ready,
        cudaEventDisableTiming
    );
    if (event_result != cudaSuccess) {
        write_error(
            error,
            error_size,
            "create decoder-ready event",
            event_result
        );
        cudaStreamDestroy(context->stream);
        delete context;
        return nullptr;
    }
    const cudaError_t completion_event_result = cudaEventCreateWithFlags(
        &context->overlay_complete,
        cudaEventDisableTiming
    );
    if (completion_event_result != cudaSuccess) {
        write_error(
            error,
            error_size,
            "create overlay-complete event",
            completion_event_result
        );
        cudaEventDestroy(context->producer_ready);
        cudaStreamDestroy(context->stream);
        delete context;
        return nullptr;
    }
    return context;
}

extern "C" int cuda_overlay_apply_nv12(
    void* opaque,
    std::uint8_t* luma,
    std::uint8_t* chroma,
    int luma_pitch,
    int chroma_pitch,
    int width,
    int height,
    const CudaOverlaySpan* fill_spans,
    std::size_t fill_count,
    const std::size_t* fill_batch_ends,
    std::size_t fill_batch_count,
    const CudaOverlaySpan* outline_spans,
    std::size_t outline_count,
    const std::size_t* outline_batch_ends,
    std::size_t outline_batch_count,
    char* error,
    std::size_t error_size
) {
    auto* context = static_cast<Context*>(opaque);
    if (!context) {
        if (error && error_size) {
            std::snprintf(error, error_size, "CUDA overlay context is null");
        }
        return 1;
    }
    // A decoded CUDA AVFrame can be returned while NVDEC work on libav's
    // stream is still in flight.  Under six-worker load our non-blocking
    // overlay stream could run first, after which NVDEC overwrote the mask.
    // Establish the producer/consumer boundary before editing the surface.
    // Recording on libav's producer stream and waiting on our stream keeps the
    // dependency entirely on the GPU; synchronizing the producer on the CPU
    // here costs measurable throughput with six workers.
    cudaError_t result = cudaEventRecord(
        context->producer_ready,
        context->producer_stream
    );
    if (result != cudaSuccess) {
        write_error(error, error_size, "record decoded-frame readiness", result);
        return 1;
    }
    result = cudaStreamWaitEvent(
        context->stream,
        context->producer_ready,
        0
    );
    if (result != cudaSuccess) {
        write_error(error, error_size, "wait for decoded CUDA frame", result);
        return 1;
    }
    result = ensure_capacity(
        &context->device_fill,
        &context->fill_capacity,
        fill_count
    );
    if (result != cudaSuccess) {
        write_error(error, error_size, "allocate fill spans", result);
        return 1;
    }
    result = ensure_capacity(
        &context->device_outline,
        &context->outline_capacity,
        outline_count
    );
    if (result != cudaSuccess) {
        write_error(error, error_size, "allocate outline spans", result);
        return 1;
    }
    if (fill_count) {
        result = cudaMemcpyAsync(
            context->device_fill,
            fill_spans,
            fill_count * sizeof(CudaOverlaySpan),
            cudaMemcpyHostToDevice,
            context->stream
        );
        if (result != cudaSuccess) {
            write_error(error, error_size, "copy fill spans", result);
            return 1;
        }
        result = launch_batches(
            luma,
            chroma,
            luma_pitch,
            chroma_pitch,
            width,
            height,
            fill_spans,
            context->device_fill,
            fill_count,
            fill_batch_ends,
            fill_batch_count,
            context->stream
        );
        if (result != cudaSuccess) {
            write_error(error, error_size, "launch fill spans", result);
            return 1;
        }
    }
    if (outline_count) {
        result = cudaMemcpyAsync(
            context->device_outline,
            outline_spans,
            outline_count * sizeof(CudaOverlaySpan),
            cudaMemcpyHostToDevice,
            context->stream
        );
        if (result != cudaSuccess) {
            write_error(error, error_size, "copy outline spans", result);
            return 1;
        }
        result = launch_batches(
            luma,
            chroma,
            luma_pitch,
            chroma_pitch,
            width,
            height,
            outline_spans,
            context->device_outline,
            outline_count,
            outline_batch_ends,
            outline_batch_count,
            context->stream
        );
        if (result != cudaSuccess) {
            write_error(error, error_size, "launch outline spans", result);
            return 1;
        }
    }
    result = cudaEventRecord(context->overlay_complete, context->stream);
    if (result != cudaSuccess) {
        write_error(error, error_size, "record overlay completion", result);
        return 1;
    }
    // Return ownership to libav's stream without stalling the CPU.  Subsequent
    // NVDEC/NVENC work using the shared device context cannot consume the
    // surface until our overlay event has completed.
    result = cudaStreamWaitEvent(
        context->producer_stream,
        context->overlay_complete,
        0
    );
    if (result != cudaSuccess) {
        write_error(error, error_size, "publish completed overlay", result);
        return 1;
    }
    return 0;
}

extern "C" void cuda_overlay_destroy(void* opaque) {
    auto* context = static_cast<Context*>(opaque);
    if (!context) {
        return;
    }
    if (context->device_fill) {
        cudaFree(context->device_fill);
    }
    if (context->device_outline) {
        cudaFree(context->device_outline);
    }
    if (context->stream) {
        cudaStreamDestroy(context->stream);
    }
    if (context->producer_ready) {
        cudaEventDestroy(context->producer_ready);
    }
    if (context->overlay_complete) {
        cudaEventDestroy(context->overlay_complete);
    }
    delete context;
}
