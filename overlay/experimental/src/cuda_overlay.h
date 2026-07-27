#pragma once

#include <cstddef>
#include <cstdint>

struct CudaOverlaySpan {
    int y;
    int first_x;
    int last_x;
    std::uint8_t luma;
    std::uint8_t u;
    std::uint8_t v;
    std::uint8_t alpha;
};

extern "C" {

void* cuda_overlay_create(
    int device_index,
    char* error,
    std::size_t error_size
);

int cuda_overlay_apply_nv12(
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
);

void cuda_overlay_destroy(void* opaque);

}
