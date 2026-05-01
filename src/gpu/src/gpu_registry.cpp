/* ============================================================================
 * UltrafastSecp256k1 -- GPU Backend Registry
 * ============================================================================
 * Compile-time registry of available GPU backends.
 * Each backend is conditionally compiled in via CMake defines:
 *   -DSECP256K1_HAVE_CUDA=1
 *   -DSECP256K1_HAVE_OPENCL=1
 *   -DSECP256K1_HAVE_METAL=1
 * ============================================================================ */

#include "gpu_backend.hpp"

/* Forward declarations for backend factories (defined in their own .cpp/.cu) */
#if defined(SECP256K1_HAVE_CUDA)
namespace secp256k1::gpu {
std::unique_ptr<GpuBackend> create_cuda_backend();
}
#endif

#if defined(SECP256K1_HAVE_OPENCL)
namespace secp256k1::gpu {
std::unique_ptr<GpuBackend> create_opencl_backend();
}
#endif

#if defined(SECP256K1_HAVE_METAL)
namespace secp256k1::gpu {
std::unique_ptr<GpuBackend> create_metal_backend();
}
#endif

namespace secp256k1 {
namespace gpu {

/* -- Backend IDs compiled in ----------------------------------------------- */

static constexpr uint32_t s_backend_ids[] = {
#if defined(SECP256K1_HAVE_CUDA)
    1, /* CUDA */
#endif
#if defined(SECP256K1_HAVE_OPENCL)
    2, /* OpenCL */
#endif
#if defined(SECP256K1_HAVE_METAL)
    3, /* Metal */
#endif
    0  /* sentinel (always present so array is never empty) */
};

static constexpr uint32_t s_num_backends =
    (sizeof(s_backend_ids) / sizeof(s_backend_ids[0])) - 1; /* exclude sentinel */

uint32_t backend_count() {
    return s_num_backends;
}

uint32_t backend_ids(uint32_t* ids, uint32_t max_ids) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < s_num_backends && n < max_ids; ++i) {
        ids[n++] = s_backend_ids[i];
    }
    return n;
}

std::unique_ptr<GpuBackend> create_backend(uint32_t backend_id) {
#if defined(SECP256K1_HAVE_CUDA)
    if (backend_id == 1) return create_cuda_backend();
#endif
#if defined(SECP256K1_HAVE_OPENCL)
    if (backend_id == 2) return create_opencl_backend();
#endif
#if defined(SECP256K1_HAVE_METAL)
    if (backend_id == 3) return create_metal_backend();
#endif
    (void)backend_id;
    return nullptr;
}

bool is_available(uint32_t backend_id) {
    auto b = create_backend(backend_id);
    return b && b->device_count() > 0;
}

} // namespace gpu
} // namespace secp256k1
