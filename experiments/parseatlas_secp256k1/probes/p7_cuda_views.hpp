#pragma once

#include <cstddef>
#include <cstdint>

// Experimental host ABI; no CUDA headers are needed by oracle callers.
struct P7Field { std::uint64_t limbs[4]; };
static_assert(sizeof(P7Field) == 32);

inline constexpr std::size_t P7_MAX_COUNT = 1048576;
inline constexpr unsigned P7_MAX_STEPS = 1048576;
inline constexpr unsigned P7_MAX_LAUNCHES = 1048576;

struct P7KernelInfo {
    int registers;
    int max_threads_per_block;
    std::size_t local_bytes;
    std::size_t shared_bytes;
    int binary_version;
    int ptx_version;
};
struct P7Info {
    char name[256];
    int device;
    int major;
    int minor;
    int multiprocessors;
    int driver_version;
    int runtime_version;
    int little_endian_checked;
    int hybrid_mul;
    int montgomery;
    P7KernelInfo field[4];
    P7KernelInfo raw[4];
};

// Routes: 0 current CUDA field_mul; 1 unchanged Comba pair order with shifts;
// 2 the same Comba arithmetic, memcpy input view; 3 existing all64 product/reducer.
// Field inputs must be canonical (<p). Raw inputs may be arbitrary 256-bit values.
// a==b and host output/input overlap are permitted: both inputs are copied before
// any host output write. Callers still provide valid, correctly typed storage.
// count/steps/launches must be nonzero and within the public caps; all pointers
// required by a call must be nonnull. Invalid arguments are rejected before CUDA.
// Bench excludes allocation/copies and one untimed warm launch from its event
// interval. Every timed launch resets x=a[i], repeats steps with fixed b[i], and
// writes all four limbs. Only the last identical launch is copied back.
// Return 0 on success, 1 on error. error() is thread-local and remains valid until
// the next API operation on that thread; success resets it to the empty string.
extern "C" {
int pa_p7_eval(unsigned route, const P7Field* a, const P7Field* b,
               P7Field* out, std::size_t count, unsigned steps);
int pa_p7_raw(unsigned route, const P7Field* a, const P7Field* b,
              std::uint64_t* out8, std::size_t count);
int pa_p7_bench(unsigned route, const P7Field* a, const P7Field* b,
                P7Field* out, std::size_t count, unsigned steps,
                unsigned launches, float* elapsed_ms);
int pa_p7_info(P7Info* out);
const char* pa_p7_error();
}
