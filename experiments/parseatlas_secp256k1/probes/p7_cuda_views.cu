#include "p7_cuda_views.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "secp256k1.cuh"

#if defined(SECP256K1_CUDA_LIMBS_32)
#error P7 requires the production four-by-64 field representation
#endif
#if !SECP256K1_CUDA_USE_HYBRID_MUL || SECP256K1_CUDA_USE_MONTGOMERY || !SECP256K1_USE_PTX
#error P7 route zero must use the standard-domain native PTX hybrid implementation
#endif

namespace {
using FE = secp256k1::cuda::FieldElement;
thread_local std::string last_error;
constexpr unsigned block_size = 128;

void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class F> int api(F operation) {
    last_error.clear();
    try { operation(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); }
    catch (...) { last_error = "unknown CUDA probe error"; }
    return 1;
}

// Only these two loaders differ. memcpy creates real uint32_t objects, without
// dereferencing a uint32_t pointer into uint64_t storage. Host and device byte
// order are checked outside timing. The pair sequence below is the production
// mul_256_comba32 sequence, including its three-word PTX accumulator.
template<bool CopyView>
__device__ __forceinline__ void comba(const FE& a, const FE& b, uint32_t t32[16]) {
    uint32_t a32[8], b32[8];
    if constexpr (CopyView) {
        memcpy(a32, a.limbs, sizeof a32);
        memcpy(b32, b.limbs, sizeof b32);
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a32[2*i] = (uint32_t)a.limbs[i];
            a32[2*i+1] = (uint32_t)(a.limbs[i] >> 32);
            b32[2*i] = (uint32_t)b.limbs[i];
            b32[2*i+1] = (uint32_t)(b.limbs[i] >> 32);
        }
    }
    uint32_t r0 = 0, r1 = 0, r2 = 0;
    #define P7_MUL32_ACC(ai, bj) { \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a32[ai]), "r"(b32[bj]) \
        ); \
    }
    P7_MUL32_ACC(0, 0);
    t32[0] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 1); P7_MUL32_ACC(1, 0);
    t32[1] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 2); P7_MUL32_ACC(1, 1); P7_MUL32_ACC(2, 0);
    t32[2] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 3); P7_MUL32_ACC(1, 2); P7_MUL32_ACC(2, 1); P7_MUL32_ACC(3, 0);
    t32[3] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 4); P7_MUL32_ACC(1, 3); P7_MUL32_ACC(2, 2); P7_MUL32_ACC(3, 1); P7_MUL32_ACC(4, 0);
    t32[4] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 5); P7_MUL32_ACC(1, 4); P7_MUL32_ACC(2, 3); P7_MUL32_ACC(3, 2); P7_MUL32_ACC(4, 1); P7_MUL32_ACC(5, 0);
    t32[5] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 6); P7_MUL32_ACC(1, 5); P7_MUL32_ACC(2, 4); P7_MUL32_ACC(3, 3); P7_MUL32_ACC(4, 2); P7_MUL32_ACC(5, 1); P7_MUL32_ACC(6, 0);
    t32[6] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(0, 7); P7_MUL32_ACC(1, 6); P7_MUL32_ACC(2, 5); P7_MUL32_ACC(3, 4); P7_MUL32_ACC(4, 3); P7_MUL32_ACC(5, 2); P7_MUL32_ACC(6, 1); P7_MUL32_ACC(7, 0);
    t32[7] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(1, 7); P7_MUL32_ACC(2, 6); P7_MUL32_ACC(3, 5); P7_MUL32_ACC(4, 4); P7_MUL32_ACC(5, 3); P7_MUL32_ACC(6, 2); P7_MUL32_ACC(7, 1);
    t32[8] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(2, 7); P7_MUL32_ACC(3, 6); P7_MUL32_ACC(4, 5); P7_MUL32_ACC(5, 4); P7_MUL32_ACC(6, 3); P7_MUL32_ACC(7, 2);
    t32[9] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(3, 7); P7_MUL32_ACC(4, 6); P7_MUL32_ACC(5, 5); P7_MUL32_ACC(6, 4); P7_MUL32_ACC(7, 3);
    t32[10] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(4, 7); P7_MUL32_ACC(5, 6); P7_MUL32_ACC(6, 5); P7_MUL32_ACC(7, 4);
    t32[11] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(5, 7); P7_MUL32_ACC(6, 6); P7_MUL32_ACC(7, 5);
    t32[12] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(6, 7); P7_MUL32_ACC(7, 6);
    t32[13] = r0; r0 = r1; r1 = r2; r2 = 0;
    P7_MUL32_ACC(7, 7);
    t32[14] = r0;
    t32[15] = r1;
    #undef P7_MUL32_ACC
}

template<unsigned Route>
__device__ __forceinline__ FE multiply(const FE& a, const FE& b) {
    FE result;
    if constexpr (Route == 0) {
        secp256k1::cuda::field_mul(&a, &b, &result);
    } else if constexpr (Route == 1 || Route == 2) {
        uint32_t product[16];
        comba<Route == 2>(a, b, product);
        secp256k1::cuda::reduce_512_to_256_32(product, &result);
    } else {
        uint64_t product[8];
        secp256k1::cuda::mul_256_512(&a, &b, product);
        secp256k1::cuda::reduce_512_to_256(product, &result);
    }
    return result;
}

__device__ __forceinline__ FE load_field(const P7Field& input) {
    FE result;
    #pragma unroll
    for (int limb = 0; limb < 4; ++limb) result.limbs[limb] = input.limbs[limb];
    return result;
}

template<unsigned Route>
__global__ void field_job(const P7Field* a, const P7Field* b, P7Field* out,
                          std::size_t count, unsigned steps) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    FE x = load_field(a[i]);
    const FE rhs = load_field(b[i]);
    // One serial field dependency per thread; independent inputs run on the GPU.
    for (unsigned step = 0; step < steps; ++step) x = multiply<Route>(x, rhs);
    #pragma unroll
    for (int limb = 0; limb < 4; ++limb) out[i].limbs[limb] = x.limbs[limb];
}

template<unsigned Route>
__global__ void raw_job(const P7Field* a, const P7Field* b, uint64_t* out,
                        std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const FE x = load_field(a[i]), y = load_field(b[i]);
    uint64_t product[8];
    if constexpr (Route == 3) {
        secp256k1::cuda::mul_256_512(&x, &y, product);
    } else {
        uint32_t words[16];
        if constexpr (Route == 0) secp256k1::cuda::mul_256_comba32(&x, &y, words);
        else comba<Route == 2>(x, y, words);
        #pragma unroll
        for (int limb = 0; limb < 8; ++limb)
            product[limb] = (uint64_t(words[2*limb+1]) << 32) | words[2*limb];
    }
    #pragma unroll
    for (int limb = 0; limb < 8; ++limb) out[8*i+limb] = product[limb];
}

__global__ void endian_probe(unsigned* out) {
    const uint64_t value = 0x0123456789abcdefULL;
    uint32_t words[2];
    memcpy(words, &value, sizeof words);
    *out = words[0] == 0x89abcdefU && words[1] == 0x01234567U;
}

struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "stream create"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value), "event create"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
};
template<class T> struct DeviceBuffer {
    T* value{};
    explicit DeviceBuffer(std::size_t n) {
        require(n <= std::numeric_limits<std::size_t>::max() / sizeof(T), "allocation size overflow");
        check(cudaMalloc(reinterpret_cast<void**>(&value), n*sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (value) cudaFree(value); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};

void check_endian(cudaStream_t stream) {
    const uint64_t value = 0x0123456789abcdefULL;
    uint32_t words[2];
    std::memcpy(words, &value, sizeof words);
    require(words[0] == 0x89abcdefU && words[1] == 0x01234567U, "host is not little endian");
    DeviceBuffer<unsigned> device(1);
    unsigned result = 0;
    endian_probe<<<1, 1, 0, stream>>>(device.value);
    check(cudaGetLastError(), "endian launch");
    check(cudaMemcpyAsync(&result, device.value, sizeof result, cudaMemcpyDeviceToHost, stream), "endian copy");
    check(cudaStreamSynchronize(stream), "endian synchronize");
    require(result == 1, "device is not little endian");
}
void arguments(unsigned route, const P7Field* a, const P7Field* b,
               const void* out, std::size_t count) {
    require(route < 4, "route must be in [0,3]");
    require(a && b && out, "required pointer is null");
    require(count && count <= P7_MAX_COUNT, "count must be in [1,P7_MAX_COUNT]");
}

void launch_field(unsigned route, const P7Field* a, const P7Field* b,
                  P7Field* out, std::size_t count, unsigned steps, cudaStream_t stream) {
    const unsigned blocks = unsigned((count + block_size - 1) / block_size);
    switch (route) {
    case 0: field_job<0><<<blocks, block_size, 0, stream>>>(a,b,out,count,steps); break;
    case 1: field_job<1><<<blocks, block_size, 0, stream>>>(a,b,out,count,steps); break;
    case 2: field_job<2><<<blocks, block_size, 0, stream>>>(a,b,out,count,steps); break;
    case 3: field_job<3><<<blocks, block_size, 0, stream>>>(a,b,out,count,steps); break;
    }
    check(cudaGetLastError(), "field launch");
}
void launch_raw(unsigned route, const P7Field* a, const P7Field* b,
                uint64_t* out, std::size_t count, cudaStream_t stream) {
    const unsigned blocks = unsigned((count + block_size - 1) / block_size);
    switch (route) {
    case 0: raw_job<0><<<blocks, block_size, 0, stream>>>(a,b,out,count); break;
    case 1: raw_job<1><<<blocks, block_size, 0, stream>>>(a,b,out,count); break;
    case 2: raw_job<2><<<blocks, block_size, 0, stream>>>(a,b,out,count); break;
    case 3: raw_job<3><<<blocks, block_size, 0, stream>>>(a,b,out,count); break;
    }
    check(cudaGetLastError(), "raw launch");
}

template<class Kernel> P7KernelInfo attributes(Kernel kernel) {
    cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr, kernel), "kernel attributes");
    return {attr.numRegs, attr.maxThreadsPerBlock, attr.localSizeBytes,
            attr.sharedSizeBytes, attr.binaryVersion, attr.ptxVersion};
}
}

extern "C" const char* pa_p7_error() { return last_error.c_str(); }

extern "C" int pa_p7_eval(unsigned route, const P7Field* a, const P7Field* b,
                           P7Field* out, std::size_t count, unsigned steps) {
    return api([&] {
        arguments(route,a,b,out,count);
        require(steps && steps <= P7_MAX_STEPS, "steps must be in [1,P7_MAX_STEPS]");
        Stream stream;
        check_endian(stream.value);
        DeviceBuffer<P7Field> da(count), db(count), dr(count);
        const auto bytes = count*sizeof(P7Field);
        check(cudaMemcpyAsync(da.value,a,bytes,cudaMemcpyHostToDevice,stream.value), "copy a");
        check(cudaMemcpyAsync(db.value,b,bytes,cudaMemcpyHostToDevice,stream.value), "copy b");
        launch_field(route,da.value,db.value,dr.value,count,steps,stream.value);
        check(cudaMemcpyAsync(out,dr.value,bytes,cudaMemcpyDeviceToHost,stream.value), "copy result");
        check(cudaStreamSynchronize(stream.value), "eval synchronize");
    });
}

extern "C" int pa_p7_raw(unsigned route, const P7Field* a, const P7Field* b,
                          std::uint64_t* out8, std::size_t count) {
    return api([&] {
        arguments(route,a,b,out8,count);
        Stream stream;
        check_endian(stream.value);
        DeviceBuffer<P7Field> da(count), db(count);
        DeviceBuffer<uint64_t> dr(count*8);
        const auto bytes = count*sizeof(P7Field);
        check(cudaMemcpyAsync(da.value,a,bytes,cudaMemcpyHostToDevice,stream.value), "copy a");
        check(cudaMemcpyAsync(db.value,b,bytes,cudaMemcpyHostToDevice,stream.value), "copy b");
        launch_raw(route,da.value,db.value,dr.value,count,stream.value);
        check(cudaMemcpyAsync(out8,dr.value,count*8*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream.value), "copy raw result");
        check(cudaStreamSynchronize(stream.value), "raw synchronize");
    });
}

extern "C" int pa_p7_bench(unsigned route, const P7Field* a, const P7Field* b,
                            P7Field* out, std::size_t count, unsigned steps,
                            unsigned launches, float* elapsed_ms) {
    return api([&] {
        arguments(route,a,b,out,count);
        require(steps && steps <= P7_MAX_STEPS, "steps must be in [1,P7_MAX_STEPS]");
        require(launches && launches <= P7_MAX_LAUNCHES, "launches must be in [1,P7_MAX_LAUNCHES]");
        require(elapsed_ms, "elapsed_ms is null");
        Stream stream;
        check_endian(stream.value);
        Event begin, end;
        DeviceBuffer<P7Field> da(count), db(count), dr(count);
        const auto bytes = count*sizeof(P7Field);
        check(cudaMemcpyAsync(da.value,a,bytes,cudaMemcpyHostToDevice,stream.value), "copy a");
        check(cudaMemcpyAsync(db.value,b,bytes,cudaMemcpyHostToDevice,stream.value), "copy b");
        // A complete untimed launch admits JIT, initializes caches and validates
        // the launch before the event interval. It is not a measured warmup row.
        launch_field(route,da.value,db.value,dr.value,count,steps,stream.value);
        check(cudaStreamSynchronize(stream.value), "untimed warm launch");
        check(cudaEventRecord(begin.value,stream.value), "begin event");
        for (unsigned launch = 0; launch < launches; ++launch)
            launch_field(route,da.value,db.value,dr.value,count,steps,stream.value);
        check(cudaEventRecord(end.value,stream.value), "end event");
        check(cudaEventSynchronize(end.value), "event synchronize");
        check(cudaEventElapsedTime(elapsed_ms,begin.value,end.value), "event elapsed");
        check(cudaMemcpyAsync(out,dr.value,bytes,cudaMemcpyDeviceToHost,stream.value), "copy bench result");
        check(cudaStreamSynchronize(stream.value), "bench result synchronize");
    });
}

extern "C" int pa_p7_info(P7Info* out) {
    return api([&] {
        require(out, "info pointer is null");
        P7Info info{};
        check(cudaGetDevice(&info.device), "get device");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop,info.device), "device properties");
        std::memcpy(info.name,prop.name,sizeof info.name);
        info.name[sizeof info.name-1] = 0;
        info.major = prop.major; info.minor = prop.minor;
        info.multiprocessors = prop.multiProcessorCount;
        check(cudaDriverGetVersion(&info.driver_version), "driver version");
        check(cudaRuntimeGetVersion(&info.runtime_version), "runtime version");
        Stream stream;
        check_endian(stream.value);
        info.little_endian_checked = 1;
        info.hybrid_mul = SECP256K1_CUDA_USE_HYBRID_MUL;
        info.montgomery = SECP256K1_CUDA_USE_MONTGOMERY;
        info.field[0] = attributes(field_job<0>); info.raw[0] = attributes(raw_job<0>);
        info.field[1] = attributes(field_job<1>); info.raw[1] = attributes(raw_job<1>);
        info.field[2] = attributes(field_job<2>); info.raw[2] = attributes(raw_job<2>);
        info.field[3] = attributes(field_job<3>); info.raw[3] = attributes(raw_job<3>);
        *out = info;
    });
}
