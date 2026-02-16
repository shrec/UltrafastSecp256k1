// =============================================================================
// UltrafastSecp256k1 OpenCL - Context Implementation
// =============================================================================
// Cross-platform OpenCL initialization and management
// Supports Windows (MSVC, Clang) and Linux (GCC, Clang)
// =============================================================================

#include "secp256k1_opencl.hpp"

// Platform detection
#ifdef _WIN32
    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

// OpenCL headers
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace secp256k1 {
namespace opencl {

// =============================================================================
// OpenCL Error Strings
// =============================================================================

const char* cl_error_string(int error_code) {
    switch (error_code) {
        case CL_SUCCESS:                         return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:    return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:           return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                     return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:             return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:        return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:              return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                 return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                  return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:           return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:       return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:          return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:           return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:         return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                   return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:               return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:               return "CL_INVALID_MIP_LEVEL";
        default:                                 return "UNKNOWN_ERROR";
    }
}

// =============================================================================
// Buffer Implementation
// =============================================================================

struct Buffer::Impl {
    cl_mem buffer = nullptr;
    std::size_t size = 0;

    ~Impl() {
        if (buffer) {
            clReleaseMemObject(buffer);
        }
    }
};

Buffer::Buffer() : impl_(std::make_unique<Impl>()) {}

Buffer::~Buffer() = default;

Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

std::size_t Buffer::size() const noexcept {
    return impl_ ? impl_->size : 0;
}

bool Buffer::is_valid() const noexcept {
    return impl_ && impl_->buffer != nullptr;
}

void* Buffer::native_handle() const noexcept {
    return impl_ ? impl_->buffer : nullptr;
}

// =============================================================================
// Context Implementation
// =============================================================================

struct Context::Impl {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    // Kernels
    cl_kernel kernel_field_add = nullptr;
    cl_kernel kernel_field_sub = nullptr;
    cl_kernel kernel_field_mul = nullptr;
    cl_kernel kernel_field_sqr = nullptr;
    cl_kernel kernel_field_inv = nullptr;
    cl_kernel kernel_point_double = nullptr;
    cl_kernel kernel_point_add = nullptr;
    cl_kernel kernel_scalar_mul = nullptr;
    cl_kernel kernel_scalar_mul_generator = nullptr;
    cl_kernel kernel_batch_inversion = nullptr;
    cl_kernel kernel_batch_jacobian_to_affine = nullptr;

    // Affine point addition kernels
    cl_kernel kernel_affine_add = nullptr;
    cl_kernel kernel_affine_add_lambda = nullptr;
    cl_kernel kernel_affine_add_x_only = nullptr;
    cl_kernel kernel_jacobian_to_affine = nullptr;

    DeviceInfo device_info;
    std::string last_error;
    DeviceConfig config;

    // Scratch buffers for temporary storage
    std::unique_ptr<Buffer> scratch_a;
    std::unique_ptr<Buffer> scratch_b;

    ~Impl() {
        // Release kernels
        if (kernel_field_add) clReleaseKernel(kernel_field_add);
        if (kernel_field_sub) clReleaseKernel(kernel_field_sub);
        if (kernel_field_mul) clReleaseKernel(kernel_field_mul);
        if (kernel_field_sqr) clReleaseKernel(kernel_field_sqr);
        if (kernel_field_inv) clReleaseKernel(kernel_field_inv);
        if (kernel_point_double) clReleaseKernel(kernel_point_double);
        if (kernel_point_add) clReleaseKernel(kernel_point_add);
        if (kernel_scalar_mul) clReleaseKernel(kernel_scalar_mul);
        if (kernel_scalar_mul_generator) clReleaseKernel(kernel_scalar_mul_generator);
        if (kernel_batch_inversion) clReleaseKernel(kernel_batch_inversion);
        if (kernel_batch_jacobian_to_affine) clReleaseKernel(kernel_batch_jacobian_to_affine);
        if (kernel_affine_add) clReleaseKernel(kernel_affine_add);
        if (kernel_affine_add_lambda) clReleaseKernel(kernel_affine_add_lambda);
        if (kernel_affine_add_x_only) clReleaseKernel(kernel_affine_add_x_only);
        if (kernel_jacobian_to_affine) clReleaseKernel(kernel_jacobian_to_affine);

        // Release program
        if (program) clReleaseProgram(program);

        // Release queue and context
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    bool init(const DeviceConfig& cfg);
    bool build_program();
    bool create_kernels();

    cl_kernel get_kernel(const char* name);
};

bool Context::Impl::init(const DeviceConfig& cfg) {
    config = cfg;
    cl_int err;

    // Get platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        last_error = "No OpenCL platforms found";
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // Find suitable platform/device
    cl_platform_id selected_platform = nullptr;
    cl_device_id selected_device = nullptr;
    int best_score = 0;
    bool found_exact = false;

    for (cl_uint i = 0; i < num_platforms; i++) {
        // Get platform name
        char platform_name[256] = {0};
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);

        bool is_intel = (std::string(platform_name).find("Intel") != std::string::npos);

        // Get devices
        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        for (cl_uint j = 0; j < num_devices; j++) {
            // Check if this is the explicitly requested device
            if (cfg.platform_id >= 0 &&
                static_cast<int>(i) == cfg.platform_id &&
                static_cast<int>(j) == cfg.device_id) {
                selected_platform = platforms[i];
                selected_device = devices[j];
                found_exact = true;
                break;
            }

            // Auto-selection: Score Intel devices higher if preferred
            if (cfg.platform_id < 0) {
                int score = 1;
                if (is_intel && cfg.prefer_intel) {
                    score = 10;
                }

                if (score > best_score) {
                    best_score = score;
                    selected_platform = platforms[i];
                    selected_device = devices[j];
                }
            }
        }

        if (found_exact) break;
    }

    if (!selected_device) {
        last_error = "No suitable GPU device found";
        return false;
    }

    platform = selected_platform;
    device = selected_device;

    // Get device info
    char buffer[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    device_info.name = buffer;

    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr);
    device_info.vendor = buffer;

    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, nullptr);
    device_info.version = buffer;

    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(buffer), buffer, nullptr);
    device_info.driver_version = buffer;

    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.global_mem_size),
                    &device_info.global_mem_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_info.local_mem_size),
                    &device_info.local_mem_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_info.max_work_group_size),
                    &device_info.max_work_group_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_info.compute_units),
                    &device_info.compute_units, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_info.max_clock_freq),
                    &device_info.max_clock_freq, nullptr);

    // Check for double precision support
    cl_device_fp_config fp_config;
    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr);
    device_info.supports_double = (fp_config != 0);

    // Detect vendor
    device_info.is_intel = (device_info.vendor.find("Intel") != std::string::npos);
    device_info.is_amd = (device_info.vendor.find("AMD") != std::string::npos ||
                          device_info.vendor.find("Advanced Micro") != std::string::npos);
    device_info.is_nvidia = (device_info.vendor.find("NVIDIA") != std::string::npos);

    if (cfg.verbose) {
        std::cout << "OpenCL Device: " << device_info.name << std::endl;
        std::cout << "  Vendor: " << device_info.vendor << std::endl;
        std::cout << "  Version: " << device_info.version << std::endl;
        std::cout << "  Driver: " << device_info.driver_version << std::endl;
        std::cout << "  Global Memory: " << (device_info.global_mem_size / (1024*1024)) << " MB" << std::endl;
        std::cout << "  Local Memory: " << (device_info.local_mem_size / 1024) << " KB" << std::endl;
        std::cout << "  Compute Units: " << device_info.compute_units << std::endl;
        std::cout << "  Max Clock: " << device_info.max_clock_freq << " MHz" << std::endl;
    }

    // Create context
    std::cerr << "[DEBUG] Creating OpenCL context for device: " << device_info.name << std::endl;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        last_error = std::string("Failed to create OpenCL context: ") + cl_error_string(err);
        std::cerr << "[DEBUG] " << last_error << std::endl;
        return false;
    }
    std::cerr << "[DEBUG] Context created successfully" << std::endl;

    // Create command queue with profiling enabled
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif

    if (err != CL_SUCCESS) {
        last_error = std::string("Failed to create command queue: ") + cl_error_string(err);
        std::cerr << "[DEBUG] " << last_error << std::endl;
        return false;
    }
    std::cerr << "[DEBUG] Command queue created successfully" << std::endl;

    // Build program
    if (!build_program()) {
        std::cerr << "[DEBUG] build_program failed: " << last_error << std::endl;
        return false;
    }
    std::cerr << "[DEBUG] Program built successfully" << std::endl;

    // Create kernels
    if (!create_kernels()) {
        std::cerr << "[DEBUG] create_kernels failed: " << last_error << std::endl;
        return false;
    }
    std::cerr << "[DEBUG] Kernels created successfully" << std::endl;

    return true;
}

// Embedded kernel source (will be generated by CMake)
// For now, include a minimal version
static const char* kernel_source = R"KERNEL(
// =============================================================================
// Secp256k1 OpenCL Kernels - Embedded Version
// =============================================================================

#define SECP256K1_P0 0xFFFFFFFEFFFFFC2FUL
#define SECP256K1_P1 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P2 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P3 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_K 0x1000003D1UL

typedef struct { ulong limbs[4]; } FieldElement;
typedef struct { ulong limbs[4]; } Scalar;
typedef struct { FieldElement x; FieldElement y; } AffinePoint;
typedef struct { FieldElement x; FieldElement y; FieldElement z; uint infinity; uint pad[7]; } JacobianPoint;

// =============================================================================
// NVIDIA PTX Optimized Helpers (hardware carry-chain multiply + reduce)
// =============================================================================
#ifdef __NV_CL_C_VERSION

#define PTX_MAD_ACC(c0, c1, c2, a, b) \
    asm volatile( \
        "mad.lo.cc.u64 %0, %3, %4, %0;\n\t" \
        "madc.hi.cc.u64 %1, %3, %4, %1;\n\t" \
        "addc.u64 %2, %2, 0;\n\t" \
        : "+l"(c0), "+l"(c1), "+l"(c2) \
        : "l"((ulong)(a)), "l"((ulong)(b)) \
    )

inline void mul_256_512_cl(const ulong* a, const ulong* b, ulong* r) {
    ulong a0=a[0],a1=a[1],a2=a[2],a3=a[3];
    ulong b0=b[0],b1=b[1],b2=b[2],b3=b[3];
    ulong c0=0,c1=0,c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,b0);
    r[0]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,b1); PTX_MAD_ACC(c0,c1,c2,a1,b0);
    r[1]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,b2); PTX_MAD_ACC(c0,c1,c2,a1,b1); PTX_MAD_ACC(c0,c1,c2,a2,b0);
    r[2]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,b3); PTX_MAD_ACC(c0,c1,c2,a1,b2); PTX_MAD_ACC(c0,c1,c2,a2,b1); PTX_MAD_ACC(c0,c1,c2,a3,b0);
    r[3]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a1,b3); PTX_MAD_ACC(c0,c1,c2,a2,b2); PTX_MAD_ACC(c0,c1,c2,a3,b1);
    r[4]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a2,b3); PTX_MAD_ACC(c0,c1,c2,a3,b2);
    r[5]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a3,b3);
    r[6]=c0; r[7]=c1;
}

inline void sqr_256_512_cl(const ulong* a, ulong* r) {
    ulong a0=a[0],a1=a[1],a2=a[2],a3=a[3];
    ulong c0=0,c1=0,c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,a0);
    r[0]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,a1); PTX_MAD_ACC(c0,c1,c2,a0,a1);
    r[1]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,a2); PTX_MAD_ACC(c0,c1,c2,a0,a2); PTX_MAD_ACC(c0,c1,c2,a1,a1);
    r[2]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a0,a3); PTX_MAD_ACC(c0,c1,c2,a0,a3); PTX_MAD_ACC(c0,c1,c2,a1,a2); PTX_MAD_ACC(c0,c1,c2,a1,a2);
    r[3]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a1,a3); PTX_MAD_ACC(c0,c1,c2,a1,a3); PTX_MAD_ACC(c0,c1,c2,a2,a2);
    r[4]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a2,a3); PTX_MAD_ACC(c0,c1,c2,a2,a3);
    r[5]=c0; c0=c1; c1=c2; c2=0;
    PTX_MAD_ACC(c0,c1,c2,a3,a3);
    r[6]=c0; r[7]=c1;
}

inline void reduce_512_to_256_cl(ulong* t, FieldElement* r) {
    ulong t0=t[0],t1=t[1],t2=t[2],t3=t[3],t4=t[4],t5=t[5],t6=t[6],t7=t[7];
    ulong a0,a1,a2,a3,a4;
    asm volatile("mul.lo.u64 %0, %5, 977;\n\t" "mul.hi.u64 %1, %5, 977;\n\t"
        "mad.lo.cc.u64 %1, %6, 977, %1;\n\t" "madc.hi.u64 %2, %6, 977, 0;\n\t"
        "mad.lo.cc.u64 %2, %7, 977, %2;\n\t" "madc.hi.u64 %3, %7, 977, 0;\n\t"
        "mad.lo.cc.u64 %3, %8, 977, %3;\n\t" "madc.hi.u64 %4, %8, 977, 0;\n\t"
        :"=l"(a0),"=l"(a1),"=l"(a2),"=l"(a3),"=l"(a4):"l"(t4),"l"(t5),"l"(t6),"l"(t7));
    ulong carry_a;
    asm volatile("add.cc.u64 %0,%0,%5;\n\t" "addc.cc.u64 %1,%1,%6;\n\t"
        "addc.cc.u64 %2,%2,%7;\n\t" "addc.cc.u64 %3,%3,%8;\n\t" "addc.u64 %4,0,0;\n\t"
        :"+l"(t0),"+l"(t1),"+l"(t2),"+l"(t3),"=l"(carry_a):"l"(a0),"l"(a1),"l"(a2),"l"(a3));
    ulong s0=(t4<<32),s1=(t4>>32)|(t5<<32),s2=(t5>>32)|(t6<<32),s3=(t6>>32)|(t7<<32);
    ulong carry_s;
    asm volatile("add.cc.u64 %0,%0,%5;\n\t" "addc.cc.u64 %1,%1,%6;\n\t"
        "addc.cc.u64 %2,%2,%7;\n\t" "addc.cc.u64 %3,%3,%8;\n\t" "addc.u64 %4,0,0;\n\t"
        :"+l"(t0),"+l"(t1),"+l"(t2),"+l"(t3),"=l"(carry_s):"l"(s0),"l"(s1),"l"(s2),"l"(s3));
    ulong extra = a4 + (t7>>32) + carry_a + carry_s;
    ulong e_lo=extra*977, e_sh=extra<<32, e_sh_hi=extra>>32;
    ulong c=0, ct=0;
    asm volatile("add.cc.u64 %0,%0,%5;\n\t" "addc.cc.u64 %1,%1,0;\n\t"
        "addc.cc.u64 %2,%2,0;\n\t" "addc.cc.u64 %3,%3,0;\n\t" "addc.u64 %4,0,0;\n\t"
        :"+l"(t0),"+l"(t1),"+l"(t2),"+l"(t3),"=l"(ct):"l"(e_lo)); c+=ct;
    asm volatile("add.cc.u64 %0,%0,%5;\n\t" "addc.cc.u64 %1,%1,%6;\n\t"
        "addc.cc.u64 %2,%2,0;\n\t" "addc.cc.u64 %3,%3,0;\n\t" "addc.u64 %4,0,0;\n\t"
        :"+l"(t0),"+l"(t1),"+l"(t2),"+l"(t3),"=l"(ct):"l"(e_sh),"l"(e_sh_hi)); c+=ct;
    if (c) { ulong kv=SECP256K1_K;
        asm volatile("add.cc.u64 %0,%0,%4;\n\t" "addc.cc.u64 %1,%1,0;\n\t"
            "addc.cc.u64 %2,%2,0;\n\t" "addc.u64 %3,%3,0;\n\t"
            :"+l"(t0),"+l"(t1),"+l"(t2),"+l"(t3):"l"(kv));
    }
    ulong r0,r1,r2,r3,bw;
    ulong p0=SECP256K1_P0,p1=SECP256K1_P1,p2=SECP256K1_P2,p3=SECP256K1_P3;
    asm volatile("sub.cc.u64 %0,%5,%9;\n\t" "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t" "subc.cc.u64 %3,%8,%12;\n\t" "subc.u64 %4,0,0;\n\t"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(bw)
        :"l"(t0),"l"(t1),"l"(t2),"l"(t3),"l"(p0),"l"(p1),"l"(p2),"l"(p3));
    if (bw==0) { r->limbs[0]=r0; r->limbs[1]=r1; r->limbs[2]=r2; r->limbs[3]=r3; }
    else { r->limbs[0]=t0; r->limbs[1]=t1; r->limbs[2]=t2; r->limbs[3]=t3; }
}

#endif // __NV_CL_C_VERSION
// =============================================================================

inline ulong2 mul64_full(ulong a, ulong b) {
    return (ulong2)(a * b, mul_hi(a, b));
}

inline ulong add_with_carry(ulong a, ulong b, ulong ci, ulong* co) {
    ulong s = a + b; ulong c1 = (s < a) ? 1UL : 0UL;
    s += ci; ulong c2 = (s < ci) ? 1UL : 0UL;
    *co = c1 + c2; return s;
}

inline ulong sub_with_borrow(ulong a, ulong b, ulong bi, ulong* bo) {
    ulong d = a - b; ulong b1 = (a < b) ? 1UL : 0UL;
    ulong t = d; d -= bi; ulong b2 = (t < bi) ? 1UL : 0UL;
    *bo = b1 + b2; return d;
}

inline void field_reduce(FieldElement* r, const ulong* a8) {
    ulong carry = 0, c1; ulong temp[5]; ulong2 prod;
    prod = mul64_full(SECP256K1_K, a8[4]); temp[0] = a8[0] + prod.x; carry = (temp[0] < a8[0]) ? 1UL : 0UL; carry += prod.y;
    prod = mul64_full(SECP256K1_K, a8[5]); temp[1] = a8[1] + carry; c1 = (temp[1] < carry) ? 1UL : 0UL; temp[1] += prod.x; c1 += (temp[1] < prod.x) ? 1UL : 0UL; carry = c1 + prod.y;
    prod = mul64_full(SECP256K1_K, a8[6]); temp[2] = a8[2] + carry; c1 = (temp[2] < carry) ? 1UL : 0UL; temp[2] += prod.x; c1 += (temp[2] < prod.x) ? 1UL : 0UL; carry = c1 + prod.y;
    prod = mul64_full(SECP256K1_K, a8[7]); temp[3] = a8[3] + carry; c1 = (temp[3] < carry) ? 1UL : 0UL; temp[3] += prod.x; c1 += (temp[3] < prod.x) ? 1UL : 0UL; temp[4] = c1 + prod.y;
    if (temp[4] != 0) { prod = mul64_full(SECP256K1_K, temp[4]); temp[0] += prod.x; carry = (temp[0] < prod.x) ? 1UL : 0UL; carry += prod.y; temp[1] += carry; carry = (temp[1] < carry) ? 1UL : 0UL; temp[2] += carry; carry = (temp[2] < carry) ? 1UL : 0UL; temp[3] += carry; }
    ulong borrow = 0; ulong diff[4];
    diff[0] = sub_with_borrow(temp[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(temp[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(temp[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(temp[3], SECP256K1_P3, borrow, &borrow);
    ulong mask = (borrow == 0) ? ~0UL : 0UL;
    r->limbs[0] = (diff[0] & mask) | (temp[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (temp[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (temp[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (temp[3] & ~mask);
}

inline void field_add_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    ulong a0=a->limbs[0],a1=a->limbs[1],a2=a->limbs[2],a3=a->limbs[3];
    ulong b0=b->limbs[0],b1=b->limbs[1],b2=b->limbs[2],b3=b->limbs[3];
    ulong s0,s1,s2,s3,carry;
    asm volatile("add.cc.u64 %0,%5,%9;\n\t" "addc.cc.u64 %1,%6,%10;\n\t"
        "addc.cc.u64 %2,%7,%11;\n\t" "addc.cc.u64 %3,%8,%12;\n\t" "addc.u64 %4,0,0;\n\t"
        :"=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(carry)
        :"l"(a0),"l"(a1),"l"(a2),"l"(a3),"l"(b0),"l"(b1),"l"(b2),"l"(b3));
    ulong d0,d1,d2,d3,borrow;
    ulong p0=SECP256K1_P0,p1=SECP256K1_P1,p2=SECP256K1_P2,p3=SECP256K1_P3;
    asm volatile("sub.cc.u64 %0,%5,%9;\n\t" "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t" "subc.cc.u64 %3,%8,%12;\n\t" "subc.u64 %4,0,0;\n\t"
        :"=l"(d0),"=l"(d1),"=l"(d2),"=l"(d3),"=l"(borrow)
        :"l"(s0),"l"(s1),"l"(s2),"l"(s3),"l"(p0),"l"(p1),"l"(p2),"l"(p3));
    ulong use_diff = (carry != 0) | (borrow == 0);
    ulong mask = use_diff ? ~0UL : 0UL;
    r->limbs[0]=(d0&mask)|(s0&~mask); r->limbs[1]=(d1&mask)|(s1&~mask);
    r->limbs[2]=(d2&mask)|(s2&~mask); r->limbs[3]=(d3&mask)|(s3&~mask);
#else
    ulong carry = 0; ulong sum[4];
    sum[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    sum[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    sum[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    sum[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    ulong borrow = 0; ulong diff[4];
    diff[0] = sub_with_borrow(sum[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(sum[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(sum[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(sum[3], SECP256K1_P3, borrow, &borrow);
    ulong use_diff = (carry != 0) | (borrow == 0);
    ulong mask = use_diff ? ~0UL : 0UL;
    r->limbs[0] = (diff[0] & mask) | (sum[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (sum[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (sum[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (sum[3] & ~mask);
#endif
}

inline void field_sub_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    ulong a0=a->limbs[0],a1=a->limbs[1],a2=a->limbs[2],a3=a->limbs[3];
    ulong b0=b->limbs[0],b1=b->limbs[1],b2=b->limbs[2],b3=b->limbs[3];
    ulong d0,d1,d2,d3,borrow;
    asm volatile("sub.cc.u64 %0,%5,%9;\n\t" "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t" "subc.cc.u64 %3,%8,%12;\n\t" "subc.u64 %4,0,0;\n\t"
        :"=l"(d0),"=l"(d1),"=l"(d2),"=l"(d3),"=l"(borrow)
        :"l"(a0),"l"(a1),"l"(a2),"l"(a3),"l"(b0),"l"(b1),"l"(b2),"l"(b3));
    ulong mp0 = (borrow!=0) ? (ulong)SECP256K1_P0 : 0UL;
    ulong mp1 = (borrow!=0) ? (ulong)SECP256K1_P1 : 0UL;
    ulong mp2 = (borrow!=0) ? (ulong)SECP256K1_P2 : 0UL;
    ulong mp3 = (borrow!=0) ? (ulong)SECP256K1_P3 : 0UL;
    asm volatile("add.cc.u64 %0,%0,%4;\n\t" "addc.cc.u64 %1,%1,%5;\n\t"
        "addc.cc.u64 %2,%2,%6;\n\t" "addc.u64 %3,%3,%7;\n\t"
        :"+l"(d0),"+l"(d1),"+l"(d2),"+l"(d3):"l"(mp0),"l"(mp1),"l"(mp2),"l"(mp3));
    r->limbs[0]=d0; r->limbs[1]=d1; r->limbs[2]=d2; r->limbs[3]=d3;
#else
    ulong borrow = 0; ulong diff[4];
    diff[0] = sub_with_borrow(a->limbs[0], b->limbs[0], 0, &borrow);
    diff[1] = sub_with_borrow(a->limbs[1], b->limbs[1], borrow, &borrow);
    diff[2] = sub_with_borrow(a->limbs[2], b->limbs[2], borrow, &borrow);
    diff[3] = sub_with_borrow(a->limbs[3], b->limbs[3], borrow, &borrow);
    ulong mask = borrow ? ~0UL : 0UL;
    ulong carry = 0;
    r->limbs[0] = add_with_carry(diff[0], SECP256K1_P0 & mask, 0, &carry);
    r->limbs[1] = add_with_carry(diff[1], SECP256K1_P1 & mask, carry, &carry);
    r->limbs[2] = add_with_carry(diff[2], SECP256K1_P2 & mask, carry, &carry);
    r->limbs[3] = add_with_carry(diff[3], SECP256K1_P3 & mask, carry, &carry);
#endif
}

inline void field_mul_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    ulong t[8]; mul_256_512_cl(a->limbs, b->limbs, t); reduce_512_to_256_cl(t, r);
#else
    ulong product[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong2 mul = mul64_full(a->limbs[i], b->limbs[j]);
            ulong sum = product[i+j] + mul.x; ulong c1 = (sum < product[i+j]) ? 1UL : 0UL;
            sum += carry; ulong c2 = (sum < carry) ? 1UL : 0UL;
            product[i+j] = sum; carry = mul.y + c1 + c2;
        }
        product[i+4] += carry;
    }
    field_reduce(r, product);
#endif
}

inline void field_sqr_impl(FieldElement* r, const FieldElement* a) {
#ifdef __NV_CL_C_VERSION
    ulong t[8]; sqr_256_512_cl(a->limbs, t); reduce_512_to_256_cl(t, r);
#else
    field_mul_impl(r, a, a);
#endif
}

inline void field_sqr_n_impl(FieldElement* r, int n) {
    for (int i = 0; i < n; i++) { FieldElement tmp = *r; field_sqr_impl(r, &tmp); }
}

inline void field_inv_impl(FieldElement* r, const FieldElement* a) {
    FieldElement x2,x3,x6,x12,x24,x48,x96,x192,x7,x31,x223,x5,x11,x22,t;
    field_sqr_impl(&x2, a); field_mul_impl(&x2, &x2, a);
    field_sqr_impl(&x3, &x2); field_mul_impl(&x3, &x3, a);
    t=x3; field_sqr_n_impl(&t,3); field_mul_impl(&x6,&t,&x3);
    t=x6; field_sqr_n_impl(&t,6); field_mul_impl(&x12,&t,&x6);
    t=x12; field_sqr_n_impl(&t,12); field_mul_impl(&x24,&t,&x12);
    t=x24; field_sqr_n_impl(&t,24); field_mul_impl(&x48,&t,&x24);
    t=x48; field_sqr_n_impl(&t,48); field_mul_impl(&x96,&t,&x48);
    t=x96; field_sqr_n_impl(&t,96); field_mul_impl(&x192,&t,&x96);
    field_sqr_impl(&x7,&x6); field_mul_impl(&x7,&x7,a);
    t=x24; field_sqr_n_impl(&t,7); field_mul_impl(&x31,&t,&x7);
    t=x192; field_sqr_n_impl(&t,31); field_mul_impl(&x223,&t,&x31);
    t=x3; field_sqr_n_impl(&t,2); field_mul_impl(&x5,&t,&x2);
    t=x6; field_sqr_n_impl(&t,5); field_mul_impl(&x11,&t,&x5);
    t=x11; field_sqr_n_impl(&t,11); field_mul_impl(&x22,&t,&x11);
    field_sqr_impl(&t,&x223);
    field_sqr_n_impl(&t,22); field_mul_impl(&t,&t,&x22);
    field_sqr_n_impl(&t,4);
    field_sqr_impl(&t,&t); field_mul_impl(&t,&t,a);
    field_sqr_impl(&t,&t);
    field_sqr_impl(&t,&t); field_mul_impl(&t,&t,a);
    field_sqr_impl(&t,&t); field_mul_impl(&t,&t,a);
    field_sqr_impl(&t,&t);
    field_sqr_impl(&t,&t); field_mul_impl(r,&t,a);
}

__kernel void field_add(__global const FieldElement* a, __global const FieldElement* b, __global FieldElement* r, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    FieldElement a_local = a[gid]; FieldElement b_local = b[gid];
    FieldElement res; field_add_impl(&res, &a_local, &b_local); r[gid] = res;
}

__kernel void field_sub(__global const FieldElement* a, __global const FieldElement* b, __global FieldElement* r, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    FieldElement a_local = a[gid]; FieldElement b_local = b[gid];
    FieldElement res; field_sub_impl(&res, &a_local, &b_local); r[gid] = res;
}

__kernel void field_mul(__global const FieldElement* a, __global const FieldElement* b, __global FieldElement* r, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    FieldElement a_local = a[gid]; FieldElement b_local = b[gid];
    FieldElement res; field_mul_impl(&res, &a_local, &b_local); r[gid] = res;
}

__kernel void field_sqr(__global const FieldElement* a, __global FieldElement* r, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    FieldElement a_local = a[gid];
    FieldElement res; field_sqr_impl(&res, &a_local); r[gid] = res;
}

__kernel void field_inv(__global const FieldElement* a, __global FieldElement* r, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    FieldElement a_local = a[gid];
    FieldElement res; field_inv_impl(&res, &a_local); r[gid] = res;
}

)KERNEL"

// ---- second segment (point operations + scalar mul + batch) ----
R"KERNEL(

// Point operations - simplified versions
inline int point_is_infinity(const JacobianPoint* p) { return p->infinity || ((p->z.limbs[0] | p->z.limbs[1] | p->z.limbs[2] | p->z.limbs[3]) == 0); }
inline void point_set_infinity(JacobianPoint* p) { p->x.limbs[0]=0; p->x.limbs[1]=0; p->x.limbs[2]=0; p->x.limbs[3]=0; p->y.limbs[0]=1; p->y.limbs[1]=0; p->y.limbs[2]=0; p->y.limbs[3]=0; p->z.limbs[0]=0; p->z.limbs[1]=0; p->z.limbs[2]=0; p->z.limbs[3]=0; p->infinity=1; }

// Point doubling: dbl-2007-a for a=0 (secp256k1)
// CUDA-matched: 3 temps, write directly into r
inline void point_double_impl(JacobianPoint* r, const JacobianPoint* p) {
    if (point_is_infinity(p)) { point_set_infinity(r); return; }
    if ((p->y.limbs[0] | p->y.limbs[1] | p->y.limbs[2] | p->y.limbs[3]) == 0) {
        point_set_infinity(r); return;
    }

    FieldElement t1, t2, t3;

    // Z' = 2*Y*Z (compute first, only reads p->y, p->z)
    field_mul_impl(&r->z, &p->y, &p->z);
    field_add_impl(&r->z, &r->z, &r->z);

    // t1 = X^2 (A)
    field_sqr_impl(&t1, &p->x);

    // t2 = Y^2 (B)
    field_sqr_impl(&t2, &p->y);

    // t3 = Y^4 (C = B^2)
    field_sqr_impl(&t3, &t2);

    // D = 2*((X+B)^2 - A - C), store in r->x
    field_add_impl(&r->x, &p->x, &t2);
    field_sqr_impl(&r->x, &r->x);
    field_sub_impl(&r->x, &r->x, &t1);
    field_sub_impl(&r->x, &r->x, &t3);
    field_add_impl(&r->x, &r->x, &r->x);

    // E = 3*A, store in t1 (reuse t2 as scratch)
    field_add_impl(&t2, &t1, &t1);       // 2A
    field_add_impl(&t1, &t2, &t1);       // 3A = E

    // F = E^2, store in t2
    field_sqr_impl(&t2, &t1);

    // Save D in r->y
    r->y = r->x;

    // X' = F - 2*D
    field_add_impl(&r->x, &r->x, &r->x);
    field_sub_impl(&r->x, &t2, &r->x);

    // Y' = E*(D - X') - 8*C
    field_sub_impl(&r->y, &r->y, &r->x);
    field_mul_impl(&r->y, &t1, &r->y);
    field_add_impl(&t3, &t3, &t3);       // 2C
    field_add_impl(&t3, &t3, &t3);       // 4C
    field_add_impl(&t3, &t3, &t3);       // 8C
    field_sub_impl(&r->y, &r->y, &t3);

    r->infinity = 0;
}

// Mixed addition: Jacobian + Affine → Jacobian (CUDA-matched, write directly to r)
inline void point_add_mixed_impl(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
    if (point_is_infinity(p)) { r->x = q->x; r->y = q->y; r->z.limbs[0]=1; r->z.limbs[1]=0; r->z.limbs[2]=0; r->z.limbs[3]=0; r->infinity=0; return; }
    FieldElement z1z1, u2, s2, h, hh, i, j, rr, v, t1;
    field_sqr_impl(&z1z1, &p->z);
    field_mul_impl(&u2, &q->x, &z1z1);
    field_mul_impl(&t1, &p->z, &z1z1);       // Z1^3
    field_mul_impl(&s2, &q->y, &t1);
    field_sub_impl(&h, &u2, &p->x);
    if ((h.limbs[0]|h.limbs[1]|h.limbs[2]|h.limbs[3]) == 0) { field_sub_impl(&t1, &s2, &p->y); if ((t1.limbs[0]|t1.limbs[1]|t1.limbs[2]|t1.limbs[3]) == 0) { point_double_impl(r, p); return; } point_set_infinity(r); return; }
    field_sqr_impl(&hh, &h);
    field_add_impl(&i, &hh, &hh); field_add_impl(&i, &i, &i);  // I = 4*HH
    field_mul_impl(&j, &h, &i);
    field_sub_impl(&rr, &s2, &p->y); field_add_impl(&rr, &rr, &rr);  // rr = 2*(S2-Y1)
    field_mul_impl(&v, &p->x, &i);
    // X3 = rr^2 - J - 2*V
    field_sqr_impl(&r->x, &rr);
    field_sub_impl(&r->x, &r->x, &j);
    field_add_impl(&t1, &v, &v);
    field_sub_impl(&r->x, &r->x, &t1);
    // Y3 = rr*(V - X3) - 2*Y1*J  (compute Y1*J BEFORE writing r->y for r==p safety)
    field_mul_impl(&t1, &p->y, &j);
    field_add_impl(&t1, &t1, &t1);
    field_sub_impl(&v, &v, &r->x);
    field_mul_impl(&r->y, &rr, &v);
    field_sub_impl(&r->y, &r->y, &t1);
    // Z3 = (Z1+H)^2 - Z1^2 - HH  (reads p->z before writing r->z)
    field_add_impl(&t1, &p->z, &h);
    field_sqr_impl(&r->z, &t1);
    field_sub_impl(&r->z, &r->z, &z1z1);
    field_sub_impl(&r->z, &r->z, &hh);
    r->infinity = 0;
}

inline void point_add_impl(JacobianPoint* r, const JacobianPoint* p, const JacobianPoint* q) {
    if (point_is_infinity(p)) { *r = *q; return; }
    if (point_is_infinity(q)) { *r = *p; return; }
    FieldElement U1,U2,S1,S2,H,I,J,rr,V,X3,Y3,Z3,Z1Z1,Z2Z2,t1,t2;
    field_sqr_impl(&Z1Z1, &p->z); field_sqr_impl(&Z2Z2, &q->z);
    field_mul_impl(&U1, &p->x, &Z2Z2); field_mul_impl(&U2, &q->x, &Z1Z1);
    field_mul_impl(&t1, &p->y, &q->z); field_mul_impl(&S1, &t1, &Z2Z2);
    field_mul_impl(&t1, &q->y, &p->z); field_mul_impl(&S2, &t1, &Z1Z1);
    field_sub_impl(&H, &U2, &U1);
    if ((H.limbs[0]|H.limbs[1]|H.limbs[2]|H.limbs[3]) == 0) { field_sub_impl(&t1, &S2, &S1); if ((t1.limbs[0]|t1.limbs[1]|t1.limbs[2]|t1.limbs[3]) == 0) { point_double_impl(r, p); return; } point_set_infinity(r); return; }
    field_add_impl(&I, &H, &H); field_sqr_impl(&I, &I); field_mul_impl(&J, &H, &I);
    field_sub_impl(&rr, &S2, &S1); field_add_impl(&rr, &rr, &rr); field_mul_impl(&V, &U1, &I);
    field_sqr_impl(&X3, &rr); field_sub_impl(&X3, &X3, &J); field_add_impl(&t1, &V, &V); field_sub_impl(&X3, &X3, &t1);
    field_sub_impl(&t1, &V, &X3); field_mul_impl(&Y3, &rr, &t1); field_mul_impl(&t2, &S1, &J); field_add_impl(&t2, &t2, &t2); field_sub_impl(&Y3, &Y3, &t2);
    field_add_impl(&t1, &p->z, &q->z); field_sqr_impl(&t1, &t1); field_sub_impl(&t1, &t1, &Z1Z1); field_sub_impl(&t1, &t1, &Z2Z2); field_mul_impl(&Z3, &t1, &H);
    r->x = X3; r->y = Y3; r->z = Z3; r->infinity = 0;
}

// =============================================================================
// Field negation: r = -a mod p  (delegates to PTX-optimized field_sub)
// =============================================================================
inline void field_neg_impl(FieldElement* r, const FieldElement* a) {
    FieldElement zero;
    zero.limbs[0]=0; zero.limbs[1]=0; zero.limbs[2]=0; zero.limbs[3]=0;
    field_sub_impl(r, &zero, a);
}

// =============================================================================
// Scalar helpers for wNAF encoding (plain integer arithmetic, no mod-order)
// =============================================================================
inline int scalar_is_zero_cl(const Scalar* s) {
    return (s->limbs[0] | s->limbs[1] | s->limbs[2] | s->limbs[3]) == 0;
}

inline void scalar_add_u64_cl(Scalar* a, ulong val) {
    ulong s = a->limbs[0] + val;
    ulong c = (s < a->limbs[0]) ? 1UL : 0UL;
    a->limbs[0] = s;
    s = a->limbs[1] + c; c = (s < a->limbs[1]) ? 1UL : 0UL;
    a->limbs[1] = s;
    s = a->limbs[2] + c; c = (s < a->limbs[2]) ? 1UL : 0UL;
    a->limbs[2] = s;
    a->limbs[3] += c;
}

inline void scalar_sub_u64_cl(Scalar* a, ulong val) {
    ulong d = a->limbs[0] - val;
    ulong b = (a->limbs[0] < val) ? 1UL : 0UL;
    a->limbs[0] = d;
    d = a->limbs[1] - b; b = (a->limbs[1] < b) ? 1UL : 0UL;
    a->limbs[1] = d;
    d = a->limbs[2] - b; b = (a->limbs[2] < b) ? 1UL : 0UL;
    a->limbs[2] = d;
    a->limbs[3] -= b;
}

inline void scalar_shr1_cl(Scalar* s) {
    s->limbs[0] = (s->limbs[0] >> 1) | (s->limbs[1] << 63);
    s->limbs[1] = (s->limbs[1] >> 1) | (s->limbs[2] << 63);
    s->limbs[2] = (s->limbs[2] >> 1) | (s->limbs[3] << 63);
    s->limbs[3] = s->limbs[3] >> 1;
}

// wNAF encoding (window width 5): digits in {-15,...,15}, ~51 non-zero digits
inline int scalar_to_wnaf_cl(const Scalar* k, int* wnaf) {
    Scalar temp = *k;
    int len = 0;
    while (!scalar_is_zero_cl(&temp) && len < 260) {
        if (temp.limbs[0] & 1UL) {
            int digit = (int)(temp.limbs[0] & 31UL);
            if (digit >= 16) {
                digit -= 32;
                scalar_add_u64_cl(&temp, (ulong)(-digit));
            } else {
                scalar_sub_u64_cl(&temp, (ulong)digit);
            }
            wnaf[len] = digit;
        } else {
            wnaf[len] = 0;
        }
        scalar_shr1_cl(&temp);
        len++;
    }
    return len;
}

// wNAF scalar multiplication core: precompute table [P,3P,...,15P], scan MSB→LSB
inline void scalar_mul_wnaf(JacobianPoint* r, const Scalar* k, const JacobianPoint* base_jac) {
    // wNAF encode
    int wnaf[260];
    int wnaf_len = scalar_to_wnaf_cl(k, wnaf);

    // Precompute odd multiples: table[i] = (2i+1)*base
    JacobianPoint table[8];
    JacobianPoint dbl;
    table[0] = *base_jac;
    point_double_impl(&dbl, base_jac);
    for (int i = 1; i < 8; i++)
        point_add_impl(&table[i], &table[i-1], &dbl);

    // Scan from MSB
    point_set_infinity(r);
    for (int i = wnaf_len - 1; i >= 0; --i) {
        point_double_impl(r, r);
        int digit = wnaf[i];
        if (digit > 0) {
            point_add_impl(r, r, &table[(digit - 1) >> 1]);
        } else if (digit < 0) {
            JacobianPoint neg = table[(-digit - 1) >> 1];
            field_neg_impl(&neg.y, &neg.y);
            point_add_impl(r, r, &neg);
        }
    }
}

#define GX0 0x59F2815B16F81798UL
#define GX1 0x029BFCDB2DCE28D9UL
#define GX2 0x55A06295CE870B07UL
#define GX3 0x79BE667EF9DCBBACUL
#define GY0 0x9C47D08FFB10D4B8UL
#define GY1 0xFD17B448A6855419UL
#define GY2 0x5DA4FBFC0E1108A8UL
#define GY3 0x483ADA7726A3C465UL

inline void get_generator(AffinePoint* g) { g->x.limbs[0]=GX0; g->x.limbs[1]=GX1; g->x.limbs[2]=GX2; g->x.limbs[3]=GX3; g->y.limbs[0]=GY0; g->y.limbs[1]=GY1; g->y.limbs[2]=GY2; g->y.limbs[3]=GY3; }

__kernel void scalar_mul_generator(__global const Scalar* scalars, __global JacobianPoint* results, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    Scalar k = scalars[gid];
    JacobianPoint R;
    if ((k.limbs[0]|k.limbs[1]|k.limbs[2]|k.limbs[3]) == 0) { point_set_infinity(&R); results[gid] = R; return; }
    point_set_infinity(&R);
    AffinePoint G; get_generator(&G);
    int started = 0;
    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            if (started) point_double_impl(&R, &R);
            if ((k.limbs[limb] >> bit) & 1) { if (!started) { R.x = G.x; R.y = G.y; R.z.limbs[0]=1; R.z.limbs[1]=0; R.z.limbs[2]=0; R.z.limbs[3]=0; R.infinity=0; started=1; } else { point_add_mixed_impl(&R, &R, &G); } }
        }
    }
    results[gid] = R;
}

__kernel void point_double(__global const JacobianPoint* points, __global JacobianPoint* results, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    JacobianPoint p_local = points[gid];
    JacobianPoint r; point_double_impl(&r, &p_local); results[gid] = r;
}

__kernel void point_add(__global const JacobianPoint* p, __global const JacobianPoint* q, __global JacobianPoint* results, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    JacobianPoint p_local = p[gid]; JacobianPoint q_local = q[gid];
    JacobianPoint r; point_add_impl(&r, &p_local, &q_local); results[gid] = r;
}

inline void scalar_mul_impl(JacobianPoint* r, const Scalar* k, const AffinePoint* p) {
    if ((k->limbs[0]|k->limbs[1]|k->limbs[2]|k->limbs[3]) == 0) { point_set_infinity(r); return; }
    point_set_infinity(r);
    int started = 0;
    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            if (started) point_double_impl(r, r);
            if ((k->limbs[limb] >> bit) & 1) {
                if (!started) { r->x = p->x; r->y = p->y; r->z.limbs[0]=1; r->z.limbs[1]=0; r->z.limbs[2]=0; r->z.limbs[3]=0; r->infinity=0; started=1; }
                else { point_add_mixed_impl(r, r, p); }
            }
        }
    }
}

__kernel void scalar_mul(__global const Scalar* scalars, __global const AffinePoint* points, __global JacobianPoint* results, uint count) {
    uint gid = get_global_id(0); if (gid >= count) return;
    Scalar k = scalars[gid]; AffinePoint p = points[gid];
    JacobianPoint r; scalar_mul_impl(&r, &k, &p); results[gid] = r;
}

__kernel void batch_jacobian_to_affine_kernel(
    __global const JacobianPoint* jacobians,
    __global AffinePoint* affines,
    uint count
) {
    uint gid = get_global_id(0); if (gid >= count) return;
    JacobianPoint p = jacobians[gid];
    if (point_is_infinity(&p)) {
        affines[gid].x.limbs[0]=0; affines[gid].x.limbs[1]=0; affines[gid].x.limbs[2]=0; affines[gid].x.limbs[3]=0;
        affines[gid].y.limbs[0]=0; affines[gid].y.limbs[1]=0; affines[gid].y.limbs[2]=0; affines[gid].y.limbs[3]=0;
        return;
    }
    if (p.z.limbs[0]==1 && p.z.limbs[1]==0 && p.z.limbs[2]==0 && p.z.limbs[3]==0) {
        affines[gid].x = p.x; affines[gid].y = p.y; return;
    }
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv_impl(&z_inv, &p.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv2, &z_inv);
    FieldElement ax, ay;
    field_mul_impl(&ax, &p.x, &z_inv2);
    field_mul_impl(&ay, &p.y, &z_inv3);
    affines[gid].x = ax;
    affines[gid].y = ay;
}

// ---- Affine point addition kernels ----

// affine_add_impl: P + Q -> R, all affine (2M + 1S + inv)
inline void affine_add_impl(AffinePoint* r,
                             const FieldElement* px, const FieldElement* py,
                             const FieldElement* qx, const FieldElement* qy) {
    FieldElement h, rr, t, lam;
    field_sub_impl(&h, qx, px);
    field_sub_impl(&rr, qy, py);
    field_inv_impl(&t, &h);
    field_mul_impl(&lam, &rr, &t);
    field_sqr_impl(&r->x, &lam);
    field_sub_impl(&r->x, &r->x, px);
    field_sub_impl(&r->x, &r->x, qx);
    field_sub_impl(&r->y, px, &r->x);
    field_mul_impl(&r->y, &lam, &r->y);
    field_sub_impl(&r->y, &r->y, py);
}

// affine_add_lambda_impl: with pre-inverted H (2M + 1S)
inline void affine_add_lambda_impl(AffinePoint* r,
                                    const FieldElement* px, const FieldElement* py,
                                    const FieldElement* qx, const FieldElement* qy,
                                    const FieldElement* h_inv) {
    FieldElement rr, lam;
    field_sub_impl(&rr, qy, py);
    field_mul_impl(&lam, &rr, h_inv);
    field_sqr_impl(&r->x, &lam);
    field_sub_impl(&r->x, &r->x, px);
    field_sub_impl(&r->x, &r->x, qx);
    field_sub_impl(&r->y, px, &r->x);
    field_mul_impl(&r->y, &lam, &r->y);
    field_sub_impl(&r->y, &r->y, py);
}

// affine_add_x_only_impl: X-only with pre-inverted H (1M + 1S)
inline void affine_add_x_only_impl(FieldElement* rx,
                                    const FieldElement* px, const FieldElement* py,
                                    const FieldElement* qx, const FieldElement* qy,
                                    const FieldElement* h_inv) {
    FieldElement rr, lam;
    field_sub_impl(&rr, qy, py);
    field_mul_impl(&lam, &rr, h_inv);
    field_sqr_impl(rx, &lam);
    field_sub_impl(rx, rx, px);
    field_sub_impl(rx, rx, qx);
}

// jacobian_to_affine_convert_impl: single point
inline void jacobian_to_affine_convert_impl(AffinePoint* r,
                                             const FieldElement* x,
                                             const FieldElement* y,
                                             const FieldElement* z) {
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv_impl(&z_inv, z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv, &z_inv2);
    field_mul_impl(&r->x, x, &z_inv2);
    field_mul_impl(&r->y, y, &z_inv3);
}

__kernel void affine_add(
    __global const FieldElement* px, __global const FieldElement* py,
    __global const FieldElement* qx, __global const FieldElement* qy,
    __global FieldElement* rx, __global FieldElement* ry,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    FieldElement lpx = px[gid], lpy = py[gid];
    FieldElement lqx = qx[gid], lqy = qy[gid];
    AffinePoint r;
    affine_add_impl(&r, &lpx, &lpy, &lqx, &lqy);
    rx[gid] = r.x;
    ry[gid] = r.y;
}

__kernel void affine_add_lambda(
    __global const FieldElement* px, __global const FieldElement* py,
    __global const FieldElement* qx, __global const FieldElement* qy,
    __global const FieldElement* h_inv,
    __global FieldElement* rx, __global FieldElement* ry,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    FieldElement lpx = px[gid], lpy = py[gid];
    FieldElement lqx = qx[gid], lqy = qy[gid];
    FieldElement lhinv = h_inv[gid];
    AffinePoint r;
    affine_add_lambda_impl(&r, &lpx, &lpy, &lqx, &lqy, &lhinv);
    rx[gid] = r.x;
    ry[gid] = r.y;
}

__kernel void affine_add_x_only(
    __global const FieldElement* px, __global const FieldElement* py,
    __global const FieldElement* qx, __global const FieldElement* qy,
    __global const FieldElement* h_inv,
    __global FieldElement* rx,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    FieldElement lpx = px[gid], lpy = py[gid];
    FieldElement lqx = qx[gid], lqy = qy[gid];
    FieldElement lhinv = h_inv[gid];
    FieldElement lrx;
    affine_add_x_only_impl(&lrx, &lpx, &lpy, &lqx, &lqy, &lhinv);
    rx[gid] = lrx;
}

__kernel void jacobian_to_affine(
    __global const FieldElement* jx,
    __global const FieldElement* jy,
    __global const FieldElement* jz,
    __global FieldElement* ax, __global FieldElement* ay,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    FieldElement lx = jx[gid], ly = jy[gid], lz = jz[gid];
    AffinePoint r;
    jacobian_to_affine_convert_impl(&r, &lx, &ly, &lz);
    ax[gid] = r.x;
    ay[gid] = r.y;
}
)KERNEL";

bool Context::Impl::build_program() {
    cl_int err;

    // Create program from source
    const char* sources[] = {kernel_source};
    std::size_t lengths[] = {std::strlen(kernel_source)};

    program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
    if (err != CL_SUCCESS) {
        last_error = std::string("Failed to create program: ") + cl_error_string(err);
        return false;
    }

    // Build options
    std::string build_options = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";

    // NVIDIA-specific optimization flags
    if (device_info.is_nvidia) {
        build_options += " -cl-nv-opt-level=3";
    }

    // Build program
    err = clBuildProgram(program, 1, &device, build_options.c_str(), nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // Get build log
        std::size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);

        last_error = std::string("Build failed: ") + build_log.data();
        return false;
    }

    return true;
}

bool Context::Impl::create_kernels() {
    cl_int err;

    kernel_field_add = clCreateKernel(program, "field_add", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create field_add kernel"; return false; }

    kernel_field_sub = clCreateKernel(program, "field_sub", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create field_sub kernel"; return false; }

    kernel_field_mul = clCreateKernel(program, "field_mul", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create field_mul kernel"; return false; }

    kernel_field_sqr = clCreateKernel(program, "field_sqr", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create field_sqr kernel"; return false; }

    kernel_field_inv = clCreateKernel(program, "field_inv", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create field_inv kernel"; return false; }

    kernel_point_double = clCreateKernel(program, "point_double", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create point_double kernel"; return false; }

    kernel_point_add = clCreateKernel(program, "point_add", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create point_add kernel"; return false; }

    kernel_scalar_mul_generator = clCreateKernel(program, "scalar_mul_generator", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create scalar_mul_generator kernel"; return false; }

    kernel_scalar_mul = clCreateKernel(program, "scalar_mul", &err);
    if (err != CL_SUCCESS) { last_error = "Failed to create scalar_mul kernel"; return false; }

    kernel_batch_jacobian_to_affine = clCreateKernel(program, "batch_jacobian_to_affine", &err);
    if (err != CL_SUCCESS) {
        // Non-fatal — kernel may not exist in older builds
        kernel_batch_jacobian_to_affine = nullptr;
    }

    // Affine point addition kernels (optional — benchmark/utility)
    err = CL_SUCCESS;
    kernel_affine_add = clCreateKernel(program, "affine_add", &err);
    if (err != CL_SUCCESS) {
        if (config.verbose) std::cerr << "[DEBUG] affine_add kernel: err=" << err << "\n";
        kernel_affine_add = nullptr;
    }
    kernel_affine_add_lambda = clCreateKernel(program, "affine_add_lambda", &err);
    if (err != CL_SUCCESS) {
        if (config.verbose) std::cerr << "[DEBUG] affine_add_lambda kernel: err=" << err << "\n";
        kernel_affine_add_lambda = nullptr;
    }
    kernel_affine_add_x_only = clCreateKernel(program, "affine_add_x_only", &err);
    if (err != CL_SUCCESS) {
        if (config.verbose) std::cerr << "[DEBUG] affine_add_x_only kernel: err=" << err << "\n";
        kernel_affine_add_x_only = nullptr;
    }
    kernel_jacobian_to_affine = clCreateKernel(program, "jacobian_to_affine", &err);
    if (err != CL_SUCCESS) {
        if (config.verbose) std::cerr << "[DEBUG] jacobian_to_affine kernel: err=" << err << "\n";
        kernel_jacobian_to_affine = nullptr;
    }

    return true;
}

// =============================================================================
// Context Public Interface
// =============================================================================

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

std::unique_ptr<Context> Context::create(const DeviceConfig& config) {
    auto ctx = std::unique_ptr<Context>(new Context());
    if (!ctx->impl_->init(config)) {
        return nullptr;
    }
    return ctx;
}

const DeviceInfo& Context::device_info() const noexcept {
    return impl_->device_info;
}

bool Context::is_valid() const noexcept {
    return impl_ && impl_->context != nullptr;
}

const std::string& Context::last_error() const noexcept {
    return impl_->last_error;
}

// =============================================================================
// Single Field Operations
// =============================================================================

FieldElement Context::field_add(const FieldElement& a, const FieldElement& b) {
    FieldElement result;
    cl_int err;

    // Create buffers
    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&a, &err);
    cl_mem buf_b = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&b, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(FieldElement), nullptr, &err);

    // Set kernel args
    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_field_add, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_add, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(impl_->kernel_field_add, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_add, 3, sizeof(cl_uint), &count);

    // Execute
    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_add, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);

    // Read result
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(FieldElement), &result, 0, nullptr, nullptr);

    // Cleanup
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_r);

    return result;
}

FieldElement Context::field_sub(const FieldElement& a, const FieldElement& b) {
    FieldElement result;
    cl_int err;

    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&a, &err);
    cl_mem buf_b = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&b, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(FieldElement), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_field_sub, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_sub, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(impl_->kernel_field_sub, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_sub, 3, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_sub, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(FieldElement), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_r);

    return result;
}

FieldElement Context::field_mul(const FieldElement& a, const FieldElement& b) {
    FieldElement result;
    cl_int err;

    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&a, &err);
    cl_mem buf_b = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&b, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(FieldElement), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_field_mul, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_mul, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(impl_->kernel_field_mul, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_mul, 3, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_mul, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(FieldElement), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_r);

    return result;
}

FieldElement Context::field_sqr(const FieldElement& a) {
    FieldElement result;
    cl_int err;

    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&a, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(FieldElement), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_field_sqr, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_sqr, 1, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_sqr, 2, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_sqr, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(FieldElement), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_r);

    return result;
}

FieldElement Context::field_inv(const FieldElement& a) {
    FieldElement result;
    cl_int err;

    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(FieldElement), (void*)&a, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(FieldElement), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_field_inv, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_inv, 1, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_inv, 2, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_inv, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(FieldElement), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_r);

    return result;
}

// =============================================================================
// Point Operations
// =============================================================================

JacobianPoint Context::point_double(const JacobianPoint& p) {
    JacobianPoint result;
    cl_int err;

    cl_mem buf_p = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(JacobianPoint), (void*)&p, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(JacobianPoint), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_point_double, 0, sizeof(cl_mem), &buf_p);
    clSetKernelArg(impl_->kernel_point_double, 1, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_point_double, 2, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_point_double, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(JacobianPoint), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_p);
    clReleaseMemObject(buf_r);

    return result;
}

JacobianPoint Context::point_add(const JacobianPoint& p, const JacobianPoint& q) {
    JacobianPoint result;
    cl_int err;

    cl_mem buf_p = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(JacobianPoint), (void*)&p, &err);
    cl_mem buf_q = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(JacobianPoint), (void*)&q, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(JacobianPoint), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_point_add, 0, sizeof(cl_mem), &buf_p);
    clSetKernelArg(impl_->kernel_point_add, 1, sizeof(cl_mem), &buf_q);
    clSetKernelArg(impl_->kernel_point_add, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_point_add, 3, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_point_add, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(JacobianPoint), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_p);
    clReleaseMemObject(buf_q);
    clReleaseMemObject(buf_r);

    return result;
}

JacobianPoint Context::scalar_mul(const Scalar& k, const AffinePoint& p) {
    JacobianPoint result;
    cl_int err;

    cl_mem buf_k = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(Scalar), (void*)&k, &err);
    cl_mem buf_p = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(AffinePoint), (void*)&p, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(JacobianPoint), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_scalar_mul, 0, sizeof(cl_mem), &buf_k);
    clSetKernelArg(impl_->kernel_scalar_mul, 1, sizeof(cl_mem), &buf_p);
    clSetKernelArg(impl_->kernel_scalar_mul, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_scalar_mul, 3, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_scalar_mul, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(JacobianPoint), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_k);
    clReleaseMemObject(buf_p);
    clReleaseMemObject(buf_r);

    return result;
}

JacobianPoint Context::scalar_mul_generator(const Scalar& k) {
    JacobianPoint result;
    cl_int err;

    cl_mem buf_k = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(Scalar), (void*)&k, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   sizeof(JacobianPoint), nullptr, &err);

    cl_uint count = 1;
    clSetKernelArg(impl_->kernel_scalar_mul_generator, 0, sizeof(cl_mem), &buf_k);
    clSetKernelArg(impl_->kernel_scalar_mul_generator, 1, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_scalar_mul_generator, 2, sizeof(cl_uint), &count);

    std::size_t global_size = 1;
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_scalar_mul_generator, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0, sizeof(JacobianPoint), &result, 0, nullptr, nullptr);

    clReleaseMemObject(buf_k);
    clReleaseMemObject(buf_r);

    return result;
}

// =============================================================================
// Batch Operations
// =============================================================================

// Helper: compute work sizes for batch dispatch
static void compute_work_sizes(std::size_t count, std::size_t max_wg, std::size_t& local, std::size_t& global) {
    local = std::min(static_cast<std::size_t>(256), max_wg);
    global = ((count + local - 1) / local) * local;
}

void Context::batch_field_add(const FieldElement* a, const FieldElement* b,
                               FieldElement* results, std::size_t count) {
    if (count == 0) return;
    cl_int err;
    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)a, &err);
    cl_mem buf_b = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)b, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   count * sizeof(FieldElement), nullptr, &err);
    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_field_add, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_add, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(impl_->kernel_field_add, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_add, 3, sizeof(cl_uint), &cnt);
    std::size_t local_size, global_size;
    compute_work_sizes(count, impl_->device_info.max_work_group_size, local_size, global_size);
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_add, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0,
                        count * sizeof(FieldElement), results, 0, nullptr, nullptr);
    clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); clReleaseMemObject(buf_r);
}

void Context::batch_field_sub(const FieldElement* a, const FieldElement* b,
                               FieldElement* results, std::size_t count) {
    if (count == 0) return;
    cl_int err;
    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)a, &err);
    cl_mem buf_b = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)b, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   count * sizeof(FieldElement), nullptr, &err);
    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_field_sub, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_sub, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(impl_->kernel_field_sub, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_sub, 3, sizeof(cl_uint), &cnt);
    std::size_t local_size, global_size;
    compute_work_sizes(count, impl_->device_info.max_work_group_size, local_size, global_size);
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_sub, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0,
                        count * sizeof(FieldElement), results, 0, nullptr, nullptr);
    clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); clReleaseMemObject(buf_r);
}

void Context::batch_field_mul(const FieldElement* a, const FieldElement* b,
                               FieldElement* results, std::size_t count) {
    if (count == 0) return;
    cl_int err;
    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)a, &err);
    cl_mem buf_b = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)b, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   count * sizeof(FieldElement), nullptr, &err);
    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_field_mul, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_mul, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(impl_->kernel_field_mul, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_mul, 3, sizeof(cl_uint), &cnt);
    std::size_t local_size, global_size;
    compute_work_sizes(count, impl_->device_info.max_work_group_size, local_size, global_size);
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_mul, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0,
                        count * sizeof(FieldElement), results, 0, nullptr, nullptr);
    clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); clReleaseMemObject(buf_r);
}

void Context::batch_field_sqr(const FieldElement* inputs, FieldElement* results, std::size_t count) {
    if (count == 0) return;
    cl_int err;
    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(FieldElement), (void*)inputs, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   count * sizeof(FieldElement), nullptr, &err);
    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_field_sqr, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_field_sqr, 1, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_field_sqr, 2, sizeof(cl_uint), &cnt);
    std::size_t local_size, global_size;
    compute_work_sizes(count, impl_->device_info.max_work_group_size, local_size, global_size);
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_sqr, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0,
                        count * sizeof(FieldElement), results, 0, nullptr, nullptr);
    clReleaseMemObject(buf_a); clReleaseMemObject(buf_r);
}

void Context::batch_point_double(const JacobianPoint* inputs, JacobianPoint* results, std::size_t count) {
    if (count == 0) return;
    cl_int err;
    cl_mem buf_a = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(JacobianPoint), (void*)inputs, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   count * sizeof(JacobianPoint), nullptr, &err);
    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_point_double, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(impl_->kernel_point_double, 1, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_point_double, 2, sizeof(cl_uint), &cnt);
    std::size_t local_size, global_size;
    compute_work_sizes(count, impl_->device_info.max_work_group_size, local_size, global_size);
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_point_double, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0,
                        count * sizeof(JacobianPoint), results, 0, nullptr, nullptr);
    clReleaseMemObject(buf_a); clReleaseMemObject(buf_r);
}

void Context::batch_point_add(const JacobianPoint* p, const JacobianPoint* q,
                               JacobianPoint* results, std::size_t count) {
    if (count == 0) return;
    cl_int err;
    cl_mem buf_p = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(JacobianPoint), (void*)p, &err);
    cl_mem buf_q = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   count * sizeof(JacobianPoint), (void*)q, &err);
    cl_mem buf_r = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                   count * sizeof(JacobianPoint), nullptr, &err);
    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_point_add, 0, sizeof(cl_mem), &buf_p);
    clSetKernelArg(impl_->kernel_point_add, 1, sizeof(cl_mem), &buf_q);
    clSetKernelArg(impl_->kernel_point_add, 2, sizeof(cl_mem), &buf_r);
    clSetKernelArg(impl_->kernel_point_add, 3, sizeof(cl_uint), &cnt);
    std::size_t local_size, global_size;
    compute_work_sizes(count, impl_->device_info.max_work_group_size, local_size, global_size);
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_point_add, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_r, CL_TRUE, 0,
                        count * sizeof(JacobianPoint), results, 0, nullptr, nullptr);
    clReleaseMemObject(buf_p); clReleaseMemObject(buf_q); clReleaseMemObject(buf_r);
}

void Context::batch_scalar_mul_generator(const Scalar* scalars, JacobianPoint* results, std::size_t count) {
    if (count == 0) return;

    cl_int err;

    cl_mem buf_scalars = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         count * sizeof(Scalar), (void*)scalars, &err);
    cl_mem buf_results = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                         count * sizeof(JacobianPoint), nullptr, &err);

    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_scalar_mul_generator, 0, sizeof(cl_mem), &buf_scalars);
    clSetKernelArg(impl_->kernel_scalar_mul_generator, 1, sizeof(cl_mem), &buf_results);
    clSetKernelArg(impl_->kernel_scalar_mul_generator, 2, sizeof(cl_uint), &cnt);

    // Calculate work group size
    std::size_t local_size = impl_->config.local_work_size;
    if (local_size == 0 || local_size > impl_->device_info.max_work_group_size) {
        local_size = std::min(static_cast<std::size_t>(256), impl_->device_info.max_work_group_size);
    }

    std::size_t global_size = ((count + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_scalar_mul_generator, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_results, CL_TRUE, 0,
                        count * sizeof(JacobianPoint), results, 0, nullptr, nullptr);

    clReleaseMemObject(buf_scalars);
    clReleaseMemObject(buf_results);
}

void Context::batch_scalar_mul(const Scalar* scalars, const AffinePoint* points,
                                JacobianPoint* results, std::size_t count) {
    if (count == 0) return;

    cl_int err;

    cl_mem buf_scalars = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         count * sizeof(Scalar), (void*)scalars, &err);
    cl_mem buf_points = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        count * sizeof(AffinePoint), (void*)points, &err);
    cl_mem buf_results = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                         count * sizeof(JacobianPoint), nullptr, &err);

    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_scalar_mul, 0, sizeof(cl_mem), &buf_scalars);
    clSetKernelArg(impl_->kernel_scalar_mul, 1, sizeof(cl_mem), &buf_points);
    clSetKernelArg(impl_->kernel_scalar_mul, 2, sizeof(cl_mem), &buf_results);
    clSetKernelArg(impl_->kernel_scalar_mul, 3, sizeof(cl_uint), &cnt);

    std::size_t local_size = std::min(static_cast<std::size_t>(256), impl_->device_info.max_work_group_size);
    std::size_t global_size = ((count + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_scalar_mul, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_results, CL_TRUE, 0,
                        count * sizeof(JacobianPoint), results, 0, nullptr, nullptr);

    clReleaseMemObject(buf_scalars);
    clReleaseMemObject(buf_points);
    clReleaseMemObject(buf_results);
}

void Context::batch_field_inv(const FieldElement* inputs, FieldElement* outputs, std::size_t count) {
    if (count == 0) return;

    cl_int err;

    cl_mem buf_in = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    count * sizeof(FieldElement), (void*)inputs, &err);
    cl_mem buf_out = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                     count * sizeof(FieldElement), nullptr, &err);

    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_field_inv, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(impl_->kernel_field_inv, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(impl_->kernel_field_inv, 2, sizeof(cl_uint), &cnt);

    std::size_t local_size = std::min(static_cast<std::size_t>(256), impl_->device_info.max_work_group_size);
    std::size_t global_size = ((count + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_field_inv, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_out, CL_TRUE, 0,
                        count * sizeof(FieldElement), outputs, 0, nullptr, nullptr);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
}

void Context::batch_jacobian_to_affine(const JacobianPoint* jacobians, AffinePoint* affines, std::size_t count) {
    if (count == 0) return;

    cl_int err;

    cl_mem buf_jac = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     count * sizeof(JacobianPoint), (void*)jacobians, &err);
    cl_mem buf_aff = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY,
                                     count * sizeof(AffinePoint), nullptr, &err);

    cl_uint cnt = static_cast<cl_uint>(count);
    clSetKernelArg(impl_->kernel_batch_jacobian_to_affine, 0, sizeof(cl_mem), &buf_jac);
    clSetKernelArg(impl_->kernel_batch_jacobian_to_affine, 1, sizeof(cl_mem), &buf_aff);
    clSetKernelArg(impl_->kernel_batch_jacobian_to_affine, 2, sizeof(cl_uint), &cnt);

    std::size_t local_size = std::min(static_cast<std::size_t>(256), impl_->device_info.max_work_group_size);
    std::size_t global_size = ((count + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel_batch_jacobian_to_affine, 1, nullptr,
                           &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->queue, buf_aff, CL_TRUE, 0,
                        count * sizeof(AffinePoint), affines, 0, nullptr, nullptr);

    clReleaseMemObject(buf_jac);
    clReleaseMemObject(buf_aff);
}

void Context::async_batch_scalar_mul_generator(const Scalar* scalars, JacobianPoint* results, std::size_t count) {
    // For now, just call sync version
    batch_scalar_mul_generator(scalars, results, count);
}

void Context::sync() {
    clFinish(impl_->queue);
}

void Context::flush() {
    clFinish(impl_->queue);
}

void* Context::native_context() const {
    return impl_->context;
}

void* Context::native_queue() const {
    return impl_->queue;
}

void* Context::native_kernel(const char* name) const {
    std::string n(name);
    if (n == "field_add") return impl_->kernel_field_add;
    if (n == "field_sub") return impl_->kernel_field_sub;
    if (n == "field_mul") return impl_->kernel_field_mul;
    if (n == "field_sqr") return impl_->kernel_field_sqr;
    if (n == "field_inv") return impl_->kernel_field_inv;
    if (n == "point_double") return impl_->kernel_point_double;
    if (n == "point_add") return impl_->kernel_point_add;
    if (n == "scalar_mul") return impl_->kernel_scalar_mul;
    if (n == "scalar_mul_generator") return impl_->kernel_scalar_mul_generator;
    if (n == "affine_add") return impl_->kernel_affine_add;
    if (n == "affine_add_lambda") return impl_->kernel_affine_add_lambda;
    if (n == "affine_add_x_only") return impl_->kernel_affine_add_x_only;
    if (n == "jacobian_to_affine") return impl_->kernel_jacobian_to_affine;
    return nullptr;
}

std::unique_ptr<Buffer> Context::allocate(std::size_t size) {
    auto buffer = std::unique_ptr<Buffer>(new Buffer());
    cl_int err;
    buffer->impl_->buffer = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE, size, nullptr, &err);
    buffer->impl_->size = size;
    return buffer;
}

void Context::upload(Buffer& buffer, const void* data, std::size_t size) {
    clEnqueueWriteBuffer(impl_->queue, static_cast<cl_mem>(buffer.native_handle()),
                         CL_TRUE, 0, size, data, 0, nullptr, nullptr);
}

void Context::download(const Buffer& buffer, void* data, std::size_t size) {
    clEnqueueReadBuffer(impl_->queue, static_cast<cl_mem>(buffer.native_handle()),
                        CL_TRUE, 0, size, data, 0, nullptr, nullptr);
}

// =============================================================================
// Utility Functions
// =============================================================================

std::vector<std::pair<std::string, std::vector<DeviceInfo>>> enumerate_devices() {
    std::vector<std::pair<std::string, std::vector<DeviceInfo>>> result;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return result;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);

        std::vector<DeviceInfo> devices;

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);

        if (num_devices > 0) {
            std::vector<cl_device_id> device_ids(num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr);

            for (cl_uint j = 0; j < num_devices; j++) {
                DeviceInfo info;
                char buffer[256];

                clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
                info.name = buffer;

                clGetDeviceInfo(device_ids[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr);
                info.vendor = buffer;

                clGetDeviceInfo(device_ids[j], CL_DEVICE_VERSION, sizeof(buffer), buffer, nullptr);
                info.version = buffer;

                clGetDeviceInfo(device_ids[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info.global_mem_size), &info.global_mem_size, nullptr);
                clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.compute_units), &info.compute_units, nullptr);

                info.is_intel = (info.vendor.find("Intel") != std::string::npos);
                info.is_amd = (info.vendor.find("AMD") != std::string::npos);
                info.is_nvidia = (info.vendor.find("NVIDIA") != std::string::npos);

                devices.push_back(info);
            }
        }

        result.emplace_back(platform_name, devices);
    }

    return result;
}

AffinePoint get_generator() {
    AffinePoint g;
    g.x.limbs[0] = 0x59F2815B16F81798ULL;
    g.x.limbs[1] = 0x029BFCDB2DCE28D9ULL;
    g.x.limbs[2] = 0x55A06295CE870B07ULL;
    g.x.limbs[3] = 0x79BE667EF9DCBBACULL;
    g.y.limbs[0] = 0x9C47D08FFB10D4B8ULL;
    g.y.limbs[1] = 0xFD17B448A6855419ULL;
    g.y.limbs[2] = 0x5DA4FBFC0E1108A8ULL;
    g.y.limbs[3] = 0x483ADA7726A3C465ULL;
    return g;
}

// Field prime p = 2^256 - 0x1000003D1
static const std::uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// MSVC-compatible 128-bit multiplication helpers
#ifdef _MSC_VER
#include <intrin.h>

static inline void mul64_host(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
    lo = _umul128(a, b, &hi);
}

static inline std::uint64_t add64_host(std::uint64_t a, std::uint64_t b, std::uint64_t carry_in, std::uint64_t& carry_out) {
    std::uint64_t sum = a + b;
    std::uint64_t c1 = (sum < a) ? 1 : 0;
    sum += carry_in;
    std::uint64_t c2 = (sum < carry_in) ? 1 : 0;
    carry_out = c1 + c2;
    return sum;
}

#else
// GCC/Clang with __int128
static inline void mul64_host(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
    unsigned __int128 r = static_cast<unsigned __int128>(a) * b;
    lo = static_cast<std::uint64_t>(r);
    hi = static_cast<std::uint64_t>(r >> 64);
}

static inline std::uint64_t add64_host(std::uint64_t a, std::uint64_t b, std::uint64_t carry_in, std::uint64_t& carry_out) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + carry_in;
    carry_out = static_cast<std::uint64_t>(sum >> 64);
    return static_cast<std::uint64_t>(sum);
}
#endif

// Helper: field multiplication (host-side) - MSVC compatible
static void field_mul_host(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    // Schoolbook multiplication with reduction
    std::uint64_t product[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        std::uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            std::uint64_t lo, hi;
            mul64_host(a.limbs[i], b.limbs[j], lo, hi);

            // Accumulate: product[i+j] += lo + carry, propagate overflow
            std::uint64_t c1 = 0, c2 = 0;
            product[i + j] = add64_host(product[i + j], lo, 0, c1);
            product[i + j] = add64_host(product[i + j], carry, 0, c2);
            carry = hi + c1 + c2;
        }
        product[i + 4] += carry;
    }

    // Reduction using secp256k1's special form: 2^256 = K (mod p), K = 0x1000003D1
    const std::uint64_t K = 0x1000003D1ULL;

    // First pass: fold high limbs [4..7] into [0..3] via K multiplication
    std::uint64_t temp[5];
    temp[0] = product[0]; temp[1] = product[1]; temp[2] = product[2]; temp[3] = product[3]; temp[4] = 0;

    for (int i = 4; i < 8; i++) {
        if (product[i] == 0) continue;
        std::uint64_t mul_lo, mul_hi;
        mul64_host(product[i], K, mul_lo, mul_hi);
        std::uint64_t c = 0;
        temp[i - 4] = add64_host(temp[i - 4], mul_lo, 0, c);
        if (i - 3 < 5) temp[i - 3] = add64_host(temp[i - 3], mul_hi, c, c);
        // Propagate remaining carry
        for (int k = i - 2; k < 5 && c != 0; k++) {
            std::uint64_t old = temp[k];
            temp[k] += c;
            c = (temp[k] < old) ? 1 : 0;
        }
    }

    // Second reduction if temp[4] > 0 (result overflowed 256 bits)
    while (temp[4] > 0) {
        std::uint64_t overflow = temp[4];
        temp[4] = 0;
        std::uint64_t mul_lo, mul_hi;
        mul64_host(overflow, K, mul_lo, mul_hi);
        std::uint64_t c = 0;
        temp[0] = add64_host(temp[0], mul_lo, 0, c);
        temp[1] = add64_host(temp[1], mul_hi, c, c);
        temp[2] = add64_host(temp[2], 0, c, c);
        temp[3] = add64_host(temp[3], 0, c, c);
        temp[4] += c;
    }

    r.limbs[0] = temp[0];
    r.limbs[1] = temp[1];
    r.limbs[2] = temp[2];
    r.limbs[3] = temp[3];

    // Final reduction: branchless subtract p if result >= p
    std::uint64_t borrow = 0;
    std::uint64_t diff[4];
    for (int i = 0; i < 4; i++) {
        std::uint64_t d = r.limbs[i] - SECP256K1_P[i] - borrow;
        borrow = (r.limbs[i] < SECP256K1_P[i] + borrow || (borrow && SECP256K1_P[i] == 0xFFFFFFFFFFFFFFFFULL)) ? 1 : 0;
        diff[i] = d;
    }
    // Branchless: if no borrow, use diff (result was >= p)
    std::uint64_t mask = borrow ? 0 : ~static_cast<std::uint64_t>(0);
    r.limbs[0] = (diff[0] & mask) | (r.limbs[0] & ~mask);
    r.limbs[1] = (diff[1] & mask) | (r.limbs[1] & ~mask);
    r.limbs[2] = (diff[2] & mask) | (r.limbs[2] & ~mask);
    r.limbs[3] = (diff[3] & mask) | (r.limbs[3] & ~mask);
}

// Helper: field squaring (host-side)
static void field_sqr_host(FieldElement& r, const FieldElement& a) {
    field_mul_host(r, a, a);
}

// Helper: field inversion using Fermat (a^(p-2) mod p)
static void field_inv_host(FieldElement& r, const FieldElement& a) {
    // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    FieldElement result = FieldElement::one();
    FieldElement base = a;

    const std::uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    for (int limb = 0; limb < 4; limb++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[limb] >> bit) & 1) {
                field_mul_host(result, result, base);
            }
            field_sqr_host(base, base);
        }
    }
    r = result;
}

AffinePoint jacobian_to_affine(const JacobianPoint& p) {
    AffinePoint result;

    // Check for infinity
    if (p.infinity ||
        (p.z.limbs[0] == 0 && p.z.limbs[1] == 0 &&
         p.z.limbs[2] == 0 && p.z.limbs[3] == 0)) {
        result.x = FieldElement::zero();
        result.y = FieldElement::zero();
        return result;
    }

    // If Z = 1, just copy X and Y
    if (p.z.limbs[0] == 1 && p.z.limbs[1] == 0 &&
        p.z.limbs[2] == 0 && p.z.limbs[3] == 0) {
        result.x = p.x;
        result.y = p.y;
        return result;
    }

    // Compute Z^(-1)
    FieldElement z_inv;
    field_inv_host(z_inv, p.z);

    // z_inv_2 = Z^(-2)
    FieldElement z_inv_2;
    field_sqr_host(z_inv_2, z_inv);

    // z_inv_3 = Z^(-3)
    FieldElement z_inv_3;
    field_mul_host(z_inv_3, z_inv_2, z_inv);

    // x = X * Z^(-2)
    field_mul_host(result.x, p.x, z_inv_2);

    // y = Y * Z^(-3)
    field_mul_host(result.y, p.y, z_inv_3);

    return result;
}

JacobianPoint affine_to_jacobian(const AffinePoint& p) {
    JacobianPoint result;
    result.x = p.x;
    result.y = p.y;
    result.z = FieldElement::one();
    result.infinity = 0;
    return result;
}

Scalar scalar_from_u64(std::uint64_t value) {
    return Scalar{{value, 0, 0, 0}};
}

FieldElement field_from_u64(std::uint64_t value) {
    return FieldElement{{value, 0, 0, 0}};
}

} // namespace opencl
} // namespace secp256k1

