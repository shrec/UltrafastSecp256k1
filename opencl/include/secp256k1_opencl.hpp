// =============================================================================
// UltrafastSecp256k1 OpenCL - Main Header
// =============================================================================
// Cross-platform GPU acceleration for secp256k1 operations
// Supports Intel, AMD, NVIDIA GPUs via OpenCL 1.2+
// Zero external dependencies beyond OpenCL runtime
// =============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <optional>
#include <functional>

namespace secp256k1 {
namespace opencl {

// =============================================================================
// Forward Declarations
// =============================================================================

class Context;
class Program;
class Kernel;
class Buffer;

// =============================================================================
// Configuration
// =============================================================================

struct DeviceConfig {
    int device_id = 0;              // GPU device index
    int platform_id = 0;            // Platform index (e.g., Intel, AMD)
    std::size_t max_batch_size = 65536;  // Max points per batch
    std::size_t local_work_size = 256;   // Work group size (auto if 0)
    bool prefer_intel = true;       // Prefer Intel GPU if available
    bool verbose = false;           // Print device info on init
};

// =============================================================================
// Data Types (GPU-compatible, same layout as CPU)
// =============================================================================

// Field element: 256-bit integer mod p
// Little-endian: limbs[0] is least significant
struct alignas(32) FieldElement {
    std::uint64_t limbs[4];

    bool operator==(const FieldElement& rhs) const noexcept {
        return limbs[0] == rhs.limbs[0] && limbs[1] == rhs.limbs[1] &&
               limbs[2] == rhs.limbs[2] && limbs[3] == rhs.limbs[3];
    }

    static FieldElement zero() noexcept {
        return {{0, 0, 0, 0}};
    }

    static FieldElement one() noexcept {
        return {{1, 0, 0, 0}};
    }
};

// 32-bit view of field element (for optimized operations)
struct alignas(32) MidFieldElement {
    std::uint32_t limbs[8];
};

// Scalar: 256-bit integer mod n (curve order)
struct alignas(32) Scalar {
    std::uint64_t limbs[4];

    static Scalar zero() noexcept {
        return {{0, 0, 0, 0}};
    }

    static Scalar one() noexcept {
        return {{1, 0, 0, 0}};
    }
};

// Affine point (x, y)
struct alignas(64) AffinePoint {
    FieldElement x;
    FieldElement y;
};

// Jacobian point (X, Y, Z) where affine (x, y) = (X/Z², Y/Z³)
struct alignas(128) JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    std::uint32_t infinity;  // 1 if point at infinity, 0 otherwise
    std::uint32_t padding[3]; // Alignment padding
};

// =============================================================================
// Device Information
// =============================================================================

struct DeviceInfo {
    std::string name;
    std::string vendor;
    std::string version;
    std::string driver_version;
    std::size_t global_mem_size;     // bytes
    std::size_t local_mem_size;      // bytes
    std::size_t max_work_group_size;
    std::uint32_t compute_units;
    std::uint32_t max_clock_freq;    // MHz
    bool supports_double;
    bool is_intel;
    bool is_amd;
    bool is_nvidia;
};

// =============================================================================
// Context (Manages OpenCL State)
// =============================================================================

class Context {
public:
    // Create context with configuration
    static std::unique_ptr<Context> create(const DeviceConfig& config = {});

    // Destructor
    ~Context();

    // Non-copyable, moveable
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    // Device information
    const DeviceInfo& device_info() const noexcept;

    // Check if context is valid
    bool is_valid() const noexcept;

    // Get last error message
    const std::string& last_error() const noexcept;

    // ==========================================================================
    // Single Operations (for testing/verification)
    // ==========================================================================

    // Field operations
    FieldElement field_add(const FieldElement& a, const FieldElement& b);
    FieldElement field_sub(const FieldElement& a, const FieldElement& b);
    FieldElement field_mul(const FieldElement& a, const FieldElement& b);
    FieldElement field_sqr(const FieldElement& a);
    FieldElement field_inv(const FieldElement& a);

    // Point operations
    JacobianPoint point_double(const JacobianPoint& p);
    JacobianPoint point_add(const JacobianPoint& p, const JacobianPoint& q);
    JacobianPoint scalar_mul(const Scalar& k, const AffinePoint& p);
    JacobianPoint scalar_mul_generator(const Scalar& k);

    // ==========================================================================
    // Batch Operations (High Performance)
    // ==========================================================================

    // Batch field operations: result[i] = a[i] op b[i]
    void batch_field_add(const FieldElement* a, const FieldElement* b, FieldElement* results, std::size_t count);
    void batch_field_sub(const FieldElement* a, const FieldElement* b, FieldElement* results, std::size_t count);
    void batch_field_mul(const FieldElement* a, const FieldElement* b, FieldElement* results, std::size_t count);
    void batch_field_sqr(const FieldElement* inputs, FieldElement* results, std::size_t count);

    // Batch point operations
    void batch_point_double(const JacobianPoint* inputs, JacobianPoint* results, std::size_t count);
    void batch_point_add(const JacobianPoint* p, const JacobianPoint* q, JacobianPoint* results, std::size_t count);

    // Batch scalar multiplication: result[i] = scalars[i] * G
    void batch_scalar_mul_generator(
        const Scalar* scalars,
        JacobianPoint* results,
        std::size_t count
    );

    // Batch scalar multiplication: result[i] = scalars[i] * points[i]
    void batch_scalar_mul(
        const Scalar* scalars,
        const AffinePoint* points,
        JacobianPoint* results,
        std::size_t count
    );

    // Batch field inversion: result[i] = a[i]^(-1)
    void batch_field_inv(
        const FieldElement* inputs,
        FieldElement* outputs,
        std::size_t count
    );

    // Batch Jacobian to Affine conversion
    void batch_jacobian_to_affine(
        const JacobianPoint* jacobians,
        AffinePoint* affines,
        std::size_t count
    );

    // ==========================================================================
    // Asynchronous Operations
    // ==========================================================================

    // Start async batch operation, returns immediately
    // Call wait() or sync() to get results
    void async_batch_scalar_mul_generator(
        const Scalar* scalars,
        JacobianPoint* results,
        std::size_t count
    );

    // Wait for all pending operations to complete
    void sync();

    // Flush command queue (clFinish)
    void flush();

    // Native handle access (for benchmarking / advanced use)
    void* native_context() const;
    void* native_queue() const;
    void* native_kernel(const char* name) const;

    // ==========================================================================
    // Memory Management
    // ==========================================================================

    // Allocate GPU buffer
    std::unique_ptr<Buffer> allocate(std::size_t size);

    // Upload data to GPU
    void upload(Buffer& buffer, const void* data, std::size_t size);

    // Download data from GPU
    void download(const Buffer& buffer, void* data, std::size_t size);

private:
    Context();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Buffer (GPU Memory)
// =============================================================================

class Buffer {
public:
    ~Buffer();

    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    std::size_t size() const noexcept;
    bool is_valid() const noexcept;

    // Get raw OpenCL handle (for advanced use)
    void* native_handle() const noexcept;

private:
    friend class Context;
    Buffer();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Utility Functions
// =============================================================================

// List available OpenCL platforms and devices
std::vector<std::pair<std::string, std::vector<DeviceInfo>>> enumerate_devices();

// Get generator point G
AffinePoint get_generator();

// Convert between formats
AffinePoint jacobian_to_affine(const JacobianPoint& p);
JacobianPoint affine_to_jacobian(const AffinePoint& p);

// Create scalar from uint64_t
Scalar scalar_from_u64(std::uint64_t value);

// Create field element from uint64_t
FieldElement field_from_u64(std::uint64_t value);

// Self-test: returns true if all tests pass
bool selftest(bool verbose = false, int platform_id = -1, int device_id = 0);

// =============================================================================
// Error Handling
// =============================================================================

// Get OpenCL error string
const char* cl_error_string(int error_code);

} // namespace opencl
} // namespace secp256k1

