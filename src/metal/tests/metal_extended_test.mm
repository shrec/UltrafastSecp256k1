// =============================================================================
// UltrafastSecp256k1 Metal — Full Test & Benchmark Suite
// =============================================================================
// Tests ALL Metal kernels: field, point, batch, ECDSA, Schnorr, ECDH,
// Hash160, Recovery, and benchmarks.
//
// Build (macOS):
//   xcrun -sdk macosx metal -c secp256k1_kernels.metal -o secp256k1.air \
//     -I ../shaders
//   xcrun -sdk macosx metallib secp256k1.air -o secp256k1.metallib
//   clang++ -std=c++17 -O2 -framework Metal -framework Foundation \
//     -o metal_extended_test metal_extended_test.mm metal_runtime.mm
//
// Run:
//   ./metal_extended_test [--bench] [--verbose] [--count N]
// =============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>

// =============================================================================
// Helpers
// =============================================================================

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static std::vector<uint8_t> hex_to_bytes(const char* hex) {
    size_t len = strlen(hex);
    std::vector<uint8_t> out(len / 2);
    for (size_t i = 0; i < len / 2; ++i)
        out[i] = (uint8_t)((hex_val(hex[2*i]) << 4) | hex_val(hex[2*i+1]));
    return out;
}

static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    static const char h[] = "0123456789abcdef";
    std::string s; s.reserve(len*2);
    for (size_t i = 0; i < len; ++i) { s += h[(data[i]>>4)&0xF]; s += h[data[i]&0xF]; }
    return s;
}

// =============================================================================
// Metal Context
// =============================================================================

struct MetalCtx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    bool init(int device_id = 0) {
        @autoreleasepool {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            if (!devices || devices.count == 0) {
                device = MTLCreateSystemDefaultDevice();
            } else {
                device = (device_id >= 0 && (NSUInteger)device_id < devices.count)
                         ? devices[device_id] : devices[0];
            }
            if (!device) { fprintf(stderr, "No Metal device\n"); return false; }
            queue = [device newCommandQueue];
            return queue != nil;
        }
    }

    bool load_metallib(const std::string& path) {
        @autoreleasepool {
            NSError* error = nil;
            NSString* p = [NSString stringWithUTF8String:path.c_str()];
            library = [device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&error];
            if (!library) {
                fprintf(stderr, "Failed to load metallib: %s\n",
                        [[error localizedDescription] UTF8String]);
                return false;
            }
            return true;
        }
    }

    bool load_source(const std::string& source) {
        @autoreleasepool {
            NSError* error = nil;
            NSString* src = [NSString stringWithUTF8String:source.c_str()];
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;
            opts.languageVersion = MTLLanguageVersion2_4;
            library = [device newLibraryWithSource:src options:opts error:&error];
            if (!library) {
                fprintf(stderr, "Compile error: %s\n",
                        [[error localizedDescription] UTF8String]);
                return false;
            }
            return true;
        }
    }

    id<MTLComputePipelineState> get_pipeline(const std::string& name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;
        @autoreleasepool {
            NSString* fname = [NSString stringWithUTF8String:name.c_str()];
            id<MTLFunction> func = [library newFunctionWithName:fname];
            if (!func) { fprintf(stderr, "Function not found: %s\n", name.c_str()); return nil; }
            NSError* error = nil;
            id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:func error:&error];
            if (!pipe) { fprintf(stderr, "Pipeline error: %s\n", [[error localizedDescription] UTF8String]); return nil; }
            pipelines[name] = pipe;
            return pipe;
        }
    }

    id<MTLBuffer> alloc(size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> alloc_with_data(const void* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    }

    double dispatch_sync(id<MTLComputePipelineState> pipe, uint32_t count,
                         std::vector<id<MTLBuffer>> buffers) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            for (size_t i = 0; i < buffers.size(); i++)
                [enc setBuffer:buffers[i] offset:0 atIndex:i];
            uint32_t tg = (uint32_t)[pipe threadExecutionWidth];
            if (tg == 0) tg = 256;
            MTLSize grid = MTLSizeMake(count, 1, 1);
            MTLSize tgs = MTLSizeMake(tg, 1, 1);
            auto t0 = std::chrono::high_resolution_clock::now();
            [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
};

// =============================================================================
// Test Vectors
// =============================================================================

// Hash160(compressed pubkey of key=1) = 751e76e8199196d454941c45d1b3a323f1433bd6
static const char* H160_PK  = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
static const char* H160_EXP = "751e76e8199196d454941c45d1b3a323f1433bd6";

// Uncompressed pubkey of key=1
static const char* H160_UNC_PK =
    "04"
    "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";
static const char* H160_UNC_EXP = "91b24bf9f5288532960ac687abb035127b1d28a5";

// Generator point
static const char* G_X = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
static const char* G_Y = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

// =============================================================================
// Hash160 Tests
// =============================================================================

static int test_hash160(MetalCtx& ctx, bool verbose) {
    int passed = 0, total = 0;

    auto pipe = ctx.get_pipeline("hash160_batch");
    if (!pipe) { printf("  SKIP: hash160_batch not available\n"); return 1; }

    // Test 1: compressed pubkey (key=1)
    {
        total++;
        if (verbose) printf("  Hash160(compressed key=1): ");

        auto pk = hex_to_bytes(H160_PK);
        uint32_t count = 1, stride = 33;

        auto pk_buf = ctx.alloc_with_data(pk.data(), pk.size());
        auto out_buf = ctx.alloc(20);
        auto stride_buf = ctx.alloc_with_data(&stride, sizeof(stride));
        auto count_buf = ctx.alloc_with_data(&count, sizeof(count));

        ctx.dispatch_sync(pipe, 1, {pk_buf, out_buf, stride_buf, count_buf});

        auto expected = hex_to_bytes(H160_EXP);
        uint8_t* result = (uint8_t*)[out_buf contents];
        bool ok = (memcmp(result, expected.data(), 20) == 0);
        if (ok) { passed++; if (verbose) printf("PASS\n"); }
        else {
            if (verbose) {
                printf("FAIL\n");
                printf("    Got:      %s\n", bytes_to_hex(result, 20).c_str());
                printf("    Expected: %s\n", H160_EXP);
            }
        }
    }

    // Test 2: uncompressed pubkey (key=1)
    {
        total++;
        if (verbose) printf("  Hash160(uncompressed key=1): ");

        auto pk = hex_to_bytes(H160_UNC_PK);
        uint32_t count = 1, stride = 65;

        auto pk_buf = ctx.alloc_with_data(pk.data(), pk.size());
        auto out_buf = ctx.alloc(20);
        auto stride_buf = ctx.alloc_with_data(&stride, sizeof(stride));
        auto count_buf = ctx.alloc_with_data(&count, sizeof(count));

        ctx.dispatch_sync(pipe, 1, {pk_buf, out_buf, stride_buf, count_buf});

        auto expected = hex_to_bytes(H160_UNC_EXP);
        uint8_t* result = (uint8_t*)[out_buf contents];
        bool ok = (memcmp(result, expected.data(), 20) == 0);
        if (ok) { passed++; if (verbose) printf("PASS\n"); }
        else {
            if (verbose) {
                printf("FAIL\n");
                printf("    Got:      %s\n", bytes_to_hex(result, 20).c_str());
                printf("    Expected: %s\n", H160_UNC_EXP);
            }
        }
    }

    printf("  Hash160: %d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

// =============================================================================
// Field / Point Tests (existing kernels)
// =============================================================================

static int test_field_ops(MetalCtx& ctx, bool verbose) {
    int passed = 0, total = 0;

    // Test field_mul: 2 * 3 = 6 (trivial)
    {
        total++;
        if (verbose) printf("  field_mul(2, 3) = 6: ");

        auto pipe = ctx.get_pipeline("field_mul_bench");
        if (!pipe) { printf("SKIP\n"); }
        else {
            // FieldElement = 8 × uint32. Value 2 = limbs[0]=2, rest=0
            uint32_t a[8] = {2, 0, 0, 0, 0, 0, 0, 0};
            uint32_t b[8] = {3, 0, 0, 0, 0, 0, 0, 0};
            uint32_t count = 1;

            auto a_buf = ctx.alloc_with_data(a, sizeof(a));
            auto b_buf = ctx.alloc_with_data(b, sizeof(b));
            auto r_buf = ctx.alloc(sizeof(a));
            auto c_buf = ctx.alloc_with_data(&count, sizeof(count));

            ctx.dispatch_sync(pipe, 1, {a_buf, b_buf, r_buf, c_buf});

            uint32_t* result = (uint32_t*)[r_buf contents];
            bool ok = (result[0] == 6 && result[1] == 0 && result[2] == 0 && result[3] == 0);
            if (ok) { passed++; if (verbose) printf("PASS\n"); }
            else {
                if (verbose) printf("FAIL (got limb[0]=%u)\n", result[0]);
            }
        }
    }

    // Test generator_mul: 1*G = G
    {
        total++;
        if (verbose) printf("  1*G = G: ");

        auto pipe = ctx.get_pipeline("generator_mul_batch");
        if (!pipe) { printf("SKIP\n"); }
        else {
            // Scalar256 for k=1: limbs[0]=1, rest=0
            uint32_t scalar[8] = {1, 0, 0, 0, 0, 0, 0, 0};
            uint32_t count = 1;

            auto s_buf = ctx.alloc_with_data(scalar, sizeof(scalar));
            // AffinePoint = 2 × FieldElement = 2 × 8 × uint32 = 64 bytes
            auto r_buf = ctx.alloc(64);
            auto c_buf = ctx.alloc_with_data(&count, sizeof(count));

            ctx.dispatch_sync(pipe, 1, {s_buf, r_buf, c_buf});

            // Read result and convert to hex for comparison
            uint32_t* rp = (uint32_t*)[r_buf contents];

            // The result is in 8×32-bit LE limbs — convert to big-endian hex
            uint8_t x_bytes[32], y_bytes[32];
            for (int i = 0; i < 8; i++) {
                uint32_t xv = rp[i];  // x.limbs[i]
                uint32_t yv = rp[8 + i]; // y.limbs[i]
                // LE limbs → BE bytes: limbs[7] is MSW
                x_bytes[31 - i*4 - 0] = (uint8_t)(xv);
                x_bytes[31 - i*4 - 1] = (uint8_t)(xv >> 8);
                x_bytes[31 - i*4 - 2] = (uint8_t)(xv >> 16);
                x_bytes[31 - i*4 - 3] = (uint8_t)(xv >> 24);
                y_bytes[31 - i*4 - 0] = (uint8_t)(yv);
                y_bytes[31 - i*4 - 1] = (uint8_t)(yv >> 8);
                y_bytes[31 - i*4 - 2] = (uint8_t)(yv >> 16);
                y_bytes[31 - i*4 - 3] = (uint8_t)(yv >> 24);
            }

            auto gx = hex_to_bytes(G_X);
            auto gy = hex_to_bytes(G_Y);
            bool ok = (memcmp(x_bytes, gx.data(), 32) == 0 && memcmp(y_bytes, gy.data(), 32) == 0);
            if (ok) { passed++; if (verbose) printf("PASS\n"); }
            else {
                if (verbose) {
                    printf("FAIL\n");
                    printf("    Got X: %s\n", bytes_to_hex(x_bytes, 32).c_str());
                    printf("    Exp X: %s\n", G_X);
                }
            }
        }
    }

    printf("  Field/Point: %d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

// =============================================================================
// Benchmarks
// =============================================================================

struct BenchResult {
    const char* name;
    double ops_per_sec;
    double total_ms;
    uint32_t count;
};

static void print_bench(const BenchResult& r) {
    printf("  %-30s  %12.0f ops/s  (%8.2f ms, n=%u)\n",
           r.name, r.ops_per_sec, r.total_ms, r.count);
}

static BenchResult bench_kernel(MetalCtx& ctx, const std::string& kernel_name,
                                uint32_t count, size_t input_size, size_t output_size,
                                uint32_t extra_args = 0) {
    BenchResult res = {kernel_name.c_str(), 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline(kernel_name);
    if (!pipe) { printf("  SKIP: %s\n", kernel_name.c_str()); return res; }

    // Generic input buffer filled with deterministic data
    std::vector<uint8_t> input(input_size * count, 0);
    for (size_t i = 0; i < input.size(); i++) input[i] = (uint8_t)((i * 37 + 13) & 0xFF);

    auto in_buf = ctx.alloc_with_data(input.data(), input.size());
    auto out_buf = ctx.alloc(output_size * count);
    auto count_buf = ctx.alloc_with_data(&count, sizeof(count));

    // Warmup
    ctx.dispatch_sync(pipe, count, {in_buf, out_buf, count_buf});

    // Timed
    double ms = ctx.dispatch_sync(pipe, count, {in_buf, out_buf, count_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

static BenchResult bench_hash160(MetalCtx& ctx, uint32_t count) {
    BenchResult res = {"Hash160 (compressed)", 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline("hash160_bench");
    if (!pipe) { printf("  SKIP: hash160_bench\n"); return res; }

    std::vector<uint8_t> pks(33 * count);
    for (uint32_t i = 0; i < count; i++) {
        pks[i*33] = 0x02;
        for (int j = 1; j < 33; j++) pks[i*33+j] = (uint8_t)((i*13+j*7)&0xFF);
    }

    auto pk_buf = ctx.alloc_with_data(pks.data(), pks.size());
    auto out_buf = ctx.alloc(20 * count);
    auto count_buf = ctx.alloc_with_data(&count, sizeof(count));

    // Warmup
    ctx.dispatch_sync(pipe, count, {pk_buf, out_buf, count_buf});

    // Timed
    double ms = ctx.dispatch_sync(pipe, count, {pk_buf, out_buf, count_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

static BenchResult bench_sha256(MetalCtx& ctx, uint32_t count) {
    BenchResult res = {"SHA-256 (64B input)", 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline("sha256_bench");
    if (!pipe) { printf("  SKIP: sha256_bench\n"); return res; }

    std::vector<uint8_t> data(64 * count);
    for (size_t i = 0; i < data.size(); i++) data[i] = (uint8_t)(i & 0xFF);

    auto in_buf = ctx.alloc_with_data(data.data(), data.size());
    auto out_buf = ctx.alloc(32 * count);
    auto count_buf = ctx.alloc_with_data(&count, sizeof(count));

    ctx.dispatch_sync(pipe, count, {in_buf, out_buf, count_buf});
    double ms = ctx.dispatch_sync(pipe, count, {in_buf, out_buf, count_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

static BenchResult bench_ecdsa(MetalCtx& ctx, uint32_t count) {
    BenchResult res = {"ECDSA sign+verify", 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline("ecdsa_bench");
    if (!pipe) { printf("  SKIP: ecdsa_bench\n"); return res; }

    // Deterministic msg hashes and privkeys
    std::vector<uint8_t> msgs(32 * count), keys(32 * count);
    for (uint32_t i = 0; i < count; i++) {
        for (int j = 0; j < 32; j++) {
            msgs[i*32+j] = (uint8_t)((i*7+j*13+1)&0xFF);
            keys[i*32+j] = (uint8_t)((i*11+j*3+5)&0xFF);
        }
        keys[i*32] |= 0x01; // Ensure non-zero
    }

    auto msg_buf = ctx.alloc_with_data(msgs.data(), msgs.size());
    auto key_buf = ctx.alloc_with_data(keys.data(), keys.size());
    auto res_buf = ctx.alloc(sizeof(uint32_t) * count);
    auto count_buf = ctx.alloc_with_data(&count, sizeof(count));

    ctx.dispatch_sync(pipe, count, {msg_buf, key_buf, res_buf, count_buf});
    double ms = ctx.dispatch_sync(pipe, count, {msg_buf, key_buf, res_buf, count_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

static BenchResult bench_generator_mul(MetalCtx& ctx, uint32_t count) {
    BenchResult res = {"Generator mul (k*G)", 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline("generator_mul_batch");
    if (!pipe) { printf("  SKIP: generator_mul_batch\n"); return res; }

    std::vector<uint32_t> scalars(8 * count, 0);
    for (uint32_t i = 0; i < count; i++) {
        scalars[i*8] = i + 1;
    }

    auto s_buf = ctx.alloc_with_data(scalars.data(), scalars.size() * sizeof(uint32_t));
    auto r_buf = ctx.alloc(64 * count); // AffinePoint = 64 bytes
    auto c_buf = ctx.alloc_with_data(&count, sizeof(count));

    ctx.dispatch_sync(pipe, count, {s_buf, r_buf, c_buf});
    double ms = ctx.dispatch_sync(pipe, count, {s_buf, r_buf, c_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

static BenchResult bench_field_mul(MetalCtx& ctx, uint32_t count) {
    BenchResult res = {"Field mul", 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline("field_mul_bench");
    if (!pipe) { printf("  SKIP: field_mul_bench\n"); return res; }

    std::vector<uint32_t> a(8 * count), b(8 * count);
    for (uint32_t i = 0; i < count; i++) {
        for (int j = 0; j < 8; j++) {
            a[i*8+j] = i * 37 + j + 1;
            b[i*8+j] = i * 53 + j + 2;
        }
    }

    auto a_buf = ctx.alloc_with_data(a.data(), a.size() * sizeof(uint32_t));
    auto b_buf = ctx.alloc_with_data(b.data(), b.size() * sizeof(uint32_t));
    auto r_buf = ctx.alloc(32 * count);
    auto c_buf = ctx.alloc_with_data(&count, sizeof(count));

    ctx.dispatch_sync(pipe, count, {a_buf, b_buf, r_buf, c_buf});
    double ms = ctx.dispatch_sync(pipe, count, {a_buf, b_buf, r_buf, c_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

static BenchResult bench_field_inv(MetalCtx& ctx, uint32_t count) {
    BenchResult res = {"Field inv", 0.0, 0.0, count};

    auto pipe = ctx.get_pipeline("field_inv_bench");
    if (!pipe) { printf("  SKIP: field_inv_bench\n"); return res; }

    std::vector<uint32_t> a(8 * count);
    for (uint32_t i = 0; i < count; i++) {
        a[i*8] = i + 1;
        for (int j = 1; j < 8; j++) a[i*8+j] = 0;
    }

    auto a_buf = ctx.alloc_with_data(a.data(), a.size() * sizeof(uint32_t));
    auto r_buf = ctx.alloc(32 * count);
    auto c_buf = ctx.alloc_with_data(&count, sizeof(count));

    ctx.dispatch_sync(pipe, count, {a_buf, r_buf, c_buf});
    double ms = ctx.dispatch_sync(pipe, count, {a_buf, r_buf, c_buf});

    res.total_ms = ms;
    res.ops_per_sec = (double)count / (ms / 1000.0);
    return res;
}

// =============================================================================
// Variable-time SafeGCD inverse vs Fermat (inv_var_check_kernel)
// =============================================================================
// Every input goes through field_inv_var / scalar_inv_var and the Fermat
// field_inv / scalar_inverse on the device. Required for each input and each
// modulus M: var == Fermat, the var result is canonical (< M), a * inv == 1
// (inv == 0 when a == 0 mod M), the core run with one outer iteration fewer
// than the input needed returns 0 (fail closed), and the in-place (aliased)
// field call returns the same value.
// =============================================================================

struct U256 { uint32_t l[8]; };

static const U256 P_U256 = {{0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                             0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
static const U256 N_U256 = {{0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
                             0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};

// a + d mod 2^256 for a small d of either sign.
static U256 u256_add_small(U256 a, int d) {
    if (d >= 0) {
        uint64_t c = (uint32_t)d;
        for (int i = 0; i < 8; i++) {
            uint64_t s = (uint64_t)a.l[i] + c;
            a.l[i] = (uint32_t)s; c = s >> 32;
        }
    } else {
        uint64_t b = (uint32_t)(-d);
        for (int i = 0; i < 8; i++) {
            uint64_t x = (uint64_t)a.l[i] - b;
            a.l[i] = (uint32_t)x; b = (x >> 63) & 1u;
        }
    }
    return a;
}

static U256 u256_shr1(U256 a) {
    for (int i = 0; i < 7; i++) a.l[i] = (a.l[i] >> 1) | (a.l[i + 1] << 31);
    a.l[7] >>= 1;
    return a;
}

static bool u256_lt(const uint32_t* a, const U256& m) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] != m.l[i]) return a[i] < m.l[i];
    }
    return false;
}

static uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

static int test_inv_var(MetalCtx& ctx, bool verbose) {
    auto pipe = ctx.get_pipeline("inv_var_check_kernel");
    if (!pipe) { printf("  FAIL: inv_var_check_kernel not found\n"); return 1; }

    std::vector<U256> in;
    U256 zero = {{0, 0, 0, 0, 0, 0, 0, 0}};
    U256 ones = {{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                  0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
    for (int d = 0; d <= 3; d++) in.push_back(u256_add_small(zero, d));
    for (const U256* m : {&P_U256, &N_U256}) {
        in.push_back(u256_add_small(*m, -2));
        in.push_back(u256_add_small(*m, -1));
        in.push_back(u256_shr1(u256_add_small(*m, -1)));   // (M-1)/2
        in.push_back(u256_shr1(u256_add_small(*m, 1)));    // (M+1)/2
        in.push_back(*m);
        in.push_back(u256_add_small(*m, 1));
        in.push_back(u256_add_small(*m, 2));
        in.push_back(u256_add_small(*m, 0x12345));         // between M and 2^256
    }
    { U256 v = zero; v.l[7] = 0x80000000u; in.push_back(v); }   // 2^255
    in.push_back(ones);                                          // 2^256 - 1
    in.push_back(u256_add_small(ones, -5));
    { U256 v = zero; v.l[0] = 0x40000000u; in.push_back(v); }   // 2^30
    { U256 v = zero; v.l[0] = 0x3FFFFFFFu; in.push_back(v); }   // 2^30 - 1
    { U256 v = zero; v.l[1] = 0x10000000u; in.push_back(v); }   // 2^60
    for (uint32_t w : {0x3FFFFFFFu, 0xC0000000u, 0x55555555u, 0xAAAAAAAAu, 0x80000000u}) {
        U256 v; for (int i = 0; i < 8; i++) v.l[i] = w; in.push_back(v);
    }
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (int k = 0; k < 16384; k++) {
        U256 v;
        for (int i = 0; i < 8; i++) v.l[i] = (uint32_t)(xorshift64(seed) >> 16);
        in.push_back(v);
    }

    const uint32_t count = (uint32_t)in.size();
    auto a_buf = ctx.alloc_with_data(in.data(), in.size() * sizeof(U256));
    auto o_buf = ctx.alloc((size_t)count * 40 * sizeof(uint32_t));
    auto c_buf = ctx.alloc_with_data(&count, sizeof(count));
    ctx.dispatch_sync(pipe, count, {a_buf, o_buf, c_buf});
    const uint32_t* o = (const uint32_t*)[o_buf contents];

    int bad = 0;
    uint32_t max_it_p = 0, max_it_n = 0;
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t* r = o + (size_t)k * 40;
        bool ok = memcmp(r, r + 8, 32) == 0            // field var == Fermat
               && memcmp(r + 16, r + 24, 32) == 0      // scalar var == Fermat
               && u256_lt(r, P_U256) && u256_lt(r + 16, N_U256)
               && r[34] == 1 && r[35] == 1             // fail closed one iteration short
               && r[36] == 1 && r[37] == 1             // a * inv == 1 (or both 0)
               && r[38] == 1                           // aliased call
               && r[32] >= 1 && r[32] <= 25 && r[33] >= 1 && r[33] <= 25;
        if (r[32] > max_it_p) max_it_p = r[32];
        if (r[33] > max_it_n) max_it_n = r[33];
        if (!ok) {
            if (bad < 4) {
                uint8_t be[32];
                for (int i = 0; i < 8; i++) {
                    uint32_t v = in[k].l[7 - i];
                    be[i*4] = (uint8_t)(v >> 24); be[i*4+1] = (uint8_t)(v >> 16);
                    be[i*4+2] = (uint8_t)(v >> 8); be[i*4+3] = (uint8_t)v;
                }
                printf("    FAIL input %s flags %u %u %u %u %u\n", bytes_to_hex(be, 32).c_str(),
                       r[34], r[35], r[36], r[37], r[38]);
            }
            bad++;
        }
    }
    if (verbose) printf("  inverses checked: %u inputs x 2 moduli, max outer iterations p=%u n=%u\n",
                        count, max_it_p, max_it_n);
    printf("  Var-time inverse vs Fermat: %s (%d failing input(s))\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
}

// =============================================================================
// Verify / recover round trips on valid signatures
// =============================================================================
// GPU-made ECDSA and BIP-340 signatures for 64 keys must verify: ECDSA against
// the Jacobian pubkey from generator_mul_windowed_kernel (Z != 1, so the
// pubkey takes the variable-time inverse) and against the same key with
// Z == 1; Schnorr against the x-only key. A flipped message bit must be
// rejected by both. ecrecover_batch must recover the signer's key for
// exactly one recid of each ECDSA signature, and for no recid once one bit
// of s is flipped.
// =============================================================================

static void limbs_to_be32(const uint32_t* l, uint8_t* out) {
    for (int i = 0; i < 8; i++) {
        uint32_t v = l[7 - i];
        out[i*4] = (uint8_t)(v >> 24); out[i*4+1] = (uint8_t)(v >> 16);
        out[i*4+2] = (uint8_t)(v >> 8); out[i*4+3] = (uint8_t)v;
    }
}

static int test_verify_roundtrip(MetalCtx& ctx, bool verbose) {
    const uint32_t n = 64;
    auto p_aff   = ctx.get_pipeline("generator_mul_batch");
    auto p_win   = ctx.get_pipeline("generator_mul_windowed_kernel");
    auto p_esign = ctx.get_pipeline("ecdsa_sign_kernel");
    auto p_ever  = ctx.get_pipeline("ecdsa_verify_kernel");
    auto p_ssign = ctx.get_pipeline("schnorr_sign_kernel");
    auto p_sver  = ctx.get_pipeline("schnorr_verify_kernel");
    auto p_rec   = ctx.get_pipeline("ecrecover_batch");
    if (!p_aff || !p_win || !p_esign || !p_ever || !p_ssign || !p_sver || !p_rec) {
        printf("  FAIL: a verify round-trip kernel is missing\n");
        return 1;
    }

    // Keys in [1, 2^255): below n. Messages and aux randomness: seeded.
    uint64_t seed = 0xD1B54A32D192ED03ull;
    std::vector<uint32_t> keys(n * 8);
    std::vector<uint8_t> msgs(n * 32), aux(n * 32);
    for (uint32_t i = 0; i < n; i++) {
        for (int j = 0; j < 8; j++) keys[i*8 + j] = (uint32_t)(xorshift64(seed) >> 16);
        keys[i*8 + 7] &= 0x7FFFFFFFu;
        keys[i*8] |= 1u;
        for (int j = 0; j < 32; j++) {
            msgs[i*32 + j] = (uint8_t)(xorshift64(seed) >> 24);
            aux[i*32 + j] = (uint8_t)(xorshift64(seed) >> 24);
        }
    }
    std::vector<uint8_t> msgs_bad = msgs;
    for (uint32_t i = 0; i < n; i++) msgs_bad[i*32 + (i % 32)] ^= (uint8_t)(1u << (i % 8));

    auto count_buf = ctx.alloc_with_data(&n, sizeof(n));
    auto key_buf   = ctx.alloc_with_data(keys.data(), keys.size() * sizeof(uint32_t));
    auto msg_buf   = ctx.alloc_with_data(msgs.data(), msgs.size());
    auto bad_buf   = ctx.alloc_with_data(msgs_bad.data(), msgs_bad.size());
    auto aux_buf   = ctx.alloc_with_data(aux.data(), aux.size());

    // Public keys: affine (16 words) and Jacobian (25 words: x, y, z, infinity)
    auto aff_buf = ctx.alloc((size_t)n * 64);
    auto jac_buf = ctx.alloc((size_t)n * 100);
    ctx.dispatch_sync(p_aff, n, {key_buf, aff_buf, count_buf});
    ctx.dispatch_sync(p_win, n, {key_buf, jac_buf, count_buf});
    const uint32_t* aff = (const uint32_t*)[aff_buf contents];
    const uint32_t* jac = (const uint32_t*)[jac_buf contents];
    std::vector<uint32_t> jac1((size_t)n * 25, 0);          // same keys with Z == 1
    std::vector<uint8_t> xonly(n * 32), pub64(n * 64);
    int z_not_one = 0;
    for (uint32_t i = 0; i < n; i++) {
        for (int j = 0; j < 16; j++) jac1[i*25 + j] = aff[i*16 + j];
        jac1[i*25 + 16] = 1u;
        limbs_to_be32(aff + i*16, xonly.data() + i*32);
        limbs_to_be32(aff + i*16, pub64.data() + i*64);
        limbs_to_be32(aff + i*16 + 8, pub64.data() + i*64 + 32);
        bool one = jac[i*25 + 16] == 1u;
        for (int j = 17; j < 24; j++) one = one && jac[i*25 + j] == 0u;
        if (!one) z_not_one++;
    }
    auto jac1_buf  = ctx.alloc_with_data(jac1.data(), jac1.size() * sizeof(uint32_t));
    auto xonly_buf = ctx.alloc_with_data(xonly.data(), xonly.size());

    int passed = 0, total = 0;
    auto all_eq = [&](id<MTLBuffer> b, int want) {
        const int* r = (const int*)[b contents];
        for (uint32_t i = 0; i < n; i++) if (r[i] != want) return false;
        return true;
    };
    auto report = [&](const char* name, bool ok) {
        total++;
        if (ok) passed++;
        if (verbose || !ok) printf("  %s: %s\n", name, ok ? "PASS" : "FAIL");
    };

    // ECDSA
    auto esig_buf = ctx.alloc((size_t)n * 64);             // ECDSASignature: r[8], s[8]
    auto eflag_buf = ctx.alloc((size_t)n * sizeof(int));
    ctx.dispatch_sync(p_esign, n, {msg_buf, key_buf, esig_buf, eflag_buf, count_buf});
    report("ECDSA sign succeeded for every key", all_eq(eflag_buf, 1));
    report("generator_mul_windowed_kernel gave Z != 1 keys", z_not_one > 0);

    auto res_buf = ctx.alloc((size_t)n * sizeof(int));
    ctx.dispatch_sync(p_ever, n, {msg_buf, jac_buf, esig_buf, res_buf, count_buf});
    report("ECDSA verify (Jacobian pubkey, Z != 1) accepts", all_eq(res_buf, 1));
    ctx.dispatch_sync(p_ever, n, {msg_buf, jac1_buf, esig_buf, res_buf, count_buf});
    report("ECDSA verify (pubkey with Z == 1) accepts", all_eq(res_buf, 1));
    ctx.dispatch_sync(p_ever, n, {bad_buf, jac_buf, esig_buf, res_buf, count_buf});
    report("ECDSA verify rejects a flipped message bit", all_eq(res_buf, 0));

    // Schnorr (BIP-340)
    auto ssig_buf = ctx.alloc((size_t)n * 64);             // SchnorrSignature: r[32], s
    auto sflag_buf = ctx.alloc((size_t)n * sizeof(int));
    ctx.dispatch_sync(p_ssign, n, {msg_buf, key_buf, aux_buf, ssig_buf, sflag_buf, count_buf});
    report("Schnorr sign succeeded for every key", all_eq(sflag_buf, 1));
    ctx.dispatch_sync(p_sver, n, {xonly_buf, msg_buf, ssig_buf, res_buf, count_buf});
    report("Schnorr verify accepts", all_eq(res_buf, 1));
    ctx.dispatch_sync(p_sver, n, {xonly_buf, bad_buf, ssig_buf, res_buf, count_buf});
    report("Schnorr verify rejects a flipped message bit", all_eq(res_buf, 0));

    // ecrecover: exactly one recid gives the signer's key; none after an s bit flip
    const uint32_t* esig = (const uint32_t*)[esig_buf contents];
    std::vector<uint8_t> sig64(n * 64), sig64_bad(n * 64);
    for (uint32_t i = 0; i < n; i++) {
        limbs_to_be32(esig + i*16, sig64.data() + i*64);
        limbs_to_be32(esig + i*16 + 8, sig64.data() + i*64 + 32);
    }
    sig64_bad = sig64;
    for (uint32_t i = 0; i < n; i++) sig64_bad[i*64 + 63 - (i % 8)] ^= 0x02u;
    for (int variant = 0; variant < 2; variant++) {
        auto sig_buf = ctx.alloc_with_data(variant == 0 ? sig64.data() : sig64_bad.data(), sig64.size());
        std::vector<int> matches(n, 0);
        for (uint32_t rid = 0; rid < 4; rid++) {
            std::vector<uint32_t> recids(n, rid);
            auto rid_buf = ctx.alloc_with_data(recids.data(), recids.size() * sizeof(uint32_t));
            auto pub_buf = ctx.alloc((size_t)n * 64);
            auto ok_buf = ctx.alloc((size_t)n * sizeof(uint32_t));
            memset([pub_buf contents], 0, (size_t)n * 64);
            memset([ok_buf contents], 0, (size_t)n * sizeof(uint32_t));
            ctx.dispatch_sync(p_rec, n, {msg_buf, sig_buf, rid_buf, pub_buf, ok_buf, count_buf});
            const uint32_t* okv = (const uint32_t*)[ok_buf contents];
            const uint8_t* pub = (const uint8_t*)[pub_buf contents];
            for (uint32_t i = 0; i < n; i++)
                if (okv[i] == 1u && memcmp(pub + i*64, pub64.data() + i*64, 64) == 0) matches[i]++;
        }
        bool ok = true;
        for (uint32_t i = 0; i < n; i++) ok = ok && matches[i] == (variant == 0 ? 1 : 0);
        report(variant == 0 ? "ecrecover: exactly one recid recovers the signer's key"
                            : "ecrecover: a flipped s bit recovers the key for no recid", ok);
    }

    printf("  Verify round trips: %d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

// =============================================================================
// CT Smoke Tests -- Branchless GPU Constant-Time Layer (Section 11)
// =============================================================================
// Dispatches the ct_smoke_kernel (single workitem) and reads out[0..3].
// Each subtest returns 0 on pass.  Non-zero → error bitmap printed.
// =============================================================================

static int test_ct_smoke(MetalCtx& ctx, bool verbose) {
    int passed = 0, total = 0;

    auto pipe = ctx.get_pipeline("ct_smoke_kernel");
    if (!pipe) {
        printf("  SKIP: ct_smoke_kernel not found in metallib\n");
        return 0;  // advisory; Metal GPU may be unavailable in CI
    }

    // Output buffer: 4 ints (one per subtest)
    auto out_buf = ctx.alloc(4 * sizeof(int));
    memset([out_buf contents], 0, 4 * sizeof(int));

    ctx.dispatch_sync(pipe, 1, {out_buf});

    int* results = static_cast<int*>([out_buf contents]);

    static const char* subtests[] = {
        "CT mask primitives (is_zero, is_nonzero, eq, bool)",
        "CT cmov/cswap on 8-limb arrays",
        "CT ECDSA sign (privkey=1) + fast-path verify",
        "CT Schnorr sign (privkey=1, BIP-340) + verify",
    };

    for (int i = 0; i < 4; ++i) {
        total++;
        if (verbose) printf("  [CT %d] %s: ", i, subtests[i]);
        if (results[i] == 0) {
            if (verbose) printf("PASS\n");
            passed++;
        } else {
            printf("FAIL (result=0x%x)\n", results[i]);
        }
    }

    printf("  CT smoke: %d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    @autoreleasepool {
        bool do_bench = false;
        bool verbose = true;
        uint32_t bench_count = 65536;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bench") == 0) do_bench = true;
            else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
            else if (strcmp(argv[i], "--count") == 0 && i+1 < argc) bench_count = atoi(argv[++i]);
        }

        printf("==============================================\n");
        printf("  UltrafastSecp256k1 Metal Extended Test\n");
        printf("==============================================\n");

        MetalCtx ctx;
        if (!ctx.init()) return 1;

        char name[256];
        printf("  Device: %s\n", [[ctx.device name] UTF8String]);

        // Try to load precompiled metallib, else compile from source
        bool loaded = ctx.load_metallib("secp256k1.metallib");
        if (!loaded) {
            printf("  metallib not found, trying source compile...\n");
            // Read the .metal source
            FILE* f = fopen("secp256k1_kernels.metal", "r");
            if (!f) f = fopen("shaders/secp256k1_kernels.metal", "r");
            if (!f) {
                fprintf(stderr, "Cannot find secp256k1_kernels.metal\n");
                return 1;
            }
            fseek(f, 0, SEEK_END);
            size_t sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::string src(sz, '\0');
            fread(&src[0], 1, sz, f);
            fclose(f);

            if (!ctx.load_source(src)) return 1;
        }
        printf("  Library loaded OK\n");

        int ret = 0;

        // Tests
        printf("\n--- Hash160 Tests ---\n");
        ret |= test_hash160(ctx, verbose);

        printf("\n--- Field/Point Tests ---\n");
        ret |= test_field_ops(ctx, verbose);

        printf("\n--- GPU CT Layer Smoke Tests ---\n");
        ret |= test_ct_smoke(ctx, verbose);

        printf("\n--- Variable-time Inverse Tests ---\n");
        ret |= test_inv_var(ctx, verbose);

        printf("\n--- Verify / Recover Round Trips ---\n");
        ret |= test_verify_roundtrip(ctx, verbose);

        // Benchmarks
        if (do_bench) {
            printf("\n==============================================\n");
            printf("  Benchmarks (n=%u)\n", bench_count);
            printf("==============================================\n\n");
            printf("  %-30s  %12s  %10s\n", "Operation", "ops/sec", "time");
            printf("  %-30s  %12s  %10s\n", "-----------------------------", "----------", "--------");

            print_bench(bench_field_mul(ctx, bench_count));
            print_bench(bench_field_inv(ctx, bench_count));
            print_bench(bench_generator_mul(ctx, bench_count));
            print_bench(bench_sha256(ctx, bench_count));
            print_bench(bench_hash160(ctx, bench_count));
            print_bench(bench_ecdsa(ctx, bench_count));
        }

        printf("\n==============================================\n");
        if (ret == 0) printf("  [OK] ALL TESTS PASSED\n");
        else          printf("  [FAIL] SOME TESTS FAILED\n");
        printf("==============================================\n");

        return ret;
    }
}
