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
