// =============================================================================
// UltrafastSecp256k1 OpenCL - Test Runner
// =============================================================================
// Runs the complete OpenCL self-test suite (32+ tests)
// Same test vectors as CPU and CUDA implementations
// Optionally cross-verifies against CPU library when available
// =============================================================================

#include "secp256k1_opencl.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

#define SELFTEST_PRINT(...) printf(__VA_ARGS__)

#ifdef HAVE_CPU_LIB
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"
#endif

// Cross-verify OpenCL results against CPU library
#ifdef HAVE_CPU_LIB
static bool cross_verify_scalar_mul(secp256k1::opencl::Context& ctx, bool verbose) {
    using namespace secp256k1::opencl;

    int passed = 0;
    int total = 0;

    if (verbose) {
        SELFTEST_PRINT("\nCPU Scalar Mul Cross-Verification:\n");
    }

    // Test scalars: small, medium, and large values
    const char* test_scalars[] = {
        "0000000000000000000000000000000000000000000000000000000000000001",
        "0000000000000000000000000000000000000000000000000000000000000007",
        "000000000000000000000000000000000000000000000000000000000000002a",
        "4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591",
        "c77835cf72699d217c2bbe6c59811b7a599bb640f0a16b3a332ebe64f20b1afa",
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
    };

    for (const char* hex : test_scalars) {
        total++;

        // OpenCL computation
        secp256k1::opencl::Scalar k_ocl{};
        auto bytes = [](const char* h) {
            std::array<uint8_t, 32> b{};
            for (int i = 0; i < 32; ++i) {
                auto nib = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return 0;
                };
                b[i] = static_cast<uint8_t>((nib(h[2*i]) << 4) | nib(h[2*i+1]));
            }
            return b;
        }(hex);

        // Big-endian bytes -> little-endian limbs
        for (int i = 0; i < 4; ++i) {
            uint64_t limb = 0;
            for (int j = 0; j < 8; ++j) {
                limb |= static_cast<uint64_t>(bytes[31 - (i * 8 + j)]) << (j * 8);
            }
            k_ocl.limbs[i] = limb;
        }

        auto result_jac = ctx.scalar_mul_generator(k_ocl);
        auto result_aff = secp256k1::opencl::jacobian_to_affine(result_jac);

        // CPU computation
        auto cpu_scalar = secp256k1::fast::Scalar::from_hex(hex);
        auto cpu_result = secp256k1::fast::Point::generator().scalar_mul(cpu_scalar);
        auto cpu_x_hex = cpu_result.x().to_hex();
        auto cpu_y_hex = cpu_result.y().to_hex();

        // Convert OpenCL result to hex
        auto fe_to_hex = [](const secp256k1::opencl::FieldElement& fe) -> std::string {
            uint8_t buf[32];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 8; ++j) {
                    buf[31 - (i * 8 + j)] = static_cast<uint8_t>(fe.limbs[i] >> (j * 8));
                }
            }
            static const char* hc = "0123456789abcdef";
            std::string r;
            r.reserve(64);
            for (int i = 0; i < 32; ++i) {
                r += hc[(buf[i] >> 4) & 0xF];
                r += hc[buf[i] & 0xF];
            }
            return r;
        };

        std::string ocl_x = fe_to_hex(result_aff.x);
        std::string ocl_y = fe_to_hex(result_aff.y);

        bool pass = (ocl_x == cpu_x_hex && ocl_y == cpu_y_hex);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: k=%.16s...\n", hex);
            if (pass) {
                SELFTEST_PRINT("    PASS\n");
            } else {
                SELFTEST_PRINT("    FAIL\n");
                SELFTEST_PRINT("      Expected X: %s\n", cpu_x_hex.c_str());
                SELFTEST_PRINT("      Got      X: %s\n", ocl_x.c_str());
                SELFTEST_PRINT("      Expected Y: %s\n", cpu_y_hex.c_str());
                SELFTEST_PRINT("      Got      Y: %s\n", ocl_y.c_str());
            }
        }
    }

    if (verbose) {
        SELFTEST_PRINT("  Results: %d/%d cross-verified\n", passed, total);
    }

    return (passed == total);
}

static bool cross_verify_field_ops(secp256k1::opencl::Context& ctx, bool verbose) {
    using namespace secp256k1::opencl;

    int passed = 0;
    int total = 0;

    if (verbose) {
        SELFTEST_PRINT("\nCPU Field Operations Cross-Verification:\n");
    }

    // Test field multiplication with large values
    {
        total++;
        auto a_cpu = secp256k1::fast::FieldElement::from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
        auto b_cpu = secp256k1::fast::FieldElement::from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");
        auto c_cpu = a_cpu * b_cpu;
        auto cpu_hex = c_cpu.to_hex();

        FieldElement a_ocl{}, b_ocl{};
        a_ocl.limbs[0] = 0x59F2815B16F81798; a_ocl.limbs[1] = 0x029BFCDB2DCE28D9;
        a_ocl.limbs[2] = 0x55A06295CE870B07; a_ocl.limbs[3] = 0x79BE667EF9DCBBAC;
        b_ocl.limbs[0] = 0x9C47D08FFB10D4B8; b_ocl.limbs[1] = 0xFD17B448A6855419;
        b_ocl.limbs[2] = 0x5DA4FBFC0E1108A8; b_ocl.limbs[3] = 0x483ADA7726A3C465;

        auto c_ocl = ctx.field_mul(a_ocl, b_ocl);

        auto fe_to_hex = [](const FieldElement& fe) -> std::string {
            uint8_t buf[32];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 8; ++j)
                    buf[31 - (i * 8 + j)] = static_cast<uint8_t>(fe.limbs[i] >> (j * 8));
            static const char* hc = "0123456789abcdef";
            std::string r; r.reserve(64);
            for (int i = 0; i < 32; ++i) { r += hc[(buf[i]>>4)&0xF]; r += hc[buf[i]&0xF]; }
            return r;
        };

        std::string ocl_hex = fe_to_hex(c_ocl);
        bool pass = (ocl_hex == cpu_hex);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Mul (Gx * Gy)\n");
            if (pass) {
                SELFTEST_PRINT("    PASS\n");
            } else {
                SELFTEST_PRINT("    FAIL\n");
                SELFTEST_PRINT("      Expected: %s\n", cpu_hex.c_str());
                SELFTEST_PRINT("      Got:      %s\n", ocl_hex.c_str());
            }
        }
    }

    // Test field inversion with large values
    {
        total++;
        auto a_cpu = secp256k1::fast::FieldElement::from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
        auto inv_cpu = a_cpu.inverse();
        auto product_cpu = a_cpu * inv_cpu;
        bool cpu_ok = (product_cpu == secp256k1::fast::FieldElement::one());

        FieldElement a_ocl{};
        a_ocl.limbs[0] = 0x59F2815B16F81798; a_ocl.limbs[1] = 0x029BFCDB2DCE28D9;
        a_ocl.limbs[2] = 0x55A06295CE870B07; a_ocl.limbs[3] = 0x79BE667EF9DCBBAC;

        auto inv_ocl = ctx.field_inv(a_ocl);
        auto prod_ocl = ctx.field_mul(a_ocl, inv_ocl);

        bool pass = cpu_ok &&
                    (prod_ocl.limbs[0] == 1 && prod_ocl.limbs[1] == 0 &&
                     prod_ocl.limbs[2] == 0 && prod_ocl.limbs[3] == 0);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Inv (Gx * Gx^-1 = 1)\n");
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    if (verbose) {
        SELFTEST_PRINT("  Results: %d/%d field cross-verified\n", passed, total);
    }

    return (passed == total);
}
#endif // HAVE_CPU_LIB

int main(int argc, char* argv[]) {
    bool verbose = true;
    int platform_id = -1;  // -1 = auto (prefer Intel)
    int device_id = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-q" || arg == "--quiet") {
            verbose = false;
        } else if (arg == "--platform" && i + 1 < argc) {
            platform_id = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (arg == "--intel") {
            platform_id = 1;  // Intel Graphics is usually platform 1
        } else if (arg == "--nvidia") {
            platform_id = 0;  // NVIDIA is usually platform 0
        }
    }

    SELFTEST_PRINT("\n");
    SELFTEST_PRINT("==============================================\n");
    SELFTEST_PRINT("  SECP256K1 OpenCL Test Suite\n");
    SELFTEST_PRINT("==============================================\n\n");

    SELFTEST_PRINT("Available OpenCL Devices:\n");
    auto platforms = secp256k1::opencl::enumerate_devices();

    if (platforms.empty()) {
        SELFTEST_PRINT("  No OpenCL platforms found!\n");
        SELFTEST_PRINT("  Please install OpenCL drivers:\n");
        SELFTEST_PRINT("  - Intel: Install Intel GPU driver\n");
        SELFTEST_PRINT("  - NVIDIA: Install CUDA toolkit\n");
        SELFTEST_PRINT("  - AMD: Install AMD GPU driver\n");
        return 1;
    }

    for (const auto& [platform_name, devices] : platforms) {
        SELFTEST_PRINT("  Platform: %s\n", platform_name.c_str());
        for (const auto& dev : devices) {
            SELFTEST_PRINT("    - %s (%s)\n", dev.name.c_str(), dev.vendor.c_str());
            SELFTEST_PRINT("      Memory: %llu MB, Compute Units: %u\n",
                           (unsigned long long)(dev.global_mem_size / (1024*1024)),
                           dev.compute_units);
        }
    }
    SELFTEST_PRINT("\n");

    // Run self-test (32+ tests)
    bool success = secp256k1::opencl::selftest(verbose, platform_id, device_id);

#ifdef HAVE_CPU_LIB
    // Cross-verify with CPU implementation
    if (success) {
        secp256k1::opencl::DeviceConfig config;
        config.verbose = false;
        config.prefer_intel = true;
        if (platform_id >= 0) {
            config.platform_id = platform_id;
            config.device_id = device_id;
        }

        auto ctx = secp256k1::opencl::Context::create(config);
        if (ctx) {
            bool field_ok = cross_verify_field_ops(*ctx, verbose);
            bool scalar_ok = cross_verify_scalar_mul(*ctx, verbose);
            success = success && field_ok && scalar_ok;
        }
    }
#endif

    if (success) {
        SELFTEST_PRINT("\n==============================================\n");
        SELFTEST_PRINT("  [OK] ALL TESTS PASSED\n");
        SELFTEST_PRINT("==============================================\n\n");
        return 0;
    } else {
        SELFTEST_PRINT("\n==============================================\n");
        SELFTEST_PRINT("  [FAIL] SOME TESTS FAILED\n");
        SELFTEST_PRINT("==============================================\n\n");
        return 1;
    }
}

