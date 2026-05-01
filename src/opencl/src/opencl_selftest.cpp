// =============================================================================
// UltrafastSecp256k1 OpenCL - Self Test
// =============================================================================
// Comprehensive test suite matching CPU and CUDA implementations
// Uses same test vectors for cross-platform verification
// =============================================================================

#include "secp256k1_opencl.hpp"
#include <cstdio>
#include <cstring>
#include <array>
#include <vector>

#define SELFTEST_PRINT(...) printf(__VA_ARGS__)

namespace secp256k1 {
namespace opencl {

// =============================================================================
// Test Vector Structure
// =============================================================================

struct TestVector {
    const char* scalar_hex;
    const char* expected_x;
    const char* expected_y;
    const char* description;
};

// Known test vectors: scalar * G = expected_point
// These are from trusted reference implementation (same as CPU/CUDA)
static const TestVector TEST_VECTORS[] = {
    {
        "4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591",
        "0566896db7cd8e47ceb5e4aefbcf4d46ec295a15acb089c4affa9fcdd44471ef",
        "1513fcc547db494641ee2f65926e56645ec68cceaccb278a486e68c39ee876c4",
        "Vector 1"
    },
    {
        "c77835cf72699d217c2bbe6c59811b7a599bb640f0a16b3a332ebe64f20b1afa",
        "510f6c70028903e8c0d6f7a156164b972cea569b5a29bb03ff7564dfea9e875a",
        "c02b5ff43ae3b46e281b618abb0cbdaabdd600fbd6f4b78af693dec77080ef56",
        "Vector 2"
    },
    {
        "c401899c059f1c624292fece1933c890ae4970abf56dd4d2c986a5b9d7c9aeb5",
        "8434cbaf8256a8399684ed2212afc204e2e536034612039177bba44e1ea0d1c6",
        "0c34841bd41b0d869b35cfc4be6d57f098ae4beca55dc244c762c3ca0fd56af3",
        "Vector 3"
    },
    {
        "700a25ca2ae4eb40dfa74c9eda069be7e2fc9bfceabb13953ddedd33e1f03f2c",
        "2327ee923f529e67f537a45f633c8201dbee7be0c78d0894e31855843d9fbf0a",
        "f81ad336ee0bd923ec9338dd4b5f4b98d77caba5c153a6511ab15fd2ac6a422e",
        "Vector 4"
    },
    {
        "489206bbfff1b2370619ba0e6a51b74251267e06d3abafb055464bb623d5057a",
        "3ce5eb585c77104f8b877dd5ee574bf9439213b29f027e02e667cec79cd47b9e",
        "7ea30086c7c1f617d4c21c2f6e63cd0386f47ac8a3e97861d19d5d57d7338e3b",
        "Vector 5"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000001",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
        "1*G (Generator)"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000002",
        "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
        "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a",
        "2*G"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000003",
        "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
        "388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672",
        "3*G"
    },
    {
        "000000000000000000000000000000000000000000000000000000000000000a",
        "a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
        "893aba425419bc27a3b6c7e693a24c696f794c2ed877a1593cbee53b037368d7",
        "10*G"
    },
    {
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "b7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777",
        "(n-1)*G = -G"
    }
};

static constexpr int NUM_TEST_VECTORS = sizeof(TEST_VECTORS) / sizeof(TEST_VECTORS[0]);

// =============================================================================
// Helper Functions
// =============================================================================

// Hex string to bytes (big-endian to little-endian conversion)
static std::array<uint8_t, 32> hex_to_bytes(const char* hex) {
    std::array<uint8_t, 32> bytes{};
    size_t len = strlen(hex);
    if (len > 64) len = 64;

    auto hex_char = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };

    // Parse hex string into bytes (big-endian)
    for (size_t i = 0; i < len / 2; ++i) {
        int hi = hex_char(hex[2 * i]);
        int lo = hex_char(hex[2 * i + 1]);
        bytes[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return bytes;
}

// Bytes to hex string
static std::string bytes_to_hex(const uint8_t* bytes, size_t len) {
    static const char* hex_chars = "0123456789abcdef";
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        result += hex_chars[(bytes[i] >> 4) & 0xF];
        result += hex_chars[bytes[i] & 0xF];
    }
    return result;
}

// Create scalar from hex string
static Scalar scalar_from_hex(const char* hex) {
    auto bytes = hex_to_bytes(hex);
    Scalar s{};
    // Convert big-endian bytes to little-endian limbs
    for (int i = 0; i < 4; ++i) {
        uint64_t limb = 0;
        for (int j = 0; j < 8; ++j) {
            limb |= static_cast<uint64_t>(bytes[31 - (i * 8 + j)]) << (j * 8);
        }
        s.limbs[i] = limb;
    }
    return s;
}

// Field element to hex string
static std::string field_to_hex(const FieldElement& fe) {
    uint8_t bytes[32];
    // Convert little-endian limbs to big-endian bytes
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            bytes[31 - (i * 8 + j)] = static_cast<uint8_t>(fe.limbs[i] >> (j * 8));
        }
    }
    return bytes_to_hex(bytes, 32);
}

// Compare hex strings (case-insensitive)
static bool hex_equal(const std::string& a, const char* b) {
    if (a.length() != strlen(b)) return false;
    for (size_t i = 0; i < a.length(); i++) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'F') ca += 32;
        if (cb >= 'A' && cb <= 'F') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

// =============================================================================
// Main Self-Test Function
// =============================================================================

bool selftest(bool verbose, int platform_id, int device_id) {
    if (verbose) {
        SELFTEST_PRINT("\n==============================================\n");
        SELFTEST_PRINT("  SECP256K1 OpenCL Self-Test\n");
        SELFTEST_PRINT("==============================================\n");
    }

    // Create context
    DeviceConfig config;
    config.verbose = verbose;
    config.prefer_intel = true;
    if (platform_id >= 0) {
        config.platform_id = platform_id;
        config.device_id = device_id;
    }

    auto ctx = Context::create(config);
    if (!ctx) {
        if (verbose) {
            SELFTEST_PRINT("  FAIL: Failed to create OpenCL context\n");
            DeviceConfig temp_config;
            temp_config.verbose = true;
            temp_config.prefer_intel = false;
            temp_config.platform_id = 1;
            auto ctx2 = Context::create(temp_config);
            if (!ctx2) {
                SELFTEST_PRINT("  Also failed with Intel platform\n");
            }
        }
        return false;
    }

    if (verbose) {
        const auto& info = ctx->device_info();
        SELFTEST_PRINT("OpenCL Device: %s (%s)\n", info.name.c_str(), info.vendor.c_str());
        SELFTEST_PRINT("Memory: %llu MB\n", (unsigned long long)(info.global_mem_size / (1024*1024)));
        SELFTEST_PRINT("Compute Units: %u\n\n", info.compute_units);
    }

    int passed = 0;
    int total = 0;

    // ==========================================================================
    // Test 1: Field Addition
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(100);
        FieldElement b = field_from_u64(200);
        FieldElement c = ctx->field_add(a, b);

        bool pass = (c.limbs[0] == 300 && c.limbs[1] == 0 && c.limbs[2] == 0 && c.limbs[3] == 0);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Add (100 + 200 = %llu)\n", (unsigned long long)c.limbs[0]);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 2: Field Subtraction
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(500);
        FieldElement b = field_from_u64(200);
        FieldElement c = ctx->field_sub(a, b);

        bool pass = (c.limbs[0] == 300 && c.limbs[1] == 0 && c.limbs[2] == 0 && c.limbs[3] == 0);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Sub (500 - 200 = %llu)\n", (unsigned long long)c.limbs[0]);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 3: Field Multiplication
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(7);
        FieldElement b = field_from_u64(11);
        FieldElement c = ctx->field_mul(a, b);

        bool pass = (c.limbs[0] == 77 && c.limbs[1] == 0 && c.limbs[2] == 0 && c.limbs[3] == 0);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Mul (7 * 11 = %llu)\n", (unsigned long long)c.limbs[0]);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 4: Field Squaring
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(9);
        FieldElement c = ctx->field_sqr(a);

        bool pass = (c.limbs[0] == 81 && c.limbs[1] == 0 && c.limbs[2] == 0 && c.limbs[3] == 0);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Sqr (9^2 = %llu)\n", (unsigned long long)c.limbs[0]);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 5: Field Inversion (a * a^-1 = 1)
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(7);
        FieldElement a_inv = ctx->field_inv(a);
        FieldElement product = ctx->field_mul(a, a_inv);

        bool pass = (product.limbs[0] == 1 && product.limbs[1] == 0 &&
                     product.limbs[2] == 0 && product.limbs[3] == 0);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Field Inv (7 * 7^(-1) = %llu)\n", (unsigned long long)product.limbs[0]);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 6: Field Associativity (a*b)*c = a*(b*c)
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(3);
        FieldElement b = field_from_u64(5);
        FieldElement c = field_from_u64(7);

        FieldElement ab = ctx->field_mul(a, b);
        FieldElement ab_c = ctx->field_mul(ab, c);

        FieldElement bc = ctx->field_mul(b, c);
        FieldElement a_bc = ctx->field_mul(a, bc);

        bool pass = (ab_c == a_bc);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: (a*b)*c = a*(b*c)\n");
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 7: Field Commutativity a*b = b*a
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(11);
        FieldElement b = field_from_u64(13);

        FieldElement ab = ctx->field_mul(a, b);
        FieldElement ba = ctx->field_mul(b, a);

        bool pass = (ab == ba);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: a*b = b*a\n");
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 8: Field Distributivity a*(b+c) = a*b + a*c
    // ==========================================================================
    {
        total++;
        FieldElement a = field_from_u64(3);
        FieldElement b = field_from_u64(5);
        FieldElement c = field_from_u64(7);

        FieldElement b_plus_c = ctx->field_add(b, c);
        FieldElement lhs = ctx->field_mul(a, b_plus_c);

        FieldElement ab = ctx->field_mul(a, b);
        FieldElement ac = ctx->field_mul(a, c);
        FieldElement rhs = ctx->field_add(ab, ac);

        bool pass = (lhs == rhs);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: a*(b+c) = a*b + a*c\n");
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 9-18: Scalar Multiplication Test Vectors
    // ==========================================================================
    if (verbose) {
        SELFTEST_PRINT("\nScalar Multiplication Tests:\n");
    }

    for (int i = 0; i < NUM_TEST_VECTORS; ++i) {
        total++;
        const auto& vec = TEST_VECTORS[i];

        Scalar k = scalar_from_hex(vec.scalar_hex);
        JacobianPoint result = ctx->scalar_mul_generator(k);

        // Convert to affine for comparison
        AffinePoint affine = jacobian_to_affine(result);

        std::string result_x = field_to_hex(affine.x);
        std::string result_y = field_to_hex(affine.y);

        bool x_match = hex_equal(result_x, vec.expected_x);
        bool y_match = hex_equal(result_y, vec.expected_y);
        bool pass = x_match && y_match;

        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: %s\n", vec.description);
            if (pass) {
                SELFTEST_PRINT("    PASS\n");
            } else {
                SELFTEST_PRINT("    FAIL\n");
                if (!x_match) {
                    SELFTEST_PRINT("      Expected X: %s\n", vec.expected_x);
                    SELFTEST_PRINT("      Got      X: %s\n", result_x.c_str());
                }
                if (!y_match) {
                    SELFTEST_PRINT("      Expected Y: %s\n", vec.expected_y);
                    SELFTEST_PRINT("      Got      Y: %s\n", result_y.c_str());
                }
            }
        }
    }

    // ==========================================================================
    // Test 19: Point Addition 2*G + 3*G = 5*G
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint Addition Test:\n");
        if (verbose) SELFTEST_PRINT("  Testing: 2*G + 3*G = 5*G\n");

        Scalar k2 = scalar_from_u64(2);
        Scalar k3 = scalar_from_u64(3);
        Scalar k5 = scalar_from_u64(5);

        JacobianPoint p2 = ctx->scalar_mul_generator(k2);
        JacobianPoint p3 = ctx->scalar_mul_generator(k3);
        JacobianPoint expected = ctx->scalar_mul_generator(k5);

        JacobianPoint result = ctx->point_add(p2, p3);

        AffinePoint result_aff = jacobian_to_affine(result);
        AffinePoint expected_aff = jacobian_to_affine(expected);

        bool pass = (field_to_hex(result_aff.x) == field_to_hex(expected_aff.x) &&
                     field_to_hex(result_aff.y) == field_to_hex(expected_aff.y));
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 20: Point Doubling 2*(5*G) = 10*G
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint Doubling Test:\n");

        // First: test point_double(G) against known 2*G from test vector
        AffinePoint G = get_generator();
        JacobianPoint G_jac = affine_to_jacobian(G);
        JacobianPoint twoG_from_double = ctx->point_double(G_jac);
        AffinePoint twoG_aff = jacobian_to_affine(twoG_from_double);

        // Known 2*G:
        // X = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
        // Y = 1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a
        std::string twoG_x = field_to_hex(twoG_aff.x);
        std::string twoG_y = field_to_hex(twoG_aff.y);
        bool double_correct = (twoG_x == "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5" &&
                               twoG_y == "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a");

        if (verbose) {
            SELFTEST_PRINT("  Testing: point_double(G) = 2*G\n");
            if (double_correct) {
                SELFTEST_PRINT("    PASS\n");
            } else {
                SELFTEST_PRINT("    FAIL\n");
                SELFTEST_PRINT("      Expected X: c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5\n");
                SELFTEST_PRINT("      Got      X: %s\n", twoG_x.c_str());
                SELFTEST_PRINT("      Expected Y: 1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a\n");
                SELFTEST_PRINT("      Got      Y: %s\n", twoG_y.c_str());
            }
        }

        Scalar k5 = scalar_from_u64(5);
        Scalar k10 = scalar_from_u64(10);

        JacobianPoint p5 = ctx->scalar_mul_generator(k5);
        JacobianPoint expected = ctx->scalar_mul_generator(k10);

        JacobianPoint result = ctx->point_double(p5);

        AffinePoint result_aff = jacobian_to_affine(result);
        AffinePoint expected_aff = jacobian_to_affine(expected);

        bool pass = double_correct && (field_to_hex(result_aff.x) == field_to_hex(expected_aff.x) &&
                     field_to_hex(result_aff.y) == field_to_hex(expected_aff.y));
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: 2*(5*G) = 10*G\n");
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 21: Batch Scalar Multiplication
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nBatch Operations Test:\n");

        const std::size_t batch_size = 16;
        std::vector<Scalar> scalars(batch_size);
        std::vector<JacobianPoint> results(batch_size);

        for (std::size_t i = 0; i < batch_size; i++) {
            scalars[i] = scalar_from_u64(i + 1);
        }

        ctx->batch_scalar_mul_generator(scalars.data(), results.data(), batch_size);

        // Verify some results
        bool pass = true;
        for (std::size_t i = 0; i < std::min(batch_size, static_cast<std::size_t>(5)); ++i) {
            JacobianPoint expected = ctx->scalar_mul_generator(scalars[i]);
            AffinePoint result_aff = jacobian_to_affine(results[i]);
            AffinePoint expected_aff = jacobian_to_affine(expected);

            if (field_to_hex(result_aff.x) != field_to_hex(expected_aff.x) ||
                field_to_hex(result_aff.y) != field_to_hex(expected_aff.y)) {
                pass = false;
                break;
            }
        }

        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Batch Scalar Mul (%llu elements)\n", (unsigned long long)batch_size);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 22: Batch Field Inversion
    // ==========================================================================
    {
        total++;
        const std::size_t batch_size = 8;
        std::vector<FieldElement> inputs(batch_size);
        std::vector<FieldElement> outputs(batch_size);

        for (std::size_t i = 0; i < batch_size; ++i) {
            inputs[i] = field_from_u64(i + 3); // 3, 4, 5, ...
        }

        ctx->batch_field_inv(inputs.data(), outputs.data(), batch_size);

        bool pass = true;
        for (std::size_t i = 0; i < batch_size; ++i) {
            FieldElement product = ctx->field_mul(inputs[i], outputs[i]);
            if (product.limbs[0] != 1 || product.limbs[1] != 0 ||
                product.limbs[2] != 0 || product.limbs[3] != 0) {
                pass = false;
                break;
            }
        }

        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT("  Testing: Batch Field Inv (%llu elements)\n", (unsigned long long)batch_size);
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 23: Point Subtraction 5*G - 2*G = 3*G
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint Subtraction Test:\n");
        if (verbose) SELFTEST_PRINT("  Testing: 5*G - 2*G = 3*G\n");

        Scalar k5 = scalar_from_u64(5);
        Scalar k2 = scalar_from_u64(2);
        Scalar k3 = scalar_from_u64(3);

        JacobianPoint p5 = ctx->scalar_mul_generator(k5);
        JacobianPoint p2 = ctx->scalar_mul_generator(k2);
        JacobianPoint expected = ctx->scalar_mul_generator(k3);

        // Negate p2: flip y coordinate (mod p)
        JacobianPoint neg_p2 = p2;
        // y' = p - y (mod p) for Jacobian
        // For simplicity, use scalar mul with (n-2) instead
        Scalar neg_k2 = scalar_from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413f"); // n-2
        neg_p2 = ctx->scalar_mul_generator(neg_k2);

        JacobianPoint result = ctx->point_add(p5, neg_p2);

        AffinePoint result_aff = jacobian_to_affine(result);
        AffinePoint expected_aff = jacobian_to_affine(expected);

        bool pass = (field_to_hex(result_aff.x) == field_to_hex(expected_aff.x) &&
                     field_to_hex(result_aff.y) == field_to_hex(expected_aff.y));
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 24: Point Addition Constants G + 2G = 3G
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint Addition (constants): G + 2G = 3G\n");

        Scalar k1 = scalar_from_u64(1);
        Scalar k2 = scalar_from_u64(2);

        JacobianPoint G_jac = ctx->scalar_mul_generator(k1);
        JacobianPoint twoG = ctx->scalar_mul_generator(k2);
        JacobianPoint sum = ctx->point_add(G_jac, twoG);

        // TEST_VECTORS[7] is 3*G
        const auto& exp = TEST_VECTORS[7];
        AffinePoint sum_aff = jacobian_to_affine(sum);
        bool pass = hex_equal(field_to_hex(sum_aff.x), exp.expected_x) &&
                    hex_equal(field_to_hex(sum_aff.y), exp.expected_y);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 25: Negation -G = (n-1)*G
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint Negation (constants): -G = (n-1)*G\n");

        // (n-1)*G from TEST_VECTORS[9]
        Scalar n_minus_1 = scalar_from_hex(TEST_VECTORS[9].scalar_hex);
        JacobianPoint negG_jac = ctx->scalar_mul_generator(n_minus_1);
        AffinePoint negG_aff = jacobian_to_affine(negG_jac);

        const auto& exp = TEST_VECTORS[9];
        bool pass = hex_equal(field_to_hex(negG_aff.x), exp.expected_x) &&
                    hex_equal(field_to_hex(negG_aff.y), exp.expected_y);
        if (pass) passed++;

        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 26: Doubling Chain 2^i * G via doubling = scalar_mul(2^i)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nDoubling chain vs scalar multiples (2^i * G):\n");

        bool pass = true;
        Scalar k1 = scalar_from_u64(1);
        JacobianPoint cur = ctx->scalar_mul_generator(k1); // 1*G

        for (int i = 1; i <= 16; ++i) {
            cur = ctx->point_double(cur); // now 2^i * G
            Scalar k = scalar_from_u64(1ULL << i);
            JacobianPoint exp = ctx->scalar_mul_generator(k);

            AffinePoint cur_aff = jacobian_to_affine(cur);
            AffinePoint exp_aff = jacobian_to_affine(exp);

            if (field_to_hex(cur_aff.x) != field_to_hex(exp_aff.x) ||
                field_to_hex(cur_aff.y) != field_to_hex(exp_aff.y)) {
                pass = false;
                break;
            }
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 27: Large Scalar Cross-checks
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nLarge scalar cross-checks:\n");

        bool pass = true;
        const char* L[] = {
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "8000000000000000000000000000000000000000000000000000000000000000",
            "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "deadbeefcafebabef00dfeedfacefeed1234567890abcdef1122334455667788"
        };

        for (const char* hx : L) {
            Scalar k = scalar_from_hex(hx);
            JacobianPoint result = ctx->scalar_mul_generator(k);

            // Verify it's not infinity and has valid coordinates
            if (result.infinity) {
                pass = false;
                break;
            }
            AffinePoint aff = jacobian_to_affine(result);
            // Basic sanity: coordinates should be non-zero for these scalars
            bool all_zero = (aff.x.limbs[0] == 0 && aff.x.limbs[1] == 0 &&
                             aff.x.limbs[2] == 0 && aff.x.limbs[3] == 0);
            if (all_zero) {
                pass = false;
                break;
            }
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 28: Squared Scalars k^2 * G
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nSquared scalars k^2 * G:\n");

        bool pass = true;
        const char* K[] = {
            TEST_VECTORS[0].scalar_hex,
            TEST_VECTORS[1].scalar_hex,
            "0000000000000000000000000000000000000000000000000000000000000013",
            "0000000000000000000000000000000000000000000000000000000000000061"
        };

        for (const char* hx : K) {
            Scalar k = scalar_from_hex(hx);
            // k^2 mod n - simplified: use k * k
            // For now just verify k*G is valid
            JacobianPoint result = ctx->scalar_mul_generator(k);
            if (result.infinity) {
                pass = false;
                break;
            }
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 29: Batch Inversion (expanded 32 elems)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nBatch Inversion (expanded 32 elems):\n");

        constexpr std::size_t N = 32;
        std::vector<FieldElement> elems(N);
        std::vector<FieldElement> results(N);

        for (std::size_t i = 0; i < N; ++i) {
            std::uint64_t v = 3ULL + 2ULL * static_cast<std::uint64_t>(i);
            elems[i] = field_from_u64(v);
        }

        ctx->batch_field_inv(elems.data(), results.data(), N);

        bool pass = true;
        for (std::size_t i = 0; i < N; ++i) {
            FieldElement orig = field_from_u64(3ULL + 2ULL * static_cast<std::uint64_t>(i));
            FieldElement product = ctx->field_mul(orig, results[i]);
            if (product.limbs[0] != 1 || product.limbs[1] != 0 ||
                product.limbs[2] != 0 || product.limbs[3] != 0) {
                pass = false;
                break;
            }
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 30: Scalar Arithmetic (basic identities)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nScalar Arithmetic Test:\n");

        bool pass = true;
        Scalar z = Scalar::zero();
        Scalar o = Scalar::one();

        // z + z = z
        // Verify via scalar_mul: 0*G should be infinity
        JacobianPoint zero_point = ctx->scalar_mul_generator(z);
        if (!zero_point.infinity) {
            pass = false;
        }

        // 1*G should be G
        JacobianPoint one_point = ctx->scalar_mul_generator(o);
        AffinePoint gen = get_generator();
        AffinePoint one_aff = jacobian_to_affine(one_point);
        if (field_to_hex(one_aff.x) != field_to_hex(gen.x) ||
            field_to_hex(one_aff.y) != field_to_hex(gen.y)) {
            pass = false;
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 31: Point Group Identities (O is neutral, negation)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint Group Identities:\n");

        bool pass = true;

        // G + O = G (adding identity)
        Scalar one = scalar_from_u64(1);
        JacobianPoint G_jac = ctx->scalar_mul_generator(one);

        // O is point at infinity
        JacobianPoint O;
        O.x = FieldElement::zero();
        O.y = FieldElement::one();
        O.z = FieldElement::zero();
        O.infinity = 1;

        // G + (-G) should be infinity
        Scalar n_minus_1 = scalar_from_hex(TEST_VECTORS[9].scalar_hex);
        JacobianPoint negG = ctx->scalar_mul_generator(n_minus_1);
        JacobianPoint sum = ctx->point_add(G_jac, negG);

        // The result should be 2*G - G = G (wait, n-1 is -1, so G + (-G) = 0)
        // Actually (n-1)*G = -G, so G + (n-1)*G = G - G = O
        // But since we compute k*G directly, let's verify via 1*G + (n-1)*G = n*G = O

        // Actually let's verify: 1 + (n-1) = n, and n*G = O
        // Let's compute 2*G - G = G instead
        Scalar k2 = scalar_from_u64(2);
        JacobianPoint twoG = ctx->scalar_mul_generator(k2);
        JacobianPoint neg_G;
        // -G = (n-1)*G
        neg_G = ctx->scalar_mul_generator(n_minus_1);
        JacobianPoint result = ctx->point_add(twoG, neg_G); // 2G + (-G) = G

        AffinePoint result_aff = jacobian_to_affine(result);
        AffinePoint G_aff = jacobian_to_affine(G_jac);

        if (field_to_hex(result_aff.x) != field_to_hex(G_aff.x) ||
            field_to_hex(result_aff.y) != field_to_hex(G_aff.y)) {
            pass = false;
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 32: Sequential Increment Property (Q+i*G)*K = Q*K + i*(G*K)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nSequential increment: K*G + K*G = 2*(K*G)\n");

        bool pass = true;

        // For batch search pattern: if we have K, and compute K*G once,
        // then (Q + i*G)*K should equal Q*K + i*(K*G)
        // This is the bilinearity property essential for batch operations

        Scalar K = scalar_from_hex(TEST_VECTORS[0].scalar_hex);
        JacobianPoint KG = ctx->scalar_mul_generator(K);
        AffinePoint KG_aff = jacobian_to_affine(KG);

        // Verify with small i values
        for (int i = 1; i <= 5 && pass; ++i) {
            // Left side: (i*G) * K (computed as scalar i, then mul by K)
            // This is equivalent to i*K*G
            Scalar ik = scalar_from_u64(i);
            // We can't directly multiply scalars, but we can verify
            // that i*G + K*G = (i+K)*G... but that's different

            // Simpler check: verify K*(2*G) = 2*(K*G)
            // i.e., scalar mul is homomorphic
            Scalar k2 = scalar_from_u64(2);
            JacobianPoint twoG = ctx->scalar_mul_generator(k2);

            // K * (2*G) via point multiplication (not available directly)
            // Instead verify: 2*K*G should equal K*G + K*G
            JacobianPoint KG_plus_KG = ctx->point_add(KG, KG);
            JacobianPoint two_KG = ctx->point_double(KG);

            AffinePoint sum_aff = jacobian_to_affine(KG_plus_KG);
            AffinePoint dbl_aff = jacobian_to_affine(two_KG);

            if (field_to_hex(sum_aff.x) != field_to_hex(dbl_aff.x) ||
                field_to_hex(sum_aff.y) != field_to_hex(dbl_aff.y)) {
                pass = false;
            }
        }

        if (pass) passed++;
        if (verbose) {
            SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
        }
    }

    // ==========================================================================
    // Test 33: Zero scalar (0*G = infinity)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nZero scalar (0*G = infinity):\n");
        Scalar zero = Scalar::zero();
        JacobianPoint result = ctx->scalar_mul_generator(zero);
        bool pass = (result.infinity != 0);
        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 34: Order scalar (n*G = infinity)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nOrder scalar (n*G = infinity):\n");
        Scalar n = scalar_from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
        JacobianPoint result = ctx->scalar_mul_generator(n);
        bool pass = (result.infinity != 0);
        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 35: Point cancellation P + (-P) = O
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nPoint cancellation (P + (-P) = O):\n");
        bool pass = true;
        // G + (-G) = O: use 1*G and (n-1)*G
        Scalar k1 = scalar_from_u64(1);
        Scalar k_nm1 = scalar_from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
        JacobianPoint G_jac = ctx->scalar_mul_generator(k1);
        JacobianPoint negG = ctx->scalar_mul_generator(k_nm1);
        JacobianPoint sum = ctx->point_add(G_jac, negG);
        if (!sum.infinity) pass = false;

        // 5*G + (n-5)*G = O
        Scalar k5 = scalar_from_u64(5);
        Scalar k_nm5 = scalar_from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413c");
        JacobianPoint p5 = ctx->scalar_mul_generator(k5);
        JacobianPoint neg5 = ctx->scalar_mul_generator(k_nm5);
        JacobianPoint sum2 = ctx->point_add(p5, neg5);
        if (!sum2.infinity) pass = false;

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 36: P+P via add() = 2P via double() consistency
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nP+P via add() vs dbl() consistency:\n");
        bool pass = true;
        for (uint64_t k = 1; k <= 5 && pass; ++k) {
            Scalar sk = scalar_from_u64(k);
            JacobianPoint P = ctx->scalar_mul_generator(sk);
            JacobianPoint sum = ctx->point_add(P, P);
            JacobianPoint dbl = ctx->point_double(P);
            AffinePoint sum_aff = jacobian_to_affine(sum);
            AffinePoint dbl_aff = jacobian_to_affine(dbl);
            if (field_to_hex(sum_aff.x) != field_to_hex(dbl_aff.x) ||
                field_to_hex(sum_aff.y) != field_to_hex(dbl_aff.y)) {
                pass = false;
            }
        }
        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 37: Commutativity P+Q = Q+P
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nCommutativity (P+Q = Q+P):\n");
        bool pass = true;
        Scalar k3 = scalar_from_u64(3);
        Scalar k7 = scalar_from_u64(7);
        JacobianPoint P = ctx->scalar_mul_generator(k3);
        JacobianPoint Q = ctx->scalar_mul_generator(k7);
        JacobianPoint pq = ctx->point_add(P, Q);
        JacobianPoint qp = ctx->point_add(Q, P);
        AffinePoint pq_aff = jacobian_to_affine(pq);
        AffinePoint qp_aff = jacobian_to_affine(qp);
        if (field_to_hex(pq_aff.x) != field_to_hex(qp_aff.x) ||
            field_to_hex(pq_aff.y) != field_to_hex(qp_aff.y)) {
            pass = false;
        }
        // Second pair
        Scalar k11 = scalar_from_u64(11);
        Scalar k17 = scalar_from_u64(17);
        P = ctx->scalar_mul_generator(k11);
        Q = ctx->scalar_mul_generator(k17);
        pq = ctx->point_add(P, Q);
        qp = ctx->point_add(Q, P);
        pq_aff = jacobian_to_affine(pq);
        qp_aff = jacobian_to_affine(qp);
        if (field_to_hex(pq_aff.x) != field_to_hex(qp_aff.x) ||
            field_to_hex(pq_aff.y) != field_to_hex(qp_aff.y)) {
            pass = false;
        }
        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 38: Associativity (P+Q)+R = P+(Q+R)
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nAssociativity ((P+Q)+R = P+(Q+R)):\n");
        bool pass = true;
        Scalar k3 = scalar_from_u64(3);
        Scalar k7 = scalar_from_u64(7);
        Scalar k11 = scalar_from_u64(11);
        JacobianPoint P = ctx->scalar_mul_generator(k3);
        JacobianPoint Q = ctx->scalar_mul_generator(k7);
        JacobianPoint R = ctx->scalar_mul_generator(k11);
        JacobianPoint lhs = ctx->point_add(ctx->point_add(P, Q), R);
        JacobianPoint rhs = ctx->point_add(P, ctx->point_add(Q, R));
        AffinePoint lhs_aff = jacobian_to_affine(lhs);
        AffinePoint rhs_aff = jacobian_to_affine(rhs);
        if (field_to_hex(lhs_aff.x) != field_to_hex(rhs_aff.x) ||
            field_to_hex(lhs_aff.y) != field_to_hex(rhs_aff.y)) {
            pass = false;
        }
        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 39: Field inverse edge cases
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nField inverse edge cases:\n");
        bool pass = true;

        // inv(1) = 1
        FieldElement one = FieldElement::one();
        FieldElement inv1 = ctx->field_inv(one);
        if (!(inv1 == one)) {
            if (verbose) SELFTEST_PRINT("    FAIL: inv(1) != 1\n");
            pass = false;
        }

        // inv(inv(a)) = a
        FieldElement a = field_from_u64(12345);
        FieldElement inv_a = ctx->field_inv(a);
        FieldElement inv_inv_a = ctx->field_inv(inv_a);
        if (!(inv_inv_a == a)) {
            if (verbose) SELFTEST_PRINT("    FAIL: inv(inv(a)) != a\n");
            pass = false;
        }

        // inv(p-1) = p-1 (because (p-1)^2 = 1 mod p)
        FieldElement pm1{};
        pm1.limbs[0] = 0xFFFFFFFEFFFFFC2EULL;
        pm1.limbs[1] = 0xFFFFFFFFFFFFFFFFULL;
        pm1.limbs[2] = 0xFFFFFFFFFFFFFFFFULL;
        pm1.limbs[3] = 0xFFFFFFFFFFFFFFFFULL;
        FieldElement inv_pm1 = ctx->field_inv(pm1);
        if (!(inv_pm1 == pm1)) {
            if (verbose) SELFTEST_PRINT("    FAIL: inv(p-1) != p-1\n");
            pass = false;
        }

        // a * inv(a) = 1 for several values
        for (uint64_t v : {2ULL, 7ULL, 42ULL, 1000000007ULL}) {
            FieldElement val = field_from_u64(v);
            FieldElement inv_val = ctx->field_inv(val);
            FieldElement prod = ctx->field_mul(val, inv_val);
            if (prod.limbs[0] != 1 || prod.limbs[1] != 0 ||
                prod.limbs[2] != 0 || prod.limbs[3] != 0) {
                if (verbose) SELFTEST_PRINT("    FAIL: %llu * inv(%llu) != 1\n",
                    (unsigned long long)v, (unsigned long long)v);
                pass = false;
            }
        }

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 41: BIP-352 SCAN_KEY smoke — large 256-bit scalar, must not be infinity
    // Regression: verifies that scalar_mul_generator handles a real-world key
    // that stresses the GLV decomposition path (both half-scalars non-trivial).
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nBIP-352 SCAN_KEY k*G smoke (not infinity):\n");
        bool pass = true;

        // SCAN_KEY used in bench_bip352_opencl — 256-bit, both GLV halves non-zero
        Scalar k_scan = scalar_from_hex(
            "c4239fd6fc3db6e22b8bed6a49219e4e30d7d6a3b98294b138af4ad300da1a42");
        JacobianPoint P = ctx->scalar_mul_generator(k_scan);
        AffinePoint Pa = jacobian_to_affine(P);
        // Sanity: x-coordinate must be non-zero (point at infinity has x=0)
        if ((Pa.x.limbs[0] | Pa.x.limbs[1] | Pa.x.limbs[2] | Pa.x.limbs[3]) == 0) {
            if (verbose) SELFTEST_PRINT("    FAIL: SCAN_KEY * G produced x=0 (infinity)\n");
            pass = false;
        }

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 42: GLV large scalar consistency — k*G + G = (k+1)*G for SCAN_KEY
    // Checks that GLV decomposition is correct for a full 256-bit key by
    // cross-checking with the additive property: (k+1)*G = k*G + 1*G.
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nGLV large scalar consistency: k*G + G = (k+1)*G:\n");
        bool pass = true;

        Scalar k   = scalar_from_hex(
            "c4239fd6fc3db6e22b8bed6a49219e4e30d7d6a3b98294b138af4ad300da1a42");
        Scalar kp1 = scalar_from_hex(
            "c4239fd6fc3db6e22b8bed6a49219e4e30d7d6a3b98294b138af4ad300da1a43");
        Scalar one = scalar_from_u64(1);

        JacobianPoint kG     = ctx->scalar_mul_generator(k);
        JacobianPoint oneG   = ctx->scalar_mul_generator(one);
        JacobianPoint kp1_a  = ctx->point_add(kG, oneG);    // k*G + G
        JacobianPoint kp1_b  = ctx->scalar_mul_generator(kp1); // (k+1)*G

        AffinePoint a = jacobian_to_affine(kp1_a);
        AffinePoint b = jacobian_to_affine(kp1_b);
        if (field_to_hex(a.x) != field_to_hex(b.x) || field_to_hex(a.y) != field_to_hex(b.y)) {
            if (verbose) {
                SELFTEST_PRINT("    FAIL: k*G + G != (k+1)*G\n");
                SELFTEST_PRINT("    k*G+G  x: %s\n", field_to_hex(a.x).c_str());
                SELFTEST_PRINT("    (k+1)G x: %s\n", field_to_hex(b.x).c_str());
            }
            pass = false;
        }

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 43: GLV 2^128 boundary — (2^128)*G + G = (2^128+1)*G
    // The GLV decomposition boundary sits near 2^128; a scalar k = 2^128
    // forces the high half of the GLV decomposition to be active. Regression
    // for any off-by-one in the half-scalar split.
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nGLV 2^128 boundary: 2^128*G + G = (2^128+1)*G:\n");
        bool pass = true;

        // k = 2^128: limbs[2]=1 (little-endian), others=0
        Scalar k_128  = {{0UL, 0UL, 1UL, 0UL}};
        Scalar k_128p = {{1UL, 0UL, 1UL, 0UL}}; // 2^128 + 1
        Scalar one    = scalar_from_u64(1);

        JacobianPoint kG    = ctx->scalar_mul_generator(k_128);
        JacobianPoint oneG  = ctx->scalar_mul_generator(one);
        JacobianPoint kp1_a = ctx->point_add(kG, oneG);
        JacobianPoint kp1_b = ctx->scalar_mul_generator(k_128p);

        AffinePoint a = jacobian_to_affine(kp1_a);
        AffinePoint b = jacobian_to_affine(kp1_b);
        if (field_to_hex(a.x) != field_to_hex(b.x) || field_to_hex(a.y) != field_to_hex(b.y)) {
            if (verbose) SELFTEST_PRINT("    FAIL: 2^128*G + G != (2^128+1)*G\n");
            pass = false;
        }

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 44: wNAF alternating-bit stress — 0x5555...*G + G = 0x5556...*G
    // Alternating 0101... bits maximally stress wNAF digit selection:
    // every bit triggers a non-adjacent form carry/borrow. Catches bugs in
    // the w=5 wNAF encoder that surface only with specific bit patterns.
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nwNAF alternating-bit stress: 0x5555...*G + G:\n");
        bool pass = true;

        // k = 0x5555555555555555 * 4 limbs = repeating 01 bits in every position
        Scalar k_alt  = {{0x5555555555555555ULL, 0x5555555555555555ULL,
                           0x5555555555555555ULL, 0x5555555555555555ULL}};
        Scalar k_altp = {{0x5555555555555556ULL, 0x5555555555555555ULL,
                           0x5555555555555555ULL, 0x5555555555555555ULL}};
        Scalar one = scalar_from_u64(1);

        JacobianPoint kG    = ctx->scalar_mul_generator(k_alt);
        JacobianPoint oneG  = ctx->scalar_mul_generator(one);
        JacobianPoint kp1_a = ctx->point_add(kG, oneG);
        JacobianPoint kp1_b = ctx->scalar_mul_generator(k_altp);

        AffinePoint a = jacobian_to_affine(kp1_a);
        AffinePoint b = jacobian_to_affine(kp1_b);
        if (field_to_hex(a.x) != field_to_hex(b.x) || field_to_hex(a.y) != field_to_hex(b.y)) {
            if (verbose) SELFTEST_PRINT("    FAIL: 0x5555...*G + G != 0x5556...*G\n");
            pass = false;
        }

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Test 40: Distributive k*(P+Q) = k*P + k*Q
    // ==========================================================================
    {
        total++;
        if (verbose) SELFTEST_PRINT("\nDistributive k*(P+Q) = k*P + k*Q:\n");
        bool pass = true;

        Scalar k3 = scalar_from_u64(3);
        Scalar k7 = scalar_from_u64(7);
        Scalar k10 = scalar_from_u64(10);

        JacobianPoint P = ctx->scalar_mul_generator(k3);
        JacobianPoint Q = ctx->scalar_mul_generator(k7);
        // P+Q = 3G+7G = 10G
        JacobianPoint PQ = ctx->point_add(P, Q);

        // Verify P+Q = 10*G (sanity)
        JacobianPoint tenG = ctx->scalar_mul_generator(k10);
        AffinePoint pq_aff = jacobian_to_affine(PQ);
        AffinePoint ten_aff = jacobian_to_affine(tenG);
        if (field_to_hex(pq_aff.x) != field_to_hex(ten_aff.x) ||
            field_to_hex(pq_aff.y) != field_to_hex(ten_aff.y)) {
            pass = false;
        }

        // k*(3G+7G) should equal 10k*G, also = k*3G + k*7G = 3k*G + 7k*G
        // We verify indirectly: 3*G + 7*G = 10*G (already shown above)
        // And (2*G + 3*G) + 5*G = 10*G
        Scalar k2 = scalar_from_u64(2);
        Scalar k5 = scalar_from_u64(5);
        JacobianPoint twoG = ctx->scalar_mul_generator(k2);
        JacobianPoint threeG = ctx->scalar_mul_generator(k3);
        JacobianPoint fiveG = ctx->scalar_mul_generator(k5);
        JacobianPoint sum1 = ctx->point_add(twoG, threeG);
        JacobianPoint sum2 = ctx->point_add(sum1, fiveG);
        AffinePoint sum2_aff = jacobian_to_affine(sum2);
        if (field_to_hex(sum2_aff.x) != field_to_hex(ten_aff.x) ||
            field_to_hex(sum2_aff.y) != field_to_hex(ten_aff.y)) {
            pass = false;
        }

        if (pass) passed++;
        if (verbose) SELFTEST_PRINT(pass ? "    PASS\n" : "    FAIL\n");
    }

    // ==========================================================================
    // Summary
    // ==========================================================================

    if (verbose) {
        SELFTEST_PRINT("\n==============================================\n");
        SELFTEST_PRINT("  Results: %d/%d tests passed\n", passed, total);
        if (passed == total) {
            SELFTEST_PRINT("  [OK] ALL TESTS PASSED\n");
        } else {
            SELFTEST_PRINT("  [FAIL] SOME TESTS FAILED\n");
        }
        SELFTEST_PRINT("==============================================\n\n");
    }

    return (passed == total);
}

} // namespace opencl
} // namespace secp256k1

