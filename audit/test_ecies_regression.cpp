// ============================================================================
// audit/test_ecies_regression.cpp -- Regression tests for ECIES hardening
// ============================================================================
// Tests:
//   A. Parity-byte tamper (0x02 <-> 0x03) must fail decryption
//   B. Invalid pubkey prefix (0x00, 0x04, 0xFF) must fail
//   C. Truncated envelope matrix (0, 1, 32, 33, 49, 81 bytes)
//   D. Tamper matrix (flip ephemeral pubkey / IV / ciphertext / tag byte)
//   E. Round-trip KAT with fixed decrypt vector
//   F. C ABI strict prefix rejection (ecdh, ecies_encrypt, silent_payment)
//   G. Pubkey parser consistency across endpoints
//   H. RNG fail-closed seam (fork + seccomp + SIGABRT) [Linux x86_64 only]
// ============================================================================

#include "audit_check.hpp"
#include "ufsecp/ufsecp.h"
#include <cstring>
#include <cstdint>
#include <array>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

#if defined(__linux__)
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <stddef.h>
#endif

static int g_pass = 0, g_fail = 0;

// -- Helpers ----------------------------------------------------------------

static ufsecp_ctx* make_ctx() {
    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);
    return ctx;
}

// Known test private key (valid, non-zero, < n)
static const uint8_t TEST_PRIVKEY[32] = {
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01
};

// Second test private key
static const uint8_t TEST_PRIVKEY2[32] = {
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02
};

static void get_pubkey(ufsecp_ctx* ctx, const uint8_t privkey[32], uint8_t pubkey33[33]) {
    ufsecp_pubkey_create(ctx, privkey, pubkey33);
}

// Create a valid ECIES envelope for subsequent tamper tests
static std::vector<uint8_t> make_valid_envelope(ufsecp_ctx* ctx, const uint8_t pubkey33[33]) {
    const uint8_t plaintext[] = "ECIES regression test payload";
    const size_t pt_len = sizeof(plaintext) - 1; // exclude null
    size_t env_len = pt_len + UFSECP_ECIES_OVERHEAD;
    std::vector<uint8_t> envelope(env_len);

    const ufsecp_error_t err = ufsecp_ecies_encrypt(ctx, pubkey33, plaintext, pt_len,
                                               envelope.data(), &env_len);
    if (err != UFSECP_OK) envelope.clear();
    return envelope;
}

// ============================================================================
// A. Parity-byte tamper: flip 0x02 <-> 0x03 on ephemeral pubkey
// ============================================================================
static void test_ecies_parity_tamper(ufsecp_ctx* ctx) {
    AUDIT_LOG("  A. ECIES parity-byte tamper...\n");

    uint8_t pubkey33[33];
    get_pubkey(ctx, TEST_PRIVKEY, pubkey33);

    auto envelope = make_valid_envelope(ctx, pubkey33);
    CHECK(!envelope.empty(), "valid envelope created");

    // Verify round-trip works with untampered envelope
    {
        size_t pt_len = envelope.size() - UFSECP_ECIES_OVERHEAD;
        std::vector<uint8_t> pt_out(pt_len);
        const ufsecp_error_t err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY,
            envelope.data(), envelope.size(), pt_out.data(), &pt_len);
        CHECK(err == UFSECP_OK, "untampered decrypt OK");
    }

    // Flip parity byte: 0x02 <-> 0x03
    auto tampered = envelope;
    tampered[0] ^= 0x01;  // 0x02->0x03 or 0x03->0x02

    {
        size_t pt_len = tampered.size() - UFSECP_ECIES_OVERHEAD;
        std::vector<uint8_t> pt_out(pt_len);
        const ufsecp_error_t err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY,
            tampered.data(), tampered.size(), pt_out.data(), &pt_len);
        CHECK(err != UFSECP_OK, "parity-flipped ephemeral pubkey -> decrypt fails");
    }
}

// ============================================================================
// B. Invalid pubkey prefix: 0x00, 0x04, 0xFF
// ============================================================================
static void test_ecies_invalid_prefix(ufsecp_ctx* ctx) {
    AUDIT_LOG("  B. ECIES invalid pubkey prefix...\n");

    uint8_t pubkey33[33];
    get_pubkey(ctx, TEST_PRIVKEY, pubkey33);

    auto envelope = make_valid_envelope(ctx, pubkey33);
    CHECK(!envelope.empty(), "valid envelope created for prefix test");

    const uint8_t bad_prefixes[] = {0x00, 0x04, 0xFF};
    for (const uint8_t prefix : bad_prefixes) {
        auto tampered = envelope;
        tampered[0] = prefix;

        size_t pt_len = tampered.size() - UFSECP_ECIES_OVERHEAD;
        std::vector<uint8_t> pt_out(pt_len);
        const ufsecp_error_t err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY,
            tampered.data(), tampered.size(), pt_out.data(), &pt_len);
        char msg[64];
        (void)std::snprintf(msg, sizeof(msg), "prefix 0x%02X -> decrypt fails", prefix);
        CHECK(err != UFSECP_OK, msg);
    }
}

// ============================================================================
// C. Truncated envelope matrix
// ============================================================================
static void test_ecies_truncated_envelope(ufsecp_ctx* ctx) {
    AUDIT_LOG("  C. ECIES truncated envelope matrix...\n");

    const size_t truncated_sizes[] = {0, 1, 32, 33, 49, 81};
    uint8_t junk[128];
    std::memset(junk, 0xAA, sizeof(junk));
    // Put a valid-looking compressed prefix to avoid prefix rejection before length check
    junk[0] = 0x02;

    for (const size_t sz : truncated_sizes) {
        size_t pt_len = 256;
        uint8_t pt_out[256];
        ufsecp_error_t err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY,
            junk, sz, pt_out, &pt_len);
        char msg[64];
        std::snprintf(msg, sizeof(msg), "truncated len=%zu -> fails cleanly", sz);
        CHECK(err != UFSECP_OK, msg);
    }
}

// ============================================================================
// D. Tamper matrix: flip single byte in each envelope field
// ============================================================================
static void test_ecies_tamper_matrix(ufsecp_ctx* ctx) {
    AUDIT_LOG("  D. ECIES tamper matrix...\n");

    uint8_t pubkey33[33];
    get_pubkey(ctx, TEST_PRIVKEY, pubkey33);

    auto envelope = make_valid_envelope(ctx, pubkey33);
    CHECK(!envelope.empty(), "valid envelope for tamper matrix");

    size_t const ct_len = envelope.size() - UFSECP_ECIES_OVERHEAD;

    // Offset for each field:
    struct { const char* name; size_t offset; } fields[] = {
        {"ephemeral pubkey[16]", 16},         // middle of pubkey (not prefix)
        {"IV[0]",                33},         // first IV byte
        {"ciphertext[0]",       49},          // first ciphertext byte
        {"HMAC tag[0]",         49 + ct_len}, // first tag byte
    };

    for (auto& f : fields) {
        auto tampered = envelope;
        if (f.offset < tampered.size())
            tampered[f.offset] ^= 0x01; // flip one bit

        size_t pt_len = ct_len;
        std::vector<uint8_t> pt_out(pt_len);
        ufsecp_error_t err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY,
            tampered.data(), tampered.size(), pt_out.data(), &pt_len);
        char msg[80];
        std::snprintf(msg, sizeof(msg), "tamper %s -> decrypt fails", f.name);
        CHECK(err != UFSECP_OK, msg);
    }
}

// ============================================================================
// E. Round-trip KAT with fixed key decrypt
// ============================================================================
static void test_ecies_roundtrip_kat(ufsecp_ctx* ctx) {
    AUDIT_LOG("  E. ECIES round-trip KAT...\n");

    uint8_t pubkey33[33];
    get_pubkey(ctx, TEST_PRIVKEY, pubkey33);

    // Multiple plaintext sizes to test padding behavior
    const char* test_vectors[] = {
        "A",            // 1 byte
        "Hello, ECIES!", // 13 bytes
        "0123456789abcdef0123456789abcdef", // 32 bytes (AES block aligned)
    };

    for (const char* pt_str : test_vectors) {
        size_t pt_len = std::strlen(pt_str);
        size_t env_len = pt_len + UFSECP_ECIES_OVERHEAD;
        std::vector<uint8_t> envelope(env_len);

        ufsecp_error_t err = ufsecp_ecies_encrypt(ctx, pubkey33,
            reinterpret_cast<const uint8_t*>(pt_str), pt_len,
            envelope.data(), &env_len);
        CHECK(err == UFSECP_OK, "encrypt OK");

        // Verify envelope structure
        CHECK(env_len == pt_len + UFSECP_ECIES_OVERHEAD, "envelope size correct");
        CHECK(envelope[0] == 0x02 || envelope[0] == 0x03, "valid compressed prefix");

        // Decrypt
        size_t dec_len = pt_len;
        std::vector<uint8_t> dec_out(dec_len);
        err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY,
            envelope.data(), env_len, dec_out.data(), &dec_len);
        CHECK(err == UFSECP_OK, "decrypt OK");
        CHECK(dec_len == pt_len, "decrypted length matches");
        CHECK(std::memcmp(dec_out.data(), pt_str, pt_len) == 0, "plaintext matches");

        // Wrong key must fail
        size_t dec_len2 = pt_len;
        std::vector<uint8_t> dec_out2(dec_len2);
        err = ufsecp_ecies_decrypt(ctx, TEST_PRIVKEY2,
            envelope.data(), env_len, dec_out2.data(), &dec_len2);
        CHECK(err != UFSECP_OK, "wrong key -> decrypt fails");
    }
}

// ============================================================================
// F. C ABI strict prefix rejection across endpoints
// ============================================================================
static void test_abi_prefix_rejection(ufsecp_ctx* ctx) {
    AUDIT_LOG("  F. C ABI strict prefix rejection...\n");

    // Construct malformed 33-byte pubkeys with bad prefixes
    const uint8_t bad_prefixes[] = {0x00, 0x01, 0x04, 0x05, 0x06, 0xFF};

    // Use a valid pubkey as base, then corrupt prefix
    uint8_t good_pubkey33[33];
    get_pubkey(ctx, TEST_PRIVKEY, good_pubkey33);

    for (uint8_t prefix : bad_prefixes) {
        uint8_t bad_pk[33];
        std::memcpy(bad_pk, good_pubkey33, 33);
        bad_pk[0] = prefix;

        char msg[80];

        // F1: ufsecp_ecdh
        {
            uint8_t secret[32];
            ufsecp_error_t err = ufsecp_ecdh(ctx, TEST_PRIVKEY2, bad_pk, secret);
            std::snprintf(msg, sizeof(msg), "ecdh rejects prefix 0x%02X", prefix);
            CHECK(err == UFSECP_ERR_BAD_PUBKEY, msg);
        }

        // F2: ufsecp_ecies_encrypt
        {
            uint8_t pt[] = "test";
            size_t env_len = sizeof(pt) - 1 + UFSECP_ECIES_OVERHEAD;
            std::vector<uint8_t> env(env_len);
            ufsecp_error_t err = ufsecp_ecies_encrypt(ctx, bad_pk, pt, sizeof(pt)-1,
                                                       env.data(), &env_len);
            std::snprintf(msg, sizeof(msg), "ecies_encrypt rejects prefix 0x%02X", prefix);
            CHECK(err == UFSECP_ERR_BAD_PUBKEY, msg);
        }

        // F3: ufsecp_pubkey_parse (should reject or produce different error)
        {
            uint8_t parsed[33];
            ufsecp_error_t err = ufsecp_pubkey_parse(ctx, bad_pk, 33, parsed);
            std::snprintf(msg, sizeof(msg), "pubkey_parse rejects prefix 0x%02X", prefix);
            CHECK(err != UFSECP_OK, msg);
        }

        // F4: ufsecp_ecdh_xonly
        {
            uint8_t secret[32];
            ufsecp_error_t err = ufsecp_ecdh_xonly(ctx, TEST_PRIVKEY2, bad_pk, secret);
            std::snprintf(msg, sizeof(msg), "ecdh_xonly rejects prefix 0x%02X", prefix);
            CHECK(err == UFSECP_ERR_BAD_PUBKEY, msg);
        }

        // F5: ufsecp_ecdh_raw
        {
            uint8_t secret[32];
            ufsecp_error_t err = ufsecp_ecdh_raw(ctx, TEST_PRIVKEY2, bad_pk, secret);
            std::snprintf(msg, sizeof(msg), "ecdh_raw rejects prefix 0x%02X", prefix);
            CHECK(err == UFSECP_ERR_BAD_PUBKEY, msg);
        }
    }
}

// ============================================================================
// G. Pubkey parser consistency: same bad input => same result everywhere
// ============================================================================
static void test_pubkey_parser_consistency(ufsecp_ctx* ctx) {
    AUDIT_LOG("  G. Pubkey parser consistency...\n");

    // Construct deterministic malformed compressed pubkeys
    // Type 1: valid prefix, x >= p (all FF in x-coordinate)
    uint8_t bad_x_ff[33];
    bad_x_ff[0] = 0x02;
    std::memset(bad_x_ff + 1, 0xFF, 32);

    // Type 2: valid prefix, x = 0 (not a valid x-coordinate on secp256k1)
    uint8_t bad_x_zero[33];
    bad_x_zero[0] = 0x02;
    std::memset(bad_x_zero + 1, 0x00, 32);

    // Type 3: valid prefix, x >= p (p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F)
    // Byte 24 is 0xFF instead of p's 0xFE, making this >= p
    uint8_t bad_x_gte_p[33];
    bad_x_gte_p[0] = 0x02;
    std::memset(bad_x_gte_p + 1, 0xFF, 32);
    // Fix last 8 bytes to be exactly p+1 (still >= p, but distinct from all-FF)
    bad_x_gte_p[25] = 0xFF; // p has 0xFE at this position
    bad_x_gte_p[26] = 0xFF;
    bad_x_gte_p[27] = 0xFF;
    bad_x_gte_p[28] = 0xFF;
    bad_x_gte_p[29] = 0xFF;
    bad_x_gte_p[30] = 0xFC;
    bad_x_gte_p[31] = 0x30; // p ends with FC2F, this is FC30 (p+1)

    uint8_t* bad_keys[] = {bad_x_ff, bad_x_zero, bad_x_gte_p};
    const char* names[] = {"x=0xFF..FF", "x=0x00..00", "x>=p"};

    for (int i = 0; i < 3; ++i) {
        uint8_t* bk = bad_keys[i];
        char msg[128];

        // All endpoints must reject consistently
        uint8_t parsed[33];
        ufsecp_error_t e_parse = ufsecp_pubkey_parse(ctx, bk, 33, parsed);

        uint8_t secret[32];
        ufsecp_error_t e_ecdh = ufsecp_ecdh(ctx, TEST_PRIVKEY, bk, secret);

        uint8_t pt[] = "x";
        size_t env_len = 1 + UFSECP_ECIES_OVERHEAD;
        std::vector<uint8_t> env(env_len);
        ufsecp_error_t e_ecies = ufsecp_ecies_encrypt(ctx, bk, pt, 1,
                                                       env.data(), &env_len);

        // All must fail (none should be UFSECP_OK)
        std::snprintf(msg, sizeof(msg), "%s: pubkey_parse fails", names[i]);
        CHECK(e_parse != UFSECP_OK, msg);

        std::snprintf(msg, sizeof(msg), "%s: ecdh fails", names[i]);
        CHECK(e_ecdh != UFSECP_OK, msg);

        std::snprintf(msg, sizeof(msg), "%s: ecies_encrypt fails", names[i]);
        CHECK(e_ecies != UFSECP_OK, msg);

        // Consistency: all should return same error category (BAD_PUBKEY)
        std::snprintf(msg, sizeof(msg), "%s: ecdh returns BAD_PUBKEY", names[i]);
        CHECK(e_ecdh == UFSECP_ERR_BAD_PUBKEY, msg);

        std::snprintf(msg, sizeof(msg), "%s: ecies returns BAD_PUBKEY", names[i]);
        CHECK(e_ecies == UFSECP_ERR_BAD_PUBKEY, msg);
    }
}

// ============================================================================
// H. RNG fail-closed seam test (fork + seccomp blocks getrandom -> SIGABRT)
// ============================================================================
#if defined(__linux__) && defined(__x86_64__)
static void test_rng_fail_closed(ufsecp_ctx* ctx) {
    AUDIT_LOG("  H. RNG fail-closed seam...\n");

    // Get pubkey before fork (avoid any RNG in parent after seccomp)
    uint8_t pubkey33[33];
    get_pubkey(ctx, TEST_PRIVKEY, pubkey33);

    pid_t pid = fork();
    if (pid == 0) {
        // Child: install seccomp-bpf filter that makes getrandom return ENOSYS
        struct sock_filter filter[] = {
            // Load syscall number
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                     (unsigned int)offsetof(struct seccomp_data, nr)),
            // If getrandom (nr=318 on x86_64), return errno ENOSYS
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_getrandom, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (38 & SECCOMP_RET_DATA)),
            // Otherwise allow
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        };
        struct sock_fprog prog = {
            .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
            .filter = filter,
        };
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog, 0, 0);

        // Now getrandom will fail -> csprng_fill -> std::abort()
        uint8_t pt[] = "rng-test";
        std::size_t env_len = 8 + UFSECP_ECIES_OVERHEAD;
        std::vector<uint8_t> env(env_len);
        ufsecp_ecies_encrypt(ctx, pubkey33, pt, 8, env.data(), &env_len);
        // If we reach here, RNG failure was silently swallowed
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);

    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
        CHECK(true, "RNG failure causes abort (fail-closed)");
    } else if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        CHECK(false, "RNG failure was silently swallowed (NOT fail-closed!)");
    } else {
        // SIGKILL, SIGSYS, or non-zero exit also means fail-closed behavior
        CHECK(true, "RNG failure caused process termination (fail-closed)");
    }
}
#endif // __linux__ && __x86_64__

// ============================================================================
// Entry point
// ============================================================================
int test_ecies_regression_run() {
    AUDIT_LOG("\n[ECIES Regression + C ABI Prefix Enforcement]\n");

    auto ctx = make_ctx();
    if (!ctx) {
        AUDIT_LOG("  [FAIL] ctx creation\n");
        return 1;
    }

    test_ecies_parity_tamper(ctx);
    test_ecies_invalid_prefix(ctx);
    test_ecies_truncated_envelope(ctx);
    test_ecies_tamper_matrix(ctx);
    test_ecies_roundtrip_kat(ctx);
    test_abi_prefix_rejection(ctx);
    test_pubkey_parser_consistency(ctx);
#if defined(__linux__) && defined(__x86_64__)
    test_rng_fail_closed(ctx);
#endif

    ufsecp_ctx_destroy(ctx);

    AUDIT_LOG("  -- ECIES regression: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}

#ifdef STANDALONE_ECIES_REGRESSION
int main() {
    int fails = test_ecies_regression_run();
    return fails ? 1 : 0;
}
#endif
