// ============================================================================
// test_regression_schnorr_r_zero_ct.cpp
// Regression: shim_schnorr.cpp R-zero check uses CT OR accumulator instead
// of variable-time for+break loop (SEC-006).
//
// Both secp256k1_schnorrsig_sign32 and secp256k1_schnorrsig_sign_custom had:
//   for (int i=0;i<32;i++) { if (sig.r[i]!=0) { r_all_zero=false; break; } }
// Fixed to:
//   uint32_t r_nonzero=0; for (int i=0;i<32;i++) r_nonzero|=sig.r[i];
//
// Advisory: returns 77 if shim not linked (GitHub CI without shim).
//
// SRC-1: sign32 produces non-zero r, returns 1
// SRC-2: different keys produce different sigs
// SRC-3: sign32 is deterministic
// SRC-4: sign_custom (32-byte) delegates to sign32
// SRC-5: sign_custom (64-byte) produces valid sig
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstring>
#include <cstdio>

static constexpr int ADVISORY_SKIP_CODE = 77;
static int g_fail = 0;
#define ASSERT_TRUE(cond, msg)  do { if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)
#define ASSERT_FALSE(cond, msg) do { if ( (cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)

// Weak attribute so macOS ld64 doesn't fail when the libsecp256k1 shim is
// not linked into the unified_audit_runner target. On platforms where the
// shim is absent these resolve to nullptr → ctx_create returns null →
// ADVISORY_SKIP_CODE. Linux ld permits undefined weaks by default; macOS
// requires the explicit __attribute__((weak_import)) (alias of weak).
#if defined(__APPLE__)
#  define SHIM_WEAK __attribute__((weak_import))
#else
#  define SHIM_WEAK __attribute__((weak))
#endif

extern "C" {
    typedef struct { unsigned char data[64]; } secp256k1_pubkey;
    typedef struct { unsigned char data[96]; } secp256k1_keypair;
    typedef struct secp256k1_context_struct secp256k1_context;
    SHIM_WEAK secp256k1_context* secp256k1_context_create(unsigned int flags);
    SHIM_WEAK void secp256k1_context_destroy(secp256k1_context* ctx);
    SHIM_WEAK int secp256k1_keypair_create(const secp256k1_context*, secp256k1_keypair*, const unsigned char*);
    SHIM_WEAK int secp256k1_schnorrsig_sign32(const secp256k1_context*, unsigned char*, const unsigned char*, const secp256k1_keypair*, const unsigned char*);
    SHIM_WEAK int secp256k1_schnorrsig_sign_custom(const secp256k1_context*, unsigned char*, const unsigned char*, size_t, const secp256k1_keypair*, void*);
}

static constexpr unsigned int CTX_SIGN = 0x0101;
static constexpr unsigned int CTX_VERIFY = 0x0102;

static void test_sign32_normal(secp256k1_context* ctx) {
    unsigned char sk[32] = {}; sk[31] = 7;
    secp256k1_keypair kp{};
    if (secp256k1_keypair_create(ctx, &kp, sk) != 1) { std::printf("SKIP SRC-1\n"); return; }
    unsigned char msg[32] = {}; msg[0] = 0xAB;
    unsigned char aux[32] = {}; aux[0] = 0x11;
    unsigned char sig[64] = {};
    int rc = secp256k1_schnorrsig_sign32(ctx, sig, msg, &kp, aux);
    ASSERT_TRUE(rc == 1, "[SRC-1] sign32 must return 1");
    unsigned int r = 0; for (int i=0;i<32;i++) r |= sig[i];
    ASSERT_TRUE(r != 0, "[SRC-1] r must not be all-zero");
}

static void test_sign32_different_keys(secp256k1_context* ctx) {
    unsigned char sk1[32] = {}; sk1[31] = 7;
    unsigned char sk2[32] = {}; sk2[31] = 11;
    secp256k1_keypair kp1{}, kp2{};
    if (secp256k1_keypair_create(ctx,&kp1,sk1)!=1||secp256k1_keypair_create(ctx,&kp2,sk2)!=1) return;
    unsigned char msg[32] = {}; msg[0] = 0xCC;
    unsigned char s1[64] = {}, s2[64] = {};
    ASSERT_TRUE(secp256k1_schnorrsig_sign32(ctx,s1,msg,&kp1,nullptr)==1, "[SRC-2] sign1 ok");
    ASSERT_TRUE(secp256k1_schnorrsig_sign32(ctx,s2,msg,&kp2,nullptr)==1, "[SRC-2] sign2 ok");
    ASSERT_TRUE(std::memcmp(s1,s2,64)!=0, "[SRC-2] different keys → different sigs");
}

static void test_sign32_deterministic(secp256k1_context* ctx) {
    unsigned char sk[32] = {}; sk[31] = 5;
    secp256k1_keypair kp{};
    if (secp256k1_keypair_create(ctx,&kp,sk)!=1) return;
    unsigned char msg[32] = {}; msg[15] = 0xFF;
    unsigned char aux[32] = {}; aux[0] = 0x42;
    unsigned char s1[64] = {}, s2[64] = {};
    ASSERT_TRUE(secp256k1_schnorrsig_sign32(ctx,s1,msg,&kp,aux)==1, "[SRC-3] sign1 ok");
    ASSERT_TRUE(secp256k1_schnorrsig_sign32(ctx,s2,msg,&kp,aux)==1, "[SRC-3] sign2 ok");
    ASSERT_TRUE(std::memcmp(s1,s2,64)==0, "[SRC-3] sign32 must be deterministic");
}

static void test_sign_custom_32byte(secp256k1_context* ctx) {
    unsigned char sk[32] = {}; sk[31] = 9;
    secp256k1_keypair kp{};
    if (secp256k1_keypair_create(ctx,&kp,sk)!=1) return;
    unsigned char msg[32] = {}; msg[31] = 0x77;
    unsigned char sig[64] = {};
    int rc = secp256k1_schnorrsig_sign_custom(ctx, sig, msg, 32, &kp, nullptr);
    ASSERT_TRUE(rc == 1, "[SRC-4] sign_custom 32-byte must return 1");
    unsigned int r = 0; for (int i=0;i<32;i++) r |= sig[i];
    ASSERT_TRUE(r != 0, "[SRC-4] r not all-zero");
}

static void test_sign_custom_64byte(secp256k1_context* ctx) {
    unsigned char sk[32] = {}; sk[31] = 13;
    secp256k1_keypair kp{};
    if (secp256k1_keypair_create(ctx,&kp,sk)!=1) return;
    unsigned char msg[64] = {}; msg[0] = 0xDE; msg[32] = 0xAD;
    unsigned char sig[64] = {};
    int rc = secp256k1_schnorrsig_sign_custom(ctx, sig, msg, 64, &kp, nullptr);
    ASSERT_TRUE(rc == 1, "[SRC-5] sign_custom 64-byte must return 1");
    unsigned int r = 0; for (int i=0;i<32;i++) r |= sig[i];
    ASSERT_TRUE(r != 0, "[SRC-5] r not all-zero");
}

int test_regression_schnorr_r_zero_ct_run() {
    g_fail = 0;
    // When the libsecp256k1 shim is not linked into this binary the weak
    // function pointers resolve to nullptr — guard against that before calling.
    if (!secp256k1_context_create || !secp256k1_context_destroy ||
        !secp256k1_keypair_create || !secp256k1_schnorrsig_sign32 ||
        !secp256k1_schnorrsig_sign_custom) {
        std::printf("SKIP SEC-006: shim not linked\n");
        return ADVISORY_SKIP_CODE;
    }
    secp256k1_context* ctx = secp256k1_context_create(CTX_SIGN | CTX_VERIFY);
    if (!ctx) {
        std::printf("SKIP SEC-006: shim not linked\n");
        return ADVISORY_SKIP_CODE;
    }
    test_sign32_normal(ctx);
    test_sign32_different_keys(ctx);
    test_sign32_deterministic(ctx);
    test_sign_custom_32byte(ctx);
    test_sign_custom_64byte(ctx);
    secp256k1_context_destroy(ctx);
    if (g_fail == 0)
        std::printf("PASS: Schnorr shim R-zero CT check (SEC-006, SRC-1..5)\n");
    else
        std::printf("FAIL: Schnorr shim R-zero CT check: %d failure(s)\n", g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_schnorr_r_zero_ct_run(); }
#endif
