// Runtime provider/hook smoke for the libbitcoin direct GPU-accelerated column
// verify path. This test lives IN the libbitcoin direct profile itself (not the
// C ABI audit path) so the direct-GPU opt-in profile carries its own acceptance
// smoke, visible in `ctest -N` / target help as `lbtc_direct_gpu_columns_hook`.
//
// It asserts the engine's GpuColumnsVerifyHook is self-installed at process
// startup — BEFORE this test installs any double — by the GPU-host self-installer
// (EngineGpuColumnsInstaller in secp256k1_gpu_host), which this executable retains
// at link via the targeted `--undefined=secp256k1_gpu_columns_provider_anchor`
// anchor. A null hook here means that provider object was dropped at link and
// "transparent GPU column verify" silently degraded to CPU-only — exactly what
// this smoke guards. It then runs a small valid column batch through the unified
// engine call (GPU when a device exists, transparent CPU fallback otherwise) and
// a tampered-row fail-closed check — all through the ONE caller-visible API, with
// no CPU/GPU split and no caller-visible GPU status.
//
// Build: linked only in the SECP256K1_BUILD_LIBBITCOIN_GPU profile with a GPU
// backend compiled (see compat/libbitcoin_direct/CMakeLists.txt). Returns 0 on
// success, 1 on any failure.
//
// Test-data generation uses CT-backed ufsecp::lbtc::* entrypoints so this file
// emits no deprecated non-CT signing/keypair warnings.
#include "ufsecp/libbitcoin.hpp"
#include "secp256k1/batch_verify.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
std::uint64_t g_xs = 0x243F6A8885A308D3ull;
std::uint8_t nb() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return static_cast<std::uint8_t>(g_xs); }
int fails = 0;
void check(bool cond, const char* what) { if (!cond) { std::printf("FAIL: %s\n", what); ++fails; } }

// Generate a valid random secret key (CT-backed via ufsecp::lbtc::seckey_verify).
void rand_sk(std::uint8_t sk[32]) {
    do { for (int i = 0; i < 32; ++i) sk[i] = nb(); } while (!ufsecp::lbtc::seckey_verify(sk));
}
} // namespace

int main() {
    // (1) STARTUP ASSERTION — read BEFORE any test double touches the hook.
    // Non-destructive read: swap the installed hook out to null, capturing it, then
    // restore it immediately. The GPU-host installer runs at load, before main, so a
    // non-null capture proves the provider TU is linked and self-installed.
    secp256k1::GpuColumnsVerifyHook startup_hook =
        secp256k1::install_gpu_columns_verify_hook(nullptr);
    secp256k1::install_gpu_columns_verify_hook(startup_hook);  // restore immediately
    check(startup_hook != nullptr,
          "GpuColumnsVerifyHook self-installed at startup by secp256k1_gpu_host (provider TU retained)");

    // (2) Transparent accelerated path: a small valid ECDSA + Schnorr column batch
    // through the unified engine surface. With the hook installed the engine
    // dispatches to the GPU backend when a device exists and transparently falls
    // back to CPU otherwise; the caller sees ONE API and all-valid results either way.
    constexpr int N = 256;
    std::vector<std::uint8_t> cd(N * 32), cp(N * 33), cs(N * 64);  // ecdsa digests/pubkeys/sigs
    std::vector<std::uint8_t> sx(N * 32), ss(N * 64);              // schnorr xonly/sigs (shares cd digests)
    std::uint8_t aux[32]{};
    for (int i = 0; i < N; ++i) {
        std::uint8_t sk[32], msg[32], pub33[33], esig[64], xonly[32], ssig[64];
        rand_sk(sk);
        for (int j = 0; j < 32; ++j) msg[j] = nb();
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "smoke ecdsa pubkey_create");   // CT-backed
        check(ufsecp::lbtc::ecdsa_sign(msg, sk, esig), "smoke ecdsa_sign");           // CT-backed
        std::memcpy(cd.data() + i * 32, msg, 32);
        std::memcpy(cp.data() + i * 33, pub33, 33);
        std::memcpy(cs.data() + i * 64, esig, 64);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, xonly), "smoke schnorr keypair_create");  // CT-backed
        check(ufsecp::lbtc::schnorr_sign(xonly, sk, msg, aux, ssig), "smoke schnorr_sign");      // CT-backed
        std::memcpy(sx.data() + i * 32, xonly, 32);
        std::memcpy(ss.data() + i * 64, ssig, 64);
    }
    std::vector<std::uint8_t> er(N, 0), sr(N, 0);
    check(secp256k1::ecdsa_batch_verify_opaque_columns(cd.data(), cp.data(), cs.data(), N, er.data(), 0),
          "engine ecdsa columns all-valid via installed hook");
    { long ok = 0; for (auto v : er) ok += v; check(ok == N, "ecdsa columns per-row all-valid"); }
    check(secp256k1::schnorr_batch_verify_bip340_columns(cd.data(), sx.data(), ss.data(), N, sr.data(), 0),
          "engine schnorr columns all-valid via installed hook");
    { long ok = 0; for (auto v : sr) ok += v; check(ok == N, "schnorr columns per-row all-valid"); }

    // (3) Fail-closed through the same accelerated path: tamper one row.
    cs[13 * 64] ^= 1;
    std::vector<std::uint8_t> er2(N, 0);
    check(!secp256k1::ecdsa_batch_verify_opaque_columns(cd.data(), cp.data(), cs.data(), N, er2.data(), 0),
          "engine ecdsa columns fail-closed on tamper via installed hook");
    check(er2[13] == 0, "ecdsa tampered row marked invalid");

    if (fails == 0)
        std::printf("test_direct_gpu_columns_hook: ALL PASS (hook installed at startup + transparent columns + fail-closed)\n");
    return fails == 0 ? 0 : 1;
}
