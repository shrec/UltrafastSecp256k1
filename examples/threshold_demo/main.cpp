// ============================================================================
// UltrafastSecp256k1 -- FROST Threshold Signature Demo (2-of-3)
// ============================================================================
// Demonstrates a full FROST 2-of-3 threshold signing ceremony:
//   1. Distributed Key Generation (DKG): 3 participants
//   2. Signing Round 1: Nonce commitment (2 signers)
//   3. Signing Round 2: Partial signatures
//   4. Aggregation into standard BIP-340 Schnorr signature
//   5. Verification with standard schnorr_verify
//
// Build: cmake --build <build_dir> --target example_threshold_demo
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/frost.hpp"

using namespace secp256k1;
using namespace secp256k1::fast;

// -- helpers ------------------------------------------------------------------

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("  %s: ", label);
    for (size_t i = 0; i < len; ++i) printf("%02x", data[i]);
    printf("\n");
}

// Deterministic seed per participant (DEMO ONLY — use CSPRNG in production!)
static std::array<uint8_t, 32> make_seed(uint8_t id, uint8_t round) {
    std::array<uint8_t, 32> seed{};
    seed[0] = id;
    seed[1] = round;
    for (int i = 2; i < 32; ++i) {
        seed[i] = static_cast<uint8_t>((id * 0x6d + round * 0xb5 + i) & 0xff);
    }
    return seed;
}

// -- main ---------------------------------------------------------------------

int main() {
    printf("=== UltrafastSecp256k1 -- FROST 2-of-3 Threshold Demo ===\n\n");

    constexpr uint32_t T = 2;   // threshold
    constexpr uint32_t N = 3;   // total participants

    // ── Phase 1: Distributed Key Generation (DKG) ────────────────────────────

    printf("[Phase 1] Distributed Key Generation (DKG)\n");
    printf("  Threshold: %u-of-%u\n\n", T, N);

    // Round 1: Each participant generates commitments + shares
    std::vector<FrostCommitment> all_commitments(N);
    // shares[i][j] = share from participant i+1 for participant j+1
    std::vector<std::vector<FrostShare>> all_shares(N);

    for (uint32_t i = 0; i < N; ++i) {
        auto id = static_cast<ParticipantId>(i + 1);
        auto seed = make_seed(static_cast<uint8_t>(id), 1);

        auto [commitment, shares] = frost_keygen_begin(id, T, N, seed);
        all_commitments[i] = commitment;
        all_shares[i] = shares;

        printf("  Participant %u: generated commitment (%zu coeffs) + %zu shares\n",
               id, commitment.coeffs.size(), shares.size());
    }
    printf("\n");

    // Round 2: Each participant receives shares and finalizes key
    std::vector<FrostKeyPackage> key_packages(N);

    for (uint32_t i = 0; i < N; ++i) {
        auto id = static_cast<ParticipantId>(i + 1);

        // Collect shares destined for participant id
        std::vector<FrostShare> my_shares;
        for (uint32_t j = 0; j < N; ++j) {
            for (const auto& share : all_shares[j]) {
                if (share.id == id) {
                    my_shares.push_back(share);
                }
            }
        }

        auto [key_pkg, ok] = frost_keygen_finalize(
            id, all_commitments, my_shares, T, N);

        if (!ok) {
            printf("  ERROR: Participant %u DKG failed!\n", id);
            return 1;
        }

        key_packages[i] = key_pkg;

        auto vshare_bytes = key_pkg.verification_share.to_compressed();
        printf("  Participant %u: DKG success, verification share = ",  id);
        for (int b = 0; b < 6; ++b) printf("%02x", vshare_bytes[b]);
        printf("...\n");
    }

    // Verify all participants derived the same group public key
    auto gpk = key_packages[0].group_public_key;
    for (uint32_t i = 1; i < N; ++i) {
        auto gpk_i_bytes = key_packages[i].group_public_key.to_compressed();
        auto gpk_bytes = gpk.to_compressed();
        if (gpk_bytes != gpk_i_bytes) {
            printf("  ERROR: Group public key mismatch!\n");
            return 1;
        }
    }

    auto gpk_bytes = gpk.to_compressed();
    printf("\n  Group public key: ");
    for (int b = 0; b < 8; ++b) printf("%02x", gpk_bytes[b]);
    printf("...\n\n");

    // ── Phase 2: Signing Ceremony ────────────────────────────────────────────
    //
    // Participants 1 and 2 sign (any 2-of-3 subset works)

    printf("[Phase 2] Signing Ceremony (participants 1 & 2)\n");

    // Message to sign
    std::array<uint8_t, 32> msg{};
    const char* text = "FROST threshold signing demo";
    memcpy(msg.data(), text, strlen(text) < 32 ? strlen(text) : 32);
    print_hex("Message    ", msg.data(), msg.size());
    printf("\n");

    // Signing Round 1: Generate nonce commitments
    printf("  [Round 1] Nonce generation\n");

    std::vector<ParticipantId> signers = {1, 2};
    std::vector<FrostNonce> nonces(2);
    std::vector<FrostNonceCommitment> nonce_commitments(2);

    for (size_t i = 0; i < signers.size(); ++i) {
        auto nonce_seed = make_seed(static_cast<uint8_t>(signers[i]), 2);
        auto [nonce, commitment] = frost_sign_nonce_gen(signers[i], nonce_seed);
        nonces[i] = nonce;
        nonce_commitments[i] = commitment;
        printf("    Signer %u: nonce commitment generated\n", signers[i]);
    }
    printf("\n");

    // Signing Round 2: Compute partial signatures
    printf("  [Round 2] Partial signatures\n");
    std::vector<FrostPartialSig> partial_sigs;

    for (size_t i = 0; i < signers.size(); ++i) {
        uint32_t idx = signers[i] - 1;  // 0-based index into key_packages

        auto partial = frost_sign(
            key_packages[idx], nonces[i], msg, nonce_commitments);

        partial_sigs.push_back(partial);

        auto z_hex = partial.z_i.to_hex();
        printf("    Signer %u: z_i = %s...%s\n", signers[i],
               z_hex.substr(0, 8).c_str(),
               z_hex.substr(z_hex.size() - 8).c_str());
    }
    printf("\n");

    // ── Phase 3: Aggregation ─────────────────────────────────────────────────

    printf("[Phase 3] Signature Aggregation\n");

    auto final_sig = frost_aggregate(
        partial_sigs, nonce_commitments, gpk, msg);

    auto sig_bytes = final_sig.to_bytes();
    print_hex("Final signature", sig_bytes.data(), sig_bytes.size());
    printf("  Format: Standard BIP-340 Schnorr (64 bytes)\n\n");

    // ── Phase 4: Verification ────────────────────────────────────────────────

    printf("[Phase 4] Verification\n");

    // Extract x-only key (32 bytes) for BIP-340 verification
    auto gpk_x = gpk.x().to_bytes();
    bool valid = schnorr_verify(gpk_x, msg, final_sig);
    printf("  schnorr_verify(group_pubkey, msg, sig) = %s\n",
           valid ? "PASS" : "FAIL");

    // Tampered message
    auto bad_msg = msg;
    bad_msg[0] ^= 0x01;
    bool bad_valid = schnorr_verify(gpk_x, bad_msg, final_sig);
    printf("  Tampered message verify = %s (expected FAIL)\n",
           bad_valid ? "PASS" : "FAIL");
    printf("\n");

    // ── Summary ──────────────────────────────────────────────────────────────

    bool all_pass = valid && !bad_valid;
    printf("=== %s ===\n", all_pass
           ? "FROST 2-of-3 demo completed successfully"
           : "FROST demo FAILED");

    return all_pass ? 0 : 1;
}
