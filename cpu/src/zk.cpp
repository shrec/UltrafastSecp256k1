// ============================================================================
// Zero-Knowledge Proof Layer -- Implementation
// ============================================================================

#include "secp256k1/zk.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace secp256k1 {
namespace zk {

using fast::Point;
using fast::Scalar;
using fast::FieldElement;

// -- Tagged hash midstates for ZK proofs --------------------------------------

namespace {

const SHA256 g_knowledge_midstate = detail::make_tag_midstate("ZK/knowledge");
const SHA256 g_dleq_midstate      = detail::make_tag_midstate("ZK/dleq");
const SHA256 g_bp_A_midstate      = detail::make_tag_midstate("Bulletproof/A");
const SHA256 g_bp_S_midstate      = detail::make_tag_midstate("Bulletproof/S");
const SHA256 g_bp_y_midstate      = detail::make_tag_midstate("Bulletproof/y");
const SHA256 g_bp_z_midstate      = detail::make_tag_midstate("Bulletproof/z");
const SHA256 g_bp_x_midstate      = detail::make_tag_midstate("Bulletproof/x");
const SHA256 g_bp_ip_midstate     = detail::make_tag_midstate("Bulletproof/ip");
const SHA256 g_bp_gen_midstate    = detail::make_tag_midstate("Bulletproof/gen");
const SHA256 g_nonce_midstate     = detail::make_tag_midstate("ZK/nonce");

// Deterministic nonce derivation: k = H("ZK/nonce" || secret || point_ser || msg || aux)
Scalar derive_nonce(const Scalar& secret, const Point& point,
                    const std::uint8_t* msg32,
                    const std::uint8_t* aux32) {
    auto sec_bytes = secret.to_bytes();
    auto pt_comp = point.to_compressed();

    // XOR secret with H(aux) for nonce hedging
    auto aux_hash = SHA256::hash(aux32, 32);
    std::uint8_t masked[32];
    for (int i = 0; i < 32; ++i) {
        masked[i] = sec_bytes[i] ^ aux_hash[i];
    }

    std::uint8_t buf[32 + 33 + 32 + 32]; // masked || pt_comp || msg || aux
    std::memcpy(buf, masked, 32);
    std::memcpy(buf + 32, pt_comp.data(), 33);
    std::memcpy(buf + 65, msg32, 32);
    std::memcpy(buf + 97, aux32, 32);

    auto hash = detail::cached_tagged_hash(g_nonce_midstate, buf, sizeof(buf));
    detail::secure_erase(sec_bytes.data(), sec_bytes.size());
    detail::secure_erase(masked, sizeof(masked));
    detail::secure_erase(buf, sizeof(buf));
    return Scalar::from_bytes(hash);
}

// lift_x: recover point from x-coordinate (even Y)
Point lift_x_even(const FieldElement& x_in) {
    FieldElement x = x_in;
    for (int attempt = 0; attempt < 256; ++attempt) {
        FieldElement const x2 = x * x;
        FieldElement const x3 = x2 * x;
        FieldElement const rhs = x3 + FieldElement::from_uint64(7);
        FieldElement y = rhs.sqrt();
        if (y.square() == rhs) {
            auto y_bytes = y.to_bytes();
            if (y_bytes[31] & 1) {
                y = FieldElement::zero() - y;
            }
            return Point::from_affine(x, y);
        }
        x = x + FieldElement::one();
    }
    return Point::infinity();
}

} // anonymous namespace


// ============================================================================
// KnowledgeProof serialization
// ============================================================================

std::array<std::uint8_t, 64> KnowledgeProof::serialize() const {
    std::array<std::uint8_t, 64> out{};
    std::memcpy(out.data(), rx.data(), 32);
    auto s_bytes = s.to_bytes();
    std::memcpy(out.data() + 32, s_bytes.data(), 32);
    return out;
}

bool KnowledgeProof::deserialize(const std::uint8_t* data64, KnowledgeProof& out) {
    std::memcpy(out.rx.data(), data64, 32);
    auto s_bytes = std::array<std::uint8_t, 32>{};
    std::memcpy(s_bytes.data(), data64 + 32, 32);
    out.s = Scalar::from_bytes(s_bytes);
    return !out.s.is_zero();
}


// ============================================================================
// 1. Schnorr Knowledge Proof
// ============================================================================

KnowledgeProof knowledge_prove(const Scalar& secret,
                                const Point& pubkey,
                                const std::array<std::uint8_t, 32>& msg,
                                const std::array<std::uint8_t, 32>& aux_rand) {
    return knowledge_prove_base(secret, pubkey, Point::generator(), msg, aux_rand);
}

bool knowledge_verify(const KnowledgeProof& proof,
                      const Point& pubkey,
                      const std::array<std::uint8_t, 32>& msg) {
    return knowledge_verify_base(proof, pubkey, Point::generator(), msg);
}

KnowledgeProof knowledge_prove_base(const Scalar& secret,
                                     const Point& point,
                                     const Point& base,
                                     const std::array<std::uint8_t, 32>& msg,
                                     const std::array<std::uint8_t, 32>& aux_rand) {
    // k = deterministic nonce
    Scalar const k = derive_nonce(secret, point, msg.data(), aux_rand.data());

    // R = k * base (constant-time: k is secret)
    Point R = ct::scalar_mul(base, k);

    // Ensure even Y for R (BIP-340 style)
    auto R_comp = R.to_compressed();
    Scalar k_eff = k;
    if (R_comp[0] == 0x03) {
        R = R.negate();
        k_eff = k_eff.negate();
        R_comp = R.to_compressed();
    }

    // Extract R.x
    std::array<std::uint8_t, 32> rx{};
    std::memcpy(rx.data(), R_comp.data() + 1, 32);

    // e = H("ZK/knowledge" || R.x || P_compressed || base_compressed || msg)
    auto P_comp = point.to_compressed();
    auto B_comp = base.to_compressed();
    std::uint8_t buf[32 + 33 + 33 + 32]; // rx || P || B || msg
    std::memcpy(buf, rx.data(), 32);
    std::memcpy(buf + 32, P_comp.data(), 33);
    std::memcpy(buf + 65, B_comp.data(), 33);
    std::memcpy(buf + 98, msg.data(), 32);

    auto e_hash = detail::cached_tagged_hash(g_knowledge_midstate, buf, sizeof(buf));
    Scalar const e = Scalar::from_bytes(e_hash);

    // s = k + e * secret (constant-time: both k and secret are secrets)
    Scalar const s = k_eff + e * secret;

    detail::secure_erase(const_cast<Scalar*>(&k), sizeof(k));
    detail::secure_erase(&k_eff, sizeof(k_eff));

    return KnowledgeProof{rx, s};
}

bool knowledge_verify_base(const KnowledgeProof& proof,
                           const Point& point,
                           const Point& base,
                           const std::array<std::uint8_t, 32>& msg) {
    // Reconstruct challenge e
    auto P_comp = point.to_compressed();
    auto B_comp = base.to_compressed();
    std::uint8_t buf[32 + 33 + 33 + 32];
    std::memcpy(buf, proof.rx.data(), 32);
    std::memcpy(buf + 32, P_comp.data(), 33);
    std::memcpy(buf + 65, B_comp.data(), 33);
    std::memcpy(buf + 98, msg.data(), 32);

    auto e_hash = detail::cached_tagged_hash(g_knowledge_midstate, buf, sizeof(buf));
    Scalar const e = Scalar::from_bytes(e_hash);

    // Verify: s*base == R + e*point
    // => s*base - e*point == R
    // => Check that (s*base - e*point).x == proof.rx with even Y
    Point const sB = base.scalar_mul(proof.s);
    Point const eP = point.scalar_mul(e);
    Point const R_check = sB.add(eP.negate());

    if (R_check.is_infinity()) return false;

    auto R_comp = R_check.to_compressed();
    // Must have even Y
    if (R_comp[0] != 0x02) return false;
    // Check x-coordinate matches
    return std::memcmp(R_comp.data() + 1, proof.rx.data(), 32) == 0;
}


// ============================================================================
// 2. DLEQ Proof
// ============================================================================

std::array<std::uint8_t, 64> DLEQProof::serialize() const {
    std::array<std::uint8_t, 64> out{};
    auto e_bytes = e.to_bytes();
    auto s_bytes = s.to_bytes();
    std::memcpy(out.data(), e_bytes.data(), 32);
    std::memcpy(out.data() + 32, s_bytes.data(), 32);
    return out;
}

bool DLEQProof::deserialize(const std::uint8_t* data64, DLEQProof& out) {
    auto e_bytes = std::array<std::uint8_t, 32>{};
    auto s_bytes = std::array<std::uint8_t, 32>{};
    std::memcpy(e_bytes.data(), data64, 32);
    std::memcpy(s_bytes.data(), data64 + 32, 32);
    out.e = Scalar::from_bytes(e_bytes);
    out.s = Scalar::from_bytes(s_bytes);
    return true;
}

DLEQProof dleq_prove(const Scalar& secret,
                      const Point& G,
                      const Point& H,
                      const Point& P,
                      const Point& Q,
                      const std::array<std::uint8_t, 32>& aux_rand) {
    // Deterministic nonce: k = H("ZK/nonce" || secret || P || aux)
    Scalar const k = derive_nonce(secret, P, Q.to_compressed().data(), aux_rand.data());

    // R1 = k*G, R2 = k*H (constant-time: k is secret)
    Point const R1 = ct::scalar_mul(G, k);
    Point const R2 = ct::scalar_mul(H, k);

    // e = H("ZK/dleq" || G_comp || H_comp || P_comp || Q_comp || R1_comp || R2_comp)
    auto G_comp  = G.to_compressed();
    auto H_comp  = H.to_compressed();
    auto P_comp  = P.to_compressed();
    auto Q_comp  = Q.to_compressed();
    auto R1_comp = R1.to_compressed();
    auto R2_comp = R2.to_compressed();

    std::uint8_t buf[33 * 6]; // 6 compressed points
    std::memcpy(buf,       G_comp.data(),  33);
    std::memcpy(buf + 33,  H_comp.data(),  33);
    std::memcpy(buf + 66,  P_comp.data(),  33);
    std::memcpy(buf + 99,  Q_comp.data(),  33);
    std::memcpy(buf + 132, R1_comp.data(), 33);
    std::memcpy(buf + 165, R2_comp.data(), 33);

    auto e_hash = detail::cached_tagged_hash(g_dleq_midstate, buf, sizeof(buf));
    Scalar const e = Scalar::from_bytes(e_hash);

    // s = k + e * secret (constant-time)
    Scalar const s = k + e * secret;

    detail::secure_erase(const_cast<Scalar*>(&k), sizeof(k));

    return DLEQProof{e, s};
}

bool dleq_verify(const DLEQProof& proof,
                 const Point& G,
                 const Point& H,
                 const Point& P,
                 const Point& Q) {
    // Reconstruct R1 = s*G - e*P, R2 = s*H - e*Q
    Point const sG = G.scalar_mul(proof.s);
    Point const eP = P.scalar_mul(proof.e);
    Point const R1 = sG.add(eP.negate());

    Point const sH = H.scalar_mul(proof.s);
    Point const eQ = Q.scalar_mul(proof.e);
    Point const R2 = sH.add(eQ.negate());

    if (R1.is_infinity() || R2.is_infinity()) return false;

    // Recompute challenge
    auto G_comp  = G.to_compressed();
    auto H_comp  = H.to_compressed();
    auto P_comp  = P.to_compressed();
    auto Q_comp  = Q.to_compressed();
    auto R1_comp = R1.to_compressed();
    auto R2_comp = R2.to_compressed();

    std::uint8_t buf[33 * 6];
    std::memcpy(buf,       G_comp.data(),  33);
    std::memcpy(buf + 33,  H_comp.data(),  33);
    std::memcpy(buf + 66,  P_comp.data(),  33);
    std::memcpy(buf + 99,  Q_comp.data(),  33);
    std::memcpy(buf + 132, R1_comp.data(), 33);
    std::memcpy(buf + 165, R2_comp.data(), 33);

    auto e_hash = detail::cached_tagged_hash(g_dleq_midstate, buf, sizeof(buf));
    Scalar const e_check = Scalar::from_bytes(e_hash);

    // e must match
    return proof.e == e_check;
}


// ============================================================================
// Bulletproof Generator Vectors
// ============================================================================

const GeneratorVectors& get_generator_vectors() {
    static const GeneratorVectors vecs = []() {
        GeneratorVectors v{};
        for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
            // G_i = lift_x(H("Bulletproof/gen" || "G" || LE32(i)))
            std::uint8_t buf_g[5];
            buf_g[0] = 'G';
            buf_g[1] = static_cast<std::uint8_t>(i & 0xFF);
            buf_g[2] = static_cast<std::uint8_t>((i >> 8) & 0xFF);
            buf_g[3] = static_cast<std::uint8_t>((i >> 16) & 0xFF);
            buf_g[4] = static_cast<std::uint8_t>((i >> 24) & 0xFF);
            auto hash_g = detail::cached_tagged_hash(g_bp_gen_midstate, buf_g, 5);
            v.G[i] = lift_x_even(FieldElement::from_bytes(hash_g));

            // H_i = lift_x(H("Bulletproof/gen" || "H" || LE32(i)))
            buf_g[0] = 'H';
            auto hash_h = detail::cached_tagged_hash(g_bp_gen_midstate, buf_g, 5);
            v.H[i] = lift_x_even(FieldElement::from_bytes(hash_h));
        }
        return v;
    }();
    return vecs;
}


// ============================================================================
// 3. Bulletproof Range Proof
// ============================================================================

// Helper: hash point for Fiat-Shamir transcript
[[maybe_unused]]
static void transcript_append_point(SHA256& ctx, const Point& p) {
    auto comp = p.to_compressed();
    ctx.update(comp.data(), 33);
}

[[maybe_unused]]
static void transcript_append_scalar(SHA256& ctx, const Scalar& s) {
    auto bytes = s.to_bytes();
    ctx.update(bytes.data(), 32);
}

RangeProof range_prove(std::uint64_t value,
                        const Scalar& blinding,
                        const PedersenCommitment& commitment,
                        const std::array<std::uint8_t, 32>& aux_rand) {
    const auto& gens = get_generator_vectors();
    const Point& H_ped = pedersen_generator_H();
    RangeProof proof{};

    // Bit decomposition: a_L[i] = (value >> i) & 1
    // a_R[i] = a_L[i] - 1
    Scalar a_L[RANGE_PROOF_BITS];
    Scalar a_R[RANGE_PROOF_BITS];
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        std::uint64_t const bit = (value >> i) & 1;
        a_L[i] = Scalar::from_uint64(bit);
        a_R[i] = Scalar::from_uint64(bit) - Scalar::one();
    }

    // Random blinding scalars for vector commitments
    const Scalar alpha = derive_nonce(blinding, commitment.point,
                                aux_rand.data(), aux_rand.data());
    // Derive more randomness
    auto alpha_bytes = alpha.to_bytes();
    const Scalar rho = Scalar::from_bytes(SHA256::hash(alpha_bytes.data(), 32));

    // Random blinding vectors s_L, s_R
    Scalar s_L[RANGE_PROOF_BITS];
    Scalar s_R[RANGE_PROOF_BITS];
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        std::uint8_t buf[32 + 1 + 1];
        std::memcpy(buf, alpha_bytes.data(), 32);
        buf[32] = static_cast<std::uint8_t>(i);
        buf[33] = 'L';
        s_L[i] = Scalar::from_bytes(SHA256::hash(buf, 34));
        buf[33] = 'R';
        s_R[i] = Scalar::from_bytes(SHA256::hash(buf, 34));
    }

    // A = alpha*G + sum(a_L[i]*G_i + a_R[i]*H_i)
    Point A_pt = ct::generator_mul(alpha);
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        if (!a_L[i].is_zero()) {
            A_pt = A_pt.add(gens.G[i].scalar_mul(a_L[i]));
        }
        A_pt = A_pt.add(gens.H[i].scalar_mul(a_R[i]));
    }
    proof.A = A_pt;
    Point const& A = proof.A;

    // S = rho*G + sum(s_L[i]*G_i + s_R[i]*H_i)
    Point S_pt = ct::generator_mul(rho);
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        S_pt = S_pt.add(gens.G[i].scalar_mul(s_L[i]));
        S_pt = S_pt.add(gens.H[i].scalar_mul(s_R[i]));
    }
    proof.S = S_pt;
    Point const& S = proof.S;

    // Fiat-Shamir: y = H(A || S || V)
    std::uint8_t fs_buf[33 + 33 + 33];
    auto A_comp = A.to_compressed();
    auto S_comp = S.to_compressed();
    auto V_comp = commitment.to_compressed();
    std::memcpy(fs_buf, A_comp.data(), 33);
    std::memcpy(fs_buf + 33, S_comp.data(), 33);
    std::memcpy(fs_buf + 66, V_comp.data(), 33);

    auto y_hash = detail::cached_tagged_hash(g_bp_y_midstate, fs_buf, sizeof(fs_buf));
    Scalar const y = Scalar::from_bytes(y_hash);

    auto z_hash = detail::cached_tagged_hash(g_bp_z_midstate, fs_buf, sizeof(fs_buf));
    Scalar const z = Scalar::from_bytes(z_hash);

    // Compute powers of y and z
    Scalar y_powers[RANGE_PROOF_BITS]; // y^0, y^1, ..., y^{n-1}
    y_powers[0] = Scalar::one();
    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        y_powers[i] = y_powers[i - 1] * y;
    }

    Scalar const z2 = z * z;

    // 2^i scalars
    Scalar two_powers[RANGE_PROOF_BITS];
    two_powers[0] = Scalar::one();
    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        two_powers[i] = two_powers[i - 1] + two_powers[i - 1];
    }

    // l(x) = (a_L - z*1) + s_L*x
    // r(x) = y^n * (a_R + z*1 + s_R*x) + z^2 * 2^n
    // t(x) = <l(x), r(x)> = t_0 + t_1*x + t_2*x^2

    // Compute t_1 and t_2 coefficients
    // t_1 = <a_L - z, y^n * s_R> + <s_L, y^n * (a_R + z) + z^2 * 2^n>
    // t_2 = <s_L, y^n * s_R>
    Scalar t1 = Scalar::zero();
    Scalar t2 = Scalar::zero();
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        Scalar const l0_i = a_L[i] - z;
        Scalar const r0_i = y_powers[i] * (a_R[i] + z) + z2 * two_powers[i];
        Scalar const l1_i = s_L[i];
        Scalar const r1_i = y_powers[i] * s_R[i];

        t1 = t1 + l0_i * r1_i + l1_i * r0_i;
        t2 = t2 + l1_i * r1_i;
    }

    // Commit to t_1, t_2
    const Scalar tau1_bytes_raw = Scalar::from_bytes(
        SHA256::hash(rho.to_bytes().data(), 32));
    const Scalar tau2_bytes_raw = Scalar::from_bytes(
        SHA256::hash(tau1_bytes_raw.to_bytes().data(), 32));
    Scalar const tau1 = tau1_bytes_raw;
    Scalar const tau2 = tau2_bytes_raw;

    // T1 = t_1*H + tau_1*G, T2 = t_2*H + tau_2*G
    proof.T1 = H_ped.scalar_mul(t1).add(ct::generator_mul(tau1));
    proof.T2 = H_ped.scalar_mul(t2).add(ct::generator_mul(tau2));

    // Fiat-Shamir: x = H(T1 || T2 || y || z)
    auto T1_comp = proof.T1.to_compressed();
    auto T2_comp = proof.T2.to_compressed();
    std::uint8_t x_buf[33 + 33 + 32 + 32];
    std::memcpy(x_buf, T1_comp.data(), 33);
    std::memcpy(x_buf + 33, T2_comp.data(), 33);
    auto y_bytes = y.to_bytes();
    auto z_bytes = z.to_bytes();
    std::memcpy(x_buf + 66, y_bytes.data(), 32);
    std::memcpy(x_buf + 98, z_bytes.data(), 32);

    auto x_hash = detail::cached_tagged_hash(g_bp_x_midstate, x_buf, sizeof(x_buf));
    Scalar const x = Scalar::from_bytes(x_hash);

    // Evaluate l(x), r(x)
    Scalar l_x[RANGE_PROOF_BITS];
    Scalar r_x[RANGE_PROOF_BITS];
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        l_x[i] = (a_L[i] - z) + s_L[i] * x;
        r_x[i] = y_powers[i] * (a_R[i] + z + s_R[i] * x) + z2 * two_powers[i];
    }

    // t_hat = <l(x), r(x)>
    Scalar t_hat = Scalar::zero();
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        t_hat = t_hat + l_x[i] * r_x[i];
    }
    proof.t_hat = t_hat;

    // tau_x = tau_2 * x^2 + tau_1 * x + z^2 * blinding
    Scalar const x2 = x * x;
    proof.tau_x = tau2 * x2 + tau1 * x + z2 * blinding;

    // mu = alpha + rho * x
    proof.mu = alpha + rho * x;

    // Inner product argument
    // Reduce l_x, r_x vectors using recursive halving
    Scalar a_vec[RANGE_PROOF_BITS];
    Scalar b_vec[RANGE_PROOF_BITS];
    std::memcpy(a_vec, l_x, sizeof(l_x));
    std::memcpy(b_vec, r_x, sizeof(r_x));

    // Compute modified generators: H'_i = H_i * y^{-i}
    const Scalar y_inv = y.inverse();
    Scalar y_inv_powers[RANGE_PROOF_BITS];
    y_inv_powers[0] = Scalar::one();
    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        y_inv_powers[i] = y_inv_powers[i - 1] * y_inv;
    }

    Point G_vec[RANGE_PROOF_BITS];
    Point H_vec[RANGE_PROOF_BITS];
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        G_vec[i] = gens.G[i];
        H_vec[i] = gens.H[i].scalar_mul(y_inv_powers[i]);
    }

    std::size_t n = RANGE_PROOF_BITS;
    for (std::size_t round = 0; round < RANGE_PROOF_LOG2; ++round) {
        n /= 2;

        // L = <a_lo, G_hi> + <b_hi, H'_lo> + <a_lo, b_hi>*U
        // R = <a_hi, G_lo> + <b_lo, H'_hi> + <a_hi, b_lo>*U
        // U = H_ped (inner product base point)
        Scalar c_L = Scalar::zero();
        Scalar c_R = Scalar::zero();
        Point L = Point::infinity();
        Point R_pt = Point::infinity();

        for (std::size_t i = 0; i < n; ++i) {
            L = L.add(G_vec[n + i].scalar_mul(a_vec[i]));
            L = L.add(H_vec[i].scalar_mul(b_vec[n + i]));
            c_L = c_L + a_vec[i] * b_vec[n + i];

            R_pt = R_pt.add(G_vec[i].scalar_mul(a_vec[n + i]));
            R_pt = R_pt.add(H_vec[n + i].scalar_mul(b_vec[i]));
            c_R = c_R + a_vec[n + i] * b_vec[i];
        }
        L = L.add(H_ped.scalar_mul(c_L));
        R_pt = R_pt.add(H_ped.scalar_mul(c_R));

        proof.L[round] = L;
        proof.R[round] = R_pt;

        // Fiat-Shamir: x_round = H("Bulletproof/ip" || L || R)
        auto L_comp = L.to_compressed();
        auto R_comp = R_pt.to_compressed();
        std::uint8_t ip_buf[33 + 33];
        std::memcpy(ip_buf, L_comp.data(), 33);
        std::memcpy(ip_buf + 33, R_comp.data(), 33);
        auto x_r_hash = detail::cached_tagged_hash(g_bp_ip_midstate, ip_buf, sizeof(ip_buf));
        Scalar const x_r = Scalar::from_bytes(x_r_hash);
        Scalar const x_r_inv = x_r.inverse();

        // Fold vectors: a' = a_lo*x + a_hi*x^{-1}, b' = b_lo*x^{-1} + b_hi*x
        for (std::size_t i = 0; i < n; ++i) {
            a_vec[i] = a_vec[i] * x_r + a_vec[n + i] * x_r_inv;
            b_vec[i] = b_vec[i] * x_r_inv + b_vec[n + i] * x_r;
        }
        // G' = G_lo*x^{-1} + G_hi*x, H' = H_lo*x + H_hi*x^{-1}
        for (std::size_t i = 0; i < n; ++i) {
            G_vec[i] = G_vec[i].scalar_mul(x_r_inv).add(G_vec[n + i].scalar_mul(x_r));
            H_vec[i] = H_vec[i].scalar_mul(x_r).add(H_vec[n + i].scalar_mul(x_r_inv));
        }
    }

    proof.a = a_vec[0];
    proof.b = b_vec[0];

    // Erase secret state from stack
    detail::secure_erase(a_L, sizeof(a_L));
    detail::secure_erase(a_R, sizeof(a_R));
    detail::secure_erase(s_L, sizeof(s_L));
    detail::secure_erase(s_R, sizeof(s_R));
    detail::secure_erase(l_x, sizeof(l_x));
    detail::secure_erase(r_x, sizeof(r_x));
    detail::secure_erase(a_vec, sizeof(a_vec));
    detail::secure_erase(b_vec, sizeof(b_vec));
    detail::secure_erase(alpha_bytes.data(), alpha_bytes.size());

    return proof;
}

bool range_verify(const PedersenCommitment& commitment,
                  const RangeProof& proof) {
    const auto& gens = get_generator_vectors();
    const Point& H_ped = pedersen_generator_H();

    // Recompute Fiat-Shamir challenges
    auto A_comp = proof.A.to_compressed();
    auto S_comp = proof.S.to_compressed();
    auto V_comp = commitment.to_compressed();

    std::uint8_t fs_buf[33 + 33 + 33];
    std::memcpy(fs_buf, A_comp.data(), 33);
    std::memcpy(fs_buf + 33, S_comp.data(), 33);
    std::memcpy(fs_buf + 66, V_comp.data(), 33);

    auto y_hash = detail::cached_tagged_hash(g_bp_y_midstate, fs_buf, sizeof(fs_buf));
    Scalar const y = Scalar::from_bytes(y_hash);

    auto z_hash = detail::cached_tagged_hash(g_bp_z_midstate, fs_buf, sizeof(fs_buf));
    Scalar const z = Scalar::from_bytes(z_hash);

    auto T1_comp = proof.T1.to_compressed();
    auto T2_comp = proof.T2.to_compressed();
    std::uint8_t x_buf[33 + 33 + 32 + 32];
    std::memcpy(x_buf, T1_comp.data(), 33);
    std::memcpy(x_buf + 33, T2_comp.data(), 33);
    auto y_bytes = y.to_bytes();
    auto z_bytes = z.to_bytes();
    std::memcpy(x_buf + 66, y_bytes.data(), 32);
    std::memcpy(x_buf + 98, z_bytes.data(), 32);

    auto x_hash = detail::cached_tagged_hash(g_bp_x_midstate, x_buf, sizeof(x_buf));
    Scalar const x = Scalar::from_bytes(x_hash);

    Scalar const x2 = x * x;
    Scalar const z2 = z * z;

    // Verify polynomial commitment:
    // t_hat * H_ped + tau_x * G == z^2 * V + delta(y,z) * H_ped + x * T1 + x^2 * T2
    // where delta(y,z) = (z - z^2) * <1, y^n> - z^3 * <1, 2^n>
    Scalar y_powers[RANGE_PROOF_BITS];
    y_powers[0] = Scalar::one();
    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        y_powers[i] = y_powers[i - 1] * y;
    }

    Scalar two_powers[RANGE_PROOF_BITS];
    two_powers[0] = Scalar::one();
    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        two_powers[i] = two_powers[i - 1] + two_powers[i - 1];
    }

    Scalar sum_y = Scalar::zero();
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        sum_y = sum_y + y_powers[i];
    }

    Scalar sum_2 = Scalar::zero();
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        sum_2 = sum_2 + two_powers[i];
    }

    Scalar const z3 = z2 * z;
    Scalar const delta = (z - z2) * sum_y - z3 * sum_2;

    // Polynomial check via single MSM:
    // (t_hat - delta)*H_ped + tau_x*G - z^2*V - x*T1 - x^2*T2 == 0
    {
        Scalar poly_s[5] = {
            proof.t_hat - delta,   // H_ped coeff
            proof.tau_x,           // G coeff
            z2.negate(),           // V coeff
            x.negate(),            // T1 coeff
            x2.negate()            // T2 coeff
        };
        Point poly_p[5] = {
            H_ped, Point::generator(), commitment.point, proof.T1, proof.T2
        };
        const Point poly_check = msm(poly_s, poly_p, 5);
        if (!poly_check.is_infinity()) return false;
    }

    // Verify inner product argument
    // Reconstruct challenges from L, R pairs
    Scalar x_rounds[RANGE_PROOF_LOG2];
    for (std::size_t round = 0; round < RANGE_PROOF_LOG2; ++round) {
        auto L_comp = proof.L[round].to_compressed();
        auto R_comp = proof.R[round].to_compressed();
        std::uint8_t ip_buf[33 + 33];
        std::memcpy(ip_buf, L_comp.data(), 33);
        std::memcpy(ip_buf + 33, R_comp.data(), 33);
        auto x_r_hash = detail::cached_tagged_hash(g_bp_ip_midstate, ip_buf, sizeof(ip_buf));
        x_rounds[round] = Scalar::from_bytes(x_r_hash);
    }

    // Compute scalar coefficients for each generator
    // s_i = product_{j: bit j of i is 0} x_j^{-1} * product_{j: bit j of i is 1} x_j
    const Scalar y_inv = y.inverse();
    Scalar y_inv_powers[RANGE_PROOF_BITS];
    y_inv_powers[0] = Scalar::one();
    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        y_inv_powers[i] = y_inv_powers[i - 1] * y_inv;
    }

    Scalar x_inv_rounds[RANGE_PROOF_LOG2];
    for (std::size_t j = 0; j < RANGE_PROOF_LOG2; ++j) {
        x_inv_rounds[j] = x_rounds[j].inverse();
    }

    // Compute s_i via product tree (much faster than per-index loop)
    // s_0 = prod(x_inv_rounds[j]), then propagate: flip x_inv->x for each set bit
    Scalar s_coeff[RANGE_PROOF_BITS];
    s_coeff[0] = Scalar::one();
    for (std::size_t j = 0; j < RANGE_PROOF_LOG2; ++j) {
        s_coeff[0] = s_coeff[0] * x_inv_rounds[j];
    }

    for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
        // s[i] = s[i-1] * x_rounds[j] / x_inv_rounds[j] = s[i-1] * x_rounds[j]^2
        // But we also need to undo all lower bits that went from 1->0
        // Simpler: s[i] = s[i & (i-1)] * x_rounds[j]^2 ... actually
        // The butterfly approach: s_i = s_{i without highest changed bit} * ratio
        // Use standard butterfly construction
        s_coeff[i] = Scalar::one();
        for (std::size_t jj = 0; jj < RANGE_PROOF_LOG2; ++jj) {
            if ((i >> (RANGE_PROOF_LOG2 - 1 - jj)) & 1) {
                s_coeff[i] = s_coeff[i] * x_rounds[jj];
            }
            else {
                s_coeff[i] = s_coeff[i] * x_inv_rounds[jj];
            }
        }
    }

    // Merged verification: compute P_check - expected == 0 as single MSM
    // P_check = A + x*S + sum((-z - a*s_i)*G_i) + sum((z + z2*2^i*y^{-i} - b*s_inv_i*y^{-i})*H_i)
    //         - mu*G + (t_hat - a*b)*U + sum(x_j^2*L_j + x_j^{-2}*R_j)
    //
    // Total points: 2 (A,S) + 128 (G_i) + 128 (H_i) + 1 (G) + 1 (U) + 12 (L,R) = ~144
    constexpr std::size_t MSM_SIZE = 2 + 2*RANGE_PROOF_BITS + 1 + 1 + 2*RANGE_PROOF_LOG2;
    Scalar msm_s[MSM_SIZE];
    Point  msm_p[MSM_SIZE];

    std::size_t idx = 0;

    // A (coefficient 1)
    msm_s[idx] = Scalar::one();
    msm_p[idx] = proof.A;
    ++idx;

    // x * S
    msm_s[idx] = x;
    msm_p[idx] = proof.S;
    ++idx;

    // Compute s_inv using batch approach: accumulate products
    Scalar s_inv[RANGE_PROOF_BITS];
    {
        // Product tree for batch inversion
        Scalar acc[RANGE_PROOF_BITS];
        acc[0] = s_coeff[0];
        for (std::size_t i = 1; i < RANGE_PROOF_BITS; ++i) {
            acc[i] = acc[i - 1] * s_coeff[i];
        }

        Scalar inv_acc = acc[RANGE_PROOF_BITS - 1].inverse();  // single inversion!
        for (std::size_t i = RANGE_PROOF_BITS; i-- > 1; ) {
            s_inv[i] = inv_acc * acc[i - 1];
            inv_acc = inv_acc * s_coeff[i];
        }
        s_inv[0] = inv_acc;
    }

    Scalar const neg_z = z.negate();
    Scalar const ab = proof.a * proof.b;

    // G_i coefficients: -z - a*s_i  (P_check: -z*G_i, expected: a*s_i*G_i, diff: -z - a*s_i)
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        msm_s[idx] = neg_z - proof.a * s_coeff[i];
        msm_p[idx] = gens.G[i];
        ++idx;
    }

    // H_i coefficients: (z + z2*2^i*y_inv^i) - b*s_inv[i]*y_inv^i
    for (std::size_t i = 0; i < RANGE_PROOF_BITS; ++i) {
        Scalar const h_pcheck = z + z2 * two_powers[i] * y_inv_powers[i];
        Scalar const h_expect = proof.b * s_inv[i] * y_inv_powers[i];
        msm_s[idx] = h_pcheck - h_expect;
        msm_p[idx] = gens.H[i];
        ++idx;
    }

    // -mu * G (generator)
    msm_s[idx] = proof.mu.negate();
    msm_p[idx] = Point::generator();
    ++idx;

    // (t_hat - a*b) * U  (H_ped)
    msm_s[idx] = proof.t_hat - ab;
    msm_p[idx] = H_ped;
    ++idx;

    // L_j and R_j contributions
    for (std::size_t j = 0; j < RANGE_PROOF_LOG2; ++j) {
        msm_s[idx] = x_rounds[j] * x_rounds[j];
        msm_p[idx] = proof.L[j];
        ++idx;

        msm_s[idx] = x_inv_rounds[j] * x_inv_rounds[j];
        msm_p[idx] = proof.R[j];
        ++idx;
    }

    // Single MSM: if result is infinity, verification passes
    Point const final_check = msm(msm_s, msm_p, MSM_SIZE);
    return final_check.is_infinity();
}


// ============================================================================
// Batch Operations
// ============================================================================

bool batch_range_verify(const PedersenCommitment* commitments,
                        const RangeProof* proofs,
                        std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!range_verify(commitments[i], proofs[i])) {
            return false;
        }
    }
    return true;
}

void batch_commit(const Scalar* values,
                  const Scalar* blindings,
                  PedersenCommitment* commitments_out,
                  std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        commitments_out[i] = pedersen_commit(values[i], blindings[i]);
    }
}


// ============================================================================
// 4. ECDSA-in-SNARK Foreign-Field Witness  (eprint 2025/695)
// ============================================================================

namespace {

// Convert a 4×64-bit scalar (little-endian limbs) to 5×52-bit ForeignFieldLimbs.
// Implements the canonical 52-bit decomposition used by PLONK foreign-field
// gadgets for tight range constraints.
ForeignFieldLimbs scalar_to_ff_limbs(const Scalar& s) noexcept {
    static constexpr std::uint64_t MASK52 = (1ULL << 52) - 1;
    const auto& L = s.limbs();  // 4×64-bit, little-endian
    ForeignFieldLimbs fl{};
    fl.limbs[0] =  L[0]                          & MASK52;
    fl.limbs[1] = ((L[0] >> 52) | (L[1] << 12)) & MASK52;
    fl.limbs[2] = ((L[1] >> 40) | (L[2] << 24)) & MASK52;
    fl.limbs[3] = ((L[2] >> 28) | (L[3] << 36)) & MASK52;
    fl.limbs[4] =   L[3] >> 16;  // at most 48 bits for values < n < p
    return fl;
}

// Convert a FieldElement (mod p, 4×64-bit canonical limbs) to 5×52-bit
// ForeignFieldLimbs.  Implemented directly on the 4×64 limbs to avoid a
// dependency on FieldElement52 (which requires __int128 and is therefore
// unavailable on MSVC / 32-bit targets).  The bit-extraction is identical
// to FieldElement52::from_fe() followed by normalize() for a canonical FE.
ForeignFieldLimbs fe_to_ff_limbs(const FieldElement& fe) noexcept {
    static constexpr std::uint64_t MASK52 = (1ULL << 52) - 1;
    const auto& L = fe.limbs();  // canonical 4×64-bit, little-endian
    ForeignFieldLimbs fl{};
    fl.limbs[0] =  L[0]                          & MASK52;
    fl.limbs[1] = ((L[0] >> 52) | (L[1] << 12)) & MASK52;
    fl.limbs[2] = ((L[1] >> 40) | (L[2] << 24)) & MASK52;
    fl.limbs[3] = ((L[2] >> 28) | (L[3] << 36)) & MASK52;
    fl.limbs[4] =  L[3] >> 16;  // top limb: at most 48 bits for any value < p
    return fl;
}

} // anonymous namespace

EcdsaSnarkWitness ecdsa_snark_witness(
    const std::array<std::uint8_t, 32>& msg_hash,
    const fast::Point& pubkey,
    const fast::Scalar& sig_r,
    const fast::Scalar& sig_s) {

    EcdsaSnarkWitness w{};

    // ---- 1. Encode public inputs in foreign-field limb form ----
    Scalar msg_scalar = Scalar::from_bytes(msg_hash);
    w.msg    = scalar_to_ff_limbs(msg_scalar);
    w.sig_r  = scalar_to_ff_limbs(sig_r);
    w.sig_s  = scalar_to_ff_limbs(sig_s);

    if (!pubkey.is_infinity()) {
        w.pub_x = fe_to_ff_limbs(pubkey.x());
        w.pub_y = fe_to_ff_limbs(pubkey.y());
    }

    // ---- 2. Reject degenerate inputs early ----
    // s_inv = 0 if s = 0 (shouldn't happen in a well-formed signature)
    if (sig_r.is_zero() || sig_s.is_zero() || pubkey.is_infinity()) {
        w.valid = false;
        return w;
    }

    // ---- 3. Compute private witness values ----

    // s_inv = s^{-1} mod n
    Scalar s_inv = sig_s.inverse();
    w.bytes_s_inv = s_inv.to_bytes();
    w.s_inv = scalar_to_ff_limbs(s_inv);

    // u1 = msg * s^{-1} mod n
    Scalar u1 = msg_scalar * s_inv;
    w.bytes_u1 = u1.to_bytes();
    w.u1 = scalar_to_ff_limbs(u1);

    // u2 = r * s^{-1} mod n
    Scalar u2 = sig_r * s_inv;
    w.bytes_u2 = u2.to_bytes();
    w.u2 = scalar_to_ff_limbs(u2);

    // ---- 4. MSM: R = u1*G + u2*P ----
    Point R = Point::dual_scalar_mul_gen_point(u1, u2, pubkey);
    if (R.is_infinity()) {
        w.valid = false;
        return w;
    }

    // R.x, R.y in Fp
    FieldElement rx = R.x();
    FieldElement ry = R.y();
    w.bytes_result_x = rx.to_bytes();
    w.bytes_result_y = ry.to_bytes();
    w.result_x = fe_to_ff_limbs(rx);
    w.result_y = fe_to_ff_limbs(ry);

    // R.x mod n  (Fp → Fr: same bytes, but reduced mod n if R.x >= n)
    Scalar rx_mod_n = Scalar::from_bytes(rx.to_bytes());
    w.bytes_result_x_mod_n = rx_mod_n.to_bytes();
    w.result_x_mod_n = scalar_to_ff_limbs(rx_mod_n);

    // ---- 5. Validity check ----
    w.valid = (rx_mod_n == sig_r);

    return w;
}

} // namespace zk
} // namespace secp256k1
