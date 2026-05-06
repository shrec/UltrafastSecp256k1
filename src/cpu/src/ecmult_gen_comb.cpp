// ============================================================================
// ecmult_gen: Lim-Lee Comb Method Implementation
// ============================================================================
// Precomputes a table of G multiples arranged in comb pattern,
// then multiplies by gathering bits from the scalar at regular spacings.
//
// Reference: Lim & Lee, "More Flexible Exponentiation with Precomputation"
//            (CRYPTO 1994) -- adapted for elliptic curves.

#include "secp256k1/ecmult_gen_comb.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/debug_invariants.hpp"
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace secp256k1::fast {

// -- Table construction -------------------------------------------------------
// For teeth=t, spacing=d:
//   We need base points: G_j = 2^(j*d) * G  for j = 0..t-1
//   Table entry for index I (t-bit) = sum_{j where bit j of I is set} G_j
//   Total entries: 2^t (entry 0 = infinity)

void CombGenContext::build_table() {
    std::size_t const table_entries = static_cast<std::size_t>(1) << teeth_;

    // Step 1: Compute base points G_j = 2^(j*d) * G
    std::vector<Point> bases(teeth_);
    bases[0] = Point::generator();
    for (unsigned j = 1; j < teeth_; ++j) {
        // G_j = 2^d * G_{j-1}  (d successive doublings)
        Point p = bases[j - 1];
        for (unsigned s = 0; s < spacing_; ++s) {
            p.dbl_inplace();
        }
        bases[j] = p;
    }

    // Step 2: Build all 2^t combinations using Gray code order for efficiency
    // We compute table[I] = sum of bases[j] where bit j of I is set
    table_.resize(table_entries);
    table_[0].infinity = true;  // entry 0 = point at infinity

    for (std::size_t i = 1; i < table_entries; ++i) {
        // Find lowest set bit -- this entry = table[i ^ (1<<lsb)] + bases[lsb]
        unsigned lsb = 0;
        while (((i >> lsb) & 1) == 0) ++lsb;

        std::size_t const prev = i ^ (static_cast<std::size_t>(1) << lsb);
        Point const sum = (prev == 0) ? bases[lsb]
                                : Point::from_jacobian_coords(
                                      table_[prev].x, table_[prev].y,
                                      FieldElement::one(), table_[prev].infinity)
                                      .add(bases[lsb]);

        // Normalize to affine for cache-friendly storage
        // (Jacobian -> affine via z inversion)
        if (sum.is_infinity()) {
            table_[i].infinity = true;
        } else {
            table_[i].x = sum.x();
            table_[i].y = sum.y();
            table_[i].infinity = false;
        }
    }
}

void CombGenContext::init(unsigned teeth) {
    if (teeth < 2 || teeth > 20) {
        throw std::runtime_error("CombGenContext: teeth must be in [2, 20]");
    }
    teeth_ = teeth;
    spacing_ = (256 + teeth_ - 1) / teeth_; // ceil(256 / teeth)
    num_combs_ = 1;
    build_table();
}

// -- Extract comb index -------------------------------------------------------
// For bit position b, gather bits at b, b+d, b+2d, ..., b+(t-1)*d
// forming a t-bit index.

uint32_t CombGenContext::extract_comb_index(const Scalar& k, unsigned b) const {
    // Direct limb access: 1-2 shifts per tooth instead of k.bit() function call.
    auto const& L = k.limbs();
    uint32_t idx = 0;
    for (unsigned j = 0; j < teeth_; ++j) {
        unsigned const pos = b + j * spacing_;
        if (pos < 256) {
            idx |= static_cast<uint32_t>((L[pos >> 6] >> (pos & 63)) & 1) << j;
        }
    }
    return idx;
}

// -- Fast Generator Multiplication (variable-time) ----------------------------
// For each bit position b from (spacing_-1) down to 0:
//   1. R = 2*R (except first)
//   2. Look up table[extract_comb_index(k, b)] and add to R

Point CombGenContext::mul(const Scalar& k) const {
    if (!ready()) {
        throw std::runtime_error("CombGenContext not initialized");
    }

    Point R = Point::infinity();

    for (int b = static_cast<int>(spacing_) - 1; b >= 0; --b) {
        // Double (skip for first iteration)
        if (b < static_cast<int>(spacing_) - 1) {
            R.dbl_inplace();
        }

        uint32_t const idx = extract_comb_index(k, static_cast<unsigned>(b));
        if (SECP256K1_UNLIKELY(idx == 0)) continue;  // ~3% of positions (all teeth==0)

        const auto& entry = table_[idx];
        if (SECP256K1_UNLIKELY(entry.infinity)) continue;

        // Mixed addition: Jacobian R + Affine table entry (Z=1, non-infinity).
        // add_mixed_inplace(x, y) uses the 7M+4S mixed-add formula directly,
        // avoiding the 120-byte intermediate Point construct and the 12M+5S Jac+Jac path.
        if (SECP256K1_UNLIKELY(R.is_infinity())) {
            R = Point::from_affine(entry.x, entry.y);
        } else {
            R.add_mixed_inplace(entry.x, entry.y);
        }
    }

    return R;
}

// -- Constant-Time Generator Multiplication -----------------------------------
// Same algorithm but:
//   - Always performs addition (adds identity if idx=0)
//   - CT table lookup (scans all entries)
//   - No early exit on zero index

Point CombGenContext::mul_ct(const Scalar& k) const {
    if (!ready()) {
        throw std::runtime_error("CombGenContext not initialized");
    }

    Point R = Point::infinity();
    std::size_t const table_entries = static_cast<std::size_t>(1) << teeth_;

    for (int b = static_cast<int>(spacing_) - 1; b >= 0; --b) {
        // Double (skip for first iteration)
        if (b < static_cast<int>(spacing_) - 1) {
            R.dbl_inplace();
        }

        uint32_t const idx = extract_comb_index(k, static_cast<unsigned>(b));

        // CT table lookup: scan all entries, select the right one
        CombAffinePoint selected;
        selected.infinity = true;

        for (std::size_t i = 0; i < table_entries; ++i) {
            uint64_t const mask = ct::eq_mask(static_cast<uint64_t>(i),
                                        static_cast<uint64_t>(idx));
            // CT conditional copy
            ct::cmov256(selected.x.limbs_mut().data(), table_[i].x.limbs().data(), mask);
            ct::cmov256(selected.y.limbs_mut().data(), table_[i].y.limbs().data(), mask);
            // CT update infinity flag
            uint64_t const inf_val = table_[i].infinity ? UINT64_MAX : 0;
            uint64_t sel_inf = selected.infinity ? UINT64_MAX : 0;
            sel_inf = ct::ct_select(inf_val, sel_inf, mask);
            selected.infinity = (sel_inf != 0);
        }

        // Use mixed-add (affine input, Z=1): 7M+4S vs 12M+5S for Jac+Jac.
        // selected is affine by construction (table entries have Z=1).
        if (SECP256K1_UNLIKELY(selected.infinity)) {
            // identity: no-op
        } else if (SECP256K1_UNLIKELY(R.is_infinity())) {
            R = Point::from_affine(selected.x, selected.y);
        } else {
            R.add_mixed_inplace(selected.x, selected.y);
        }
    }

    return R;
}

// -- Cache I/O ----------------------------------------------------------------

std::size_t CombGenContext::table_size_bytes() const noexcept {
    if (!ready()) return 0;
    return table_.size() * sizeof(CombAffinePoint);
}

bool CombGenContext::save_cache(const std::string& path) const {
    if (!ready()) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Header: teeth, spacing, entry count
    f.write(reinterpret_cast<const char*>(&teeth_), sizeof(teeth_));
    f.write(reinterpret_cast<const char*>(&spacing_), sizeof(spacing_));
    auto count = static_cast<uint32_t>(table_.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Table data
    SHA256 hasher;
    for (const auto& entry : table_) {
        uint8_t inf = entry.infinity ? 1 : 0;
        f.write(reinterpret_cast<const char*>(&inf), 1);
        hasher.update(&inf, 1);
        auto xb = entry.x.to_bytes();
        auto yb = entry.y.to_bytes();
        f.write(reinterpret_cast<const char*>(xb.data()), 32);
        f.write(reinterpret_cast<const char*>(yb.data()), 32);
        hasher.update(xb.data(), 32);
        hasher.update(yb.data(), 32);
    }

    // Write SHA256 integrity checksum
    auto checksum = hasher.finalize();
    f.write(reinterpret_cast<const char*>(checksum.data()), 32);

    return f.good();
}

bool CombGenContext::load_cache(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    unsigned teeth = 0, spacing = 0;
    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&teeth), sizeof(teeth));
    f.read(reinterpret_cast<char*>(&spacing), sizeof(spacing));
    f.read(reinterpret_cast<char*>(&count), sizeof(count));

    // spacing=0 causes a loop `for (b = spacing_-1; b>=0; --b)` to never execute
    // (unsigned wrap to UINT_MAX), returning the point at infinity for every scalar.
    // spacing > 256 would index far outside any table row. Reject both.
    if (!f || teeth < 2 || teeth > 20 || spacing < 1 || spacing > 256
           || count != (1u << teeth)) return false;

    teeth_ = teeth;
    spacing_ = spacing;
    num_combs_ = 1;
    // reserve without zero-init: avoids default-constructing count CombAffinePoints
    // (each ~72 bytes) only to immediately overwrite them with file data.
    // For count=32768 (teeth=15) this saves ~2.3 MB of zero-init on load.
    table_.clear();
    table_.reserve(count);

    SHA256 hasher;
    for (uint32_t i = 0; i < count; ++i) {
        CombAffinePoint entry;
        uint8_t inf = 0;
        f.read(reinterpret_cast<char*>(&inf), 1);
        hasher.update(&inf, 1);
        entry.infinity = (inf != 0);
        std::array<uint8_t, 32> xb{}, yb{};
        f.read(reinterpret_cast<char*>(xb.data()), 32);
        f.read(reinterpret_cast<char*>(yb.data()), 32);
        hasher.update(xb.data(), 32);
        hasher.update(yb.data(), 32);
        entry.x = FieldElement::from_bytes(xb);
        entry.y = FieldElement::from_bytes(yb);
        table_.push_back(std::move(entry));
    }

    if (!f.good()) return false;

    // Verify SHA256 integrity checksum
    std::array<uint8_t, 32> stored_checksum{};
    f.read(reinterpret_cast<char*>(stored_checksum.data()), 32);
    if (!f.good()) {
        // Legacy cache without checksum — reject for safety
        table_.clear();
        return false;
    }

    auto computed_checksum = hasher.finalize();
    if (stored_checksum != computed_checksum) {
        table_.clear();
        return false;
    }

    // Cryptographic on-curve validation: every non-infinity point must satisfy
    // y^2 = x^3 + 7 (mod p). A cache file with a valid SHA256 checksum but
    // off-curve points would allow a local attacker to backdoor all scalar
    // multiplications (signatures, ECDH) without touching private key material.
    for (uint32_t i = 0; i < count; ++i) {
        if (table_[i].infinity) continue;
        const FieldElement& x = table_[i].x;
        const FieldElement& y = table_[i].y;
        // y^2 == x^3 + 7  (secp256k1 curve equation)
        FieldElement lhs = y * y;
        FieldElement rhs = x * x * x + FieldElement::from_uint64(7);
        if (lhs != rhs) {
            table_.clear();
            return false;
        }
    }

    return true;
}

// -- Global singleton ---------------------------------------------------------
// Table is built exactly once (std::call_once) and is read-only afterward.
// Read paths hold no lock — the table is immutable after construction and
// call_once provides the happens-before edge needed for safe concurrent reads.

static std::once_flag g_comb_once;
static CombGenContext  g_comb_ctx;

// Internal: initialise with a specific teeth value (used by init_comb_gen).
// Must only be called from within call_once.
static unsigned g_comb_init_teeth = 15;

static void do_init_comb() {
    g_comb_ctx.init(g_comb_init_teeth);
}

void init_comb_gen(unsigned teeth) {
    g_comb_init_teeth = teeth;
    std::call_once(g_comb_once, do_init_comb);
}

bool comb_gen_ready() {
    // Relaxed: if the once_flag has fired, the table is ready.
    // std::call_once provides the required memory ordering.
    std::call_once(g_comb_once, do_init_comb);
    return g_comb_ctx.ready();
}

Point comb_gen_mul(const Scalar& k) {
    std::call_once(g_comb_once, do_init_comb);
    return g_comb_ctx.mul(k);
}

Point comb_gen_mul_ct(const Scalar& k) {
    std::call_once(g_comb_once, do_init_comb);
    return g_comb_ctx.mul_ct(k);
}

} // namespace secp256k1::fast
