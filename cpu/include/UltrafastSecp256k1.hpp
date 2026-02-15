// ============================================================================
// UltrafastSecp256k1 — Umbrella Header
// ============================================================================
// Includes all public API headers for iOS/macOS/SPM/CocoaPods integration.
//
//   #include "UltrafastSecp256k1.hpp"   // Everything
//   // — or individual headers —
//   #include "secp256k1/ecdsa.hpp"      // ECDSA only
//   #include "secp256k1/schnorr.hpp"    // Schnorr only
// ============================================================================

#pragma once

// ── Core Types ──────────────────────────────────────────────────────────────
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"

// ── Fast (Non-CT) Operations ────────────────────────────────────────────────
#include "secp256k1/fast.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/selftest.hpp"

// ── Constant-Time (Side-Channel Resistant) ──────────────────────────────────
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/ops.hpp"

// ── Cryptographic Protocols ─────────────────────────────────────────────────
#include "secp256k1/sha256.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
