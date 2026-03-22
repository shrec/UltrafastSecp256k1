# UltrafastSecp256k1 API Reference

Complete API documentation for CPU, CUDA, and WASM implementations.

---

## Table of Contents

1. [CPU API](#cpu-api)
   - [FieldElement](#fieldelement)
   - [Scalar](#scalar)
   - [Point](#point)
   - [ECDSA (RFC 6979)](#ecdsa-rfc-6979)
   - [Schnorr (BIP-340)](#schnorr-bip-340)
   - [SHA-256](#sha-256)
   - [Constant-Time Layer](#constant-time-layer)
   - [Utility Functions](#utility-functions)
2. [Address Generation API](#address-generation-api)
   - [Address Types](#address-types)
   - [Encoding Functions](#encoding-functions)
   - [WIF (Wallet Import Format)](#wif-wallet-import-format)
   - [BIP-352 Silent Payments](#bip-352-silent-payments)
3. [Multi-Chain Coins API](#multi-chain-coins-api)
   - [Coin Parameters](#coin-parameters)
   - [Coin Address Functions](#coin-address-functions)
   - [Coin Key Generation](#coin-key-generation)
4. [Unified Wallet API](#unified-wallet-api)
   - [Wallet Key Management](#wallet-key-management)
   - [Wallet Address Generation](#wallet-address-generation)
   - [Wallet Signing](#wallet-signing)
   - [Wallet Recovery](#wallet-recovery)
5. [BIP-39 Mnemonic Seed Phrases](#bip-39-mnemonic-seed-phrases)
6. [Message Signing API](#message-signing-api)
7. [Zero-Knowledge Proof API](#zero-knowledge-proof-api)
   - [Knowledge Proof (Schnorr Sigma)](#knowledge-proof-schnorr-sigma)
   - [DLEQ Proof (Discrete Log Equality)](#dleq-proof-discrete-log-equality)
   - [Bulletproof Range Proof](#bulletproof-range-proof)
   - [Batch Operations](#zk-batch-operations)
8. [CUDA API](#cuda-api)
   - [Data Structures](#cuda-data-structures)
   - [Field Operations](#cuda-field-operations)
   - [Point Operations](#cuda-point-operations)
   - [Batch Operations](#cuda-batch-operations)
   - [Signature Operations](#cuda-signature-operations)
9. [WASM API](#wasm-api)
10. [C ABI (ufsecp)](#c-abi-ufsecp)
    - [Context Lifecycle](#c-abi-context-lifecycle)
    - [Private Key Operations](#c-abi-private-key-operations)
    - [Public Key Operations](#c-abi-public-key-operations)
    - [ECDSA](#c-abi-ecdsa)
    - [Schnorr / BIP-340](#c-abi-schnorr)
    - [ECDH](#c-abi-ecdh)
    - [Hashing](#c-abi-hashing)
    - [Addresses and WIF](#c-abi-addresses)
    - [BIP-32 HD Keys](#c-abi-bip32)
    - [Taproot / BIP-341](#c-abi-taproot)
    - [BIP-39 Mnemonics](#c-abi-bip39)
    - [Batch Verification](#c-abi-batch)
    - [Multi-Scalar Multiplication](#c-abi-msm)
    - [MuSig2 / BIP-327](#c-abi-musig2)
    - [FROST Threshold Signatures](#c-abi-frost)
    - [Adaptor Signatures](#c-abi-adaptor)
    - [Pedersen Commitments](#c-abi-pedersen)
    - [Zero-Knowledge Proofs](#c-abi-zk)
    - [Multi-Coin Wallet](#c-abi-multicoin)
    - [Ethereum](#c-abi-ethereum)
11. [Performance Tips](#performance-tips)
12. [Examples](#examples)

---

## CPU API

**Namespace:** `secp256k1::fast`

**Headers:**
```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
```

---

### FieldElement

256-bit field element for secp256k1 curve (mod p where p = 2^256 - 2^32 - 977).

#### Construction

```cpp
// Zero element
FieldElement a = FieldElement::zero();

// One element
FieldElement b = FieldElement::one();

// From 64-bit integer
FieldElement c = FieldElement::from_uint64(12345);

// From 4 x 64-bit limbs (little-endian, RECOMMENDED for binary I/O)
std::array<uint64_t, 4> limbs = {0x123, 0x456, 0x789, 0xABC};
FieldElement d = FieldElement::from_limbs(limbs);

// From 32 bytes (big-endian, for hex/test vectors only)
std::array<uint8_t, 32> bytes = {...};
FieldElement e = FieldElement::from_bytes(bytes);

// From hex string (developer-friendly)
FieldElement f = FieldElement::from_hex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
);
```

#### Arithmetic Operations

```cpp
FieldElement a, b;

// Basic arithmetic (immutable, returns new object)
FieldElement sum = a + b;
FieldElement diff = a - b;
FieldElement prod = a * b;
FieldElement sq = a.square();
FieldElement inv = a.inverse();

// In-place arithmetic (mutable, ~10-15% faster)
a += b;
a -= b;
a *= b;
a.square_inplace();    // a = a^2
a.inverse_inplace();   // a = a^-¹
```

#### Serialization

```cpp
FieldElement a;

// To bytes (big-endian)
std::array<uint8_t, 32> bytes = a.to_bytes();

// To bytes into existing buffer (no allocation)
uint8_t buffer[32];
a.to_bytes_into(buffer);

// To hex string
std::string hex = a.to_hex();

// Access raw limbs (little-endian)
const auto& limbs = a.limbs();  // std::array<uint64_t, 4>
```

#### Comparison

```cpp
FieldElement a, b;
if (a == b) { ... }
if (a != b) { ... }
```

---

### Scalar

256-bit scalar for secp256k1 curve (mod n where n is the group order).

#### Construction

```cpp
// Zero
Scalar a = Scalar::zero();

// One
Scalar b = Scalar::one();

// From 64-bit integer
Scalar c = Scalar::from_uint64(12345);

// From limbs (little-endian)
std::array<uint64_t, 4> limbs = {...};
Scalar d = Scalar::from_limbs(limbs);

// From bytes (big-endian)
std::array<uint8_t, 32> bytes = {...};
Scalar e = Scalar::from_bytes(bytes);

// From hex string
Scalar f = Scalar::from_hex(
    "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
);
```

#### Arithmetic Operations

```cpp
Scalar a, b;

// Basic arithmetic
Scalar sum = a + b;
Scalar diff = a - b;
Scalar prod = a * b;

// In-place
a += b;
a -= b;
a *= b;
```

#### Utility Methods

```cpp
Scalar s;

// Check if zero
bool isZero = s.is_zero();

// Get specific bit
uint8_t bit = s.bit(index);  // 0 or 1

// NAF encoding (Non-Adjacent Form)
std::vector<int8_t> naf = s.to_naf();

// wNAF encoding (width-w NAF)
std::vector<int8_t> wnaf = s.to_wnaf(4);  // width = 4
```

---

### Point

Elliptic curve point on secp256k1 (internally Jacobian coordinates).

#### Construction

```cpp
// Generator point G
Point G = Point::generator();

// Point at infinity (identity)
Point inf = Point::infinity();

// From affine coordinates
FieldElement x = FieldElement::from_hex("...");
FieldElement y = FieldElement::from_hex("...");
Point p = Point::from_affine(x, y);

// From hex strings (developer-friendly)
Point p2 = Point::from_hex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
    "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
);
```

#### Point Operations

```cpp
Point p, q;
Scalar k;

// Addition and doubling
Point sum = p.add(q);       // p + q
Point doubled = p.dbl();    // 2p

// Scalar multiplication
Point result = p.scalar_mul(k);  // k * p

// Negation
Point neg = p.negate();  // -p
```

#### Optimized Scalar Multiplication

```cpp
// For fixed K x variable Q pattern (same K, different Q points):
Scalar K = Scalar::from_hex("...");
KPlan plan = KPlan::from_scalar(K);  // Precompute once

// Then for each Q:
Point Q1 = Point::from_hex("...", "...");
Point Q2 = Point::from_hex("...", "...");

Point R1 = Q1.scalar_mul_with_plan(plan);  // Fastest!
Point R2 = Q2.scalar_mul_with_plan(plan);
```

#### In-Place Operations (Fastest)

```cpp
Point p;

// Increment/decrement by generator
p.next_inplace();    // p += G
p.prev_inplace();    // p -= G

// In-place arithmetic
p.add_inplace(q);    // p += q
p.sub_inplace(q);    // p -= q
p.dbl_inplace();     // p = 2p
p.negate_inplace();  // p = -p

// Mixed addition (when q is affine, z=1)
FieldElement qx, qy;
p.add_mixed_inplace(qx, qy);  // Branchless, ~12% faster
```

#### Serialization

```cpp
Point p;

// Get affine coordinates
FieldElement x = p.x();
FieldElement y = p.y();

// Compressed format (33 bytes: 0x02/0x03 + x)
std::array<uint8_t, 33> compressed = p.to_compressed();

// Uncompressed format (65 bytes: 0x04 + x + y)
std::array<uint8_t, 65> uncompressed = p.to_uncompressed();

// Split x-coordinate for database lookups
std::array<uint8_t, 16> first_half = p.x_first_half();
std::array<uint8_t, 16> second_half = p.x_second_half();
```

#### Properties

```cpp
Point p;

// Check if point at infinity
bool isInf = p.is_infinity();

// Direct Jacobian coordinate access
const FieldElement& X = p.X();  // Jacobian X
const FieldElement& Y = p.Y();  // Jacobian Y
const FieldElement& Z = p.z();  // Jacobian Z
```

---

### Utility Functions

#### Self-Test

```cpp
#include <secp256k1/point.hpp>

// Run correctness tests
bool passed = secp256k1::fast::Selftest(true);  // verbose=true
if (!passed) {
    std::cerr << "Self-test failed!" << std::endl;
}
```

---

### ECDSA (RFC 6979)

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/ecdsa.hpp>
```

#### ECDSASignature

```cpp
struct ECDSASignature {
    fast::Scalar r;
    fast::Scalar s;

    // DER encoding (variable length, max 72 bytes)
    std::pair<std::array<uint8_t, 72>, std::size_t> to_der() const;

    // Compact 64-byte encoding: r (32 bytes) || s (32 bytes)
    std::array<uint8_t, 64> to_compact() const;

    // Decode from compact
    static ECDSASignature from_compact(const std::array<uint8_t, 64>& data);

    // Normalize to low-S (BIP-62)
    ECDSASignature normalize() const;

    // Check low-S
    bool is_low_s() const;
};
```

#### Signing

```cpp
// Sign a 32-byte message hash with RFC 6979 deterministic nonce.
// Returns normalized (low-S) signature. Returns {0,0} on failure.
ECDSASignature ecdsa_sign(
    const std::array<uint8_t, 32>& msg_hash,
    const fast::Scalar& private_key
);
```

#### Verification

```cpp
// Verify ECDSA signature. Accepts both low-S and high-S.
bool ecdsa_verify(
    const std::array<uint8_t, 32>& msg_hash,
    const fast::Point& public_key,
    const ECDSASignature& sig
);
```

#### RFC 6979 Nonce

```cpp
// Deterministic nonce generation per RFC 6979.
fast::Scalar rfc6979_nonce(
    const fast::Scalar& private_key,
    const std::array<uint8_t, 32>& msg_hash
);
```

#### Example

```cpp
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/sha256.hpp>
#include <secp256k1/point.hpp>

using namespace secp256k1;

auto msg_hash = SHA256::hash("Hello ECDSA", 11);
fast::Scalar sk = fast::Scalar::from_hex("...");
fast::Point pk = fast::Point::generator().scalar_mul(sk);

// Sign
auto sig = ecdsa_sign(msg_hash, sk);

// Verify
bool ok = ecdsa_verify(msg_hash, pk, sig);

// Compact encoding (64 bytes)
auto compact = sig.to_compact();
auto recovered = ECDSASignature::from_compact(compact);
```

---

### Schnorr (BIP-340)

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/schnorr.hpp>
```

#### SchnorrSignature

```cpp
struct SchnorrSignature {
    std::array<uint8_t, 32> r;  // R.x (nonce point x-coordinate)
    fast::Scalar s;              // scalar s

    // 64-byte encoding: r (32) || s (32)
    std::array<uint8_t, 64> to_bytes() const;
    static SchnorrSignature from_bytes(const std::array<uint8_t, 64>& data);
};
```

#### Signing

```cpp
// BIP-340 Schnorr sign.
// aux_rand: 32 bytes of auxiliary randomness (use zeros for deterministic).
SchnorrSignature schnorr_sign(
    const fast::Scalar& private_key,
    const std::array<uint8_t, 32>& msg,
    const std::array<uint8_t, 32>& aux_rand
);
```

#### Verification

```cpp
// BIP-340 Schnorr verify with x-only public key.
bool schnorr_verify(
    const std::array<uint8_t, 32>& pubkey_x,
    const std::array<uint8_t, 32>& msg,
    const SchnorrSignature& sig
);
```

#### Utilities

```cpp
// X-only public key (BIP-340: negate if Y is odd)
std::array<uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

// Tagged hash: H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)
std::array<uint8_t, 32> tagged_hash(
    const char* tag, const void* data, std::size_t len
);
```

#### Example

```cpp
#include <secp256k1/schnorr.hpp>

using namespace secp256k1;

fast::Scalar sk = fast::Scalar::from_hex("...");
auto pk_x = schnorr_pubkey(sk);

std::array<uint8_t, 32> msg = { /* message hash */ };
std::array<uint8_t, 32> aux = {}; // zeros for deterministic

auto sig = schnorr_sign(sk, msg, aux);
bool ok = schnorr_verify(pk_x, msg, sig);
```

---

### SHA-256

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/sha256.hpp>
```

#### One-shot Hashing

```cpp
// SHA-256
SHA256::digest_type SHA256::hash(const void* data, std::size_t len);

// Double-SHA256: SHA256(SHA256(data))
SHA256::digest_type SHA256::hash256(const void* data, std::size_t len);
```

#### Streaming API

```cpp
secp256k1::SHA256 ctx;
ctx.update("part1", 5);
ctx.update("part2", 5);
auto digest = ctx.finalize();

// Reuse
ctx.reset();
ctx.update("new data", 8);
auto digest2 = ctx.finalize();
```

#### Example

```cpp
#include <secp256k1/sha256.hpp>

auto hash = secp256k1::SHA256::hash("Hello, world!", 13);
// hash is std::array<uint8_t, 32>

// Double-SHA256 (Bitcoin's hash)
auto hash256 = secp256k1::SHA256::hash256("tx_data", 7);
```

---

### Constant-Time Layer

**Namespace:** `secp256k1::fast::ct`

**Headers:**
```cpp
#include <secp256k1/ct/field.hpp>
#include <secp256k1/ct/scalar.hpp>
#include <secp256k1/ct/point.hpp>
#include <secp256k1/ct/ops.hpp>
```

The CT layer provides side-channel resistant variants of critical operations:

```cpp
namespace secp256k1::fast::ct {

// Constant-time field operations
void field_mul(const FieldElement& a, const FieldElement& b, FieldElement& out);
void field_inv(const FieldElement& a, FieldElement& out);

// Constant-time scalar multiplication (branchless double-and-add)
void scalar_mul(const Point& base, const Scalar& k, Point& out);

// Complete addition formula (handles all edge cases without branching)
void point_add_complete(const Point& p, const Point& q, Point& out);

// Constant-time point doubling
void point_dbl(const Point& p, Point& out);

} // namespace secp256k1::fast::ct
```

> [!] CT operations are ~5-7x slower than the fast variants. Use only for private key operations (signing, ECDH).

---

## Address Generation API

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/address.hpp>
```

### Address Types

All address functions take a `Network` enum (`Mainnet` or `Testnet`) and return a `std::string`.

#### P2PKH (Legacy, Base58Check)

```cpp
// "1..." (mainnet) or "m/n..." (testnet)
std::string address_p2pkh(const fast::Point& pubkey,
                          Network net = Network::Mainnet);
```

#### P2WPKH (Native SegWit v0, Bech32)

```cpp
// "bc1q..." (mainnet) or "tb1q..." (testnet)
std::string address_p2wpkh(const fast::Point& pubkey,
                           Network net = Network::Mainnet);
```

#### P2TR (Taproot, SegWit v1, Bech32m)

```cpp
// "bc1p..." (mainnet) or "tb1p..." (testnet)
std::string address_p2tr(const fast::Point& internal_key,
                         Network net = Network::Mainnet);

// From raw 32-byte x-only output key
std::string address_p2tr_raw(const std::array<uint8_t, 32>& output_key_x,
                             Network net = Network::Mainnet);
```

#### P2SH-P2WPKH (Nested/Wrapped SegWit)

```cpp
// "3..." (mainnet) -- wraps P2WPKH inside P2SH for backward compatibility (BIP-49)
std::string address_p2sh_p2wpkh(const fast::Point& pubkey,
                                Network net = Network::Mainnet);
```

#### P2SH (Pay-to-Script-Hash)

```cpp
// "3..." (mainnet) -- from raw 20-byte script hash
std::string address_p2sh(const std::array<uint8_t, 20>& script_hash,
                         Network net = Network::Mainnet);
```

#### P2WSH (Witness Script Hash)

```cpp
// "bc1q..." 32-byte program (SegWit v0)
std::string address_p2wsh(const std::array<uint8_t, 32>& witness_script_hash,
                          Network net = Network::Mainnet);
```

#### CashAddr (Bitcoin Cash BIP-0185)

```cpp
// Encode hash160 as CashAddr (type: 0=P2PKH, 1=P2SH)
std::string cashaddr_encode(const std::array<uint8_t, 20>& hash,
                            const std::string& prefix,
                            uint8_t type = 0);

// CashAddr P2PKH from public key: "bitcoincash:q..."
std::string address_cashaddr(const fast::Point& pubkey,
                             const std::string& prefix = "bitcoincash");
```

### Encoding Functions

#### Base58Check

```cpp
std::string base58check_encode(const uint8_t* data, size_t len);
std::pair<std::vector<uint8_t>, bool> base58check_decode(const std::string& encoded);
```

#### Bech32 / Bech32m (BIP-173 / BIP-350)

```cpp
std::string bech32_encode(const std::string& hrp,
                          uint8_t witness_version,
                          const uint8_t* witness_program,
                          size_t prog_len);

struct Bech32DecodeResult {
    std::string hrp;
    int witness_version;          // -1 if invalid
    std::vector<uint8_t> witness_program;
    bool valid;
};
Bech32DecodeResult bech32_decode(const std::string& addr);
```

#### HASH160

```cpp
// RIPEMD160(SHA256(data))
std::array<uint8_t, 20> hash160(const uint8_t* data, size_t len);
```

### WIF (Wallet Import Format)

```cpp
std::string wif_encode(const fast::Scalar& private_key,
                       bool compressed = true,
                       Network net = Network::Mainnet);

struct WIFDecodeResult {
    fast::Scalar key;
    bool compressed;
    Network network;
    bool valid;
};
WIFDecodeResult wif_decode(const std::string& wif);
```

### BIP-352 Silent Payments

```cpp
struct SilentPaymentAddress {
    fast::Point scan_pubkey;     // B_scan
    fast::Point spend_pubkey;    // B_spend
    std::string encode(Network net = Network::Mainnet) const;  // sp1q... or tsp1q...
};

// Generate silent payment address from scan/spend private keys
SilentPaymentAddress silent_payment_address(const fast::Scalar& scan_privkey,
                                            const fast::Scalar& spend_privkey);

// Sender: compute unique output key for recipient
std::pair<fast::Point, fast::Scalar>
silent_payment_create_output(const std::vector<fast::Scalar>& input_privkeys,
                             const SilentPaymentAddress& recipient,
                             uint32_t k = 0);

// Receiver: scan transaction to detect addressed outputs
std::vector<std::pair<uint32_t, fast::Scalar>>
silent_payment_scan(const fast::Scalar& scan_privkey,
                    const fast::Scalar& spend_privkey,
                    const std::vector<fast::Point>& input_pubkeys,
                    const std::vector<std::array<uint8_t, 32>>& output_pubkeys);
```

---

## Multi-Chain Coins API

**Namespace:** `secp256k1::coins`

**Headers:**
```cpp
#include <secp256k1/coins/coin_params.hpp>
#include <secp256k1/coins/coin_address.hpp>
```

### Coin Parameters

28 coins predefined as `constexpr CoinParams` objects:

| Coin | Ticker | Encoding | BIP-44 | Chain ID |
|------|--------|----------|--------|----------|
| Bitcoin | BTC | Bech32 | 0 | -- |
| Litecoin | LTC | Bech32 | 2 | -- |
| Dogecoin | DOGE | Base58Check | 3 | -- |
| Dash | DASH | Base58Check | 5 | -- |
| Ethereum | ETH | EIP-55 | 60 | 1 |
| Bitcoin Cash | BCH | CashAddr | 145 | -- |
| Bitcoin SV | BSV | Base58Check | 236 | -- |
| Zcash | ZEC | Base58Check | 133 | -- |
| DigiByte | DGB | Bech32 | 20 | -- |
| Namecoin | NMC | Base58Check | 7 | -- |
| Peercoin | PPC | Base58Check | 6 | -- |
| Vertcoin | VTC | Bech32 | 28 | -- |
| Viacoin | VIA | Bech32 | 14 | -- |
| Groestlcoin | GRS | Bech32 | 17 | -- |
| Syscoin | SYS | Bech32 | 57 | -- |
| BNB Smart Chain | BNB | EIP-55 | 60 | 56 |
| Polygon | POL | EIP-55 | 60 | 137 |
| Avalanche | AVAX | EIP-55 | 60 | 43114 |
| Fantom | FTM | EIP-55 | 60 | 250 |
| Arbitrum | ARB | EIP-55 | 60 | 42161 |
| Optimism | OP | EIP-55 | 60 | 10 |
| Ravencoin | RVN | Base58Check | 175 | -- |
| Flux | FLUX | Base58Check | 19167 | -- |
| Qtum | QTUM | Base58Check | 2301 | -- |
| Horizen | ZEN | Base58Check | 121 | -- |
| Bitcoin Gold | BTG | Base58Check | 156 | -- |
| Komodo | KMD | Base58Check | 141 | -- |
| Tron | TRX | TRON_BASE58 | 195 | -- |

#### Lookup Functions

```cpp
const CoinParams* find_by_coin_type(uint32_t coin_type);  // Returns nullptr if not found
const CoinParams* find_by_ticker(const char* ticker);      // Case-sensitive
```

#### Iteration

```cpp
inline constexpr const CoinParams* ALL_COINS[];            // Array of all 28 coins
inline constexpr size_t ALL_COINS_COUNT = 28;
```

### Coin Address Functions

```cpp
// Default/preferred format for each coin (Bech32 for BTC, EIP-55 for ETH, etc.)
std::string coin_address(const fast::Point& pubkey, const CoinParams& coin,
                         bool testnet = false);

// Explicit format selection
std::string coin_address_p2pkh(const fast::Point& pubkey, const CoinParams& coin,
                               bool testnet = false);
std::string coin_address_p2wpkh(const fast::Point& pubkey, const CoinParams& coin,
                                bool testnet = false);
std::string coin_address_p2tr(const fast::Point& internal_key, const CoinParams& coin,
                              bool testnet = false);
std::string coin_address_p2sh_p2wpkh(const fast::Point& pubkey, const CoinParams& coin,
                                     bool testnet = false);
std::string coin_address_p2sh(const std::array<uint8_t, 20>& script_hash,
                              const CoinParams& coin, bool testnet = false);
std::string coin_address_cashaddr(const fast::Point& pubkey, const CoinParams& coin,
                                  bool testnet = false);

// WIF encoding with coin-specific prefix
std::string coin_wif_encode(const fast::Scalar& private_key, const CoinParams& coin,
                            bool compressed = true, bool testnet = false);
```

> Functions return empty string if the coin does not support the requested format (e.g., `coin_address_p2wpkh` for Dogecoin).

### Coin Key Generation

```cpp
struct CoinKeyPair {
    fast::Scalar private_key;
    fast::Point  public_key;
    std::string  address;         // Default format for coin
    std::string  wif;             // WIF-encoded key (empty for EVM coins)
};

CoinKeyPair coin_derive(const fast::Scalar& private_key, const CoinParams& coin,
                        bool testnet = false, const CurveContext* ctx = nullptr);
```

---

## Unified Wallet API

**Namespace:** `secp256k1::coins::wallet`

**Header:**
```cpp
#include <secp256k1/coins/wallet.hpp>
```

Chain-agnostic facade over all address, signing, and recovery operations. Same API regardless of underlying chain.

### Wallet Key Management

```cpp
struct WalletKey {
    fast::Scalar priv;           // 32-byte private scalar
    fast::Point  pub;            // Public key
};

// Create from raw 32-byte key. Returns (key, success).
std::pair<WalletKey, bool> from_private_key(const uint8_t* priv32);

// Export in chain-appropriate format:
//   Bitcoin-family: WIF (Base58Check)
//   EVM-family:     0x-prefixed hex
//   Tron:           raw hex
std::string export_private_key(const CoinParams& coin, const WalletKey& key,
                               bool testnet = false);

std::string export_public_key_hex(const CoinParams& coin, const WalletKey& key);
```

### Wallet Address Generation

```cpp
// Default address format for coin
std::string get_address(const CoinParams& coin, const WalletKey& key,
                        bool testnet = false);

// Explicit format selection
std::string get_address_p2pkh(const CoinParams& coin, const WalletKey& key,
                              bool testnet = false);
std::string get_address_p2wpkh(const CoinParams& coin, const WalletKey& key,
                               bool testnet = false);
std::string get_address_p2sh_p2wpkh(const CoinParams& coin, const WalletKey& key,
                                    bool testnet = false);
std::string get_address_p2tr(const CoinParams& coin, const WalletKey& key,
                             bool testnet = false);
std::string get_address_cashaddr(const CoinParams& coin, const WalletKey& key,
                                 bool testnet = false);
```

### Wallet Signing

```cpp
struct MessageSignature {
    std::array<uint8_t, 32> r;
    std::array<uint8_t, 32> s;
    int recid;                   // Recovery ID (0-3)
    uint64_t v;                  // EIP-155 v (EVM) or 27+recid (Bitcoin)
    std::array<uint8_t, 65> to_rsv() const;  // [r:32][s:32][v:1]
};

// Chain-aware message signing:
//   Bitcoin: "\x18Bitcoin Signed Message:\n" prefix -> dSHA256
//   Ethereum: "\x19Ethereum Signed Message:\n" prefix -> Keccak-256
MessageSignature sign_message(const CoinParams& coin, const WalletKey& key,
                              const uint8_t* msg, size_t msg_len);

// Sign raw 32-byte hash (no prefix, no hashing)
MessageSignature sign_hash(const CoinParams& coin, const WalletKey& key,
                           const uint8_t* hash32);

// Verify signed message against public key
bool verify_message(const CoinParams& coin, const fast::Point& pubkey,
                    const uint8_t* msg, size_t msg_len,
                    const MessageSignature& sig);
```

### Wallet Recovery

```cpp
// Recover public key from message + signature
std::pair<fast::Point, bool>
recover_signer(const CoinParams& coin,
               const uint8_t* msg, size_t msg_len,
               const MessageSignature& sig);

// Recover address string from message + signature
std::pair<std::string, bool>
recover_address(const CoinParams& coin,
                const uint8_t* msg, size_t msg_len,
                const MessageSignature& sig);
```

---

## BIP-39 Mnemonic Seed Phrases

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/bip39.hpp>
```

BIP-39 mnemonic code for generating deterministic keys. Converts entropy to
human-readable word sequences and derives 512-bit seeds compatible with BIP-32.

### Mnemonic Generation

```cpp
// Generate mnemonic from OS CSPRNG entropy (12 words = 16 bytes)
auto [mnemonic, ok] = secp256k1::bip39_generate(16);

// Generate from explicit entropy (24 words = 32 bytes)
uint8_t entropy[32] = { /* ... */ };
auto [mnemonic24, ok2] = secp256k1::bip39_generate(32, entropy);
```

```cpp
std::pair<std::string, bool>
bip39_generate(std::size_t entropy_bytes,
               const std::uint8_t* entropy_in = nullptr);
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `entropy_bytes` | 16 (12 words), 20 (15), 24 (18), 28 (21), or 32 (24 words) |
| `entropy_in` | Optional explicit entropy; `nullptr` = use OS CSPRNG |

**Returns:** `{mnemonic_string, success}`

### Mnemonic Validation

```cpp
bool valid = secp256k1::bip39_validate("abandon abandon ... about");
```

```cpp
bool bip39_validate(const std::string& mnemonic);
```

Validates word count (12/15/18/21/24), word membership in BIP-39 English wordlist,
and SHA-256 checksum.

### Seed Derivation

```cpp
auto [seed, ok] = secp256k1::bip39_mnemonic_to_seed(mnemonic, "my passphrase");
// seed is std::array<uint8_t, 64> -- pass to bip32_master_key()
```

```cpp
std::pair<std::array<std::uint8_t, 64>, bool>
bip39_mnemonic_to_seed(const std::string& mnemonic,
                       const std::string& passphrase = "");
```

Uses PBKDF2-HMAC-SHA512 with 2048 iterations. Salt = `"mnemonic" + passphrase`.

### Mnemonic to Entropy

```cpp
auto [ent, ok] = secp256k1::bip39_mnemonic_to_entropy("abandon abandon ... about");
// ent.data = raw entropy bytes, ent.length = byte count (16-32)
```

```cpp
struct Bip39Entropy {
    std::array<std::uint8_t, 32> data{};
    std::size_t length = 0;
};

std::pair<Bip39Entropy, bool>
bip39_mnemonic_to_entropy(const std::string& mnemonic);
```

### PBKDF2-HMAC-SHA512

```cpp
void pbkdf2_hmac_sha512(const std::uint8_t* password, std::size_t password_len,
                         const std::uint8_t* salt, std::size_t salt_len,
                         std::uint32_t iterations,
                         std::uint8_t* output, std::size_t output_len);
```

Exposed for direct use and testing. Used internally by `bip39_mnemonic_to_seed()`.

### Wordlist Access

```cpp
const char* const* words = secp256k1::bip39_wordlist_english();
// words[0] == "abandon", words[2047] == "zoo"
```

Returns pointer to the 2048-word sorted English BIP-39 wordlist.

---

## Message Signing API

**Namespace:** `secp256k1::coins`

**Header:**
```cpp
#include <secp256k1/coins/message_signing.hpp>
```

Low-level Bitcoin message signing (BIP-137 / Electrum format).

#### Bitcoin Message Hash

```cpp
// SHA256(SHA256("\x18Bitcoin Signed Message:\n" + varint(msg_len) + msg))
std::array<uint8_t, 32> bitcoin_message_hash(const uint8_t* msg, size_t msg_len);
```

#### Sign / Verify / Recover

```cpp
RecoverableSignature bitcoin_sign_message(const uint8_t* msg, size_t msg_len,
                                          const fast::Scalar& private_key);

bool bitcoin_verify_message(const uint8_t* msg, size_t msg_len,
                            const fast::Point& pubkey,
                            const ECDSASignature& sig);

std::pair<fast::Point, bool>
bitcoin_recover_message(const uint8_t* msg, size_t msg_len,
                        const ECDSASignature& sig, int recid);
```

#### Base64 Encoding

```cpp
// 65-byte format: 1-byte header (recid + compression flag, range 27-34) + r + s
std::string bitcoin_sig_to_base64(const RecoverableSignature& rsig,
                                  bool compressed = true);

struct BitcoinSigDecodeResult {
    ECDSASignature sig;
    int recid;
    bool compressed;
    bool valid;
};
BitcoinSigDecodeResult bitcoin_sig_from_base64(const std::string& base64);
```

---

## Zero-Knowledge Proof API

**Namespace:** `secp256k1::zk`

**Header:**
```cpp
#include <secp256k1/zk.hpp>
```

---

### Knowledge Proof (Schnorr Sigma)

Non-interactive proof of knowledge of discrete log via Fiat-Shamir transform.

#### KnowledgeProof

```cpp
struct KnowledgeProof {
    std::array<uint8_t, 32> rx;  // R.x (nonce point x-coordinate)
    fast::Scalar s;               // response scalar

    std::array<uint8_t, 64> serialize() const;
    static bool deserialize(const uint8_t* data64, KnowledgeProof& out);
};
```

#### knowledge_prove

```cpp
// Prove: "I know x such that pubkey = x*G"
// Uses CT layer (constant-time). Nonce hedged with aux_rand.
KnowledgeProof knowledge_prove(
    const fast::Scalar& secret,
    const fast::Point& pubkey,
    const std::array<uint8_t, 32>& msg,
    const std::array<uint8_t, 32>& aux_rand);
```

#### knowledge_verify

```cpp
// Verify: s*G == R + e*P (FAST path)
bool knowledge_verify(
    const KnowledgeProof& proof,
    const fast::Point& pubkey,
    const std::array<uint8_t, 32>& msg);
```

#### knowledge_prove_base / knowledge_verify_base

Same as above but with an arbitrary base point (not just G):

```cpp
KnowledgeProof knowledge_prove_base(
    const fast::Scalar& secret,
    const fast::Point& point,     // point = secret * base
    const fast::Point& base,
    const std::array<uint8_t, 32>& msg,
    const std::array<uint8_t, 32>& aux_rand);

bool knowledge_verify_base(
    const KnowledgeProof& proof,
    const fast::Point& point,
    const fast::Point& base,
    const std::array<uint8_t, 32>& msg);
```

---

### DLEQ Proof (Discrete Log Equality)

Proves log_G(P) == log_H(Q) without revealing the secret.

#### DLEQProof

```cpp
struct DLEQProof {
    fast::Scalar e;  // challenge
    fast::Scalar s;  // response

    std::array<uint8_t, 64> serialize() const;
    static bool deserialize(const uint8_t* data64, DLEQProof& out);
};
```

#### dleq_prove

```cpp
// Prove: P = secret*G AND Q = secret*H (same secret)
// Uses CT layer. Nonce hedged with aux_rand.
DLEQProof dleq_prove(
    const fast::Scalar& secret,
    const fast::Point& G,
    const fast::Point& H,
    const fast::Point& P,
    const fast::Point& Q,
    const std::array<uint8_t, 32>& aux_rand);
```

#### dleq_verify

```cpp
// Verify: s*G == R1 + e*P AND s*H == R2 + e*Q (FAST path)
bool dleq_verify(
    const DLEQProof& proof,
    const fast::Point& G,
    const fast::Point& H,
    const fast::Point& P,
    const fast::Point& Q);
```

---

### Bulletproof Range Proof

Proves committed value is in [0, 2^64) without revealing the value. Based on
Bulletproofs (Bunz et al., 2018). Logarithmic proof size.

#### RangeProof

```cpp
static constexpr size_t RANGE_PROOF_BITS = 64;
static constexpr size_t RANGE_PROOF_LOG2 = 6;

struct RangeProof {
    fast::Point A, S;              // vector commitments
    fast::Point T1, T2;            // polynomial commitments
    fast::Scalar tau_x, mu, t_hat; // scalar responses
    std::array<fast::Point, 6> L;  // inner product argument (log2(64) rounds)
    std::array<fast::Point, 6> R;
    fast::Scalar a, b;             // final inner product scalars
};
```

#### range_prove

```cpp
// Generate Bulletproof range proof for Pedersen commitment C = value*H + blinding*G
// value must be in [0, 2^64). Uses CT layer.
RangeProof range_prove(
    uint64_t value,
    const fast::Scalar& blinding,
    const PedersenCommitment& commitment,
    const std::array<uint8_t, 32>& aux_rand);
```

#### range_verify

```cpp
// Verify range proof (MSM-optimized, FAST path)
// Returns true if committed value is in [0, 2^64)
bool range_verify(
    const PedersenCommitment& commitment,
    const RangeProof& proof);
```

---

### ZK Batch Operations

#### batch_range_verify

```cpp
// Batch-verify multiple range proofs (more efficient than individual verification)
// Returns true only if ALL proofs are valid.
bool batch_range_verify(
    const PedersenCommitment* commitments,
    const RangeProof* proofs,
    size_t count);
```

#### batch_commit

```cpp
// Batch-create Pedersen commitments
void batch_commit(
    const fast::Scalar* values,
    const fast::Scalar* blindings,
    PedersenCommitment* commitments_out,
    size_t count);
```

#### GeneratorVectors

```cpp
struct GeneratorVectors {
    std::array<fast::Point, 64> G;  // nothing-up-my-sleeve generators
    std::array<fast::Point, 64> H;
};

// Retrieve cached generator vectors (computed once on first call)
const GeneratorVectors& get_generator_vectors();
```

#### ZK Performance Summary

| Operation | Time | Throughput | Layer |
|-----------|------|------------|-------|
| Pedersen Commit | 33.0 us | 30.3K op/s | FAST |
| Knowledge Prove | 20.3 us | 49.3K op/s | CT |
| Knowledge Verify | 21.8 us | 46.0K op/s | FAST |
| DLEQ Prove | 40.0 us | 25.0K op/s | CT |
| DLEQ Verify | 56.4 us | 17.7K op/s | FAST |
| Range Prove (64b) | 13,467 us | 74 op/s | CT |
| Range Verify (64b) | 2,634 us | 380 op/s | FAST |

*Measured on i7-14400F, 11 passes, pinned core, median.*

---

## CUDA API

**Namespace:** `secp256k1::cuda`

**Header:**
```cpp
#include <secp256k1.cuh>
```

---

### CUDA Data Structures

```cpp
// Field element (4 x 64-bit limbs, little-endian)
struct FieldElement {
    uint64_t limbs[4];
};

// Scalar (4 x 64-bit limbs)
struct Scalar {
    uint64_t limbs[4];
};

// Jacobian point (X, Y, Z)
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

// Affine point (x, y)
struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

// 32-bit view for optimized operations (zero-cost conversion)
struct MidFieldElement {
    uint32_t limbs[8];
};
```

---

### CUDA Field Operations

All functions are `__device__` and can only be called from GPU kernels.

```cpp
// Initialization
__device__ void field_set_zero(FieldElement* r);
__device__ void field_set_one(FieldElement* r);

// Comparison
__device__ bool field_is_zero(const FieldElement* a);
__device__ bool field_eq(const FieldElement* a, const FieldElement* b);

// Arithmetic
__device__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sqr(const FieldElement* a, FieldElement* r);
__device__ void field_inv(const FieldElement* a, FieldElement* r);
__device__ void field_neg(const FieldElement* a, FieldElement* r);

// Domain conversion (Montgomery mode only)
__device__ void field_to_mont(const FieldElement* a, FieldElement* r);
__device__ void field_from_mont(const FieldElement* a, FieldElement* r);
```

---

### CUDA Point Operations

```cpp
// Initialization
__device__ void jacobian_set_infinity(JacobianPoint* p);
__device__ void jacobian_set_generator(JacobianPoint* p);
__device__ bool jacobian_is_infinity(const JacobianPoint* p);

// Point arithmetic
__device__ void jacobian_double(const JacobianPoint* p, JacobianPoint* r);
__device__ void jacobian_add(const JacobianPoint* p, const JacobianPoint* q, JacobianPoint* r);
__device__ void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r);

// Scalar multiplication
__device__ void scalar_mul(const JacobianPoint* p, const Scalar* k, JacobianPoint* r);
__device__ void scalar_mul_generator(const Scalar* k, JacobianPoint* r);

// Conversion
__device__ void jacobian_to_affine(const JacobianPoint* p, AffinePoint* r);
```

---

### CUDA Batch Operations

```cpp
#include <batch_inversion.cuh>

// Batch field inversion (Montgomery's trick)
// Inverts n field elements using only 1 modular inversion + 3(n-1) multiplications
__device__ void batch_invert(FieldElement* elements, int n, FieldElement* scratch);
```

---

### CUDA Hash Operations

```cpp
#include <hash160.cuh>

// Compute HASH160 = RIPEMD160(SHA256(pubkey))
__device__ void hash160_compressed(const uint8_t pubkey[33], uint8_t hash[20]);
__device__ void hash160_uncompressed(const uint8_t pubkey[65], uint8_t hash[20]);
```

---

### CUDA Signature Operations

> **World-first:** No other open-source GPU library provides secp256k1 ECDSA + Schnorr sign/verify.

#### Data Structures

```cpp
#include <ecdsa.cuh>
#include <schnorr.cuh>
#include <recovery.cuh>

// ECDSA signature (r, s as Scalars)
struct ECDSASignatureGPU {
    Scalar r;
    Scalar s;
};

// Schnorr BIP-340 signature (32-byte R x-coordinate + Scalar s)
struct SchnorrSignatureGPU {
    uint8_t r[32];  // x-coordinate of R point
    Scalar s;
};

// Recoverable ECDSA signature
struct RecoverableSignatureGPU {
    ECDSASignatureGPU sig;
    int recid;  // Recovery ID (0-3)
};
```

#### Device Functions

```cpp
// ECDSA Sign (RFC 6979 deterministic nonces, low-S normalization)
// Returns true on success
__device__ bool ecdsa_sign(
    const uint8_t msg_hash[32],   // 32-byte message hash
    const Scalar* privkey,         // Private key
    ECDSASignatureGPU* sig         // Output signature
);

// ECDSA Verify (Shamir's trick + GLV endomorphism)
// Returns true if signature is valid
__device__ bool ecdsa_verify(
    const uint8_t msg_hash[32],   // 32-byte message hash
    const JacobianPoint* pubkey,   // Public key (Jacobian)
    const ECDSASignatureGPU* sig   // Signature to verify
);

// ECDSA Sign with Recovery ID
__device__ bool ecdsa_sign_recoverable(
    const uint8_t msg_hash[32],
    const Scalar* privkey,
    RecoverableSignatureGPU* sig   // Output: signature + recid
);

// ECDSA Recover public key from signature
__device__ bool ecdsa_recover(
    const uint8_t msg_hash[32],
    const RecoverableSignatureGPU* sig,
    JacobianPoint* pubkey          // Output: recovered public key
);

// Schnorr Sign (BIP-340, tagged hash midstates for performance)
__device__ bool schnorr_sign(
    const Scalar* privkey,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],    // Auxiliary randomness
    SchnorrSignatureGPU* sig       // Output signature
);

// Schnorr Verify (BIP-340, x-only pubkey)
__device__ bool schnorr_verify(
    const uint8_t pubkey_x[32],    // X-only public key (32 bytes)
    const uint8_t msg[32],
    const SchnorrSignatureGPU* sig
);
```

#### Batch Kernel Wrappers

Host-callable kernel wrappers for batch processing:

```cpp
// Launch batch ECDSA sign (128 threads/block, 2 blocks/SM)
void ecdsa_sign_batch_kernel<<<blocks, 128>>>(
    const uint8_t* msg_hashes,     // N x 32 bytes
    const Scalar* privkeys,         // N scalars
    ECDSASignatureGPU* sigs,        // N output signatures
    int count
);

// Launch batch ECDSA verify
void ecdsa_verify_batch_kernel<<<blocks, 128>>>(
    const uint8_t* msg_hashes,
    const JacobianPoint* pubkeys,
    const ECDSASignatureGPU* sigs,
    bool* results,                  // N output booleans
    int count
);

// Launch batch Schnorr sign
void schnorr_sign_batch_kernel<<<blocks, 128>>>(
    const Scalar* privkeys,
    const uint8_t* msgs,
    const uint8_t* aux_rands,
    SchnorrSignatureGPU* sigs,
    int count
);

// Launch batch Schnorr verify
void schnorr_verify_batch_kernel<<<blocks, 128>>>(
    const uint8_t* pubkey_xs,
    const uint8_t* msgs,
    const SchnorrSignatureGPU* sigs,
    bool* results,
    int count
);
```

#### Performance

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| ECDSA Sign | 204.8 ns | 4.88 M/s |
| ECDSA Verify | 410.1 ns | 2.44 M/s |
| ECDSA Sign + Recid | 311.5 ns | 3.21 M/s |
| Schnorr Sign | 273.4 ns | 3.66 M/s |
| Schnorr Verify | 354.6 ns | 2.82 M/s |

*RTX 5060 Ti, kernel-only timing, batch 16K*

---

### CUDA CT (Constant-Time) Operations

**Namespace:** `secp256k1::cuda::ct`

**Header:**
```cpp
#include <ct/ct_sign.cuh>  // CT ECDSA/Schnorr sign
#include <ct/ct_zk.cuh>    // CT ZK knowledge/DLEQ prove
```

All CT functions use constant-time scalar multiplication and arithmetic to prevent
side-channel leakage. Proving operations are CT; verification uses the fast path
(public data only -- see `zk.cuh`).

#### CT Signing

```cpp
// CT ECDSA sign (constant-time k*G, k^-1, scalar ops)
__device__ bool ct_ecdsa_sign(
    const uint8_t msg_hash[32],
    const Scalar* privkey,
    ECDSASignatureGPU* sig
);

// CT Schnorr sign (constant-time nonce generation + signing)
__device__ bool ct_schnorr_sign(
    const Scalar* privkey,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],
    SchnorrSignatureGPU* sig
);

// CT Schnorr keypair (x-only pubkey extraction)
__device__ void ct_schnorr_keypair_create(
    const Scalar* privkey,
    SchnorrKeypairGPU* keypair
);
```

#### CT Zero-Knowledge Proofs

##### Data Structures

```cpp
#include <zk.cuh>       // KnowledgeProofGPU, DLEQProofGPU
#include <ct/ct_zk.cuh>  // CT proving functions

// Knowledge proof (Schnorr sigma protocol)
struct KnowledgeProofGPU {
    uint8_t rx[32];  // R point x-coordinate
    Scalar s;        // Response scalar
};

// DLEQ proof (discrete log equality)
struct DLEQProofGPU {
    Scalar e;  // Challenge
    Scalar s;  // Response
};
```

##### Device Functions

```cpp
// CT Knowledge Proof: proves knowledge of secret s such that P = s * B
// Uses CT scalar_mul for k*B (k is secret), CT scalar arithmetic
__device__ bool ct_knowledge_prove_device(
    const Scalar* secret,
    const JacobianPoint* pubkey,   // P = secret * base
    const JacobianPoint* base,     // Arbitrary base point B
    const uint8_t msg[32],
    const uint8_t aux[32],         // Auxiliary randomness (XOR hedging)
    KnowledgeProofGPU* proof
);

// Convenience: prove for generator G
__device__ bool ct_knowledge_prove_generator_device(
    const Scalar* secret,
    const JacobianPoint* pubkey,   // P = secret * G
    const uint8_t msg[32],
    const uint8_t aux[32],
    KnowledgeProofGPU* proof
);

// CT DLEQ Proof: proves log_G(P) == log_H(Q) without revealing the log
// Two CT scalar_mul operations (k*G, k*H), 6-point challenge hash
__device__ bool ct_dleq_prove_device(
    const Scalar* secret,
    const JacobianPoint* G,        // First base
    const JacobianPoint* H,        // Second base
    const JacobianPoint* P,        // P = secret * G
    const JacobianPoint* Q,        // Q = secret * H
    const uint8_t aux[32],
    DLEQProofGPU* proof
);
```

##### Batch Kernels

```cpp
// Batch CT knowledge prove (one thread per proof)
__global__ void ct_knowledge_prove_batch_kernel(
    const Scalar* secrets,
    const JacobianPoint* pubkeys,
    const JacobianPoint* bases,
    const uint8_t* messages,    // N * 32 bytes
    const uint8_t* aux_rands,   // N * 32 bytes
    KnowledgeProofGPU* proofs,
    bool* results,
    uint32_t count
);

// Batch CT knowledge prove with generator G
__global__ void ct_knowledge_prove_generator_batch_kernel(
    const Scalar* secrets,
    const JacobianPoint* pubkeys,
    const uint8_t* messages,
    const uint8_t* aux_rands,
    KnowledgeProofGPU* proofs,
    bool* results,
    uint32_t count
);

// Batch CT DLEQ prove
__global__ void ct_dleq_prove_batch_kernel(
    const Scalar* secrets,
    const JacobianPoint* G_pts,
    const JacobianPoint* H_pts,
    const JacobianPoint* P_pts,
    const JacobianPoint* Q_pts,
    const uint8_t* aux_rands,
    DLEQProofGPU* proofs,
    bool* results,
    uint32_t count
);
```

> **Design note:** Verification functions (`knowledge_verify_device`, `dleq_verify_device`)
> live in `zk.cuh` and use the fast (variable-time) path -- verification inputs are public.

---

### OpenCL ZK Operations

**Kernel file:** `opencl/kernels/secp256k1_zk.cl`

OpenCL ZK operations use the fast-path scalar multiplication (wNAF-5). OpenCL has no
separate CT layer -- all operations are uniform across work-items by design.

```c
// Device functions (called from kernels)
void zk_knowledge_prove_impl(...);
void zk_knowledge_verify_impl(...);
void zk_dleq_prove_impl(...);
void zk_dleq_verify_impl(...);

// Batch kernels
__kernel void zk_knowledge_prove_batch(...);
__kernel void zk_knowledge_verify_batch(...);
__kernel void zk_dleq_prove_batch(...);
__kernel void zk_dleq_verify_batch(...);
```

---

### Metal ZK Operations

**Shader file:** `metal/shaders/secp256k1_zk.h`
**Kernel file:** `metal/shaders/secp256k1_kernels.metal` (Kernels 19-22)

Metal ZK uses branchless `affine_select` in scalar multiplication (semi-CT).
8x32 limb representation.

```metal
// Device functions
bool zk_knowledge_prove(...);
bool zk_knowledge_verify(...);
bool zk_dleq_prove(...);
bool zk_dleq_verify(...);

// Kernels 19-22
kernel void zk_knowledge_prove_batch(...)  [[kernel]];
kernel void zk_knowledge_verify_batch(...) [[kernel]];
kernel void zk_dleq_prove_batch(...)       [[kernel]];
kernel void zk_dleq_verify_batch(...)      [[kernel]];
```

---

## WASM API

**Module:** `@ultrafastsecp256k1/wasm`

**Usage:**
```javascript
import { Secp256k1 } from './secp256k1.mjs';
const lib = await Secp256k1.create();
```

### Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `selftest()` | -- | `boolean` | Run built-in self-test |
| `version()` | -- | `string` | Library version (`"3.0.0"`) |
| `pubkeyCreate(seckey)` | `Uint8Array(32)` | `{x, y}` | Public key from private key |
| `pointMul(px, py, scalar)` | `Uint8Array(32)` x 3 | `{x, y}` | Scalar x Point |
| `pointAdd(px, py, qx, qy)` | `Uint8Array(32)` x 4 | `{x, y}` | Point addition |
| `ecdsaSign(msgHash, seckey)` | `Uint8Array(32)` x 2 | `Uint8Array(64)` | ECDSA sign (r‖s) |
| `ecdsaVerify(msgHash, pubX, pubY, sig)` | `Uint8Array(32)` x 3 + `Uint8Array(64)` | `boolean` | ECDSA verify |
| `schnorrSign(seckey, msg, aux?)` | `Uint8Array(32)` x 2-3 | `Uint8Array(64)` | Schnorr BIP-340 sign |
| `schnorrVerify(pubkeyX, msg, sig)` | `Uint8Array(32)` x 2 + `Uint8Array(64)` | `boolean` | Schnorr verify |
| `schnorrPubkey(seckey)` | `Uint8Array(32)` | `Uint8Array(32)` | X-only public key |
| `sha256(data)` | `Uint8Array` | `Uint8Array(32)` | SHA-256 hash |

### C API

For direct C/C++ or custom WASM bindings, see [secp256k1_wasm.h](../wasm/secp256k1_wasm.h).

### Example

```javascript
const lib = await Secp256k1.create();
console.log('v' + lib.version(), lib.selftest() ? 'OK' : 'X');

// ECDSA workflow
const privkey = new Uint8Array(32);
privkey[31] = 1;
const { x, y } = lib.pubkeyCreate(privkey);
const msgHash = lib.sha256(new TextEncoder().encode('Hello'));
const sig = lib.ecdsaSign(msgHash, privkey);
const valid = lib.ecdsaVerify(msgHash, x, y, sig);
```

See [wasm/README.md](../wasm/README.md) for detailed build and usage instructions.

---

## C ABI (ufsecp)

The stable C ABI is defined in `include/ufsecp/ufsecp.h`. All functions follow these rules:

- **Opaque context**: `ufsecp_ctx*` -- one per thread, or externally synchronised
- **Every function returns `ufsecp_error_t`** (0 = OK)
- **All I/O is `uint8_t[]`** with fixed sizes -- no internal types leak
- **Dual-layer CT**: signing/nonce/key-tweak always use the CT layer; verify/point-arith use the fast layer. No opt-in flag.
- **Caller owns all buffers** -- library never allocates on behalf of caller (except `ctx_create`/`ctx_clone`)

### Error Codes

| Code | Name | Value |
|------|------|-------|
| `UFSECP_OK` | Success | 0 |
| `UFSECP_ERR_NULL_ARG` | NULL pointer argument | 1 |
| `UFSECP_ERR_BAD_KEY` | Invalid private key | 2 |
| `UFSECP_ERR_BAD_PUBKEY` | Invalid public key | 3 |
| `UFSECP_ERR_BAD_SIG` | Invalid signature | 4 |
| `UFSECP_ERR_BAD_INPUT` | Invalid input data | 5 |
| `UFSECP_ERR_VERIFY_FAIL` | Verification failed | 6 |
| `UFSECP_ERR_ARITH` | Arithmetic error | 7 |
| `UFSECP_ERR_SELFTEST` | Self-test failure | 8 |
| `UFSECP_ERR_INTERNAL` | Internal error | 9 |
| `UFSECP_ERR_BUF_TOO_SMALL` | Output buffer too small | 10 |

### Size Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `UFSECP_PRIVKEY_LEN` | 32 | Private key |
| `UFSECP_PUBKEY_COMPRESSED_LEN` | 33 | Compressed public key |
| `UFSECP_PUBKEY_UNCOMPRESSED_LEN` | 65 | Uncompressed public key |
| `UFSECP_PUBKEY_XONLY_LEN` | 32 | x-only public key (BIP-340) |
| `UFSECP_SIG_COMPACT_LEN` | 64 | Compact R\|\|S / r\|\|s |
| `UFSECP_SIG_DER_MAX_LEN` | 72 | Maximum DER-encoded ECDSA sig |
| `UFSECP_HASH_LEN` | 32 | SHA-256 / Keccak-256 digest |
| `UFSECP_HASH160_LEN` | 20 | RIPEMD160(SHA256) digest |
| `UFSECP_SHARED_SECRET_LEN` | 32 | ECDH shared secret |
| `UFSECP_BIP32_SERIALIZED_LEN` | 78 | BIP-32 extended key |

<a id="c-abi-context-lifecycle"></a>
### Context Lifecycle

```c
// Create context (runs self-test on first call)
ufsecp_ctx* ctx = NULL;
ufsecp_error_t err = ufsecp_ctx_create(&ctx);

// Clone context (deep copy, for multi-thread use)
ufsecp_ctx* ctx2 = NULL;
ufsecp_ctx_clone(ctx, &ctx2);

// Error inspection
ufsecp_error_t last = ufsecp_last_error(ctx);
const char* msg = ufsecp_last_error_msg(ctx);

// FFI layout assertion
size_t sz = ufsecp_ctx_size();

// Destroy (NULL-safe)
ufsecp_ctx_destroy(ctx);
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_ctx_create` | `(ufsecp_ctx** ctx_out) -> error_t` | Create new context |
| `ufsecp_ctx_clone` | `(const ctx*, ufsecp_ctx** ctx_out) -> error_t` | Deep copy context |
| `ufsecp_ctx_destroy` | `(ctx*) -> void` | Free context (NULL-safe) |
| `ufsecp_last_error` | `(const ctx*) -> error_t` | Last error code |
| `ufsecp_last_error_msg` | `(const ctx*) -> const char*` | Last error message |
| `ufsecp_ctx_size` | `(void) -> size_t` | Compiled ctx struct size |

<a id="c-abi-private-key-operations"></a>
### Private Key Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_seckey_verify` | `(ctx, privkey[32]) -> error_t` | Validate private key (non-zero, < n) |
| `ufsecp_seckey_negate` | `(ctx, privkey[32]) -> error_t` | Negate in-place: key <- -key mod n |
| `ufsecp_seckey_tweak_add` | `(ctx, privkey[32], tweak[32]) -> error_t` | key <- (key + tweak) mod n |
| `ufsecp_seckey_tweak_mul` | `(ctx, privkey[32], tweak[32]) -> error_t` | key <- (key * tweak) mod n |

<a id="c-abi-public-key-operations"></a>
### Public Key Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_pubkey_create` | `(ctx, privkey[32], pubkey33_out[33]) -> error_t` | Compressed pubkey from privkey |
| `ufsecp_pubkey_create_uncompressed` | `(ctx, privkey[32], pubkey65_out[65]) -> error_t` | Uncompressed pubkey from privkey |
| `ufsecp_pubkey_parse` | `(ctx, input, input_len, pubkey33_out[33]) -> error_t` | Parse 33 or 65 bytes to compressed |
| `ufsecp_pubkey_xonly` | `(ctx, privkey[32], xonly32_out[32]) -> error_t` | x-only pubkey (BIP-340) |
| `ufsecp_pubkey_add` | `(ctx, a33[33], b33[33], out33[33]) -> error_t` | Point addition: out = a + b |
| `ufsecp_pubkey_negate` | `(ctx, pubkey33[33], out33[33]) -> error_t` | Point negation: out = -P |
| `ufsecp_pubkey_tweak_add` | `(ctx, pubkey33[33], tweak[32], out33[33]) -> error_t` | out = P + tweak*G |
| `ufsecp_pubkey_tweak_mul` | `(ctx, pubkey33[33], tweak[32], out33[33]) -> error_t` | out = tweak * P |
| `ufsecp_pubkey_combine` | `(ctx, pubkeys, n, out33[33]) -> error_t` | Sum N compressed pubkeys |

<a id="c-abi-ecdsa"></a>
### ECDSA

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_ecdsa_sign` | `(ctx, msg32[32], privkey[32], sig64_out[64]) -> error_t` | Sign (RFC 6979, low-S) |
| `ufsecp_ecdsa_sign_verified` | `(ctx, msg32[32], privkey[32], sig64_out[64]) -> error_t` | Sign + verify (fault resistance) |
| `ufsecp_ecdsa_sign_batch` | `(ctx, n, msgs32[], privkeys32[], sigs64_out[]) -> error_t` | CPU CT batch sign; repeats stable 32/32/64-byte item layout per entry |
| `ufsecp_ecdsa_verify` | `(ctx, msg32[32], sig64[64], pubkey33[33]) -> error_t` | Verify compact signature |
| `ufsecp_ecdsa_sig_to_der` | `(ctx, sig64[64], der_out, der_len*) -> error_t` | Compact to DER encoding |
| `ufsecp_ecdsa_sig_from_der` | `(ctx, der, der_len, sig64_out[64]) -> error_t` | DER to compact encoding |
| `ufsecp_ecdsa_sign_recoverable` | `(ctx, msg32, privkey, sig64_out, recid_out*) -> error_t` | Sign with recovery id (0-3) |
| `ufsecp_ecdsa_recover` | `(ctx, msg32, sig64, recid, pubkey33_out[33]) -> error_t` | Recover pubkey from recoverable sig |

<a id="c-abi-schnorr"></a>
### Schnorr / BIP-340

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_schnorr_sign` | `(ctx, msg32, privkey, aux_rand[32], sig64_out) -> error_t` | BIP-340 sign (aux_rand=zeros for deterministic) |
| `ufsecp_schnorr_sign_verified` | `(ctx, msg32, privkey, aux_rand, sig64_out) -> error_t` | Sign + verify (fault resistance) |
| `ufsecp_schnorr_sign_batch` | `(ctx, n, msgs32[], privkeys32[], aux_rands32[]?, sigs64_out[]) -> error_t` | CPU CT batch sign; `aux_rands32 == NULL` means all-zero aux for every entry |
| `ufsecp_schnorr_verify` | `(ctx, msg32, sig64, pubkey_x[32]) -> error_t` | Verify BIP-340 signature |

<a id="c-abi-ecdh"></a>
### ECDH

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_ecdh` | `(ctx, privkey, pubkey33, secret32_out) -> error_t` | SHA256(compressed shared point) |
| `ufsecp_ecdh_xonly` | `(ctx, privkey, pubkey33, secret32_out) -> error_t` | SHA256(x-coordinate) |
| `ufsecp_ecdh_raw` | `(ctx, privkey, pubkey33, secret32_out) -> error_t` | Raw x-coordinate (no hash) |

<a id="c-abi-hashing"></a>
### Hashing

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_sha256` | `(data, len, digest32_out) -> error_t` | SHA-256 (HW-accel when available) |
| `ufsecp_sha512` | `(data, len, digest64_out) -> error_t` | SHA-512 |
| `ufsecp_hash160` | `(data, len, digest20_out) -> error_t` | RIPEMD160(SHA256) |
| `ufsecp_tagged_hash` | `(tag, data, len, digest32_out) -> error_t` | BIP-340 tagged hash |

<a id="c-abi-addresses"></a>
### Addresses and WIF

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_addr_p2pkh` | `(ctx, pubkey33, network, addr_out, addr_len*) -> error_t` | P2PKH (Base58) |
| `ufsecp_addr_p2wpkh` | `(ctx, pubkey33, network, addr_out, addr_len*) -> error_t` | P2WPKH (Bech32, SegWit v0) |
| `ufsecp_addr_p2tr` | `(ctx, internal_key_x[32], network, addr_out, addr_len*) -> error_t` | P2TR (Bech32m, Taproot) |
| `ufsecp_wif_encode` | `(ctx, privkey, compressed, network, wif_out, wif_len*) -> error_t` | Private key to WIF |
| `ufsecp_wif_decode` | `(ctx, wif, privkey32_out, compressed_out*, network_out*) -> error_t` | WIF to private key |

Network constants: `UFSECP_NET_MAINNET` (0), `UFSECP_NET_TESTNET` (1).

<a id="c-abi-bip32"></a>
### BIP-32 HD Keys

Opaque key type: `ufsecp_bip32_key` (82 bytes, contains 78-byte serialised key + metadata).

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_bip32_master` | `(ctx, seed, seed_len, key_out*) -> error_t` | Master key from seed (16-64 bytes) |
| `ufsecp_bip32_derive` | `(ctx, parent*, index, child_out*) -> error_t` | Single child derivation (>=0x80000000 = hardened) |
| `ufsecp_bip32_derive_path` | `(ctx, master*, path_str, key_out*) -> error_t` | Full path, e.g. "m/44'/0'/0'/0/0" |
| `ufsecp_bip32_privkey` | `(ctx, key*, privkey32_out) -> error_t` | Extract 32-byte private key |
| `ufsecp_bip32_pubkey` | `(ctx, key*, pubkey33_out) -> error_t` | Extract 33-byte compressed public key |

<a id="c-abi-taproot"></a>
### Taproot / BIP-341

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_taproot_output_key` | `(ctx, internal_x[32], merkle_root, output_x_out[32], parity_out*) -> error_t` | Derive Taproot output key |
| `ufsecp_taproot_tweak_seckey` | `(ctx, privkey[32], merkle_root, tweaked32_out[32]) -> error_t` | Tweak privkey for key-path spend |
| `ufsecp_taproot_verify` | `(ctx, output_x, parity, internal_x, merkle_root, len) -> error_t` | Verify Taproot commitment |

<a id="c-abi-bip39"></a>
### BIP-39 Mnemonics

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_bip39_generate` | `(ctx, entropy_bytes, entropy_in, mnemonic_out, mnemonic_len*) -> error_t` | Generate mnemonic (12/15/18/21/24 words) |
| `ufsecp_bip39_validate` | `(ctx, mnemonic) -> error_t` | Validate mnemonic (checksum + wordlist) |
| `ufsecp_bip39_to_seed` | `(ctx, mnemonic, passphrase, seed64_out) -> error_t` | Mnemonic to 64-byte seed (PBKDF2) |
| `ufsecp_bip39_to_entropy` | `(ctx, mnemonic, entropy_out, entropy_len*) -> error_t` | Mnemonic back to raw entropy |

Entropy sizes: 16 (12 words), 20 (15), 24 (18), 28 (21), 32 (24 words). Pass `entropy_in=NULL` for random.

<a id="c-abi-batch"></a>
### Batch Verification

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_schnorr_batch_verify` | `(ctx, entries, n) -> error_t` | Verify N Schnorr sigs. Entry: 32 xonly + 32 msg + 64 sig = 128 bytes |
| `ufsecp_ecdsa_batch_verify` | `(ctx, entries, n) -> error_t` | Verify N ECDSA sigs. Entry: 32 msg + 33 pubkey + 64 sig = 129 bytes |
| `ufsecp_schnorr_batch_identify_invalid` | `(ctx, entries, n, invalid_out, invalid_count*) -> error_t` | Find indices of invalid Schnorr sigs |
| `ufsecp_ecdsa_batch_identify_invalid` | `(ctx, entries, n, invalid_out, invalid_count*) -> error_t` | Find indices of invalid ECDSA sigs |

<a id="c-abi-msm"></a>
### Multi-Scalar Multiplication

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_shamir_trick` | `(ctx, a[32], P33[33], b[32], Q33[33], out33[33]) -> error_t` | Compute a*P + b*Q |
| `ufsecp_multi_scalar_mul` | `(ctx, scalars, points, n, out33[33]) -> error_t` | Compute sum(scalars[i] * points[i]) |

Scalars: 32-byte big-endian, contiguous. Points: 33-byte compressed, contiguous.

<a id="c-abi-musig2"></a>
### MuSig2 / BIP-327

Size constants: `UFSECP_MUSIG2_PUBNONCE_LEN` (66), `UFSECP_MUSIG2_AGGNONCE_LEN` (66), `UFSECP_MUSIG2_KEYAGG_LEN` (165), `UFSECP_MUSIG2_SESSION_LEN` (165), `UFSECP_MUSIG2_SECNONCE_LEN` (64).

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_musig2_key_agg` | `(ctx, pubkeys, n, keyagg_out, agg_pubkey32_out) -> error_t` | Aggregate x-only pubkeys |
| `ufsecp_musig2_nonce_gen` | `(ctx, privkey, pubkey32, agg_pubkey32, msg32, extra_in, secnonce_out, pubnonce_out) -> error_t` | Generate nonce pair |
| `ufsecp_musig2_nonce_agg` | `(ctx, pubnonces, n, aggnonce_out) -> error_t` | Aggregate public nonces |
| `ufsecp_musig2_start_sign_session` | `(ctx, aggnonce, keyagg, msg32, session_out) -> error_t` | Start signing session |
| `ufsecp_musig2_partial_sign` | `(ctx, secnonce, privkey, keyagg, session, signer_idx, partial_sig32_out) -> error_t` | Produce partial signature |
| `ufsecp_musig2_partial_verify` | `(ctx, partial_sig32, pubnonce, pubkey32, keyagg, session, signer_idx) -> error_t` | Verify partial signature |
| `ufsecp_musig2_partial_sig_agg` | `(ctx, partial_sigs, n, session, sig64_out) -> error_t` | Aggregate partials to final BIP-340 sig |

<a id="c-abi-frost"></a>
### FROST Threshold Signatures

Size constants: `UFSECP_FROST_SHARE_LEN` (36), `UFSECP_FROST_KEYPKG_LEN` (141), `UFSECP_FROST_NONCE_LEN` (64), `UFSECP_FROST_NONCE_COMMIT_LEN` (70).

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_frost_keygen_begin` | `(ctx, id, threshold, n, seed, commits_out, commits_len*, shares_out, shares_len*) -> error_t` | Key generation phase 1 |
| `ufsecp_frost_keygen_finalize` | `(ctx, id, all_commits, commits_len, shares, shares_len, t, n, keypkg_out) -> error_t` | Key generation phase 2 |
| `ufsecp_frost_sign_nonce_gen` | `(ctx, id, nonce_seed, nonce_out, nonce_commit_out) -> error_t` | Generate signing nonce |
| `ufsecp_frost_sign` | `(ctx, keypkg, nonce, msg32, nonce_commits, n_signers, partial_sig_out[36]) -> error_t` | Produce partial signature |
| `ufsecp_frost_verify_partial` | `(ctx, partial_sig[36], verification_share33[33], nonce_commits, n_signers, msg32, group_pubkey32) -> error_t` | Verify partial signature |
| `ufsecp_frost_aggregate` | `(ctx, partial_sigs, n, nonce_commits, n_signers, group_pubkey32, msg32, sig64_out) -> error_t` | Aggregate to final Schnorr sig |

<a id="c-abi-adaptor"></a>
### Adaptor Signatures

Size constants: `UFSECP_SCHNORR_ADAPTOR_SIG_LEN` (97), `UFSECP_ECDSA_ADAPTOR_SIG_LEN` (130).

**Schnorr Adaptor:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_schnorr_adaptor_sign` | `(ctx, privkey, msg32, adaptor33, aux_rand, pre_sig_out[97]) -> error_t` | Pre-sign |
| `ufsecp_schnorr_adaptor_verify` | `(ctx, pre_sig[97], pubkey_x, msg32, adaptor33) -> error_t` | Verify pre-signature |
| `ufsecp_schnorr_adaptor_adapt` | `(ctx, pre_sig[97], secret[32], sig64_out) -> error_t` | Adapt to valid signature |
| `ufsecp_schnorr_adaptor_extract` | `(ctx, pre_sig[97], sig64, secret32_out) -> error_t` | Extract adaptor secret |

**ECDSA Adaptor:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_ecdsa_adaptor_sign` | `(ctx, privkey, msg32, adaptor33, pre_sig_out[130]) -> error_t` | Pre-sign |
| `ufsecp_ecdsa_adaptor_verify` | `(ctx, pre_sig[130], pubkey33, msg32, adaptor33) -> error_t` | Verify pre-signature |
| `ufsecp_ecdsa_adaptor_adapt` | `(ctx, pre_sig[130], secret[32], sig64_out) -> error_t` | Adapt to valid signature |
| `ufsecp_ecdsa_adaptor_extract` | `(ctx, pre_sig[130], sig64, secret32_out) -> error_t` | Extract adaptor secret |

<a id="c-abi-pedersen"></a>
### Pedersen Commitments

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_pedersen_commit` | `(ctx, value[32], blinding[32], commitment33_out) -> error_t` | C = value*H + blinding*G |
| `ufsecp_pedersen_verify` | `(ctx, commitment33, value[32], blinding[32]) -> error_t` | Verify commitment |
| `ufsecp_pedersen_verify_sum` | `(ctx, pos, n_pos, neg, n_neg) -> error_t` | Verify balance: sum(pos) == sum(neg) |
| `ufsecp_pedersen_blind_sum` | `(ctx, blinds_in, n_in, blinds_out, n_out, sum32_out) -> error_t` | Compute blinding factor sum |
| `ufsecp_pedersen_switch_commit` | `(ctx, value, blinding, switch_blind, commitment33_out) -> error_t` | Switch commitment (3-gen) |

<a id="c-abi-zk"></a>
### Zero-Knowledge Proofs

Size constants: `UFSECP_ZK_KNOWLEDGE_PROOF_LEN` (64), `UFSECP_ZK_DLEQ_PROOF_LEN` (64), `UFSECP_ZK_RANGE_PROOF_MAX_LEN` (675).

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_zk_knowledge_prove` | `(ctx, secret, pubkey33, msg32, aux_rand, proof_out[64]) -> error_t` | Prove knowledge of discrete log |
| `ufsecp_zk_knowledge_verify` | `(ctx, proof[64], pubkey33, msg32) -> error_t` | Verify knowledge proof |
| `ufsecp_zk_dleq_prove` | `(ctx, secret, G33, H33, P33, Q33, aux_rand, proof_out[64]) -> error_t` | Prove DLEQ: logG(P) == logH(Q) |
| `ufsecp_zk_dleq_verify` | `(ctx, proof[64], G33, H33, P33, Q33) -> error_t` | Verify DLEQ proof |
| `ufsecp_zk_range_prove` | `(ctx, value, blinding, commitment33, aux_rand, proof_out, proof_len*) -> error_t` | Bulletproof range proof |
| `ufsecp_zk_range_verify` | `(ctx, commitment33, proof, proof_len) -> error_t` | Verify Bulletproof range proof |

<a id="c-abi-multicoin"></a>
### Multi-Coin Wallet

Coin type constants (BIP-44): `UFSECP_COIN_BITCOIN` (0), `UFSECP_COIN_LITECOIN` (2), `UFSECP_COIN_DOGECOIN` (3), `UFSECP_COIN_DASH` (5), `UFSECP_COIN_ETHEREUM` (60), `UFSECP_COIN_BITCOIN_CASH` (145), `UFSECP_COIN_TRON` (195).

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_coin_address` | `(ctx, pubkey33, coin_type, testnet, addr_out, addr_len*) -> error_t` | Default address for any coin |
| `ufsecp_coin_derive_from_seed` | `(ctx, seed, seed_len, coin_type, account, change, index, testnet, privkey_out, pubkey_out, addr_out, addr_len*) -> error_t` | Full derivation from seed |
| `ufsecp_coin_wif_encode` | `(ctx, privkey, coin_type, testnet, wif_out, wif_len*) -> error_t` | WIF for any coin |
| `ufsecp_btc_message_sign` | `(ctx, msg, msg_len, privkey, base64_out, base64_len*) -> error_t` | Bitcoin message sign (BIP-137) |
| `ufsecp_btc_message_verify` | `(ctx, msg, msg_len, pubkey33, base64_sig) -> error_t` | Bitcoin message verify |
| `ufsecp_btc_message_hash` | `(msg, msg_len, digest32_out) -> error_t` | Bitcoin message hash |

<a id="c-abi-ethereum"></a>
### Ethereum

Available when built with `-DSECP256K1_BUILD_ETHEREUM=ON` (default ON).

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_keccak256` | `(data, len, digest32_out) -> error_t` | Keccak-256 hash |
| `ufsecp_eth_address` | `(ctx, pubkey33, addr20_out) -> error_t` | 20-byte Ethereum address |
| `ufsecp_eth_address_checksummed` | `(ctx, pubkey33, addr_out, addr_len*) -> error_t` | EIP-55 checksummed address string |
| `ufsecp_eth_personal_hash` | `(msg, msg_len, digest32_out) -> error_t` | EIP-191 personal_sign hash |
| `ufsecp_eth_sign` | `(ctx, msg32, privkey, r_out, s_out, v_out*, chain_id) -> error_t` | ECDSA with recovery (EIP-155 v) |
| `ufsecp_eth_ecrecover` | `(ctx, msg32, r, s, v, addr20_out) -> error_t` | Recover address from v,r,s |

<a id="c-abi-gpu"></a>
### GPU Operations

Backend-neutral GPU acceleration surface. All functions use opaque `ufsecp_gpu_ctx*` (separate from CPU `ufsecp_ctx*`). Include `<ufsecp/ufsecp_gpu.h>`.

**GPU Error Codes** (100--106): `UFSECP_ERR_GPU_UNAVAILABLE` (100), `UFSECP_ERR_GPU_DEVICE` (101), `UFSECP_ERR_GPU_LAUNCH` (102), `UFSECP_ERR_GPU_MEMORY` (103), `UFSECP_ERR_GPU_UNSUPPORTED` (104), `UFSECP_ERR_GPU_BACKEND` (105), `UFSECP_ERR_GPU_QUEUE` (106).

#### Discovery

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_gpu_backend_count` | `(ids_out*, max) -> uint32_t` | Number of compiled backends; optionally fills array |
| `ufsecp_gpu_backend_name` | `(bid) -> const char*` | Human-readable backend name ("CUDA", "OpenCL", "Metal") |
| `ufsecp_gpu_is_available` | `(bid) -> int` | 1 if backend has >= 1 usable device, 0 otherwise |
| `ufsecp_gpu_device_count` | `(bid) -> uint32_t` | Devices visible to backend `bid` |
| `ufsecp_gpu_device_info` | `(bid, dev, info_out*) -> error_t` | Fill `ufsecp_gpu_device_info_t` (name, memory, CUs, clock) |

#### Context Lifecycle

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_gpu_ctx_create` | `(ctx_out**, bid, dev) -> error_t` | Create GPU context on backend `bid`, device `dev` |
| `ufsecp_gpu_ctx_destroy` | `(ctx*) -> void` | Destroy context (NULL-safe) |
| `ufsecp_gpu_last_error` | `(ctx*) -> error_t` | Last error code from `ctx` |
| `ufsecp_gpu_last_error_msg` | `(ctx*) -> const char*` | Last error message string |
| `ufsecp_gpu_error_str` | `(code) -> const char*` | Error code to string (CPU + GPU codes) |

#### Batch Operations and Extensions

| Function | Signature | Description |
|----------|-----------|-------------|
| `ufsecp_gpu_generator_mul_batch` | `(ctx, scalars32[], n, pubkeys33_out[]) -> error_t` | k*G for n scalars |
| `ufsecp_gpu_ecdsa_verify_batch` | `(ctx, msgs32[], pubs33[], sigs64[], n, results_out[]) -> error_t` | ECDSA batch verify |
| `ufsecp_gpu_schnorr_verify_batch` | `(ctx, msgs32[], pubs_x32[], sigs64[], n, results_out[]) -> error_t` | Schnorr/BIP-340 batch verify |
| `ufsecp_gpu_ecdh_batch` | `(ctx, privkeys32[], pubs33[], n, secrets32_out[]) -> error_t` | ECDH shared secrets (SECRET-BEARING) |
| `ufsecp_gpu_hash160_pubkey_batch` | `(ctx, pubs33[], n, hashes20_out[]) -> error_t` | SHA-256 + RIPEMD-160 of pubkeys |
| `ufsecp_gpu_msm` | `(ctx, scalars32[], points33[], n, result33_out) -> error_t` | Multi-scalar multiplication |
| `ufsecp_gpu_frost_verify_partial_batch` | `(ctx, z_i32[], D_i33[], E_i33[], Y_i33[], rho_i32[], lambda_ie32[], negate_R[], negate_key[], n, results_out[]) -> error_t` | Batch FROST partial verification |
| `ufsecp_gpu_ecrecover_batch` | `(ctx, msgs32[], sigs64[], recids[], n, pubkeys33_out[], valid_out[]) -> error_t` | Batch public-key recovery from recoverable ECDSA signatures |

---

## Performance Tips

### CPU

1. **Use in-place operations** when possible:
   ```cpp
   // Slower: creates temporary
   point = point.add(other);
   
   // Faster: no allocation
   point.add_inplace(other);
   ```

2. **Use KPlan for fixed-K multiplication**:
   ```cpp
   // If K is constant and Q varies, precompute K once
   KPlan plan = KPlan::from_scalar(K);
   for (auto& Q : points) {
       result = Q.scalar_mul_with_plan(plan);
   }
   ```

3. **Use `from_limbs` for binary I/O** (not `from_bytes`):
   ```cpp
   // Database/binary files: use from_limbs (native little-endian)
   FieldElement::from_limbs(limbs);
   
   // Hex strings/test vectors: use from_bytes (big-endian)
   FieldElement::from_bytes(bytes);
   ```

### CUDA

1. **Batch operations**: Process thousands of points in parallel
2. **Avoid divergence**: Use branchless algorithms where possible
3. **Memory coalescing**: Align data structures to 32/64 bytes
4. **Use hybrid 32-bit multiplication**: Enabled by default (`SECP256K1_CUDA_USE_HYBRID_MUL=1`)

---

## Examples

### Generate Bitcoin Address (CPU)

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>

using namespace secp256k1::fast;

int main() {
    // Private key (256-bit)
    Scalar private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Public key = private_key x G
    Point G = Point::generator();
    Point public_key = G.scalar_mul(private_key);
    
    // Get compressed public key (33 bytes)
    auto compressed = public_key.to_compressed();
    
    // Print as hex
    for (auto byte : compressed) {
        printf("%02x", byte);
    }
    printf("\n");
    
    return 0;
}
```

### Batch Point Generation (CUDA)

```cpp
#include <secp256k1.cuh>

using namespace secp256k1::cuda;

__global__ void generate_points_kernel(
    const Scalar* private_keys,
    AffinePoint* public_keys,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    JacobianPoint result;
    scalar_mul_generator(&private_keys[idx], &result);
    jacobian_to_affine(&result, &public_keys[idx]);
}

void generate_points(
    const Scalar* d_private_keys,
    AffinePoint* d_public_keys,
    int count
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    generate_points_kernel<<<blocks, threads>>>(
        d_private_keys, d_public_keys, count
    );
}
```

### Verify Self-Test (CPU)

```cpp
#include <secp256k1/point.hpp>
#include <iostream>

int main() {
    bool ok = secp256k1::fast::Selftest(true);
    if (ok) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "TESTS FAILED!" << std::endl;
        return 1;
    }
}
```

---

## Build Configuration Macros

### CPU

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Enable x64/RISC-V assembly |
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction (RISC-V) |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension |

### CUDA

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_CUDA_USE_HYBRID_MUL` | 1 | 32-bit hybrid multiplication (~10% faster) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | 0 | Montgomery domain arithmetic |
| `SECP256K1_CUDA_LIMBS_32` | 0 | Use 8x32-bit limbs (experimental) |

---

## Platform Support

| Platform | Assembly | SIMD | Status |
|----------|----------|------|--------|
| x86-64 Linux/Windows/macOS | BMI2/ADX | AVX2 | [OK] Production |
| RISC-V 64 | RV64GC | RVV 1.0 | [OK] Production |
| ARM64 (Android/iOS/macOS) | MUL/UMULH | NEON | [OK] Production |
| CUDA (sm_75+) | PTX | -- | [OK] Production |
| ROCm/HIP (AMD) | Portable | -- | [OK] CI |
| OpenCL 3.0 | PTX | -- | [OK] Production |
| WebAssembly | Portable | -- | [OK] Production |
| ESP32-S3 / ESP32 | Portable | -- | [OK] Tested |
| STM32F103 (Cortex-M3) | UMULL | -- | [OK] Tested |

---

## Version

UltrafastSecp256k1 v3.22.0

For more information, see the [README](../README.md) or [GitHub repository](https://github.com/shrec/UltrafastSecp256k1).

