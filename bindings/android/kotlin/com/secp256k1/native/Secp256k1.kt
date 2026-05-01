package com.secp256k1.native

/**
 * UltrafastSecp256k1 — High-performance ECC for Android.
 *
 * Two APIs available:
 *  - **Fast**: Maximum speed, no side-channel protection (public data only)
 *  - **CT** (Constant-Time): Side-channel resistant (private keys, ECDH)
 *
 * All byte arrays use **big-endian** encoding.
 * - Scalar: 32 bytes (256-bit integer mod curve order)
 * - Point (uncompressed): 65 bytes (`04 || x[32] || y[32]`)
 * - Point (compressed): 33 bytes (`02/03 || x[32]`)
 *
 * Usage:
 * ```kotlin
 * // Initialize once (precomputes generator table)
 * Secp256k1.init()
 *
 * // Generate public key from private key (CT — side-channel safe)
 * val pubkey = Secp256k1.ctScalarMulGenerator(privkeyBytes)
 *
 * // ECDH shared secret (CT)
 * val secret = Secp256k1.ctEcdh(myPrivkey, theirPubkey)
 *
 * // Fast batch operations (public data)
 * val sum = Secp256k1.pointAdd(p1, p2)
 * val product = Secp256k1.scalarMulPoint(k, p)
 * ```
 */
object Secp256k1 {

    init {
        System.loadLibrary("secp256k1_jni")
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * Initialize the library. Must be called once before any other operation.
     * Precomputes the generator table and runs self-test.
     * @return true if self-test passes
     */
    fun init(): Boolean = nativeInit()

    /**
     * Run arithmetic self-test with known vectors.
     * @return true if all tests pass
     */
    fun selfTest(): Boolean = nativeSelfTest()

    // ========================================================================
    // Scalar Arithmetic (mod curve order n)
    // ========================================================================

    /**
     * Scalar addition: (a + b) mod n
     * @param a 32-byte scalar
     * @param b 32-byte scalar
     * @return 32-byte result
     */
    fun scalarAdd(a: ByteArray, b: ByteArray): ByteArray = scalarAdd(a, b)

    /**
     * Scalar multiplication: (a * b) mod n
     * @param a 32-byte scalar
     * @param b 32-byte scalar
     * @return 32-byte result
     */
    fun scalarMul(a: ByteArray, b: ByteArray): ByteArray = scalarMul(a, b)

    /**
     * Scalar subtraction: (a - b) mod n
     * @param a 32-byte scalar
     * @param b 32-byte scalar
     * @return 32-byte result
     */
    fun scalarSub(a: ByteArray, b: ByteArray): ByteArray = scalarSub(a, b)

    // ========================================================================
    // Point Operations (Fast — no side-channel protection)
    // ========================================================================

    /**
     * Get the secp256k1 generator point G.
     * @return 65-byte uncompressed point
     */
    external fun getGenerator(): ByteArray

    /**
     * Point addition: P1 + P2
     * @param p1 65-byte uncompressed point
     * @param p2 65-byte uncompressed point
     * @return 65-byte uncompressed result
     */
    external fun pointAdd(p1: ByteArray, p2: ByteArray): ByteArray

    /**
     * Point doubling: 2*P
     * @param p 65-byte uncompressed point
     * @return 65-byte uncompressed result
     */
    external fun pointDouble(p: ByteArray): ByteArray

    /**
     * Point negation: -P
     * @param p 65-byte uncompressed point
     * @return 65-byte uncompressed result
     */
    external fun pointNegate(p: ByteArray): ByteArray

    /**
     * Scalar multiplication: k * P (FAST — not side-channel safe!)
     * Use ctScalarMulPoint for private key operations.
     * @param scalar 32-byte scalar
     * @param point 65-byte uncompressed point
     * @return 65-byte uncompressed result
     */
    external fun scalarMulPoint(scalar: ByteArray, point: ByteArray): ByteArray

    /**
     * Scalar multiplication with generator: k * G (FAST — not side-channel safe!)
     * Use ctScalarMulGenerator for private key operations.
     * @param scalar 32-byte scalar
     * @return 65-byte uncompressed result
     */
    external fun scalarMulGenerator(scalar: ByteArray): ByteArray

    /**
     * Compress point to 33 bytes.
     * @param point 65-byte uncompressed point
     * @return 33-byte compressed point (02/03 || x)
     */
    external fun pointCompress(point: ByteArray): ByteArray

    /**
     * Check if point is the point at infinity.
     * @param point 65-byte uncompressed point
     * @return true if infinity
     */
    external fun pointIsInfinity(point: ByteArray): Boolean

    // ========================================================================
    // CT (Constant-Time) Operations — Side-Channel Resistant
    // Use these for ALL private key operations (key generation, ECDH, signing)
    // ========================================================================

    /**
     * CT scalar multiplication with generator: k * G
     * Uses fixed-window algorithm with complete addition formulas.
     * Safe for private key → public key derivation.
     * @param scalar 32-byte private key
     * @return 65-byte uncompressed public key
     */
    external fun ctScalarMulGenerator(scalar: ByteArray): ByteArray

    /**
     * CT scalar multiplication: k * P
     * Uses fixed-window algorithm with complete addition formulas.
     * Safe for private key operations.
     * @param scalar 32-byte scalar
     * @param point 65-byte uncompressed point
     * @return 65-byte uncompressed result
     */
    external fun ctScalarMulPoint(scalar: ByteArray, point: ByteArray): ByteArray

    /**
     * CT ECDH: compute shared secret = x-coordinate of (privkey * pubkey)
     * Constant-time — safe against timing/power side-channels.
     * @param privkey 32-byte private key
     * @param pubkey 65-byte uncompressed public key of the other party
     * @return 32-byte shared secret (x-coordinate)
     */
    external fun ctEcdh(privkey: ByteArray, pubkey: ByteArray): ByteArray

    // ========================================================================
    // Private native declarations
    // ========================================================================
    private external fun nativeInit(): Boolean
    private external fun nativeSelfTest(): Boolean
}
