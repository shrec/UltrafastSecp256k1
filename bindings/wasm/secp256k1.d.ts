/**
 * UltrafastSecp256k1 — TypeScript declarations
 */

export interface PointXY {
    x: Uint8Array;  // 32 bytes, big-endian
    y: Uint8Array;  // 32 bytes, big-endian
}

export declare class Secp256k1 {
    /**
     * Initialize the WASM module and return a ready-to-use instance.
     */
    static create(options?: Record<string, unknown>): Promise<Secp256k1>;

    /** Run self-test. Returns true if all internal checks pass. */
    selftest(): boolean;

    /** Library version string (e.g. "3.0.0"). */
    version(): string;

    /**
     * Derive public key from 32-byte private key.
     * @throws if key is invalid (zero or >= curve order)
     */
    pubkeyCreate(seckey: Uint8Array): PointXY;

    /**
     * Scalar × Point multiplication: R = scalar * P.
     * @throws if point is invalid or scalar is zero
     */
    pointMul(pointX: Uint8Array, pointY: Uint8Array, scalar: Uint8Array): PointXY;

    /**
     * Point addition: R = P + Q.
     * @throws if result is point at infinity
     */
    pointAdd(px: Uint8Array, py: Uint8Array, qx: Uint8Array, qy: Uint8Array): PointXY;

    /**
     * ECDSA sign (RFC 6979 deterministic nonce, low-S normalized).
     * @param msgHash 32-byte message hash
     * @param seckey 32-byte private key
     * @returns 64-byte compact signature (r || s)
     * @throws on invalid key
     */
    ecdsaSign(msgHash: Uint8Array, seckey: Uint8Array): Uint8Array;

    /**
     * ECDSA verify.
     * @returns true if signature is valid
     */
    ecdsaVerify(msgHash: Uint8Array, pubX: Uint8Array, pubY: Uint8Array, sig: Uint8Array): boolean;

    /**
     * Schnorr BIP-340 sign.
     * @param seckey 32-byte private key
     * @param msg 32-byte message
     * @param auxRand 32-byte auxiliary randomness (default: zeros)
     * @returns 64-byte signature (R.x || s)
     */
    schnorrSign(seckey: Uint8Array, msg: Uint8Array, auxRand?: Uint8Array): Uint8Array;

    /**
     * Schnorr BIP-340 verify.
     * @returns true if signature is valid
     */
    schnorrVerify(pubkeyX: Uint8Array, msg: Uint8Array, sig: Uint8Array): boolean;

    /**
     * Derive x-only public key for Schnorr (BIP-340).
     * @returns 32-byte x-only public key
     */
    schnorrPubkey(seckey: Uint8Array): Uint8Array;

    /**
     * SHA-256 hash.
     * @returns 32-byte digest
     */
    sha256(data: Uint8Array): Uint8Array;
}

export default Secp256k1;
