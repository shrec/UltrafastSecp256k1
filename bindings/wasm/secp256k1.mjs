/**
 * UltrafastSecp256k1 — High-level JavaScript/TypeScript wrapper
 *
 * Usage:
 *   import { Secp256k1 } from './secp256k1.mjs';
 *   const lib = await Secp256k1.create();
 *   const { x, y } = lib.pubkeyCreate(privateKeyBytes);
 */

import createSecp256k1 from './secp256k1_wasm.js';

/**
 * High-level secp256k1 wrapper around the WASM module.
 * All inputs/outputs are Uint8Array in big-endian byte order.
 */
export class Secp256k1 {
    /** @internal */
    constructor(mod) {
        this._mod = mod;
        this._malloc = mod._malloc;
        this._free = mod._free;
    }

    /**
     * Initialize the WASM module and return a ready-to-use instance.
     * @param {object} [options] Emscripten module options
     * @returns {Promise<Secp256k1>}
     */
    static async create(options = {}) {
        const mod = await createSecp256k1(options);
        return new Secp256k1(mod);
    }

    /** Run self-test. Returns true if all internal checks pass. */
    selftest() {
        return this._mod._secp256k1_wasm_selftest() === 1;
    }

    /** Library version string. */
    version() {
        return this._mod.UTF8ToString(this._mod._secp256k1_wasm_version());
    }

    /**
     * Derive public key from 32-byte private key.
     * @param {Uint8Array} seckey 32-byte private key
     * @returns {{ x: Uint8Array, y: Uint8Array }} 32-byte X, Y coordinates
     * @throws if key is invalid
     */
    pubkeyCreate(seckey) {
        _assertLen(seckey, 32, 'seckey');
        return this._callWithBuffers(
            (sk, px, py) => this._mod._secp256k1_wasm_pubkey_create(sk, px, py),
            [seckey],
            [32, 32],
            (result, [x, y]) => {
                if (result !== 1) throw new Error('Invalid private key');
                return { x, y };
            }
        );
    }

    /**
     * Scalar × Point multiplication: R = scalar * P.
     * @param {Uint8Array} pointX 32-byte X coordinate
     * @param {Uint8Array} pointY 32-byte Y coordinate
     * @param {Uint8Array} scalar 32-byte scalar
     * @returns {{ x: Uint8Array, y: Uint8Array }}
     */
    pointMul(pointX, pointY, scalar) {
        _assertLen(pointX, 32, 'pointX');
        _assertLen(pointY, 32, 'pointY');
        _assertLen(scalar, 32, 'scalar');
        return this._callWithBuffers(
            (px, py, s, ox, oy) => this._mod._secp256k1_wasm_point_mul(px, py, s, ox, oy),
            [pointX, pointY, scalar],
            [32, 32],
            (result, [x, y]) => {
                if (result !== 1) throw new Error('Point multiplication failed');
                return { x, y };
            }
        );
    }

    /**
     * Point addition: R = P + Q.
     * @param {Uint8Array} px 32-byte X of P
     * @param {Uint8Array} py 32-byte Y of P
     * @param {Uint8Array} qx 32-byte X of Q
     * @param {Uint8Array} qy 32-byte Y of Q
     * @returns {{ x: Uint8Array, y: Uint8Array }}
     */
    pointAdd(px, py, qx, qy) {
        _assertLen(px, 32, 'px');
        _assertLen(py, 32, 'py');
        _assertLen(qx, 32, 'qx');
        _assertLen(qy, 32, 'qy');
        return this._callWithBuffers(
            (p1, p2, q1, q2, ox, oy) =>
                this._mod._secp256k1_wasm_point_add(p1, p2, q1, q2, ox, oy),
            [px, py, qx, qy],
            [32, 32],
            (result, [x, y]) => {
                if (result !== 1) throw new Error('Point addition failed');
                return { x, y };
            }
        );
    }

    /**
     * ECDSA sign (RFC 6979, low-S normalized).
     * @param {Uint8Array} msgHash 32-byte message hash
     * @param {Uint8Array} seckey 32-byte private key
     * @returns {Uint8Array} 64-byte compact signature (r || s)
     */
    ecdsaSign(msgHash, seckey) {
        _assertLen(msgHash, 32, 'msgHash');
        _assertLen(seckey, 32, 'seckey');
        return this._callWithBuffers(
            (m, s, sig) => this._mod._secp256k1_wasm_ecdsa_sign(m, s, sig),
            [msgHash, seckey],
            [64],
            (result, [sig]) => {
                if (result !== 1) throw new Error('ECDSA sign failed');
                return sig;
            }
        );
    }

    /**
     * ECDSA verify.
     * @param {Uint8Array} msgHash 32-byte message hash
     * @param {Uint8Array} pubX 32-byte public key X
     * @param {Uint8Array} pubY 32-byte public key Y
     * @param {Uint8Array} sig 64-byte compact signature
     * @returns {boolean}
     */
    ecdsaVerify(msgHash, pubX, pubY, sig) {
        _assertLen(msgHash, 32, 'msgHash');
        _assertLen(pubX, 32, 'pubX');
        _assertLen(pubY, 32, 'pubY');
        _assertLen(sig, 64, 'sig');
        return this._callWithBuffers(
            (m, px, py, s) => this._mod._secp256k1_wasm_ecdsa_verify(m, px, py, s),
            [msgHash, pubX, pubY, sig],
            [],
            (result) => result === 1
        );
    }

    /**
     * Schnorr BIP-340 sign.
     * @param {Uint8Array} seckey 32-byte private key
     * @param {Uint8Array} msg 32-byte message
     * @param {Uint8Array} [auxRand] 32-byte auxiliary randomness (default: zeros)
     * @returns {Uint8Array} 64-byte signature
     */
    schnorrSign(seckey, msg, auxRand) {
        _assertLen(seckey, 32, 'seckey');
        _assertLen(msg, 32, 'msg');
        const aux = auxRand || new Uint8Array(32);
        _assertLen(aux, 32, 'auxRand');
        return this._callWithBuffers(
            (s, m, a, sig) => this._mod._secp256k1_wasm_schnorr_sign(s, m, a, sig),
            [seckey, msg, aux],
            [64],
            (result, [sig]) => {
                if (result !== 1) throw new Error('Schnorr sign failed');
                return sig;
            }
        );
    }

    /**
     * Schnorr BIP-340 verify.
     * @param {Uint8Array} pubkeyX 32-byte x-only public key
     * @param {Uint8Array} msg 32-byte message
     * @param {Uint8Array} sig 64-byte signature
     * @returns {boolean}
     */
    schnorrVerify(pubkeyX, msg, sig) {
        _assertLen(pubkeyX, 32, 'pubkeyX');
        _assertLen(msg, 32, 'msg');
        _assertLen(sig, 64, 'sig');
        return this._callWithBuffers(
            (pk, m, s) => this._mod._secp256k1_wasm_schnorr_verify(pk, m, s),
            [pubkeyX, msg, sig],
            [],
            (result) => result === 1
        );
    }

    /**
     * Derive x-only public key for Schnorr (BIP-340).
     * @param {Uint8Array} seckey 32-byte private key
     * @returns {Uint8Array} 32-byte x-only public key
     */
    schnorrPubkey(seckey) {
        _assertLen(seckey, 32, 'seckey');
        return this._callWithBuffers(
            (s, pk) => this._mod._secp256k1_wasm_schnorr_pubkey(s, pk),
            [seckey],
            [32],
            (result, [pk]) => {
                if (result !== 1) throw new Error('Invalid private key');
                return pk;
            }
        );
    }

    /**
     * SHA-256 hash.
     * @param {Uint8Array} data Input data (any length)
     * @returns {Uint8Array} 32-byte hash
     */
    sha256(data) {
        const inPtr = this._malloc(data.length);
        const outPtr = this._malloc(32);
        try {
            this._mod.HEAPU8.set(data, inPtr);
            this._mod._secp256k1_wasm_sha256(inPtr, data.length, outPtr);
            return new Uint8Array(this._mod.HEAPU8.slice(outPtr, outPtr + 32));
        } finally {
            this._free(inPtr);
            this._free(outPtr);
        }
    }

    /**
     * @internal
     * Generic helper: allocate input + output buffers, call fn, read outputs.
     */
    _callWithBuffers(fn, inputs, outputSizes, finish) {
        const ptrs = [];
        try {
            // Allocate input buffers and copy data
            const inPtrs = inputs.map((buf) => {
                const p = this._malloc(buf.length);
                this._mod.HEAPU8.set(buf, p);
                ptrs.push(p);
                return p;
            });

            // Allocate output buffers
            const outPtrs = outputSizes.map((sz) => {
                const p = this._malloc(sz);
                ptrs.push(p);
                return p;
            });

            // Call native function
            const result = fn(...inPtrs, ...outPtrs);

            // Read outputs
            const outputs = outPtrs.map((p, i) =>
                new Uint8Array(this._mod.HEAPU8.slice(p, p + outputSizes[i]))
            );

            return finish(result, outputs);
        } finally {
            ptrs.forEach((p) => this._free(p));
        }
    }
}

/** @internal */
function _assertLen(buf, expected, name) {
    if (!(buf instanceof Uint8Array) || buf.length !== expected) {
        throw new TypeError(`${name} must be a Uint8Array of length ${expected}`);
    }
}

export default Secp256k1;
