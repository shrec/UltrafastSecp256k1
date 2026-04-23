// UltrafastSecp256k1 — Swift Bindings
//
// High-performance secp256k1 elliptic curve cryptography.
//
// Usage:
//   import UltrafastSecp256k1
//   let lib = try Secp256k1()
//   let pubkey = try lib.ecPubkeyCreate(privkey: myPrivkey)

import Foundation
import CUltrafastSecp256k1

/// Secp256k1 error types.
public enum Secp256k1Error: Error, CustomStringConvertible {
    case initFailed
    case invalidPrivateKey
    case invalidPublicKey
    case signingFailed
    case verifyFailed
    case recoveryFailed
    case ecdhFailed
    case tweakFailed
    case addressFailed
    case wifFailed
    case bip32Failed
    case taprootFailed
    case derFailed
    case invalidInput(String)

    public var description: String {
        switch self {
        case .initFailed: return "Library initialization failed"
        case .invalidPrivateKey: return "Invalid private key"
        case .invalidPublicKey: return "Invalid public key"
        case .signingFailed: return "Signing failed"
        case .verifyFailed: return "Verification failed"
        case .recoveryFailed: return "Recovery failed"
        case .ecdhFailed: return "ECDH computation failed"
        case .tweakFailed: return "Tweak operation failed"
        case .addressFailed: return "Address generation failed"
        case .wifFailed: return "WIF encode/decode failed"
        case .bip32Failed: return "BIP-32 operation failed"
        case .taprootFailed: return "Taproot operation failed"
        case .derFailed: return "DER serialization failed"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        }
    }
}

/// Bitcoin network.
public enum Network: Int32 {
    case mainnet = 0
    case testnet = 1
}

/// Main library interface. Create one instance and reuse it (thread-safe).
public final class Secp256k1 {

    /// Initialize the library (runs selftest).
    public init() throws {
        guard CUltrafastSecp256k1.secp256k1_init() == 0 else {
            throw Secp256k1Error.initFailed
        }
    }

    /// Library version string.
    public var version: String {
        String(cString: CUltrafastSecp256k1.secp256k1_version())
    }

    // MARK: - Key Operations

    /// Compute compressed public key (33 bytes) from private key (32 bytes).
    public func ecPubkeyCreate(privkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        var out = Data(count: 33)
        let rc = privkey.withUnsafeBytes { pk in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_ec_pubkey_create(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.invalidPrivateKey }
        return out
    }

    /// Compute uncompressed public key (65 bytes).
    public func ecPubkeyCreateUncompressed(privkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        var out = Data(count: 65)
        let rc = privkey.withUnsafeBytes { pk in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_ec_pubkey_create_uncompressed(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.invalidPrivateKey }
        return out
    }

    /// Parse compressed (33) or uncompressed (65) pubkey. Returns compressed.
    public func ecPubkeyParse(input: Data) throws -> Data {
        guard input.count == 33 || input.count == 65 else {
            throw Secp256k1Error.invalidInput("pubkey must be 33 or 65 bytes")
        }
        var out = Data(count: 33)
        let rc = input.withUnsafeBytes { inp in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_ec_pubkey_parse(
                    inp.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    inp.count,
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.invalidPublicKey }
        return out
    }

    /// Check whether a private key is valid.
    public func ecSeckeyVerify(privkey: Data) -> Bool {
        guard privkey.count == 32 else { return false }
        return privkey.withUnsafeBytes { pk in
            CUltrafastSecp256k1.secp256k1_ec_seckey_verify(
                pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 1
        }
    }

    /// Negate private key.
    public func ecPrivkeyNegate(privkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        var out = privkey
        let rc = out.withUnsafeMutableBytes { buf in
            CUltrafastSecp256k1.secp256k1_ec_privkey_negate(
                buf.baseAddress!.assumingMemoryBound(to: UInt8.self))
        }
        guard rc == 0 else { throw Secp256k1Error.tweakFailed }
        return out
    }

    /// Add tweak to private key.
    public func ecPrivkeyTweakAdd(privkey: Data, tweak: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        try checkLen(tweak, 32, "tweak")
        var out = privkey
        let rc = out.withUnsafeMutableBytes { pk in
            tweak.withUnsafeBytes { tw in
                CUltrafastSecp256k1.secp256k1_ec_privkey_tweak_add(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    tw.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.tweakFailed }
        return out
    }

    /// Multiply private key by tweak.
    public func ecPrivkeyTweakMul(privkey: Data, tweak: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        try checkLen(tweak, 32, "tweak")
        var out = privkey
        let rc = out.withUnsafeMutableBytes { pk in
            tweak.withUnsafeBytes { tw in
                CUltrafastSecp256k1.secp256k1_ec_privkey_tweak_mul(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    tw.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.tweakFailed }
        return out
    }

    // MARK: - ECDSA

    /// Sign 32-byte hash with ECDSA (RFC 6979). Returns 64-byte compact signature.
    public func ecdsaSign(msgHash: Data, privkey: Data) throws -> Data {
        try checkLen(msgHash, 32, "msgHash")
        try checkLen(privkey, 32, "privkey")
        var sig = Data(count: 64)
        let rc = msgHash.withUnsafeBytes { mh in
            privkey.withUnsafeBytes { pk in
                sig.withUnsafeMutableBytes { s in
                    CUltrafastSecp256k1.secp256k1_ecdsa_sign(
                        mh.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self))
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.signingFailed }
        return sig
    }

    /// Verify ECDSA signature.
    public func ecdsaVerify(msgHash: Data, sig: Data, pubkey: Data) -> Bool {
        guard msgHash.count == 32, sig.count == 64, pubkey.count == 33 else { return false }
        return msgHash.withUnsafeBytes { mh in
            sig.withUnsafeBytes { s in
                pubkey.withUnsafeBytes { pk in
                    CUltrafastSecp256k1.secp256k1_ecdsa_verify(
                        mh.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 1
                }
            }
        }
    }

    /// Serialize compact sig to DER.
    public func ecdsaSerializeDer(sig: Data) throws -> Data {
        try checkLen(sig, 64, "sig")
        var der = Data(count: 72)
        var derLen = 72
        let rc = sig.withUnsafeBytes { s in
            der.withUnsafeMutableBytes { d in
                withUnsafeMutablePointer(to: &derLen) { dl in
                    CUltrafastSecp256k1.secp256k1_ecdsa_signature_serialize_der(
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        d.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        dl)
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.derFailed }
        return der.prefix(derLen)
    }

    // MARK: - Recovery

    /// Sign with recovery id.
    public func ecdsaSignRecoverable(msgHash: Data, privkey: Data) throws -> (sig: Data, recid: Int) {
        try checkLen(msgHash, 32, "msgHash")
        try checkLen(privkey, 32, "privkey")
        var sig = Data(count: 64)
        var recid: Int32 = 0
        let rc = msgHash.withUnsafeBytes { mh in
            privkey.withUnsafeBytes { pk in
                sig.withUnsafeMutableBytes { s in
                    withUnsafeMutablePointer(to: &recid) { rid in
                        CUltrafastSecp256k1.secp256k1_ecdsa_sign_recoverable(
                            mh.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            rid)
                    }
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.signingFailed }
        return (sig, Int(recid))
    }

    /// Recover public key from recoverable signature.
    public func ecdsaRecover(msgHash: Data, sig: Data, recid: Int) throws -> Data {
        try checkLen(msgHash, 32, "msgHash")
        try checkLen(sig, 64, "sig")
        var pubkey = Data(count: 33)
        let rc = msgHash.withUnsafeBytes { mh in
            sig.withUnsafeBytes { s in
                pubkey.withUnsafeMutableBytes { pk in
                    CUltrafastSecp256k1.secp256k1_ecdsa_recover(
                        mh.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        Int32(recid),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self))
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.recoveryFailed }
        return pubkey
    }

    // MARK: - Schnorr

    /// BIP-340 Schnorr sign. Returns 64-byte signature.
    public func schnorrSign(msg: Data, privkey: Data, auxRand: Data) throws -> Data {
        try checkLen(msg, 32, "msg")
        try checkLen(privkey, 32, "privkey")
        try checkLen(auxRand, 32, "auxRand")
        var sig = Data(count: 64)
        let rc = msg.withUnsafeBytes { m in
            privkey.withUnsafeBytes { pk in
                auxRand.withUnsafeBytes { ar in
                    sig.withUnsafeMutableBytes { s in
                        CUltrafastSecp256k1.secp256k1_schnorr_sign(
                            m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            ar.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            s.baseAddress!.assumingMemoryBound(to: UInt8.self))
                    }
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.signingFailed }
        return sig
    }

    /// Verify BIP-340 Schnorr signature.
    public func schnorrVerify(msg: Data, sig: Data, pubkeyX: Data) -> Bool {
        guard msg.count == 32, sig.count == 64, pubkeyX.count == 32 else { return false }
        return msg.withUnsafeBytes { m in
            sig.withUnsafeBytes { s in
                pubkeyX.withUnsafeBytes { px in
                    CUltrafastSecp256k1.secp256k1_schnorr_verify(
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        px.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 1
                }
            }
        }
    }

    /// Get x-only public key (32 bytes).
    public func schnorrPubkey(privkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        var out = Data(count: 32)
        let rc = privkey.withUnsafeBytes { pk in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_schnorr_pubkey(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.invalidPrivateKey }
        return out
    }

    // MARK: - ECDH

    /// ECDH shared secret: SHA256(compressed shared point).
    public func ecdh(privkey: Data, pubkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        try checkLen(pubkey, 33, "pubkey")
        var out = Data(count: 32)
        let rc = privkey.withUnsafeBytes { pk in
            pubkey.withUnsafeBytes { pub in
                out.withUnsafeMutableBytes { o in
                    CUltrafastSecp256k1.secp256k1_ecdh(
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pub.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        o.baseAddress!.assumingMemoryBound(to: UInt8.self))
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.ecdhFailed }
        return out
    }

    /// ECDH x-only.
    public func ecdhXonly(privkey: Data, pubkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        try checkLen(pubkey, 33, "pubkey")
        var out = Data(count: 32)
        let rc = privkey.withUnsafeBytes { pk in
            pubkey.withUnsafeBytes { pub in
                out.withUnsafeMutableBytes { o in
                    CUltrafastSecp256k1.secp256k1_ecdh_xonly(
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pub.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        o.baseAddress!.assumingMemoryBound(to: UInt8.self))
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.ecdhFailed }
        return out
    }

    /// ECDH raw x-coordinate.
    public func ecdhRaw(privkey: Data, pubkey: Data) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        try checkLen(pubkey, 33, "pubkey")
        var out = Data(count: 32)
        let rc = privkey.withUnsafeBytes { pk in
            pubkey.withUnsafeBytes { pub in
                out.withUnsafeMutableBytes { o in
                    CUltrafastSecp256k1.secp256k1_ecdh_raw(
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pub.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        o.baseAddress!.assumingMemoryBound(to: UInt8.self))
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.ecdhFailed }
        return out
    }

    // MARK: - Hashing

    /// SHA-256.
    public func sha256(data: Data) -> Data {
        var out = Data(count: 32)
        data.withUnsafeBytes { d in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_sha256(
                    d.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    d.count,
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        return out
    }

    /// HASH160.
    public func hash160(data: Data) -> Data {
        var out = Data(count: 20)
        data.withUnsafeBytes { d in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_hash160(
                    d.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    d.count,
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        return out
    }

    /// Tagged hash (BIP-340).
    public func taggedHash(tag: String, data: Data) -> Data {
        var out = Data(count: 32)
        data.withUnsafeBytes { d in
            out.withUnsafeMutableBytes { o in
                CUltrafastSecp256k1.secp256k1_tagged_hash(
                    tag,
                    d.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    d.count,
                    o.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        return out
    }

    // MARK: - Addresses

    /// P2PKH address.
    public func addressP2PKH(pubkey: Data, network: Network = .mainnet) throws -> String {
        try checkLen(pubkey, 33, "pubkey")
        return try getAddress { buf, len in
            pubkey.withUnsafeBytes { pk in
                CUltrafastSecp256k1.secp256k1_address_p2pkh(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    network.rawValue, buf, len)
            }
        }
    }

    /// P2WPKH address.
    public func addressP2WPKH(pubkey: Data, network: Network = .mainnet) throws -> String {
        try checkLen(pubkey, 33, "pubkey")
        return try getAddress { buf, len in
            pubkey.withUnsafeBytes { pk in
                CUltrafastSecp256k1.secp256k1_address_p2wpkh(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    network.rawValue, buf, len)
            }
        }
    }

    /// P2TR address from x-only key.
    public func addressP2TR(internalKeyX: Data, network: Network = .mainnet) throws -> String {
        try checkLen(internalKeyX, 32, "internalKeyX")
        return try getAddress { buf, len in
            internalKeyX.withUnsafeBytes { ik in
                CUltrafastSecp256k1.secp256k1_address_p2tr(
                    ik.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    network.rawValue, buf, len)
            }
        }
    }

    // MARK: - WIF

    /// Encode private key as WIF.
    public func wifEncode(privkey: Data, compressed: Bool = true, network: Network = .mainnet) throws -> String {
        try checkLen(privkey, 32, "privkey")
        return try getAddress { buf, len in
            privkey.withUnsafeBytes { pk in
                CUltrafastSecp256k1.secp256k1_wif_encode(
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    compressed ? 1 : 0, network.rawValue, buf, len)
            }
        }
    }

    /// Decode WIF string.
    public func wifDecode(wif: String) throws -> (privkey: Data, compressed: Bool, network: Network) {
        var pk = Data(count: 32)
        var comp: Int32 = 0
        var net: Int32 = 0
        let rc = pk.withUnsafeMutableBytes { pkBuf in
            withUnsafeMutablePointer(to: &comp) { compPtr in
                withUnsafeMutablePointer(to: &net) { netPtr in
                    CUltrafastSecp256k1.secp256k1_wif_decode(
                        wif,
                        pkBuf.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        compPtr, netPtr)
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.wifFailed }
        return (pk, comp == 1, Network(rawValue: net) ?? .mainnet)
    }

    // MARK: - BIP-32

    /// Create master key from seed (16-64 bytes). Returns 79-byte opaque key.
    public func bip32MasterKey(seed: Data) throws -> Data {
        guard seed.count >= 16 && seed.count <= 64 else {
            throw Secp256k1Error.invalidInput("seed must be 16-64 bytes")
        }
        var key = Data(count: 79)
        let rc = seed.withUnsafeBytes { s in
            key.withUnsafeMutableBytes { k in
                CUltrafastSecp256k1.secp256k1_bip32_master_key(
                    s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    s.count,
                    k.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.bip32Failed }
        return key
    }

    /// Derive child key by index.
    public func bip32DeriveChild(parent: Data, index: UInt32) throws -> Data {
        try checkLen(parent, 79, "parent")
        var child = Data(count: 79)
        let rc = parent.withUnsafeBytes { p in
            child.withUnsafeMutableBytes { c in
                CUltrafastSecp256k1.secp256k1_bip32_derive_child(
                    p.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self),
                    index,
                    c.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.bip32Failed }
        return child
    }

    /// Derive key from path string.
    public func bip32DerivePath(master: Data, path: String) throws -> Data {
        try checkLen(master, 79, "master")
        var key = Data(count: 79)
        let rc = master.withUnsafeBytes { m in
            key.withUnsafeMutableBytes { k in
                CUltrafastSecp256k1.secp256k1_bip32_derive_path(
                    m.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self),
                    path,
                    k.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.bip32Failed }
        return key
    }

    /// Extract private key from extended key.
    public func bip32GetPrivkey(key: Data) throws -> Data {
        try checkLen(key, 79, "key")
        var pk = Data(count: 32)
        let rc = key.withUnsafeBytes { k in
            pk.withUnsafeMutableBytes { p in
                CUltrafastSecp256k1.secp256k1_bip32_get_privkey(
                    k.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self),
                    p.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.bip32Failed }
        return pk
    }

    /// Extract compressed public key from extended key.
    public func bip32GetPubkey(key: Data) throws -> Data {
        try checkLen(key, 79, "key")
        var pub = Data(count: 33)
        let rc = key.withUnsafeBytes { k in
            pub.withUnsafeMutableBytes { p in
                CUltrafastSecp256k1.secp256k1_bip32_get_pubkey(
                    k.baseAddress!.assumingMemoryBound(to: secp256k1_bip32_key.self),
                    p.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
        }
        guard rc == 0 else { throw Secp256k1Error.bip32Failed }
        return pub
    }

    // MARK: - Taproot

    /// Derive Taproot output key.
    public func taprootOutputKey(internalKeyX: Data, merkleRoot: Data? = nil) throws -> (outputKeyX: Data, parity: Int) {
        try checkLen(internalKeyX, 32, "internalKeyX")
        var out = Data(count: 32)
        var parity: Int32 = 0
        let rc: Int32
        if let mr = merkleRoot {
            rc = internalKeyX.withUnsafeBytes { ik in
                mr.withUnsafeBytes { m in
                    out.withUnsafeMutableBytes { o in
                        withUnsafeMutablePointer(to: &parity) { p in
                            CUltrafastSecp256k1.secp256k1_taproot_output_key(
                                ik.baseAddress!.assumingMemoryBound(to: UInt8.self),
                                m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                                o.baseAddress!.assumingMemoryBound(to: UInt8.self), p)
                        }
                    }
                }
            }
        } else {
            rc = internalKeyX.withUnsafeBytes { ik in
                out.withUnsafeMutableBytes { o in
                    withUnsafeMutablePointer(to: &parity) { p in
                        CUltrafastSecp256k1.secp256k1_taproot_output_key(
                            ik.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            nil,
                            o.baseAddress!.assumingMemoryBound(to: UInt8.self), p)
                    }
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.taprootFailed }
        return (out, Int(parity))
    }

    /// Tweak private key for Taproot.
    public func taprootTweakPrivkey(privkey: Data, merkleRoot: Data? = nil) throws -> Data {
        try checkLen(privkey, 32, "privkey")
        var out = Data(count: 32)
        let rc: Int32
        if let mr = merkleRoot {
            rc = privkey.withUnsafeBytes { pk in
                mr.withUnsafeBytes { m in
                    out.withUnsafeMutableBytes { o in
                        CUltrafastSecp256k1.secp256k1_taproot_tweak_privkey(
                            pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            o.baseAddress!.assumingMemoryBound(to: UInt8.self))
                    }
                }
            }
        } else {
            rc = privkey.withUnsafeBytes { pk in
                out.withUnsafeMutableBytes { o in
                    CUltrafastSecp256k1.secp256k1_taproot_tweak_privkey(
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        nil,
                        o.baseAddress!.assumingMemoryBound(to: UInt8.self))
                }
            }
        }
        guard rc == 0 else { throw Secp256k1Error.taprootFailed }
        return out
    }

    /// Verify Taproot commitment.
    public func taprootVerifyCommitment(outputKeyX: Data, parity: Int, internalKeyX: Data, merkleRoot: Data?) -> Bool {
        guard outputKeyX.count == 32, internalKeyX.count == 32 else { return false }
        let mrPtr: UnsafePointer<UInt8>? = nil
        let mrLen = merkleRoot?.count ?? 0
        return outputKeyX.withUnsafeBytes { ok in
            internalKeyX.withUnsafeBytes { ik in
                if let mr = merkleRoot {
                    return mr.withUnsafeBytes { m in
                        CUltrafastSecp256k1.secp256k1_taproot_verify_commitment(
                            ok.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            Int32(parity),
                            ik.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            mrLen) == 1
                    }
                } else {
                    return CUltrafastSecp256k1.secp256k1_taproot_verify_commitment(
                        ok.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        Int32(parity),
                        ik.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        mrPtr, 0) == 1
                }
            }
        }
    }

    // MARK: - Internal

    private func checkLen(_ data: Data, _ expected: Int, _ name: String) throws {
        guard data.count == expected else {
            throw Secp256k1Error.invalidInput("\(name) must be \(expected) bytes, got \(data.count)")
        }
    }

    private func getAddress(_ fn: (UnsafeMutablePointer<CChar>, UnsafeMutablePointer<Int>) -> Int32) throws -> String {
        var buf = [CChar](repeating: 0, count: 128)
        var len = 128
        let rc = fn(&buf, &len)
        guard rc == 0 else { throw Secp256k1Error.addressFailed }
        return String(cString: buf)
    }
}
