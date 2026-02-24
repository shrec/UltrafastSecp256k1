/// UltrafastSecp256k1 — Swift binding (ufsecp stable C ABI v1).
///
/// High-performance secp256k1 ECC with dual-layer constant-time architecture.
/// Context-based API.
///
/// Usage:
///     let ctx = try UfsecpContext()
///     let pubkey = try ctx.pubkeyCreate(privkey: Data(repeating: 0, count: 31) + Data([0x01]))
///     ctx.destroy()

import Foundation
#if canImport(CUfsecp)
import CUfsecp
#endif

// MARK: - Error

public enum UfsecpErrorCode: Int32 {
    case ok = 0
    case nullArg = 1
    case badKey = 2
    case badPubkey = 3
    case badSig = 4
    case badInput = 5
    case verifyFail = 6
    case arith = 7
    case selftest = 8
    case `internal` = 9
    case bufTooSmall = 10
}

public struct UfsecpError: Error, CustomStringConvertible {
    public let operation: String
    public let code: UfsecpErrorCode

    public var description: String {
        "ufsecp \(operation) failed: \(code) (\(code.rawValue))"
    }
}

// MARK: - Result types

public enum Network: Int32 {
    case mainnet = 0
    case testnet = 1
}

public struct RecoverableSignature {
    public let signature: Data
    public let recoveryId: Int32
}

public struct TaprootOutputKeyResult {
    public let outputKeyX: Data
    public let parity: Int32
}

public struct WifDecoded {
    public let privkey: Data
    public let compressed: Bool
    public let network: Network
}

// MARK: - Context

public final class UfsecpContext {
    private var ctx: OpaquePointer?
    private var destroyed = false

    public init() throws {
        var ptr: OpaquePointer?
        let rc = ufsecp_ctx_create(&ptr)
        guard rc == 0, let p = ptr else {
            throw UfsecpError(operation: "ctx_create", code: UfsecpErrorCode(rawValue: rc) ?? .internal)
        }
        self.ctx = p
    }

    deinit { destroy() }

    public func destroy() {
        guard !destroyed, let c = ctx else { return }
        ufsecp_ctx_destroy(c)
        ctx = nil
        destroyed = true
    }

    // MARK: Version

    public static var version: UInt32 { ufsecp_version() }
    public static var abiVersion: UInt32 { ufsecp_abi_version() }
    public static var versionString: String { String(cString: ufsecp_version_string()) }

    public var lastError: Int32 { try! alive(); return ufsecp_last_error(ctx!) }
    public var lastErrorMsg: String { try! alive(); return String(cString: ufsecp_last_error_msg(ctx!)) }

    // MARK: Key Operations

    public func pubkeyCreate(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 33)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_create(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "pubkey_create")
        }
        return Data(out)
    }

    public func pubkeyCreateUncompressed(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 65)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_create_uncompressed(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "pubkey_create_uncompressed")
        }
        return Data(out)
    }

    public func pubkeyParse(pubkey: Data) throws -> Data {
        try alive()
        var out = [UInt8](repeating: 0, count: 33)
        try pubkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_parse(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), pubkey.count, &out), "pubkey_parse")
        }
        return Data(out)
    }

    public func pubkeyXonly(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_xonly(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "pubkey_xonly")
        }
        return Data(out)
    }

    public func seckeyVerify(privkey: Data) throws -> Bool {
        try chk(privkey, 32, "privkey"); try alive()
        return privkey.withUnsafeBytes { pk in
            ufsecp_seckey_verify(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 0
        }
    }

    public func seckeyNegate(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var buf = [UInt8](privkey)
        try throwRC(ufsecp_seckey_negate(ctx!, &buf), "seckey_negate")
        return Data(buf)
    }

    public func seckeyTweakAdd(privkey: Data, tweak: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(tweak, 32, "tweak"); try alive()
        var buf = [UInt8](privkey)
        try tweak.withUnsafeBytes { tw in
            try throwRC(ufsecp_seckey_tweak_add(ctx!, &buf, tw.baseAddress!.assumingMemoryBound(to: UInt8.self)), "seckey_tweak_add")
        }
        return Data(buf)
    }

    public func seckeyTweakMul(privkey: Data, tweak: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(tweak, 32, "tweak"); try alive()
        var buf = [UInt8](privkey)
        try tweak.withUnsafeBytes { tw in
            try throwRC(ufsecp_seckey_tweak_mul(ctx!, &buf, tw.baseAddress!.assumingMemoryBound(to: UInt8.self)), "seckey_tweak_mul")
        }
        return Data(buf)
    }

    // MARK: ECDSA

    public func ecdsaSign(msgHash: Data, privkey: Data) throws -> Data {
        try chk(msgHash, 32, "msgHash"); try chk(privkey, 32, "privkey"); try alive()
        var sig = [UInt8](repeating: 0, count: 64)
        try msgHash.withUnsafeBytes { msg in
            try privkey.withUnsafeBytes { pk in
                try throwRC(ufsecp_ecdsa_sign(ctx!,
                    msg.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &sig), "ecdsa_sign")
            }
        }
        return Data(sig)
    }

    public func ecdsaVerify(msgHash: Data, sig: Data, pubkey: Data) throws -> Bool {
        try chk(msgHash, 32, "msgHash"); try chk(sig, 64, "sig"); try chk(pubkey, 33, "pubkey"); try alive()
        return msgHash.withUnsafeBytes { msg in
            sig.withUnsafeBytes { s in
                pubkey.withUnsafeBytes { pk in
                    ufsecp_ecdsa_verify(ctx!,
                        msg.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 0
                }
            }
        }
    }

    // MARK: Schnorr

    public func schnorrSign(msg: Data, privkey: Data, auxRand: Data) throws -> Data {
        try chk(msg, 32, "msg"); try chk(privkey, 32, "privkey"); try chk(auxRand, 32, "auxRand"); try alive()
        var sig = [UInt8](repeating: 0, count: 64)
        try msg.withUnsafeBytes { m in
            try privkey.withUnsafeBytes { pk in
                try auxRand.withUnsafeBytes { ar in
                    try throwRC(ufsecp_schnorr_sign(ctx!,
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        ar.baseAddress!.assumingMemoryBound(to: UInt8.self), &sig), "schnorr_sign")
                }
            }
        }
        return Data(sig)
    }

    public func schnorrVerify(msg: Data, sig: Data, pubkeyX: Data) throws -> Bool {
        try chk(msg, 32, "msg"); try chk(sig, 64, "sig"); try chk(pubkeyX, 32, "pubkeyX"); try alive()
        return msg.withUnsafeBytes { m in
            sig.withUnsafeBytes { s in
                pubkeyX.withUnsafeBytes { pk in
                    ufsecp_schnorr_verify(ctx!,
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 0
                }
            }
        }
    }

    // MARK: ECDH

    public func ecdh(privkey: Data, pubkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(pubkey, 33, "pubkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try privkey.withUnsafeBytes { pk in
            try pubkey.withUnsafeBytes { pub in
                try throwRC(ufsecp_ecdh(ctx!,
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pub.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "ecdh")
            }
        }
        return Data(out)
    }

    public func ecdhXonly(privkey: Data, pubkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(pubkey, 33, "pubkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try privkey.withUnsafeBytes { pk in
            try pubkey.withUnsafeBytes { pub in
                try throwRC(ufsecp_ecdh_xonly(ctx!,
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pub.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "ecdh_xonly")
            }
        }
        return Data(out)
    }

    public func ecdhRaw(privkey: Data, pubkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(pubkey, 33, "pubkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try privkey.withUnsafeBytes { pk in
            try pubkey.withUnsafeBytes { pub in
                try throwRC(ufsecp_ecdh_raw(ctx!,
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pub.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "ecdh_raw")
            }
        }
        return Data(out)
    }

    // MARK: ECDSA DER

    public func ecdsaSigToDer(sig: Data) throws -> Data {
        try chk(sig, 64, "sig"); try alive()
        var der = [UInt8](repeating: 0, count: 72)
        var dlen: Int = 72
        try sig.withUnsafeBytes { s in
            try throwRC(ufsecp_ecdsa_sig_to_der(ctx!,
                s.baseAddress!.assumingMemoryBound(to: UInt8.self), &der, &dlen), "ecdsa_sig_to_der")
        }
        return Data(der.prefix(dlen))
    }

    public func ecdsaSigFromDer(der: Data) throws -> Data {
        try alive()
        var sig = [UInt8](repeating: 0, count: 64)
        try der.withUnsafeBytes { d in
            try throwRC(ufsecp_ecdsa_sig_from_der(ctx!,
                d.baseAddress!.assumingMemoryBound(to: UInt8.self), der.count, &sig), "ecdsa_sig_from_der")
        }
        return Data(sig)
    }

    // MARK: ECDSA Recovery

    public func ecdsaSignRecoverable(msgHash: Data, privkey: Data) throws -> RecoverableSignature {
        try chk(msgHash, 32, "msgHash"); try chk(privkey, 32, "privkey"); try alive()
        var sig = [UInt8](repeating: 0, count: 64)
        var recid: Int32 = 0
        try msgHash.withUnsafeBytes { msg in
            try privkey.withUnsafeBytes { pk in
                try throwRC(ufsecp_ecdsa_sign_recoverable(ctx!,
                    msg.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &sig, &recid), "ecdsa_sign_recoverable")
            }
        }
        return RecoverableSignature(signature: Data(sig), recoveryId: recid)
    }

    public func ecdsaRecover(msgHash: Data, sig: Data, recid: Int32) throws -> Data {
        try chk(msgHash, 32, "msgHash"); try chk(sig, 64, "sig"); try alive()
        var pub = [UInt8](repeating: 0, count: 33)
        try msgHash.withUnsafeBytes { msg in
            try sig.withUnsafeBytes { s in
                try throwRC(ufsecp_ecdsa_recover(ctx!,
                    msg.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    s.baseAddress!.assumingMemoryBound(to: UInt8.self), recid, &pub), "ecdsa_recover")
            }
        }
        return Data(pub)
    }

    // MARK: Hashing

    public static func sha256(_ data: Data) throws -> Data {
        var out = [UInt8](repeating: 0, count: 32)
        try data.withUnsafeBytes { d in
            try throwRC(ufsecp_sha256(d.baseAddress!.assumingMemoryBound(to: UInt8.self), data.count, &out), "sha256")
        }
        return Data(out)
    }

    public static func hash160(_ data: Data) throws -> Data {
        var out = [UInt8](repeating: 0, count: 20)
        try data.withUnsafeBytes { d in
            try throwRC(ufsecp_hash160(d.baseAddress!.assumingMemoryBound(to: UInt8.self), data.count, &out), "hash160")
        }
        return Data(out)
    }

    public static func taggedHash(tag: String, data: Data) throws -> Data {
        var out = [UInt8](repeating: 0, count: 32)
        try data.withUnsafeBytes { d in
            try throwRC(ufsecp_tagged_hash(tag,
                d.baseAddress!.assumingMemoryBound(to: UInt8.self), data.count, &out), "tagged_hash")
        }
        return Data(out)
    }

    // MARK: Addresses

    public func addrP2pkh(pubkey: Data, network: Network = .mainnet) throws -> String {
        try chk(pubkey, 33, "pubkey"); try alive()
        var buf = [CChar](repeating: 0, count: 64)
        try pubkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_addr_p2pkh(ctx!,
                pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &buf, 64, network.rawValue), "addr_p2pkh")
        }
        return String(cString: buf)
    }

    public func addrP2wpkh(pubkey: Data, network: Network = .mainnet) throws -> String {
        try chk(pubkey, 33, "pubkey"); try alive()
        var buf = [CChar](repeating: 0, count: 128)
        try pubkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_addr_p2wpkh(ctx!,
                pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &buf, 128, network.rawValue), "addr_p2wpkh")
        }
        return String(cString: buf)
    }

    public func addrP2tr(xonly: Data, network: Network = .mainnet) throws -> String {
        try chk(xonly, 32, "xonly"); try alive()
        var buf = [CChar](repeating: 0, count: 128)
        try xonly.withUnsafeBytes { x in
            try throwRC(ufsecp_addr_p2tr(ctx!,
                x.baseAddress!.assumingMemoryBound(to: UInt8.self), &buf, 128, network.rawValue), "addr_p2tr")
        }
        return String(cString: buf)
    }

    // MARK: WIF

    public func wifEncode(privkey: Data, compressed: Bool = true, network: Network = .mainnet) throws -> String {
        try chk(privkey, 32, "privkey"); try alive()
        var buf = [CChar](repeating: 0, count: 64)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_wif_encode(ctx!,
                pk.baseAddress!.assumingMemoryBound(to: UInt8.self), compressed ? 1 : 0,
                network.rawValue, &buf, 64), "wif_encode")
        }
        return String(cString: buf)
    }

    public func wifDecode(wif: String) throws -> WifDecoded {
        try alive()
        var privkey = [UInt8](repeating: 0, count: 32)
        var compressed: Int32 = 0
        var net: Int32 = 0
        try throwRC(ufsecp_wif_decode(ctx!, wif, &privkey, &compressed, &net), "wif_decode")
        return WifDecoded(privkey: Data(privkey), compressed: compressed == 1,
                          network: Network(rawValue: net) ?? .mainnet)
    }

    // MARK: BIP-32

    public func bip32Master(seed: Data) throws -> Data {
        try alive()
        var out = [UInt8](repeating: 0, count: 64)
        try seed.withUnsafeBytes { s in
            try throwRC(ufsecp_bip32_master(ctx!,
                s.baseAddress!.assumingMemoryBound(to: UInt8.self), seed.count, &out), "bip32_master")
        }
        return Data(out)
    }

    public func bip32Derive(parent: Data, index: UInt32) throws -> Data {
        try chk(parent, 64, "parent"); try alive()
        var out = [UInt8](repeating: 0, count: 64)
        try parent.withUnsafeBytes { p in
            try throwRC(ufsecp_bip32_derive(ctx!,
                p.baseAddress!.assumingMemoryBound(to: UInt8.self), index, &out), "bip32_derive")
        }
        return Data(out)
    }

    public func bip32DerivePath(master: Data, path: String) throws -> Data {
        try chk(master, 64, "master"); try alive()
        var out = [UInt8](repeating: 0, count: 64)
        try master.withUnsafeBytes { m in
            try throwRC(ufsecp_bip32_derive_path(ctx!,
                m.baseAddress!.assumingMemoryBound(to: UInt8.self), path, &out), "bip32_derive_path")
        }
        return Data(out)
    }

    public func bip32Privkey(key: Data) throws -> Data {
        try chk(key, 64, "key"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try key.withUnsafeBytes { k in
            try throwRC(ufsecp_bip32_privkey(ctx!,
                k.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "bip32_privkey")
        }
        return Data(out)
    }

    public func bip32Pubkey(key: Data) throws -> Data {
        try chk(key, 64, "key"); try alive()
        var out = [UInt8](repeating: 0, count: 33)
        try key.withUnsafeBytes { k in
            try throwRC(ufsecp_bip32_pubkey(ctx!,
                k.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "bip32_pubkey")
        }
        return Data(out)
    }

    // MARK: Taproot

    public func taprootOutputKey(internalX: Data, merkleRoot: Data?) throws -> TaprootOutputKeyResult {
        try chk(internalX, 32, "internalX"); try alive()
        var outx = [UInt8](repeating: 0, count: 32)
        var parity: Int32 = 0
        if let mr = merkleRoot {
            try internalX.withUnsafeBytes { ix in
                try mr.withUnsafeBytes { m in
                    try throwRC(ufsecp_taproot_output_key(ctx!,
                        ix.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self), &outx, &parity), "taproot_output_key")
                }
            }
        } else {
            try internalX.withUnsafeBytes { ix in
                try throwRC(ufsecp_taproot_output_key(ctx!,
                    ix.baseAddress!.assumingMemoryBound(to: UInt8.self), nil, &outx, &parity), "taproot_output_key")
            }
        }
        return TaprootOutputKeyResult(outputKeyX: Data(outx), parity: parity)
    }

    public func taprootTweakSeckey(privkey: Data, merkleRoot: Data?) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        if let mr = merkleRoot {
            try privkey.withUnsafeBytes { pk in
                try mr.withUnsafeBytes { m in
                    try throwRC(ufsecp_taproot_tweak_seckey(ctx!,
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "taproot_tweak_seckey")
                }
            }
        } else {
            try privkey.withUnsafeBytes { pk in
                try throwRC(ufsecp_taproot_tweak_seckey(ctx!,
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self), nil, &out), "taproot_tweak_seckey")
            }
        }
        return Data(out)
    }

    public func taprootVerify(outputX: Data, parity: Int32, internalX: Data, merkleRoot: Data?) throws -> Bool {
        try chk(outputX, 32, "outputX"); try chk(internalX, 32, "internalX"); try alive()
        if let mr = merkleRoot {
            return outputX.withUnsafeBytes { ox in
                internalX.withUnsafeBytes { ix in
                    mr.withUnsafeBytes { m in
                        ufsecp_taproot_verify(ctx!,
                            ox.baseAddress!.assumingMemoryBound(to: UInt8.self), parity,
                            ix.baseAddress!.assumingMemoryBound(to: UInt8.self),
                            m.baseAddress!.assumingMemoryBound(to: UInt8.self), mr.count) == 0
                    }
                }
            }
        } else {
            return outputX.withUnsafeBytes { ox in
                internalX.withUnsafeBytes { ix in
                    ufsecp_taproot_verify(ctx!,
                        ox.baseAddress!.assumingMemoryBound(to: UInt8.self), parity,
                        ix.baseAddress!.assumingMemoryBound(to: UInt8.self), nil, 0) == 0
                }
            }
        }
    }

    // MARK: Internal

    private func alive() throws {
        guard !destroyed else { throw UfsecpError(operation: "alive", code: .internal) }
    }

    private func chk(_ data: Data, _ expected: Int, _ name: String) throws {
        guard data.count == expected else {
            throw UfsecpError(operation: "\(name) size", code: .badInput)
        }
    }

    private static func throwRC(_ rc: Int32, _ op: String) throws {
        guard rc == 0 else {
            throw UfsecpError(operation: op, code: UfsecpErrorCode(rawValue: rc) ?? .internal)
        }
    }

    private func throwRC(_ rc: Int32, _ op: String) throws {
        guard rc == 0 else {
            throw UfsecpError(operation: op, code: UfsecpErrorCode(rawValue: rc) ?? .internal)
        }
    }
}
