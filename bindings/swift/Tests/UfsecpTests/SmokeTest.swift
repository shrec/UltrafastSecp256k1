// ============================================================================
// UltrafastSecp256k1 — Swift Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies FFI boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.
//
// Usage:
//   swift test --filter SmokeTest
// ============================================================================

import XCTest
@testable import Ufsecp

final class SmokeTest: XCTestCase {

    // ── Golden Vectors ──────────────────────────────────────────────────

    static let KNOWN_PRIVKEY = Data(hexString:
        "0000000000000000000000000000000000000000000000000000000000000001")!

    static let KNOWN_PUBKEY = Data(hexString:
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")!

    static let KNOWN_XONLY = Data(hexString:
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")!

    static let SHA256_EMPTY = Data(hexString:
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")!

    let msg32 = Data(count: 32)
    let aux32 = Data(count: 32)

    var ctx: UfsecpContext!

    override func setUp() {
        super.setUp()
        ctx = try! UfsecpContext()
    }

    override func tearDown() {
        ctx = nil
        super.tearDown()
    }

    // ── Tests ───────────────────────────────────────────────────────────

    func testCtxCreateAbi() throws {
        let abi = ctx.abiVersion
        XCTAssertGreaterThanOrEqual(abi, 1, "ABI version < 1")
    }

    func testPubkeyCreateGolden() throws {
        let pub = try ctx.pubkeyCreate(privkey: Self.KNOWN_PRIVKEY)
        XCTAssertEqual(pub, Self.KNOWN_PUBKEY, "compressed pubkey mismatch")
    }

    func testPubkeyXonlyGolden() throws {
        let xonly = try ctx.pubkeyXonly(privkey: Self.KNOWN_PRIVKEY)
        XCTAssertEqual(xonly, Self.KNOWN_XONLY, "x-only pubkey mismatch")
    }

    func testEcdsaSignVerify() throws {
        let sig = try ctx.ecdsaSign(msgHash: msg32, privkey: Self.KNOWN_PRIVKEY)
        XCTAssertEqual(sig.count, 64, "sig length")

        let ok = try ctx.ecdsaVerify(msgHash: msg32, sig: sig, pubkey: Self.KNOWN_PUBKEY)
        XCTAssertTrue(ok, "valid sig rejected")

        // Mutated → fail
        var bad = sig
        bad[0] ^= 0x01
        let fail = try ctx.ecdsaVerify(msgHash: msg32, sig: bad, pubkey: Self.KNOWN_PUBKEY)
        XCTAssertFalse(fail, "mutated sig accepted")
    }

    func testSchnorrSignVerify() throws {
        let sig = try ctx.schnorrSign(msg: msg32, privkey: Self.KNOWN_PRIVKEY, auxRand: aux32)
        XCTAssertEqual(sig.count, 64, "schnorr sig length")

        let ok = try ctx.schnorrVerify(msg: msg32, sig: sig, pubkeyX: Self.KNOWN_XONLY)
        XCTAssertTrue(ok, "valid schnorr sig rejected")
    }

    func testEcdsaRecover() throws {
        let rec = try ctx.ecdsaSignRecoverable(msgHash: msg32, privkey: Self.KNOWN_PRIVKEY)
        let pub = try ctx.ecdsaRecover(msgHash: msg32, sig: rec.signature, recid: rec.recoveryId)
        XCTAssertEqual(pub, Self.KNOWN_PUBKEY, "recovered pubkey mismatch")
    }

    func testSha256Golden() throws {
        let digest = try ctx.sha256(Data())
        XCTAssertEqual(digest, Self.SHA256_EMPTY, "SHA-256 empty mismatch")
    }

    func testAddrP2wpkh() throws {
        let addr = try ctx.addrP2wpkh(pubkey: Self.KNOWN_PUBKEY, network: .mainnet)
        XCTAssertTrue(addr.hasPrefix("bc1q"), "Expected bc1q..., got \(addr)")
    }

    func testWifRoundtrip() throws {
        let wif = try ctx.wifEncode(privkey: Self.KNOWN_PRIVKEY, compressed: true, network: .mainnet)
        let decoded = try ctx.wifDecode(wif: wif)
        XCTAssertEqual(decoded.privkey, Self.KNOWN_PRIVKEY, "WIF privkey")
        XCTAssertTrue(decoded.compressed, "WIF compressed")
    }

    func testEcdhSymmetric() throws {
        let k2 = Data(hexString:
            "0000000000000000000000000000000000000000000000000000000000000002")!
        let pub1 = try ctx.pubkeyCreate(privkey: Self.KNOWN_PRIVKEY)
        let pub2 = try ctx.pubkeyCreate(privkey: k2)
        let s12 = try ctx.ecdh(privkey: Self.KNOWN_PRIVKEY, pubkey: pub2)
        let s21 = try ctx.ecdh(privkey: k2, pubkey: pub1)
        XCTAssertEqual(s12, s21, "ECDH symmetric")
    }

    func testErrorPath() {
        let zeroes = Data(count: 32)
        XCTAssertThrowsError(try ctx.pubkeyCreate(privkey: zeroes),
            "zero key should throw")
    }

    func testEcdsaDeterministic() throws {
        let sig1 = try ctx.ecdsaSign(msgHash: msg32, privkey: Self.KNOWN_PRIVKEY)
        let sig2 = try ctx.ecdsaSign(msgHash: msg32, privkey: Self.KNOWN_PRIVKEY)
        XCTAssertEqual(sig1, sig2, "RFC 6979 deterministic")
    }
}

// ── Data hex extension ──────────────────────────────────────────────────
extension Data {
    init?(hexString: String) {
        let hex = hexString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard hex.count.isMultiple(of: 2) else { return nil }
        var data = Data(capacity: hex.count / 2)
        var idx = hex.startIndex
        while idx < hex.endIndex {
            let next = hex.index(idx, offsetBy: 2)
            guard let byte = UInt8(hex[idx..<next], radix: 16) else { return nil }
            data.append(byte)
            idx = next
        }
        self = data
    }
}
