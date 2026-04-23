// ============================================================================
// UfsepcBenchmark — ufsecp C ABI: Correctness Tests + P/Invoke Benchmark
// ============================================================================
// Tests every major ufsecp_* operation for correctness, then benchmarks
// the hot paths to measure P/Invoke overhead vs native C++ performance.
//
// Usage:
//   dotnet run -c Release
//
// Requires: ufsecp.dll in the output directory (copy from build-ufsecp-test/)
// ============================================================================

using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Ufsecp;

// ── Known test vector: private key = 1, generator point G ────────────────
// pubkey(1) = G = 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
byte[] privkey1 = new byte[32];
privkey1[31] = 1;

byte[] expectedPubkey = HexToBytes(
    "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

// ── Utility ──────────────────────────────────────────────────────────────
int passed = 0, failed = 0;

void Assert(bool condition, string name)
{
    if (condition)
    {
        passed++;
        Console.WriteLine($"  [OK]   {name}");
    }
    else
    {
        failed++;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"  [FAIL] {name}");
        Console.ResetColor();
    }
}

void AssertEq(byte[] a, byte[] b, string name)
{
    Assert(a.Length == b.Length && a.SequenceEqual(b), name);
}

string BytesToHex(byte[] data) =>
    BitConverter.ToString(data).Replace("-", "").ToLowerInvariant();

byte[] HexToBytes(string hex)
{
    hex = hex.Replace(" ", "");
    byte[] bytes = new byte[hex.Length / 2];
    for (int i = 0; i < bytes.Length; i++)
        bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
    return bytes;
}

// ══════════════════════════════════════════════════════════════════════════
// PHASE 1: CORRECTNESS TESTS
// ══════════════════════════════════════════════════════════════════════════

Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
Console.WriteLine("║  UltrafastSecp256k1 — ufsecp C ABI Test + Benchmark     ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
Console.WriteLine();

// ── 1. Version & Context ─────────────────────────────────────────────────
Console.WriteLine("── 1. Version & Context ─────────────────────────────────");

string version = Marshal.PtrToStringAnsi(Native.ufsecp_version_string()) ?? "?";
uint abi = Native.ufsecp_abi_version();
Console.WriteLine($"  ufsecp {version}  (ABI {abi})");
Assert(abi >= 1, "ABI version >= 1");

int rc = Native.ufsecp_ctx_create(out IntPtr ctx);
Assert(rc == 0 && ctx != IntPtr.Zero, "ctx_create succeeds");

// Clone + destroy
rc = Native.ufsecp_ctx_clone(ctx, out IntPtr ctx2);
Assert(rc == 0 && ctx2 != IntPtr.Zero, "ctx_clone succeeds");
Native.ufsecp_ctx_destroy(ctx2);
Assert(true, "ctx_destroy(clone) no crash");

Console.WriteLine();

// ── 2. Private Key Utilities ─────────────────────────────────────────────
Console.WriteLine("── 2. Private Key Utilities ─────────────────────────────");

rc = Native.ufsecp_seckey_verify(ctx, privkey1);
Assert(rc == 0, "seckey_verify(1) = valid");

byte[] zeroKey = new byte[32];
rc = Native.ufsecp_seckey_verify(ctx, zeroKey);
Assert(rc != 0, "seckey_verify(0) = invalid");

// Negate: -1 mod n
byte[] negKey = (byte[])privkey1.Clone();
rc = Native.ufsecp_seckey_negate(ctx, negKey);
Assert(rc == 0 && !negKey.SequenceEqual(privkey1), "seckey_negate changes key");

// Double negate = original
byte[] negNeg = (byte[])negKey.Clone();
rc = Native.ufsecp_seckey_negate(ctx, negNeg);
Assert(rc == 0, "double negate: seckey_negate succeeded");
AssertEq(negNeg, privkey1, "double negate = original");

// Tweak add
byte[] tweakKey = (byte[])privkey1.Clone();
byte[] tweak = new byte[32]; tweak[31] = 2;
rc = Native.ufsecp_seckey_tweak_add(ctx, tweakKey, tweak);
Assert(rc == 0, "seckey_tweak_add succeeds");
byte[] expectedTweak3 = new byte[32]; expectedTweak3[31] = 3;
AssertEq(tweakKey, expectedTweak3, "1 + 2 = 3 (mod n)");

// Tweak mul
byte[] mulKey = new byte[32]; mulKey[31] = 5;
byte[] mulTweak = new byte[32]; mulTweak[31] = 7;
rc = Native.ufsecp_seckey_tweak_mul(ctx, mulKey, mulTweak);
Assert(rc == 0, "seckey_tweak_mul succeeds");
byte[] expected35 = new byte[32]; expected35[31] = 35;
AssertEq(mulKey, expected35, "5 * 7 = 35 (mod n)");

Console.WriteLine();

// ── 3. Public Key ────────────────────────────────────────────────────────
Console.WriteLine("── 3. Public Key ────────────────────────────────────────");

byte[] pubkey = new byte[33];
rc = Native.ufsecp_pubkey_create(ctx, privkey1, pubkey);
Assert(rc == 0, "pubkey_create succeeds");
AssertEq(pubkey, expectedPubkey, "pubkey(1) = G (compressed)");

byte[] pubkey65 = new byte[65];
rc = Native.ufsecp_pubkey_create_uncompressed(ctx, privkey1, pubkey65);
Assert(rc == 0, "pubkey_create_uncompressed succeeds");
Assert(pubkey65[0] == 0x04, "uncompressed prefix = 0x04");

// Parse uncompressed → compressed
byte[] parsed = new byte[33];
rc = Native.ufsecp_pubkey_parse(ctx, pubkey65, (nuint)65, parsed);
Assert(rc == 0, "pubkey_parse(65→33) succeeds");
AssertEq(parsed, expectedPubkey, "parsed = original compressed");

// x-only
byte[] xonly = new byte[32];
rc = Native.ufsecp_pubkey_xonly(ctx, privkey1, xonly);
Assert(rc == 0, "pubkey_xonly succeeds");
AssertEq(xonly, expectedPubkey[1..33], "xonly = G.x");

Console.WriteLine();

// ── 4. ECDSA Sign / Verify ───────────────────────────────────────────────
Console.WriteLine("── 4. ECDSA Sign / Verify ───────────────────────────────");

byte[] msg = new byte[32];
Native.ufsecp_sha256(Encoding.UTF8.GetBytes("hello ufsecp"), (nuint)12, msg);

byte[] sig = new byte[64];
rc = Native.ufsecp_ecdsa_sign(ctx, msg, privkey1, sig);
Assert(rc == 0, "ecdsa_sign succeeds");
Assert(!sig.All(b => b == 0), "signature is non-zero");

rc = Native.ufsecp_ecdsa_verify(ctx, msg, sig, pubkey);
Assert(rc == 0, "ecdsa_verify(valid) = OK");

byte[] badMsg = (byte[])msg.Clone();
badMsg[0] ^= 0xFF;
rc = Native.ufsecp_ecdsa_verify(ctx, badMsg, sig, pubkey);
Assert(rc != 0, "ecdsa_verify(tampered msg) = FAIL");

byte[] badSig = (byte[])sig.Clone();
badSig[0] ^= 0xFF;
rc = Native.ufsecp_ecdsa_verify(ctx, msg, badSig, pubkey);
Assert(rc != 0, "ecdsa_verify(tampered sig) = FAIL");

// DER round-trip
byte[] der = new byte[72];
nuint derLen = 72;
rc = Native.ufsecp_ecdsa_sig_to_der(ctx, sig, der, ref derLen);
Assert(rc == 0 && derLen > 0, $"sig_to_der succeeds (len={derLen})");

byte[] sigBack = new byte[64];
rc = Native.ufsecp_ecdsa_sig_from_der(ctx, der, derLen, sigBack);
Assert(rc == 0, "sig_from_der succeeds");
AssertEq(sigBack, sig, "DER round-trip: compact→DER→compact");

Console.WriteLine();

// ── 5. ECDSA Recovery ────────────────────────────────────────────────────
Console.WriteLine("── 5. ECDSA Recovery ────────────────────────────────────");

byte[] recSig = new byte[64];
int recid = -1;
rc = Native.ufsecp_ecdsa_sign_recoverable(ctx, msg, privkey1, recSig, ref recid);
Assert(rc == 0 && recid >= 0 && recid <= 3, $"sign_recoverable OK (recid={recid})");

byte[] recovered = new byte[33];
rc = Native.ufsecp_ecdsa_recover(ctx, msg, recSig, recid, recovered);
Assert(rc == 0, "ecdsa_recover succeeds");
AssertEq(recovered, pubkey, "recovered pubkey = original");

Console.WriteLine();

// ── 6. Schnorr / BIP-340 ────────────────────────────────────────────────
Console.WriteLine("── 6. Schnorr / BIP-340 ─────────────────────────────────");

byte[] auxRand = new byte[32]; // deterministic
byte[] schnorrSig = new byte[64];
rc = Native.ufsecp_schnorr_sign(ctx, msg, privkey1, auxRand, schnorrSig);
Assert(rc == 0, "schnorr_sign succeeds");

rc = Native.ufsecp_schnorr_verify(ctx, msg, schnorrSig, xonly);
Assert(rc == 0, "schnorr_verify(valid) = OK");

byte[] badSchnorr = (byte[])schnorrSig.Clone();
badSchnorr[0] ^= 0xFF;
rc = Native.ufsecp_schnorr_verify(ctx, msg, badSchnorr, xonly);
Assert(rc != 0, "schnorr_verify(tampered) = FAIL");

Console.WriteLine();

// ── 7. ECDH ──────────────────────────────────────────────────────────────
Console.WriteLine("── 7. ECDH ──────────────────────────────────────────────");

// Use privkey=2, pubkey of privkey=1 (G) → shared = 2*G
byte[] privkey2 = new byte[32]; privkey2[31] = 2;
byte[] pubkey2 = new byte[33];
Native.ufsecp_pubkey_create(ctx, privkey2, pubkey2);

byte[] secret1 = new byte[32];
byte[] secret2 = new byte[32];
// ECDH(priv1, pub2) should equal ECDH(priv2, pub1)
rc = Native.ufsecp_ecdh(ctx, privkey1, pubkey2, secret1);
Assert(rc == 0, "ecdh(1, pub2) succeeds");
rc = Native.ufsecp_ecdh(ctx, privkey2, pubkey, secret2);
Assert(rc == 0, "ecdh(2, pub1) succeeds");
AssertEq(secret1, secret2, "ECDH commutative: ECDH(1,P2) = ECDH(2,P1)");

// x-only mode
byte[] xsec1 = new byte[32], xsec2 = new byte[32];
Native.ufsecp_ecdh_xonly(ctx, privkey1, pubkey2, xsec1);
Native.ufsecp_ecdh_xonly(ctx, privkey2, pubkey, xsec2);
AssertEq(xsec1, xsec2, "ECDH x-only commutative");

// raw mode
byte[] rsec1 = new byte[32], rsec2 = new byte[32];
Native.ufsecp_ecdh_raw(ctx, privkey1, pubkey2, rsec1);
Native.ufsecp_ecdh_raw(ctx, privkey2, pubkey, rsec2);
AssertEq(rsec1, rsec2, "ECDH raw commutative");

Console.WriteLine();

// ── 8. Hashing ───────────────────────────────────────────────────────────
Console.WriteLine("── 8. Hashing ───────────────────────────────────────────");

// SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
byte[] empty = Array.Empty<byte>();
byte[] sha = new byte[32];
rc = Native.ufsecp_sha256(empty, 0, sha);
Assert(rc == 0, "sha256(\"\") succeeds");
AssertEq(sha, HexToBytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "sha256(\"\") = known vector");

// SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
byte[] abc = Encoding.UTF8.GetBytes("abc");
byte[] shaAbc = new byte[32];
Native.ufsecp_sha256(abc, (nuint)abc.Length, shaAbc);
AssertEq(shaAbc, HexToBytes("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    "sha256(\"abc\") = known vector");

// Hash160
byte[] h160 = new byte[20];
rc = Native.ufsecp_hash160(abc, (nuint)abc.Length, h160);
Assert(rc == 0 && !h160.All(b => b == 0), "hash160(\"abc\") non-zero");

// Tagged hash
byte[] tagged = new byte[32];
rc = Native.ufsecp_tagged_hash("BIP0340/challenge", abc, (nuint)abc.Length, tagged);
Assert(rc == 0 && !tagged.All(b => b == 0), "tagged_hash non-zero");

Console.WriteLine();

// ── 9. Addresses ─────────────────────────────────────────────────────────
Console.WriteLine("── 9. Addresses ─────────────────────────────────────────");

byte[] addrBuf = new byte[128];
nuint addrLen = 128;

// P2PKH mainnet for key=1 (G)
rc = Native.ufsecp_addr_p2pkh(ctx, pubkey, 0, addrBuf, ref addrLen);
Assert(rc == 0 && addrLen > 0, "addr_p2pkh succeeds");
string p2pkh = Encoding.ASCII.GetString(addrBuf, 0, (int)addrLen);
Assert(p2pkh.StartsWith("1"), $"P2PKH starts with '1': {p2pkh}");
Console.WriteLine($"       P2PKH:  {p2pkh}");

// P2WPKH
addrLen = 128;
rc = Native.ufsecp_addr_p2wpkh(ctx, pubkey, 0, addrBuf, ref addrLen);
Assert(rc == 0 && addrLen > 0, "addr_p2wpkh succeeds");
string p2wpkh = Encoding.ASCII.GetString(addrBuf, 0, (int)addrLen);
Assert(p2wpkh.StartsWith("bc1q"), $"P2WPKH starts with 'bc1q': {p2wpkh}");
Console.WriteLine($"       P2WPKH: {p2wpkh}");

// P2TR
addrLen = 128;
rc = Native.ufsecp_addr_p2tr(ctx, xonly, 0, addrBuf, ref addrLen);
Assert(rc == 0 && addrLen > 0, "addr_p2tr succeeds");
string p2tr = Encoding.ASCII.GetString(addrBuf, 0, (int)addrLen);
Assert(p2tr.StartsWith("bc1p"), $"P2TR starts with 'bc1p': {p2tr}");
Console.WriteLine($"       P2TR:   {p2tr}");

Console.WriteLine();

// ── 10. WIF ──────────────────────────────────────────────────────────────
Console.WriteLine("── 10. WIF ──────────────────────────────────────────────");

byte[] wifBuf = new byte[128];
nuint wifLen = 128;
rc = Native.ufsecp_wif_encode(ctx, privkey1, 1, 0, wifBuf, ref wifLen);
Assert(rc == 0, "wif_encode succeeds");
string wif = Encoding.ASCII.GetString(wifBuf, 0, (int)wifLen);
Assert(wif.StartsWith("K") || wif.StartsWith("L"), $"WIF compressed mainnet: {wif}");
Console.WriteLine($"       WIF: {wif}");

// Round-trip
byte[] wifPriv = new byte[32];
int wifComp = 0, wifNet = 0;
rc = Native.ufsecp_wif_decode(ctx, wif, wifPriv, ref wifComp, ref wifNet);
Assert(rc == 0, "wif_decode succeeds");
AssertEq(wifPriv, privkey1, "WIF round-trip: privkey matches");
Assert(wifComp == 1, "WIF compressed = true");
Assert(wifNet == 0, "WIF network = mainnet");

Console.WriteLine();

// ── 11. BIP-32 ───────────────────────────────────────────────────────────
Console.WriteLine("── 11. BIP-32 ───────────────────────────────────────────");

byte[] seed16 = new byte[16];
for (int i = 0; i < 16; i++) seed16[i] = (byte)(i + 1);

byte[] masterKey = new byte[82]; // ufsecp_bip32_key = 78 + 1 + 3 = 82
rc = Native.ufsecp_bip32_master(ctx, seed16, 16, masterKey);
Assert(rc == 0, "bip32_master succeeds");

byte[] childKey = new byte[82];
rc = Native.ufsecp_bip32_derive(ctx, masterKey, 0, childKey);
Assert(rc == 0, "bip32_derive(index=0) succeeds");

byte[] bipPriv = new byte[32];
rc = Native.ufsecp_bip32_privkey(ctx, masterKey, bipPriv);
Assert(rc == 0, "bip32_privkey extracts key");
Assert(!bipPriv.All(b => b == 0), "extracted privkey non-zero");

byte[] bipPub = new byte[33];
rc = Native.ufsecp_bip32_pubkey(ctx, masterKey, bipPub);
Assert(rc == 0, "bip32_pubkey succeeds");
Assert(bipPub[0] == 0x02 || bipPub[0] == 0x03, "bip32 pubkey compressed prefix");

// Derive path
byte[] pathKey = new byte[82];
rc = Native.ufsecp_bip32_derive_path(ctx, masterKey, "m/44'/0'/0'/0/0", pathKey);
Assert(rc == 0, "bip32_derive_path(\"m/44'/0'/0'/0/0\") succeeds");

Console.WriteLine();

// ── 12. Taproot ──────────────────────────────────────────────────────────
Console.WriteLine("── 12. Taproot ──────────────────────────────────────────");

byte[] outputX = new byte[32];
int parity = 0;
rc = Native.ufsecp_taproot_output_key(ctx, xonly, null, outputX, ref parity);
Assert(rc == 0, "taproot_output_key (key-path) succeeds");
Assert(!outputX.All(b => b == 0), "taproot output key non-zero");
Assert(parity == 0 || parity == 1, $"taproot parity = {parity}");

byte[] tweaked = new byte[32];
rc = Native.ufsecp_taproot_tweak_seckey(ctx, privkey1, null, tweaked);
Assert(rc == 0, "taproot_tweak_seckey succeeds");

// Verify commitment
rc = Native.ufsecp_taproot_verify(ctx, outputX, parity, xonly, null, 0);
Assert(rc == 0, "taproot_verify(commitment) = OK");

Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════
// PHASE 2: BENCHMARKS
// ══════════════════════════════════════════════════════════════════════════

Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
Console.WriteLine("║  BENCHMARK — P/Invoke overhead measurement              ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
Console.WriteLine();

// Warm-up
for (int i = 0; i < 100; i++)
{
    byte[] tmp = new byte[64];
    Native.ufsecp_ecdsa_sign(ctx, msg, privkey1, tmp);
}

void Bench(string label, int iterations, Action action)
{
    // Warm-up
    for (int i = 0; i < Math.Min(iterations / 10, 100); i++) action();

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++) action();
    sw.Stop();

    double totalUs = sw.Elapsed.TotalMicroseconds;
    double perCallUs = totalUs / iterations;
    double perCallNs = perCallUs * 1000.0;
    double opsPerSec = iterations / sw.Elapsed.TotalSeconds;

    string unit;
    string value;
    if (perCallUs < 1.0)
    {
        unit = "ns";
        value = $"{perCallNs,8:F1}";
    }
    else
    {
        unit = "μs";
        value = $"{perCallUs,8:F2}";
    }

    Console.WriteLine($"  {label,-32} {value} {unit}/call   ({opsPerSec,12:N0} ops/s)");
}

int N_FAST = 50_000;   // for fast ops (hash, key verify)
int N_SIGN = 10_000;   // for sign/keygen
int N_VERIFY = 10_000; // for verify

Console.WriteLine($"  Iterations: sign/keygen={N_SIGN}, verify={N_VERIFY}, hash={N_FAST}");
Console.WriteLine($"  {"─",-32} {"─",-12} {"─",-20}");

// SHA-256
Bench("SHA-256 (32 bytes)", N_FAST, () =>
{
    byte[] d = new byte[32];
    Native.ufsecp_sha256(msg, 32, d);
});

// Hash160
Bench("Hash160 (33 bytes)", N_FAST, () =>
{
    byte[] d = new byte[20];
    Native.ufsecp_hash160(pubkey, 33, d);
});

// Key generation
Bench("Key Generation (pubkey_create)", N_SIGN, () =>
{
    byte[] pk = new byte[33];
    Native.ufsecp_pubkey_create(ctx, privkey1, pk);
});

// ECDSA Sign
Bench("ECDSA Sign (RFC 6979)", N_SIGN, () =>
{
    byte[] s = new byte[64];
    Native.ufsecp_ecdsa_sign(ctx, msg, privkey1, s);
});

// ECDSA Verify
Bench("ECDSA Verify", N_VERIFY, () =>
{
    Native.ufsecp_ecdsa_verify(ctx, msg, sig, pubkey);
});

// ECDSA Sign + Recover
Bench("ECDSA Sign Recoverable", N_SIGN, () =>
{
    byte[] s = new byte[64]; int rid = 0;
    Native.ufsecp_ecdsa_sign_recoverable(ctx, msg, privkey1, s, ref rid);
});

Bench("ECDSA Recover", N_VERIFY, () =>
{
    Native.ufsecp_ecdsa_recover(ctx, msg, recSig, recid, new byte[33]);
});

// Schnorr
Bench("Schnorr Sign (BIP-340)", N_SIGN, () =>
{
    byte[] s = new byte[64];
    Native.ufsecp_schnorr_sign(ctx, msg, privkey1, auxRand, s);
});

Bench("Schnorr Verify", N_VERIFY, () =>
{
    Native.ufsecp_schnorr_verify(ctx, msg, schnorrSig, xonly);
});

// ECDH
Bench("ECDH (SHA256 compressed)", N_SIGN, () =>
{
    byte[] s = new byte[32];
    Native.ufsecp_ecdh(ctx, privkey1, pubkey2, s);
});

Bench("ECDH x-only", N_SIGN, () =>
{
    byte[] s = new byte[32];
    Native.ufsecp_ecdh_xonly(ctx, privkey1, pubkey2, s);
});

Bench("ECDH raw", N_SIGN, () =>
{
    byte[] s = new byte[32];
    Native.ufsecp_ecdh_raw(ctx, privkey1, pubkey2, s);
});

// seckey_verify (very fast)
Bench("seckey_verify", N_FAST, () =>
{
    Native.ufsecp_seckey_verify(ctx, privkey1);
});

// DER round-trip
Bench("DER encode+decode", N_FAST, () =>
{
    byte[] d = new byte[72]; nuint dl = 72;
    Native.ufsecp_ecdsa_sig_to_der(ctx, sig, d, ref dl);
    byte[] sb = new byte[64];
    Native.ufsecp_ecdsa_sig_from_der(ctx, d, dl, sb);
});

// Address generation
Bench("Address P2PKH", N_SIGN, () =>
{
    byte[] a = new byte[128]; nuint l = 128;
    Native.ufsecp_addr_p2pkh(ctx, pubkey, 0, a, ref l);
});

Bench("Address P2WPKH", N_SIGN, () =>
{
    byte[] a = new byte[128]; nuint l = 128;
    Native.ufsecp_addr_p2wpkh(ctx, pubkey, 0, a, ref l);
});

Bench("Address P2TR", N_SIGN, () =>
{
    byte[] a = new byte[128]; nuint l = 128;
    Native.ufsecp_addr_p2tr(ctx, xonly, 0, a, ref l);
});

// WIF
Bench("WIF encode", N_FAST, () =>
{
    byte[] w = new byte[128]; nuint wl = 128;
    Native.ufsecp_wif_encode(ctx, privkey1, 1, 0, w, ref wl);
});

// Taproot
Bench("Taproot output key", N_SIGN, () =>
{
    byte[] ox = new byte[32]; int p = 0;
    Native.ufsecp_taproot_output_key(ctx, xonly, null, ox, ref p);
});

Console.WriteLine();

// ══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ══════════════════════════════════════════════════════════════════════════

Console.WriteLine("══════════════════════════════════════════════════════════");
Console.WriteLine($"  Tests: {passed} passed, {failed} failed, {passed + failed} total");
if (failed == 0)
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine("  ALL TESTS PASSED ✓");
}
else
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"  {failed} TEST(S) FAILED ✗");
}
Console.ResetColor();
Console.WriteLine("══════════════════════════════════════════════════════════");

// Cleanup
Native.ufsecp_ctx_destroy(ctx);

return failed > 0 ? 1 : 0;
