// ============================================================================
// UltrafastSecp256k1 — Java Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies FFI/JNI boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.
//
// Usage:
//   javac -d out SmokeTest.java && java -cp out SmokeTest
// ============================================================================

package com.ultrafast.ufsecp.tests;

import com.ultrafast.ufsecp.Ufsecp;

public class SmokeTest {

    static int passed = 0, failed = 0;

    // ── Golden Vectors ──────────────────────────────────────────────────

    static final byte[] KNOWN_PRIVKEY = hexToBytes(
        "0000000000000000000000000000000000000000000000000000000000000001");

    static final byte[] KNOWN_PUBKEY = hexToBytes(
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

    static final byte[] KNOWN_XONLY = hexToBytes(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

    static final byte[] SHA256_EMPTY = hexToBytes(
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");

    static final byte[] MSG32 = new byte[32];
    static final byte[] AUX32 = new byte[32];

    // ── Helpers ─────────────────────────────────────────────────────────

    static void check(String name, Runnable fn) {
        try {
            fn.run();
            System.out.println("  [OK] " + name);
            passed++;
        } catch (Exception e) {
            System.out.println("  [FAIL] " + name + ": " + e.getMessage());
            failed++;
        }
    }

    static void assertTrue(boolean cond, String msg) {
        if (!cond) throw new RuntimeException("Assertion failed: " + msg);
    }

    static void assertArrayEquals(byte[] a, byte[] b, String label) {
        if (a.length != b.length)
            throw new RuntimeException(label + ": length " + a.length + " != " + b.length);
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i])
                throw new RuntimeException(label + ": differ at index " + i);
        }
    }

    static byte[] hexToBytes(String hex) {
        int len = hex.length() / 2;
        byte[] out = new byte[len];
        for (int i = 0; i < len; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }

    // ── Tests ───────────────────────────────────────────────────────────

    public static void main(String[] args) {
        System.out.println("UltrafastSecp256k1 Java Smoke Test");
        System.out.println("============================================================");

        try (Ufsecp ctx = new Ufsecp()) {

            check("ctx_create_abi", () -> {
                int abi = ctx.abiVersion();
                assertTrue(abi >= 1, "ABI " + abi + " < 1");
            });

            check("pubkey_create_golden", () -> {
                byte[] pub = ctx.pubkeyCreate(KNOWN_PRIVKEY);
                assertTrue(pub != null, "pubkeyCreate returned null");
                assertArrayEquals(pub, KNOWN_PUBKEY, "compressed pubkey");
            });

            check("pubkey_xonly_golden", () -> {
                byte[] xonly = ctx.pubkeyXonly(KNOWN_PRIVKEY);
                assertTrue(xonly != null, "pubkeyXonly returned null");
                assertArrayEquals(xonly, KNOWN_XONLY, "x-only pubkey");
            });

            check("ecdsa_sign_verify", () -> {
                byte[] sig = ctx.ecdsaSign(MSG32, KNOWN_PRIVKEY);
                assertTrue(sig != null && sig.length == 64, "sig null or wrong length");
                boolean ok = ctx.ecdsaVerify(MSG32, sig, KNOWN_PUBKEY);
                assertTrue(ok, "valid sig rejected");

                // Mutated → fail
                byte[] bad = sig.clone();
                bad[0] ^= 0x01;
                boolean fail = ctx.ecdsaVerify(MSG32, bad, KNOWN_PUBKEY);
                assertTrue(!fail, "mutated sig accepted");
            });

            check("schnorr_sign_verify", () -> {
                byte[] sig = ctx.schnorrSign(MSG32, KNOWN_PRIVKEY, AUX32);
                assertTrue(sig != null && sig.length == 64, "sig null or wrong length");
                boolean ok = ctx.schnorrVerify(MSG32, sig, KNOWN_XONLY);
                assertTrue(ok, "valid schnorr sig rejected");
            });

            check("ecdsa_recover", () -> {
                var rec = ctx.ecdsaSignRecoverable(MSG32, KNOWN_PRIVKEY);
                assertTrue(rec != null, "recoverable sign returned null");
                byte[] pub = ctx.ecdsaRecover(MSG32, rec.signature, rec.recoveryId);
                assertTrue(pub != null, "recovery returned null");
                assertArrayEquals(pub, KNOWN_PUBKEY, "recovered pubkey");
            });

            check("sha256_golden", () -> {
                byte[] digest = Ufsecp.sha256(new byte[0]);
                assertTrue(digest != null, "sha256 returned null");
                assertArrayEquals(digest, SHA256_EMPTY, "SHA-256 empty");
            });

            check("addr_p2wpkh", () -> {
                String addr = ctx.addrP2wpkh(KNOWN_PUBKEY, 0);
                assertTrue(addr != null && addr.startsWith("bc1q"),
                    "Expected bc1q..., got " + addr);
            });

            check("wif_roundtrip", () -> {
                String wif = ctx.wifEncode(KNOWN_PRIVKEY, true, 0);
                assertTrue(wif != null, "wifEncode returned null");
                var decoded = ctx.wifDecode(wif);
                assertTrue(decoded != null, "wifDecode returned null");
                assertArrayEquals(decoded.privkey, KNOWN_PRIVKEY, "WIF privkey");
                assertTrue(decoded.compressed, "WIF compressed");
                assertTrue(decoded.network == 0, "WIF mainnet");
            });

            check("ecdh_symmetric", () -> {
                byte[] k2 = hexToBytes(
                    "0000000000000000000000000000000000000000000000000000000000000002");
                byte[] pub1 = ctx.pubkeyCreate(KNOWN_PRIVKEY);
                byte[] pub2 = ctx.pubkeyCreate(k2);
                byte[] s12 = ctx.ecdh(KNOWN_PRIVKEY, pub2);
                byte[] s21 = ctx.ecdh(k2, pub1);
                assertArrayEquals(s12, s21, "ECDH symmetric");
            });

            check("error_path", () -> {
                byte[] zeroes = new byte[32];
                boolean threw = false;
                try {
                    ctx.pubkeyCreate(zeroes);
                } catch (Exception e) {
                    threw = true;
                }
                assertTrue(threw, "zero key should throw/fail");
            });

            check("ecdsa_deterministic", () -> {
                byte[] sig1 = ctx.ecdsaSign(MSG32, KNOWN_PRIVKEY);
                byte[] sig2 = ctx.ecdsaSign(MSG32, KNOWN_PRIVKEY);
                assertArrayEquals(sig1, sig2, "RFC 6979 deterministic");
            });

        }

        System.out.println("============================================================");
        System.out.println("  Java smoke test: " + passed + " passed, " + failed + " failed");
        System.out.println("============================================================");
        System.exit(failed > 0 ? 1 : 0);
    }
}
