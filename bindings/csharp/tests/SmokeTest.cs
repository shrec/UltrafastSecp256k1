// ============================================================================
// UltrafastSecp256k1 -- C# Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies FFI boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.
//
// Usage:
//   dotnet run --project bindings/csharp/SmokeTest
// ============================================================================

using System;
using System.Linq;

namespace UltrafastSecp256k1.SmokeTest
{
    class Program
    {
        static int passed = 0, failed = 0;

        // ── Golden Vectors ──────────────────────────────────────────────

        static readonly byte[] KNOWN_PRIVKEY = HexToBytes(
            "0000000000000000000000000000000000000000000000000000000000000001");

        static readonly byte[] KNOWN_PUBKEY = HexToBytes(
            "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

        static readonly byte[] KNOWN_XONLY = HexToBytes(
            "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");

        static readonly byte[] SHA256_EMPTY = HexToBytes(
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");

        static readonly byte[] MSG32 = new byte[32];
        static readonly byte[] AUX32 = new byte[32];

        // ── Tests ───────────────────────────────────────────────────────

        static void Test(string name, Action fn)
        {
            try
            {
                fn();
                Console.WriteLine($"  [OK] {name}");
                passed++;
            }
            catch (Exception e)
            {
                Console.WriteLine($"  [FAIL] {name}: {e.Message}");
                failed++;
            }
        }

        static void Main()
        {
            Console.WriteLine("UltrafastSecp256k1 C# Smoke Test");
            Console.WriteLine(new string('=', 60));

            Test("ctx_create_abi", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var abi = ctx.AbiVersion;
                Assert(abi >= 1, $"ABI {abi} < 1");
            });

            Test("pubkey_create_golden", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var pub = ctx.PubkeyCreate(KNOWN_PRIVKEY);
                AssertEqual(pub, KNOWN_PUBKEY, "compressed pubkey");
            });

            Test("pubkey_xonly_golden", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var xonly = ctx.PubkeyXonly(KNOWN_PRIVKEY);
                AssertEqual(xonly, KNOWN_XONLY, "x-only pubkey");
            });

            Test("ecdsa_sign_verify", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var sig = ctx.EcdsaSign(MSG32, KNOWN_PRIVKEY);
                Assert(sig.Length == 64, "sig length");
                ctx.EcdsaVerify(MSG32, sig, KNOWN_PUBKEY);

                // Mutated → fail
                var bad = (byte[])sig.Clone();
                bad[0] ^= 0x01;
                bool threw = false;
                try { ctx.EcdsaVerify(MSG32, bad, KNOWN_PUBKEY); }
                catch { threw = true; }
                Assert(threw, "mutated sig should fail");
            });

            Test("schnorr_sign_verify", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var sig = ctx.SchnorrSign(MSG32, KNOWN_PRIVKEY, AUX32);
                Assert(sig.Length == 64, "sig length");
                ctx.SchnorrVerify(MSG32, sig, KNOWN_XONLY);
            });

            Test("ecdsa_recover", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var (sig, recid) = ctx.EcdsaSignRecoverable(MSG32, KNOWN_PRIVKEY);
                Assert(recid >= 0 && recid <= 3, "recid range");
                var pub = ctx.EcdsaRecover(MSG32, sig, recid);
                AssertEqual(pub, KNOWN_PUBKEY, "recovered pubkey");
            });

            Test("sha256_golden", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var digest = ctx.Sha256(Array.Empty<byte>());
                AssertEqual(digest, SHA256_EMPTY, "SHA-256 empty");
            });

            Test("addr_p2wpkh", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var addr = ctx.AddrP2wpkh(KNOWN_PUBKEY, 0);
                Assert(addr.StartsWith("bc1q"), $"Expected bc1q..., got {addr}");
            });

            Test("wif_roundtrip", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var wif = ctx.WifEncode(KNOWN_PRIVKEY, true, 0);
                var (key, comp, net) = ctx.WifDecode(wif);
                AssertEqual(key, KNOWN_PRIVKEY, "WIF privkey");
                Assert(comp == 1, "compressed");
                Assert(net == 0, "mainnet");
            });

            Test("ecdh_symmetric", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var k2 = HexToBytes("0000000000000000000000000000000000000000000000000000000000000002");
                var pub1 = ctx.PubkeyCreate(KNOWN_PRIVKEY);
                var pub2 = ctx.PubkeyCreate(k2);
                var s12 = ctx.Ecdh(KNOWN_PRIVKEY, pub2);
                var s21 = ctx.Ecdh(k2, pub1);
                AssertEqual(s12, s21, "ECDH symmetric");
            });

            Test("error_path", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                bool threw = false;
                try { ctx.SeckeyVerify(new byte[32]); }
                catch { threw = true; }
                Assert(threw, "zero key should throw");
            });

            Test("ecdsa_deterministic", () =>
            {
                using var ctx = new Ufsecp.Ufsecp();
                var sig1 = ctx.EcdsaSign(MSG32, KNOWN_PRIVKEY);
                var sig2 = ctx.EcdsaSign(MSG32, KNOWN_PRIVKEY);
                AssertEqual(sig1, sig2, "RFC 6979 deterministic");
            });

            Console.WriteLine(new string('=', 60));
            Console.WriteLine($"  C# smoke test: {passed} passed, {failed} failed");
            Console.WriteLine(new string('=', 60));
            Environment.Exit(failed > 0 ? 1 : 0);
        }

        // ── Helpers ─────────────────────────────────────────────────────

        static void Assert(bool cond, string msg)
        {
            if (!cond) throw new Exception($"Assertion failed: {msg}");
        }

        static void AssertEqual(byte[] a, byte[] b, string label)
        {
            if (!a.SequenceEqual(b))
                throw new Exception($"{label}: {BitConverter.ToString(a)} != {BitConverter.ToString(b)}");
        }

        static byte[] HexToBytes(string hex)
        {
            var bytes = new byte[hex.Length / 2];
            for (int i = 0; i < bytes.Length; i++)
                bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
            return bytes;
        }
    }
}
