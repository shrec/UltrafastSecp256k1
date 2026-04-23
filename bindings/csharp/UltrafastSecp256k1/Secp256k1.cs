// ============================================================================
// UltrafastSecp256k1 — C# P/Invoke Bindings
// ============================================================================
// High-performance secp256k1 elliptic curve cryptography for .NET.
//
// Usage:
//   using UltrafastSecp256k1;
//
//   Secp256k1.Init();
//   byte[] privkey = new byte[32];
//   privkey[31] = 1; // private key = 1
//
//   byte[] pubkey = Secp256k1.CreatePublicKey(privkey);
//   byte[] sig = Secp256k1.EcdsaSign(msgHash, privkey);
//   bool valid = Secp256k1.EcdsaVerify(msgHash, sig, pubkey);
// ============================================================================

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace UltrafastSecp256k1
{
    /// <summary>
    /// Native P/Invoke declarations for the C API.
    /// </summary>
    internal static class Native
    {
        private const string LibName = "ultrafast_secp256k1";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr secp256k1_version();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_init();

        // Key operations
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_pubkey_create(byte[] privkey, byte[] pubkey_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_pubkey_create_uncompressed(byte[] privkey, byte[] pubkey_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_pubkey_parse(byte[] input, UIntPtr input_len, byte[] pubkey_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_seckey_verify(byte[] privkey);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_privkey_negate(byte[] privkey);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_privkey_tweak_add(byte[] privkey, byte[] tweak);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ec_privkey_tweak_mul(byte[] privkey, byte[] tweak);

        // ECDSA
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdsa_sign(byte[] msg_hash, byte[] privkey, byte[] sig_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdsa_verify(byte[] msg_hash, byte[] sig, byte[] pubkey);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdsa_signature_serialize_der(byte[] sig, byte[] der_out, ref UIntPtr der_len);

        // Recovery
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdsa_sign_recoverable(byte[] msg_hash, byte[] privkey, byte[] sig_out, ref int recid_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdsa_recover(byte[] msg_hash, byte[] sig, int recid, byte[] pubkey_out);

        // Schnorr
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_schnorr_sign(byte[] msg, byte[] privkey, byte[] aux_rand, byte[] sig_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_schnorr_verify(byte[] msg, byte[] sig, byte[] pubkey_x);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_schnorr_pubkey(byte[] privkey, byte[] pubkey_x_out);

        // ECDH
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdh(byte[] privkey, byte[] pubkey, byte[] secret_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdh_xonly(byte[] privkey, byte[] pubkey, byte[] secret_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_ecdh_raw(byte[] privkey, byte[] pubkey, byte[] secret_out);

        // Hashing
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void secp256k1_sha256(byte[] data, UIntPtr data_len, byte[] digest_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void secp256k1_hash160(byte[] data, UIntPtr data_len, byte[] digest_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void secp256k1_tagged_hash(byte[] tag, byte[] data, UIntPtr data_len, byte[] digest_out);

        // Addresses
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_address_p2pkh(byte[] pubkey, int network, byte[] addr_out, ref UIntPtr addr_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_address_p2wpkh(byte[] pubkey, int network, byte[] addr_out, ref UIntPtr addr_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_address_p2tr(byte[] internal_key_x, int network, byte[] addr_out, ref UIntPtr addr_len);

        // WIF
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_wif_encode(byte[] privkey, int compressed, int network, byte[] wif_out, ref UIntPtr wif_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_wif_decode(byte[] wif, byte[] privkey_out, ref int compressed_out, ref int network_out);

        // BIP-32
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_bip32_master_key(byte[] seed, UIntPtr seed_len, byte[] key_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_bip32_derive_path(byte[] master, byte[] path, byte[] key_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_bip32_get_privkey(byte[] key, byte[] privkey_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_bip32_get_pubkey(byte[] key, byte[] pubkey_out);

        // Taproot
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_taproot_output_key(byte[] internal_key_x, byte[]? merkle_root, byte[] output_key_x_out, ref int parity_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_taproot_tweak_privkey(byte[] privkey, byte[]? merkle_root, byte[] tweaked_out);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int secp256k1_taproot_verify_commitment(byte[] output_key_x, int output_key_parity, byte[] internal_key_x, byte[]? merkle_root, UIntPtr merkle_root_len);
    }

    /// <summary>
    /// Network selection for address generation.
    /// </summary>
    public enum Network
    {
        Mainnet = 0,
        Testnet = 1
    }

    /// <summary>
    /// High-level C# wrapper for UltrafastSecp256k1.
    /// All methods are static — the library initializes on first use.
    /// </summary>
    public static class Secp256k1
    {
        private static bool _initialized;
        private static readonly object _lock = new();

        /// <summary>
        /// Initialize the library (runs selftest). Called automatically on first use.
        /// </summary>
        public static void Init()
        {
            if (_initialized) return;
            lock (_lock)
            {
                if (_initialized) return;
                int rc = Native.secp256k1_init();
                if (rc != 0)
                    throw new InvalidOperationException("secp256k1_init() failed: library selftest failure");
                _initialized = true;
            }
        }

        /// <summary>Return the native library version string.</summary>
        public static string Version()
        {
            Init();
            IntPtr ptr = Native.secp256k1_version();
            return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
        }

        // ── Key Operations ───────────────────────────────────────────────

        /// <summary>Compute compressed public key (33 bytes) from private key (32 bytes).</summary>
        public static byte[] CreatePublicKey(byte[] privkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            byte[] pubkey = new byte[33];
            int rc = Native.secp256k1_ec_pubkey_create(privkey, pubkey);
            if (rc != 0) throw new ArgumentException("Invalid private key");
            return pubkey;
        }

        /// <summary>Compute uncompressed public key (65 bytes) from private key (32 bytes).</summary>
        public static byte[] CreatePublicKeyUncompressed(byte[] privkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            byte[] pubkey = new byte[65];
            int rc = Native.secp256k1_ec_pubkey_create_uncompressed(privkey, pubkey);
            if (rc != 0) throw new ArgumentException("Invalid private key");
            return pubkey;
        }

        /// <summary>Parse compressed (33) or uncompressed (65) public key. Returns compressed.</summary>
        public static byte[] ParsePublicKey(byte[] pubkey)
        {
            Init();
            byte[] result = new byte[33];
            int rc = Native.secp256k1_ec_pubkey_parse(pubkey, (UIntPtr)pubkey.Length, result);
            if (rc != 0) throw new ArgumentException("Invalid public key");
            return result;
        }

        /// <summary>Check if a private key is valid (non-zero, less than curve order).</summary>
        public static bool VerifySecretKey(byte[] privkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            return Native.secp256k1_ec_seckey_verify(privkey) == 1;
        }

        /// <summary>Negate a private key (mod n). Returns new key.</summary>
        public static byte[] NegatePrivateKey(byte[] privkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            byte[] result = (byte[])privkey.Clone();
            int rc = Native.secp256k1_ec_privkey_negate(result);
            if (rc != 0)
                throw new ArgumentException("NegatePrivateKey failed: invalid (zero) key");
            return result;
        }

        /// <summary>Add tweak to private key: (key + tweak) mod n.</summary>
        public static byte[] TweakAddPrivateKey(byte[] privkey, byte[] tweak)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            Check(tweak, 32, nameof(tweak));
            byte[] result = (byte[])privkey.Clone();
            int rc = Native.secp256k1_ec_privkey_tweak_add(result, tweak);
            if (rc != 0) throw new InvalidOperationException("Tweak add produced invalid key");
            return result;
        }

        /// <summary>Multiply private key by tweak: (key * tweak) mod n.</summary>
        public static byte[] TweakMulPrivateKey(byte[] privkey, byte[] tweak)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            Check(tweak, 32, nameof(tweak));
            byte[] result = (byte[])privkey.Clone();
            int rc = Native.secp256k1_ec_privkey_tweak_mul(result, tweak);
            if (rc != 0) throw new InvalidOperationException("Tweak mul produced invalid key");
            return result;
        }

        // ── ECDSA ────────────────────────────────────────────────────────

        /// <summary>
        /// Sign a 32-byte message hash with ECDSA (RFC 6979).
        /// Returns 64-byte compact signature (R || S, low-S).
        /// </summary>
        public static byte[] EcdsaSign(byte[] msgHash, byte[] privkey)
        {
            Init();
            Check(msgHash, 32, nameof(msgHash));
            Check(privkey, 32, nameof(privkey));
            byte[] sig = new byte[64];
            int rc = Native.secp256k1_ecdsa_sign(msgHash, privkey, sig);
            if (rc != 0) throw new InvalidOperationException("ECDSA signing failed");
            return sig;
        }

        /// <summary>Verify an ECDSA compact signature. Returns true if valid.</summary>
        public static bool EcdsaVerify(byte[] msgHash, byte[] sig, byte[] pubkey)
        {
            Init();
            Check(msgHash, 32, nameof(msgHash));
            Check(sig, 64, nameof(sig));
            Check(pubkey, 33, nameof(pubkey));
            return Native.secp256k1_ecdsa_verify(msgHash, sig, pubkey) == 1;
        }

        /// <summary>Serialize compact signature to DER format.</summary>
        public static byte[] EcdsaSerializeDer(byte[] sig)
        {
            Init();
            Check(sig, 64, nameof(sig));
            byte[] der = new byte[72];
            UIntPtr len = (UIntPtr)72;
            int rc = Native.secp256k1_ecdsa_signature_serialize_der(sig, der, ref len);
            if (rc != 0) throw new InvalidOperationException("DER serialization failed");
            byte[] result = new byte[(int)len];
            Array.Copy(der, result, (int)len);
            return result;
        }

        // ── ECDSA Recovery ───────────────────────────────────────────────

        /// <summary>Sign with recovery id. Returns (64-byte sig, recid).</summary>
        public static (byte[] Signature, int RecoveryId) EcdsaSignRecoverable(byte[] msgHash, byte[] privkey)
        {
            Init();
            Check(msgHash, 32, nameof(msgHash));
            Check(privkey, 32, nameof(privkey));
            byte[] sig = new byte[64];
            int recid = 0;
            int rc = Native.secp256k1_ecdsa_sign_recoverable(msgHash, privkey, sig, ref recid);
            if (rc != 0) throw new InvalidOperationException("Recoverable signing failed");
            return (sig, recid);
        }

        /// <summary>Recover compressed public key from recoverable signature.</summary>
        public static byte[] EcdsaRecover(byte[] msgHash, byte[] sig, int recid)
        {
            Init();
            Check(msgHash, 32, nameof(msgHash));
            Check(sig, 64, nameof(sig));
            byte[] pubkey = new byte[33];
            int rc = Native.secp256k1_ecdsa_recover(msgHash, sig, recid, pubkey);
            if (rc != 0) throw new InvalidOperationException("Recovery failed");
            return pubkey;
        }

        // ── Schnorr (BIP-340) ────────────────────────────────────────────

        /// <summary>Create BIP-340 Schnorr signature. Returns 64-byte signature.</summary>
        public static byte[] SchnorrSign(byte[] msg, byte[] privkey, byte[] auxRand)
        {
            Init();
            Check(msg, 32, nameof(msg));
            Check(privkey, 32, nameof(privkey));
            Check(auxRand, 32, nameof(auxRand));
            byte[] sig = new byte[64];
            int rc = Native.secp256k1_schnorr_sign(msg, privkey, auxRand, sig);
            if (rc != 0) throw new InvalidOperationException("Schnorr signing failed");
            return sig;
        }

        /// <summary>Verify BIP-340 Schnorr signature.</summary>
        public static bool SchnorrVerify(byte[] msg, byte[] sig, byte[] pubkeyX)
        {
            Init();
            Check(msg, 32, nameof(msg));
            Check(sig, 64, nameof(sig));
            Check(pubkeyX, 32, nameof(pubkeyX));
            return Native.secp256k1_schnorr_verify(msg, sig, pubkeyX) == 1;
        }

        /// <summary>Get x-only public key (32 bytes) for Schnorr.</summary>
        public static byte[] SchnorrPubkey(byte[] privkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            byte[] pubkeyX = new byte[32];
            int rc = Native.secp256k1_schnorr_pubkey(privkey, pubkeyX);
            if (rc != 0) throw new ArgumentException("Invalid private key");
            return pubkeyX;
        }

        // ── ECDH ─────────────────────────────────────────────────────────

        /// <summary>ECDH shared secret: SHA256(compressed shared point). Returns 32 bytes.</summary>
        public static byte[] Ecdh(byte[] privkey, byte[] pubkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            Check(pubkey, 33, nameof(pubkey));
            byte[] secret = new byte[32];
            int rc = Native.secp256k1_ecdh(privkey, pubkey, secret);
            if (rc != 0) throw new InvalidOperationException("ECDH failed");
            return secret;
        }

        /// <summary>ECDH x-only: SHA256(x-coordinate). Returns 32 bytes.</summary>
        public static byte[] EcdhXonly(byte[] privkey, byte[] pubkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            Check(pubkey, 33, nameof(pubkey));
            byte[] secret = new byte[32];
            int rc = Native.secp256k1_ecdh_xonly(privkey, pubkey, secret);
            if (rc != 0) throw new InvalidOperationException("ECDH xonly failed");
            return secret;
        }

        /// <summary>ECDH raw: raw x-coordinate of shared point. Returns 32 bytes.</summary>
        public static byte[] EcdhRaw(byte[] privkey, byte[] pubkey)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            Check(pubkey, 33, nameof(pubkey));
            byte[] secret = new byte[32];
            int rc = Native.secp256k1_ecdh_raw(privkey, pubkey, secret);
            if (rc != 0) throw new InvalidOperationException("ECDH raw failed");
            return secret;
        }

        // ── Hashing ──────────────────────────────────────────────────────

        /// <summary>SHA-256 hash. Returns 32 bytes.</summary>
        public static byte[] Sha256(byte[] data)
        {
            Init();
            byte[] digest = new byte[32];
            Native.secp256k1_sha256(data, (UIntPtr)data.Length, digest);
            return digest;
        }

        /// <summary>HASH160: RIPEMD160(SHA256(data)). Returns 20 bytes.</summary>
        public static byte[] Hash160(byte[] data)
        {
            Init();
            byte[] digest = new byte[20];
            Native.secp256k1_hash160(data, (UIntPtr)data.Length, digest);
            return digest;
        }

        /// <summary>BIP-340 tagged hash. Returns 32 bytes.</summary>
        public static byte[] TaggedHash(string tag, byte[] data)
        {
            Init();
            byte[] tagBytes = Encoding.UTF8.GetBytes(tag + '\0');
            byte[] digest = new byte[32];
            Native.secp256k1_tagged_hash(tagBytes, data, (UIntPtr)data.Length, digest);
            return digest;
        }

        // ── Bitcoin Addresses ────────────────────────────────────────────

        /// <summary>Generate P2PKH address from compressed public key.</summary>
        public static string AddressP2PKH(byte[] pubkey, Network network = Network.Mainnet)
        {
            Init();
            Check(pubkey, 33, nameof(pubkey));
            byte[] buf = new byte[128];
            UIntPtr len = (UIntPtr)128;
            int rc = Native.secp256k1_address_p2pkh(pubkey, (int)network, buf, ref len);
            if (rc != 0) throw new InvalidOperationException("P2PKH address generation failed");
            return Encoding.ASCII.GetString(buf, 0, (int)len);
        }

        /// <summary>Generate P2WPKH (SegWit v0) address from compressed public key.</summary>
        public static string AddressP2WPKH(byte[] pubkey, Network network = Network.Mainnet)
        {
            Init();
            Check(pubkey, 33, nameof(pubkey));
            byte[] buf = new byte[128];
            UIntPtr len = (UIntPtr)128;
            int rc = Native.secp256k1_address_p2wpkh(pubkey, (int)network, buf, ref len);
            if (rc != 0) throw new InvalidOperationException("P2WPKH address generation failed");
            return Encoding.ASCII.GetString(buf, 0, (int)len);
        }

        /// <summary>Generate P2TR (Taproot) address from x-only public key (32 bytes).</summary>
        public static string AddressP2TR(byte[] internalKeyX, Network network = Network.Mainnet)
        {
            Init();
            Check(internalKeyX, 32, nameof(internalKeyX));
            byte[] buf = new byte[128];
            UIntPtr len = (UIntPtr)128;
            int rc = Native.secp256k1_address_p2tr(internalKeyX, (int)network, buf, ref len);
            if (rc != 0) throw new InvalidOperationException("P2TR address generation failed");
            return Encoding.ASCII.GetString(buf, 0, (int)len);
        }

        // ── WIF ──────────────────────────────────────────────────────────

        /// <summary>Encode private key as WIF string.</summary>
        public static string WifEncode(byte[] privkey, bool compressed = true, Network network = Network.Mainnet)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            byte[] buf = new byte[128];
            UIntPtr len = (UIntPtr)128;
            int rc = Native.secp256k1_wif_encode(privkey, compressed ? 1 : 0, (int)network, buf, ref len);
            if (rc != 0) throw new InvalidOperationException("WIF encoding failed");
            return Encoding.ASCII.GetString(buf, 0, (int)len);
        }

        /// <summary>Decode WIF string. Returns (privkey, compressed, network).</summary>
        public static (byte[] PrivateKey, bool Compressed, Network Network) WifDecode(string wif)
        {
            Init();
            byte[] wifBytes = Encoding.ASCII.GetBytes(wif + '\0');
            byte[] privkey = new byte[32];
            int compressed = 0;
            int network = 0;
            int rc = Native.secp256k1_wif_decode(wifBytes, privkey, ref compressed, ref network);
            if (rc != 0) throw new ArgumentException("Invalid WIF string");
            return (privkey, compressed == 1, (Network)network);
        }

        // ── BIP-32 ───────────────────────────────────────────────────────

        /// <summary>Create master key from seed. Returns opaque 79-byte key.</summary>
        public static byte[] Bip32MasterKey(byte[] seed)
        {
            Init();
            if (seed.Length < 16 || seed.Length > 64)
                throw new ArgumentException("Seed must be 16-64 bytes");
            byte[] key = new byte[79];
            int rc = Native.secp256k1_bip32_master_key(seed, (UIntPtr)seed.Length, key);
            if (rc != 0) throw new InvalidOperationException("Master key generation failed");
            return key;
        }

        /// <summary>Derive key from path string, e.g. "m/44'/0'/0'/0/0".</summary>
        public static byte[] Bip32DerivePath(byte[] masterKey, string path)
        {
            Init();
            Check(masterKey, 79, nameof(masterKey));
            byte[] pathBytes = Encoding.ASCII.GetBytes(path + '\0');
            byte[] key = new byte[79];
            int rc = Native.secp256k1_bip32_derive_path(masterKey, pathBytes, key);
            if (rc != 0) throw new InvalidOperationException($"Path derivation failed: {path}");
            return key;
        }

        /// <summary>Get private key bytes from extended key.</summary>
        public static byte[] Bip32GetPrivateKey(byte[] key)
        {
            Init();
            Check(key, 79, nameof(key));
            byte[] privkey = new byte[32];
            int rc = Native.secp256k1_bip32_get_privkey(key, privkey);
            if (rc != 0) throw new InvalidOperationException("Key is not a private key");
            return privkey;
        }

        /// <summary>Get compressed public key from extended key.</summary>
        public static byte[] Bip32GetPublicKey(byte[] key)
        {
            Init();
            Check(key, 79, nameof(key));
            byte[] pubkey = new byte[33];
            int rc = Native.secp256k1_bip32_get_pubkey(key, pubkey);
            if (rc != 0) throw new InvalidOperationException("Public key extraction failed");
            return pubkey;
        }

        // ── Taproot ──────────────────────────────────────────────────────

        /// <summary>Derive Taproot output key. Returns (xonly_key, parity).</summary>
        public static (byte[] OutputKeyX, int Parity) TaprootOutputKey(byte[] internalKeyX, byte[]? merkleRoot = null)
        {
            Init();
            Check(internalKeyX, 32, nameof(internalKeyX));
            byte[] outputKeyX = new byte[32];
            int parity = 0;
            int rc = Native.secp256k1_taproot_output_key(internalKeyX, merkleRoot, outputKeyX, ref parity);
            if (rc != 0) throw new InvalidOperationException("Taproot output key derivation failed");
            return (outputKeyX, parity);
        }

        /// <summary>Tweak private key for Taproot key-path spending.</summary>
        public static byte[] TaprootTweakPrivkey(byte[] privkey, byte[]? merkleRoot = null)
        {
            Init();
            Check(privkey, 32, nameof(privkey));
            byte[] tweaked = new byte[32];
            int rc = Native.secp256k1_taproot_tweak_privkey(privkey, merkleRoot, tweaked);
            if (rc != 0) throw new InvalidOperationException("Taproot privkey tweaking failed");
            return tweaked;
        }

        /// <summary>Verify Taproot commitment (control block validation).</summary>
        public static bool TaprootVerifyCommitment(byte[] outputKeyX, int outputKeyParity,
                                                   byte[] internalKeyX, byte[]? merkleRoot = null)
        {
            Init();
            Check(outputKeyX, 32, nameof(outputKeyX));
            Check(internalKeyX, 32, nameof(internalKeyX));
            UIntPtr mrLen = merkleRoot != null ? (UIntPtr)merkleRoot.Length : UIntPtr.Zero;
            return Native.secp256k1_taproot_verify_commitment(outputKeyX, outputKeyParity,
                internalKeyX, merkleRoot, mrLen) == 1;
        }

        // ── Helpers ──────────────────────────────────────────────────────

        private static void Check(byte[] data, int expected, string name)
        {
            if (data == null) throw new ArgumentNullException(name);
            if (data.Length != expected)
                throw new ArgumentException($"{name} must be {expected} bytes, got {data.Length}");
        }
    }
}
