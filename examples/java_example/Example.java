/**
 * UltrafastSecp256k1 -- Java Example (CPU + GPU)
 *
 * Demonstrates direct JNA FFI to the ufsecp C ABI: key ops, ECDSA, Schnorr,
 * ECDH, hashing, Bitcoin addresses, and GPU batch operations.
 *
 * Build & Run:
 *   # Download jna.jar from Maven Central (or use the one in this directory):
 *   # https://repo1.maven.org/maven2/net/java/dev/jna/jna/5.14.0/jna-5.14.0.jar
 *
 *   javac -cp jna.jar Example.java
 *   java -cp .:jna.jar \
 *       -Djna.library.path=../../build-linux/include/ufsecp \
 *       Example
 */

import com.sun.jna.*;
import com.sun.jna.ptr.*;
import java.util.Arrays;
import java.util.HexFormat;

import com.sun.jna.NativeLong;

public class Example {

    // ── Native Library Interface ──────────────────────────────────────

    public interface Ufsecp extends Library {
        Ufsecp INSTANCE = Native.load("ufsecp", Ufsecp.class);

        // Context
        int ufsecp_ctx_create(PointerByReference ctx);
        void ufsecp_ctx_destroy(Pointer ctx);
        int ufsecp_abi_version();
        String ufsecp_version_string();
        String ufsecp_error_str(int code);

        // Keys
        int ufsecp_seckey_verify(Pointer ctx, byte[] k32);
        int ufsecp_pubkey_create(Pointer ctx, byte[] sk32, byte[] pk33);
        int ufsecp_pubkey_create_uncompressed(Pointer ctx, byte[] sk32, byte[] pk65);
        int ufsecp_pubkey_xonly(Pointer ctx, byte[] sk32, byte[] xo32);

        // ECDSA
        int ufsecp_ecdsa_sign(Pointer ctx, byte[] msg32, byte[] sk32, byte[] sig64);
        int ufsecp_ecdsa_verify(Pointer ctx, byte[] msg32, byte[] sig64, byte[] pk33);
        int ufsecp_ecdsa_sign_recoverable(Pointer ctx, byte[] msg32, byte[] sk32, byte[] sig64, IntByReference recid);
        int ufsecp_ecdsa_recover(Pointer ctx, byte[] msg32, byte[] sig64, int recid, byte[] pk33);
        int ufsecp_ecdsa_sig_to_der(Pointer ctx, byte[] sig64, byte[] der, NativeLongByReference len);

        // Schnorr
        int ufsecp_schnorr_sign(Pointer ctx, byte[] msg32, byte[] sk32, byte[] aux32, byte[] sig64);
        int ufsecp_schnorr_verify(Pointer ctx, byte[] msg32, byte[] sig64, byte[] xo32);

        // ECDH
        int ufsecp_ecdh(Pointer ctx, byte[] sk32, byte[] pk33, byte[] secret32);

        // Hashing
        int ufsecp_sha256(byte[] data, long len, byte[] digest32);
        int ufsecp_hash160(byte[] data, long len, byte[] digest20);

        // Addresses
        int ufsecp_addr_p2pkh(Pointer ctx, byte[] pk33, int net, byte[] addr, NativeLongByReference len);
        int ufsecp_addr_p2wpkh(Pointer ctx, byte[] pk33, int net, byte[] addr, NativeLongByReference len);
        int ufsecp_addr_p2tr(Pointer ctx, byte[] xo32, int net, byte[] addr, NativeLongByReference len);

        // WIF
        int ufsecp_wif_encode(Pointer ctx, byte[] sk32, int comp, int net, byte[] wif, NativeLongByReference len);

        // BIP-32
        int ufsecp_bip32_master(Pointer ctx, byte[] seed, long seedLen, byte[] key82);
        int ufsecp_bip32_derive_path(Pointer ctx, byte[] master82, String path, byte[] key82);
        int ufsecp_bip32_privkey(Pointer ctx, byte[] key82, byte[] priv32);
        int ufsecp_bip32_pubkey(Pointer ctx, byte[] key82, byte[] pub33);

        // Taproot
        int ufsecp_taproot_output_key(Pointer ctx, byte[] intX32, Pointer mr, byte[] outX32, IntByReference parity);
        int ufsecp_taproot_verify(Pointer ctx, byte[] outX32, int parity, byte[] intX32, Pointer mr, long mrLen);

        // Pedersen
        int ufsecp_pedersen_commit(Pointer ctx, byte[] value32, byte[] blinding32, byte[] commit33);
        int ufsecp_pedersen_verify(Pointer ctx, byte[] commit33, byte[] value32, byte[] blinding32);

        // GPU
        int ufsecp_gpu_backend_count(int[] ids, int max);
        String ufsecp_gpu_backend_name(int bid);
        int ufsecp_gpu_is_available(int bid);
        int ufsecp_gpu_device_count(int bid);
        int ufsecp_gpu_ctx_create(PointerByReference ctx, int bid, int dev);
        void ufsecp_gpu_ctx_destroy(Pointer ctx);
        int ufsecp_gpu_generator_mul_batch(Pointer ctx, byte[] s32, long n, byte[] pk33);
        int ufsecp_gpu_ecdsa_verify_batch(Pointer ctx, byte[] msg, byte[] pk, byte[] sig, long n, byte[] res);
        int ufsecp_gpu_hash160_pubkey_batch(Pointer ctx, byte[] pk33, long n, byte[] h20);
        int ufsecp_gpu_msm(Pointer ctx, byte[] s32, byte[] p33, long n, byte[] out33);
        String ufsecp_gpu_error_str(int code);
    }

    // ── Helpers ───────────────────────────────────────────────────────

    static final Ufsecp lib = Ufsecp.INSTANCE;
    static final HexFormat HEX = HexFormat.of();

    static void check(int rc, String op) {
        if (rc != 0) {
            System.err.printf("[FAIL] %s: %s (code %d)%n", op, lib.ufsecp_error_str(rc), rc);
            System.exit(1);
        }
    }

    static String hexs(byte[] data) { return HEX.formatHex(data); }
    static String hexs(byte[] data, int off, int len) {
        return HEX.formatHex(Arrays.copyOfRange(data, off, off + len));
    }

    static String getAddr(AddrFn fn, Pointer ctx, byte[] key, int net) {
        byte[] buf = new byte[128];
        NativeLongByReference len = new NativeLongByReference(new NativeLong(128));
        check(fn.call(ctx, key, net, buf, len), "addr");
        return new String(buf, 0, (int) len.getValue().longValue());
    }

    @FunctionalInterface
    interface AddrFn {
        int call(Pointer ctx, byte[] key, int net, byte[] buf, NativeLongByReference len);
    }

    // ── Test Keys ────────────────────────────────────────────────────

    static byte[] privkey() {
        byte[] k = new byte[32];
        k[31] = 1;
        return k;
    }

    static byte[] privkey2() {
        byte[] k = new byte[32];
        k[31] = 2;
        return k;
    }

    // ── CPU Demo ─────────────────────────────────────────────────────

    static void demoCPU(Pointer ctx) {
        System.out.println("=== CPU Operations ===\n");
        byte[] sk = privkey(), sk2 = privkey2();

        // 1. Key Generation
        System.out.println("[1] Key Generation");
        byte[] pub33 = new byte[33], pub65 = new byte[65], xonly = new byte[32];
        check(lib.ufsecp_pubkey_create(ctx, sk, pub33), "pubkey_create");
        check(lib.ufsecp_pubkey_create_uncompressed(ctx, sk, pub65), "pubkey_uncompressed");
        check(lib.ufsecp_pubkey_xonly(ctx, sk, xonly), "pubkey_xonly");
        System.out.printf("  %-20s %s%n", "Private key:", hexs(sk));
        System.out.printf("  %-20s %s%n", "Compressed (33B):", hexs(pub33));
        System.out.printf("  %-20s %s%n", "Uncompressed (65B):", hexs(pub65));
        System.out.printf("  %-20s %s%n", "X-only (32B):", hexs(xonly));
        System.out.println();

        // 2. ECDSA
        System.out.println("[2] ECDSA Sign / Verify (RFC 6979)");
        byte[] msg = new byte[32];
        check(lib.ufsecp_sha256("Hello UltrafastSecp256k1!".getBytes(), 24, msg), "sha256");
        System.out.printf("  %-20s %s%n", "Message hash:", hexs(msg));

        byte[] sig = new byte[64];
        check(lib.ufsecp_ecdsa_sign(ctx, msg, sk, sig), "ecdsa_sign");
        System.out.printf("  %-20s %s%n", "ECDSA signature:", hexs(sig));

        int vrc = lib.ufsecp_ecdsa_verify(ctx, msg, sig, pub33);
        System.out.printf("  %-20s %s%n", "Verify:", vrc == 0 ? "VALID" : "INVALID");

        // DER
        byte[] der = new byte[72];
        NativeLongByReference derLen = new NativeLongByReference(new NativeLong(72));
        check(lib.ufsecp_ecdsa_sig_to_der(ctx, sig, der, derLen), "sig_to_der");
        System.out.printf("  %-20s %d bytes%n", "DER length:", derLen.getValue().longValue());

        // Recovery
        IntByReference recid = new IntByReference();
        byte[] rsig = new byte[64], recovered = new byte[33];
        check(lib.ufsecp_ecdsa_sign_recoverable(ctx, msg, sk, rsig, recid), "sign_recoverable");
        check(lib.ufsecp_ecdsa_recover(ctx, msg, rsig, recid.getValue(), recovered), "recover");
        System.out.printf("  %-20s recid=%d, match=%s%n", "Recovery:",
                recid.getValue(), Arrays.equals(recovered, pub33) ? "YES" : "NO");
        System.out.println();

        // 3. Schnorr
        System.out.println("[3] Schnorr Sign / Verify (BIP-340)");
        byte[] aux = new byte[32], schnorrSig = new byte[64];
        check(lib.ufsecp_schnorr_sign(ctx, msg, sk, aux, schnorrSig), "schnorr_sign");
        System.out.printf("  %-20s %s%n", "Schnorr signature:", hexs(schnorrSig));
        vrc = lib.ufsecp_schnorr_verify(ctx, msg, schnorrSig, xonly);
        System.out.printf("  %-20s %s%n", "Verify:", vrc == 0 ? "VALID" : "INVALID");
        System.out.println();

        // 4. ECDH
        System.out.println("[4] ECDH Key Agreement");
        byte[] pub2 = new byte[33];
        check(lib.ufsecp_pubkey_create(ctx, sk2, pub2), "pubkey2");
        byte[] secretA = new byte[32], secretB = new byte[32];
        check(lib.ufsecp_ecdh(ctx, sk, pub2, secretA), "ecdh_a");
        check(lib.ufsecp_ecdh(ctx, sk2, pub33, secretB), "ecdh_b");
        System.out.printf("  %-20s %s%n", "Secret (A->B):", hexs(secretA));
        System.out.printf("  %-20s %s%n", "Secret (B->A):", hexs(secretB));
        System.out.printf("  %-20s %s%n", "Match:", Arrays.equals(secretA, secretB) ? "YES" : "NO");
        System.out.println();

        // 5. Hashing
        System.out.println("[5] Hashing");
        byte[] sha = new byte[32], h160 = new byte[20];
        check(lib.ufsecp_sha256(pub33, 33, sha), "sha256_pub");
        check(lib.ufsecp_hash160(pub33, 33, h160), "hash160_pub");
        System.out.printf("  %-20s %s%n", "SHA-256(pubkey):", hexs(sha));
        System.out.printf("  %-20s %s%n", "Hash160(pubkey):", hexs(h160));
        System.out.println();

        // 6. Bitcoin Addresses
        System.out.println("[6] Bitcoin Addresses");
        System.out.printf("  %-20s %s%n", "P2PKH:", getAddr(lib::ufsecp_addr_p2pkh, ctx, pub33, 0));
        System.out.printf("  %-20s %s%n", "P2WPKH:", getAddr(lib::ufsecp_addr_p2wpkh, ctx, pub33, 0));
        System.out.printf("  %-20s %s%n", "P2TR:", getAddr(lib::ufsecp_addr_p2tr, ctx, xonly, 0));
        System.out.println();

        // 7. WIF
        System.out.println("[7] WIF Encoding");
        byte[] wifBuf = new byte[128];
        NativeLongByReference wifLen = new NativeLongByReference(new NativeLong(128));
        check(lib.ufsecp_wif_encode(ctx, sk, 1, 0, wifBuf, wifLen), "wif_encode");
        System.out.printf("  %-20s %s%n", "WIF:", new String(wifBuf, 0, (int) wifLen.getValue().longValue()));
        System.out.println();

        // 8. BIP-32
        System.out.println("[8] BIP-32 HD Key Derivation");
        byte[] seed = new byte[64];
        Arrays.fill(seed, (byte) 0x42);
        byte[] master = new byte[82];
        check(lib.ufsecp_bip32_master(ctx, seed, 64, master), "bip32_master");
        byte[] childKey = new byte[82];
        check(lib.ufsecp_bip32_derive_path(ctx, master, "m/44'/0'/0'/0/0", childKey), "bip32_path");
        byte[] childPriv = new byte[32], childPub = new byte[33];
        check(lib.ufsecp_bip32_privkey(ctx, childKey, childPriv), "bip32_privkey");
        check(lib.ufsecp_bip32_pubkey(ctx, childKey, childPub), "bip32_pubkey");
        System.out.printf("  %-20s %s%n", "BIP-32 child priv:", hexs(childPriv));
        System.out.printf("  %-20s %s%n", "BIP-32 child pub:", hexs(childPub));
        System.out.println();

        // 9. Taproot
        System.out.println("[9] Taproot (BIP-341)");
        byte[] outputX = new byte[32];
        IntByReference parity = new IntByReference();
        check(lib.ufsecp_taproot_output_key(ctx, xonly, null, outputX, parity), "taproot_output_key");
        System.out.printf("  %-20s %s%n", "Output key:", hexs(outputX));
        System.out.printf("  %-20s %d%n", "Parity:", parity.getValue());
        vrc = lib.ufsecp_taproot_verify(ctx, outputX, parity.getValue(), xonly, null, 0);
        System.out.printf("  %-20s %s%n", "Verify:", vrc == 0 ? "VALID" : "INVALID");
        System.out.println();

        // 10. Pedersen Commitment
        System.out.println("[10] Pedersen Commitment");
        byte[] pedValue = new byte[32]; pedValue[31] = 42;
        byte[] pedBlinding = new byte[32]; pedBlinding[31] = 7;
        byte[] pedCommit = new byte[33];
        check(lib.ufsecp_pedersen_commit(ctx, pedValue, pedBlinding, pedCommit), "pedersen_commit");
        System.out.printf("  %-20s %s%n", "Commitment:", hexs(pedCommit));
        vrc = lib.ufsecp_pedersen_verify(ctx, pedCommit, pedValue, pedBlinding);
        System.out.printf("  %-20s %s%n", "Verify:", vrc == 0 ? "VALID" : "INVALID");
        System.out.println();
    }

    // ── GPU Demo ─────────────────────────────────────────────────────

    static void demoGPU(Pointer cpuCtx) {
        System.out.println("=== GPU Operations ===\n");

        // 10. Backend Discovery
        System.out.println("[10] GPU Backend Discovery");
        int[] bids = new int[4];
        int nBackends = lib.ufsecp_gpu_backend_count(bids, 4);
        System.out.printf("  %-20s %d%n", "Backends compiled:", nBackends);

        int useBackend = 0;
        for (int i = 0; i < nBackends; i++) {
            int bid = bids[i];
            String name = lib.ufsecp_gpu_backend_name(bid);
            int avail = lib.ufsecp_gpu_is_available(bid);
            int devs = lib.ufsecp_gpu_device_count(bid);
            System.out.printf("  Backend %d: %-8s available=%d devices=%d%n", bid, name, avail, devs);
            if (avail != 0 && useBackend == 0) useBackend = bid;
        }

        if (useBackend == 0) {
            System.out.println("  No GPU backends available -- skipping GPU demos.\n");
            return;
        }

        // Create GPU context
        PointerByReference gpuRef = new PointerByReference();
        int rc = lib.ufsecp_gpu_ctx_create(gpuRef, useBackend, 0);
        if (rc != 0) {
            System.out.printf("  GPU context creation failed: %s%n", lib.ufsecp_gpu_error_str(rc));
            return;
        }
        Pointer gpu = gpuRef.getValue();

        int N = 4;

        // 11. Batch Key Generation
        System.out.println("\n[11] GPU Batch Key Generation (4 keys)");
        byte[] scalars = new byte[N * 32];
        for (int i = 0; i < N; i++) scalars[i * 32 + 31] = (byte)(i + 1);

        byte[] pubkeys = new byte[N * 33];
        rc = lib.ufsecp_gpu_generator_mul_batch(gpu, scalars, N, pubkeys);
        if (rc == 0) {
            for (int i = 0; i < N; i++)
                System.out.printf("  GPU pubkey[%d]:      %s%n", i, hexs(pubkeys, i*33, 33));
        } else {
            System.out.printf("  gpu_generator_mul_batch: %s%n", lib.ufsecp_gpu_error_str(rc));
        }

        // 12. ECDSA Batch Verify
        System.out.println("\n[12] GPU ECDSA Batch Verify");
        byte[] msgs = new byte[N * 32], sigs = new byte[N * 64], pubs = new byte[N * 33];
        for (int i = 0; i < N; i++) {
            byte[] msgHash = new byte[32];
            lib.ufsecp_sha256(new byte[]{(byte)i}, 1, msgHash);
            System.arraycopy(msgHash, 0, msgs, i * 32, 32);

            byte[] sk = new byte[32];
            sk[31] = (byte)(i + 1);
            byte[] sig = new byte[64];
            lib.ufsecp_ecdsa_sign(cpuCtx, msgHash, sk, sig);
            System.arraycopy(sig, 0, sigs, i * 64, 64);
            System.arraycopy(pubkeys, i * 33, pubs, i * 33, 33);
        }

        byte[] results = new byte[N];
        rc = lib.ufsecp_gpu_ecdsa_verify_batch(gpu, msgs, pubs, sigs, N, results);
        if (rc == 0) {
            StringBuilder sb = new StringBuilder("  Results: ");
            for (int i = 0; i < N; i++) sb.append(String.format("[%d]=%s ", i, results[i] != 0 ? "VALID" : "INVALID"));
            System.out.println(sb);
        } else {
            System.out.printf("  gpu_ecdsa_verify_batch: %s%n", lib.ufsecp_gpu_error_str(rc));
        }

        // 13. Hash160 Batch
        System.out.println("\n[13] GPU Hash160 Batch");
        byte[] hashes = new byte[N * 20];
        rc = lib.ufsecp_gpu_hash160_pubkey_batch(gpu, pubkeys, N, hashes);
        if (rc == 0) {
            for (int i = 0; i < N; i++)
                System.out.printf("  Hash160[%d]:         %s%n", i, hexs(hashes, i*20, 20));
        } else {
            System.out.printf("  gpu_hash160_pubkey_batch: %s%n", lib.ufsecp_gpu_error_str(rc));
        }

        // 14. MSM
        System.out.println("\n[14] GPU Multi-Scalar Multiplication");
        byte[] msmResult = new byte[33];
        rc = lib.ufsecp_gpu_msm(gpu, scalars, pubkeys, N, msmResult);
        if (rc == 0) {
            System.out.printf("  MSM result:         %s%n", hexs(msmResult));
        } else {
            System.out.printf("  gpu_msm: %s%n", lib.ufsecp_gpu_error_str(rc));
        }

        lib.ufsecp_gpu_ctx_destroy(gpu);
        System.out.println();
    }

    // ── Main ─────────────────────────────────────────────────────────

    public static void main(String[] args) {
        System.out.println("UltrafastSecp256k1 -- Java Example");
        System.out.printf("ABI version: %d%n", lib.ufsecp_abi_version());
        System.out.printf("Library:     %s%n%n", lib.ufsecp_version_string());

        PointerByReference ctxRef = new PointerByReference();
        check(lib.ufsecp_ctx_create(ctxRef), "ctx_create");
        Pointer ctx = ctxRef.getValue();

        demoCPU(ctx);
        demoGPU(ctx);

        lib.ufsecp_ctx_destroy(ctx);
        System.out.println("All examples completed successfully.");
    }
}
