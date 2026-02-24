/**
 * UltrafastSecp256k1 — Java binding (ufsecp stable C ABI v1).
 *
 * High-performance secp256k1 ECC with dual-layer constant-time architecture.
 * Context-based API backed by JNI native bridge.
 *
 * Usage:
 *   try (Ufsecp ctx = Ufsecp.create()) {
 *       byte[] pub = ctx.pubkeyCreate(privkey);
 *   }
 */
package com.ultrafast.ufsecp;

public final class Ufsecp implements AutoCloseable {

    static {
        System.loadLibrary("ufsecp_jni");
    }

    // ── Error codes ──────────────────────────────────────────────────

    public static final int NET_MAINNET = 0;
    public static final int NET_TESTNET = 1;

    // ── Instance ─────────────────────────────────────────────────────

    private long ptr;

    private Ufsecp(long ptr) {
        this.ptr = ptr;
    }

    public static Ufsecp create() {
        long p = nativeCreate();
        return new Ufsecp(p);
    }

    @Override
    public void close() {
        if (ptr != 0) {
            nativeDestroy(ptr);
            ptr = 0;
        }
    }

    private void alive() {
        if (ptr == 0) throw new IllegalStateException("UfsecpContext already destroyed");
    }

    // ── Version ──────────────────────────────────────────────────────

    public static int version()         { return nativeVersion(); }
    public static int abiVersion()      { return nativeAbiVersion(); }
    public static String versionString(){ return nativeVersionString(); }

    // ── Context extras ───────────────────────────────────────────────

    public Ufsecp clone_() {
        alive();
        return new Ufsecp(nativeClone(ptr));
    }

    public int lastError() {
        alive();
        return nativeLastError(ptr);
    }

    public String lastErrorMsg() {
        alive();
        return nativeLastErrorMsg(ptr);
    }

    // ── Key operations ───────────────────────────────────────────────

    public byte[] pubkeyCreate(byte[] privkey) {
        alive();
        return nativePubkeyCreate(ptr, privkey);
    }

    public byte[] pubkeyCreateUncompressed(byte[] privkey) {
        alive();
        return nativePubkeyCreateUncompressed(ptr, privkey);
    }

    public byte[] pubkeyParse(byte[] pubkey) {
        alive();
        return nativePubkeyParse(ptr, pubkey);
    }

    public byte[] pubkeyXonly(byte[] privkey) {
        alive();
        return nativePubkeyXonly(ptr, privkey);
    }

    public boolean seckeyVerify(byte[] privkey) {
        alive();
        return nativeSeckeyVerify(ptr, privkey);
    }

    public byte[] seckeyNegate(byte[] privkey) {
        alive();
        return nativeSeckeyNegate(ptr, privkey);
    }

    public byte[] seckeyTweakAdd(byte[] privkey, byte[] tweak) {
        alive();
        return nativeSeckeyTweakAdd(ptr, privkey, tweak);
    }

    public byte[] seckeyTweakMul(byte[] privkey, byte[] tweak) {
        alive();
        return nativeSeckeyTweakMul(ptr, privkey, tweak);
    }

    // ── ECDSA ────────────────────────────────────────────────────────

    public byte[] ecdsaSign(byte[] msgHash, byte[] privkey) {
        alive();
        return nativeEcdsaSign(ptr, msgHash, privkey);
    }

    public boolean ecdsaVerify(byte[] msgHash, byte[] sig, byte[] pubkey) {
        alive();
        return nativeEcdsaVerify(ptr, msgHash, sig, pubkey);
    }

    public byte[] ecdsaSigToDer(byte[] sig) {
        alive();
        return nativeEcdsaSigToDer(ptr, sig);
    }

    public byte[] ecdsaSigFromDer(byte[] der) {
        alive();
        return nativeEcdsaSigFromDer(ptr, der);
    }

    public RecoverableSignature ecdsaSignRecoverable(byte[] msgHash, byte[] privkey) {
        alive();
        return nativeEcdsaSignRecoverable(ptr, msgHash, privkey);
    }

    public byte[] ecdsaRecover(byte[] msgHash, byte[] sig, int recid) {
        alive();
        return nativeEcdsaRecover(ptr, msgHash, sig, recid);
    }

    // ── Schnorr ──────────────────────────────────────────────────────

    public byte[] schnorrSign(byte[] msg, byte[] privkey, byte[] auxRand) {
        alive();
        return nativeSchnorrSign(ptr, msg, privkey, auxRand);
    }

    public boolean schnorrVerify(byte[] msg, byte[] sig, byte[] pubkeyX) {
        alive();
        return nativeSchnorrVerify(ptr, msg, sig, pubkeyX);
    }

    // ── ECDH ─────────────────────────────────────────────────────────

    public byte[] ecdh(byte[] privkey, byte[] pubkey) {
        alive();
        return nativeEcdh(ptr, privkey, pubkey);
    }

    public byte[] ecdhXonly(byte[] privkey, byte[] pubkey) {
        alive();
        return nativeEcdhXonly(ptr, privkey, pubkey);
    }

    public byte[] ecdhRaw(byte[] privkey, byte[] pubkey) {
        alive();
        return nativeEcdhRaw(ptr, privkey, pubkey);
    }

    // ── Hashing ──────────────────────────────────────────────────────

    public static byte[] sha256(byte[] data)  { return nativeSha256(data); }
    public static byte[] hash160(byte[] data) { return nativeHash160(data); }
    public static byte[] taggedHash(byte[] tag, byte[] data) { return nativeTaggedHash(tag, data); }

    // ── Addresses ────────────────────────────────────────────────────

    public String addrP2pkh(byte[] pubkey, int network) {
        alive();
        return nativeAddrP2pkh(ptr, pubkey, network);
    }

    public String addrP2wpkh(byte[] pubkey, int network) {
        alive();
        return nativeAddrP2wpkh(ptr, pubkey, network);
    }

    public String addrP2tr(byte[] xonly, int network) {
        alive();
        return nativeAddrP2tr(ptr, xonly, network);
    }

    // ── WIF ──────────────────────────────────────────────────────────

    public String wifEncode(byte[] privkey, boolean compressed, int network) {
        alive();
        return nativeWifEncode(ptr, privkey, compressed, network);
    }

    public WifDecoded wifDecode(String wif) {
        alive();
        return nativeWifDecode(ptr, wif);
    }

    // ── BIP-32 ───────────────────────────────────────────────────────

    public byte[] bip32Master(byte[] seed) {
        alive();
        return nativeBip32Master(ptr, seed);
    }

    public byte[] bip32Derive(byte[] parent, int index) {
        alive();
        return nativeBip32Derive(ptr, parent, index);
    }

    public byte[] bip32DerivePath(byte[] master, String path) {
        alive();
        return nativeBip32DerivePath(ptr, master, path);
    }

    public byte[] bip32Privkey(byte[] key) {
        alive();
        return nativeBip32Privkey(ptr, key);
    }

    public byte[] bip32Pubkey(byte[] key) {
        alive();
        return nativeBip32Pubkey(ptr, key);
    }

    // ── Taproot ──────────────────────────────────────────────────────

    public TaprootOutputKeyResult taprootOutputKey(byte[] internalX, byte[] merkleRoot) {
        alive();
        return nativeTaprootOutputKey(ptr, internalX, merkleRoot);
    }

    public byte[] taprootTweakSeckey(byte[] privkey, byte[] merkleRoot) {
        alive();
        return nativeTaprootTweakSeckey(ptr, privkey, merkleRoot);
    }

    public boolean taprootVerify(byte[] outputX, int parity, byte[] internalX, byte[] merkleRoot) {
        alive();
        return nativeTaprootVerify(ptr, outputX, parity, internalX, merkleRoot);
    }

    // ── Native declarations ──────────────────────────────────────────

    private static native long nativeCreate();
    private static native void nativeDestroy(long ptr);
    private static native long nativeClone(long ptr);
    private static native int  nativeLastError(long ptr);
    private static native String nativeLastErrorMsg(long ptr);

    private static native int nativeVersion();
    private static native int nativeAbiVersion();
    private static native String nativeVersionString();

    private static native byte[] nativePubkeyCreate(long ctx, byte[] privkey);
    private static native byte[] nativePubkeyCreateUncompressed(long ctx, byte[] privkey);
    private static native byte[] nativePubkeyParse(long ctx, byte[] pubkey);
    private static native byte[] nativePubkeyXonly(long ctx, byte[] privkey);
    private static native boolean nativeSeckeyVerify(long ctx, byte[] privkey);
    private static native byte[] nativeSeckeyNegate(long ctx, byte[] privkey);
    private static native byte[] nativeSeckeyTweakAdd(long ctx, byte[] privkey, byte[] tweak);
    private static native byte[] nativeSeckeyTweakMul(long ctx, byte[] privkey, byte[] tweak);

    private static native byte[] nativeEcdsaSign(long ctx, byte[] msgHash, byte[] privkey);
    private static native boolean nativeEcdsaVerify(long ctx, byte[] msgHash, byte[] sig, byte[] pubkey);
    private static native byte[] nativeEcdsaSigToDer(long ctx, byte[] sig);
    private static native byte[] nativeEcdsaSigFromDer(long ctx, byte[] der);
    private static native RecoverableSignature nativeEcdsaSignRecoverable(long ctx, byte[] msgHash, byte[] privkey);
    private static native byte[] nativeEcdsaRecover(long ctx, byte[] msgHash, byte[] sig, int recid);

    private static native byte[] nativeSchnorrSign(long ctx, byte[] msg, byte[] privkey, byte[] auxRand);
    private static native boolean nativeSchnorrVerify(long ctx, byte[] msg, byte[] sig, byte[] pubkeyX);

    private static native byte[] nativeEcdh(long ctx, byte[] privkey, byte[] pubkey);
    private static native byte[] nativeEcdhXonly(long ctx, byte[] privkey, byte[] pubkey);
    private static native byte[] nativeEcdhRaw(long ctx, byte[] privkey, byte[] pubkey);

    private static native byte[] nativeSha256(byte[] data);
    private static native byte[] nativeHash160(byte[] data);
    private static native byte[] nativeTaggedHash(byte[] tag, byte[] data);

    private static native String nativeAddrP2pkh(long ctx, byte[] pubkey, int network);
    private static native String nativeAddrP2wpkh(long ctx, byte[] pubkey, int network);
    private static native String nativeAddrP2tr(long ctx, byte[] xonly, int network);

    private static native String nativeWifEncode(long ctx, byte[] privkey, boolean compressed, int network);
    private static native WifDecoded nativeWifDecode(long ctx, String wif);

    private static native byte[] nativeBip32Master(long ctx, byte[] seed);
    private static native byte[] nativeBip32Derive(long ctx, byte[] parent, int index);
    private static native byte[] nativeBip32DerivePath(long ctx, byte[] master, String path);
    private static native byte[] nativeBip32Privkey(long ctx, byte[] key);
    private static native byte[] nativeBip32Pubkey(long ctx, byte[] key);

    private static native TaprootOutputKeyResult nativeTaprootOutputKey(long ctx, byte[] internalX, byte[] merkleRoot);
    private static native byte[] nativeTaprootTweakSeckey(long ctx, byte[] privkey, byte[] merkleRoot);
    private static native boolean nativeTaprootVerify(long ctx, byte[] outputX, int parity, byte[] internalX, byte[] merkleRoot);
}
