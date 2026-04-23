/* UltrafastSecp256k1 — JNI bridge (ufsecp stable C ABI v1).
 *
 * Each native method maps 1:1 to a ufsecp_* C function.
 * The Java class holds the opaque context pointer as a long.
 */

#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include "ufsecp.h"

static void secure_zero(void *ptr, size_t len) {
    volatile uint8_t *p = (volatile uint8_t*)ptr;
    while (len--) {
        *p++ = 0;
    }
}

/* Helper: throw Java exception on non-zero return code. Returns 0 on success. */
static int throw_on_err(JNIEnv *env, int rc, const char *op) {
    if (rc == 0) return 0;
    char msg[128];
    snprintf(msg, sizeof(msg), "ufsecp %s failed: %s (%d)", op, ufsecp_error_str(rc), rc);
    jclass cls = (*env)->FindClass(env, "com/ultrafast/ufsecp/UfsecpException");
    if (cls) (*env)->ThrowNew(env, cls, msg);
    return rc;
}

/* Helper: pin byte array, copy into fixed buf, unpin. Returns pinned pointer (must release). */
static jbyte* pin(JNIEnv *env, jbyteArray arr) {
    return (*env)->GetByteArrayElements(env, arr, NULL);
}
static void unpin(JNIEnv *env, jbyteArray arr, jbyte *ptr) {
    (*env)->ReleaseByteArrayElements(env, arr, ptr, JNI_ABORT);
}
static jbyteArray mk(JNIEnv *env, const uint8_t *data, int len) {
    jbyteArray r = (*env)->NewByteArray(env, len);
    if (r) (*env)->SetByteArrayRegion(env, r, 0, len, (const jbyte*)data);
    return r;
}
static jstring mkstr_copy(JNIEnv *env, const char *text) {
    size_t len = strlen(text);
    char *copy = (char*)malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, text, len + 1);
    jstring out = (*env)->NewStringUTF(env, copy);
    secure_zero(copy, len + 1);
    free(copy);
    return out;
}

/* ── Context ─────────────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeCreate(JNIEnv *env, jclass clz) {
    (void)clz;
    ufsecp_ctx *ctx = NULL;
    int rc = ufsecp_ctx_create(&ctx);
    if (throw_on_err(env, rc, "ctx_create")) return 0;
    return (jlong)(uintptr_t)ctx;
}

JNIEXPORT void JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeDestroy(JNIEnv *env, jclass clz, jlong ptr) {
    (void)env; (void)clz;
    if (ptr) ufsecp_ctx_destroy((ufsecp_ctx*)(uintptr_t)ptr);
}

/* ── Key ops ─────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativePubkeyCreate(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t out[33];
    int rc = ufsecp_pubkey_create((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, out);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "pubkey_create")) return NULL;
    return mk(env, out, 33);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativePubkeyCreateUncompressed(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t out[65];
    int rc = ufsecp_pubkey_create_uncompressed((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, out);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "pubkey_create_uncompressed")) return NULL;
    return mk(env, out, 65);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSeckeyVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    int rc = ufsecp_seckey_verify((const ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk);
    unpin(env, privkey, pk);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ── ECDSA ───────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaSign(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msgHash, jbyteArray privkey) {
    (void)clz;
    jbyte *msg = pin(env, msgHash);
    jbyte *pk  = pin(env, privkey);
    uint8_t sig[64];
    int rc = ufsecp_ecdsa_sign((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)msg, (const uint8_t*)pk, sig);
    unpin(env, privkey, pk);
    unpin(env, msgHash, msg);
    if (throw_on_err(env, rc, "ecdsa_sign")) return NULL;
    return mk(env, sig, 64);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msgHash, jbyteArray sig, jbyteArray pubkey) {
    (void)clz;
    jbyte *msg = pin(env, msgHash);
    jbyte *s   = pin(env, sig);
    jbyte *pk  = pin(env, pubkey);
    int rc = ufsecp_ecdsa_verify((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)msg, (const uint8_t*)s, (const uint8_t*)pk);
    unpin(env, pubkey, pk);
    unpin(env, sig, s);
    unpin(env, msgHash, msg);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ── Schnorr ─────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSchnorrSign(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msg, jbyteArray privkey, jbyteArray auxRand) {
    (void)clz;
    jbyte *m  = pin(env, msg);
    jbyte *pk = pin(env, privkey);
    jbyte *ar = pin(env, auxRand);
    uint8_t sig[64];
    int rc = ufsecp_schnorr_sign((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)m, (const uint8_t*)pk, (const uint8_t*)ar, sig);
    unpin(env, auxRand, ar);
    unpin(env, privkey, pk);
    unpin(env, msg, m);
    if (throw_on_err(env, rc, "schnorr_sign")) return NULL;
    return mk(env, sig, 64);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSchnorrVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msg, jbyteArray sig, jbyteArray pubkeyX) {
    (void)clz;
    jbyte *m  = pin(env, msg);
    jbyte *s  = pin(env, sig);
    jbyte *pk = pin(env, pubkeyX);
    int rc = ufsecp_schnorr_verify((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)m, (const uint8_t*)s, (const uint8_t*)pk);
    unpin(env, pubkeyX, pk);
    unpin(env, sig, s);
    unpin(env, msg, m);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ── ECDH ────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdh(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray pubkey) {
    (void)clz;
    jbyte *pk  = pin(env, privkey);
    jbyte *pub = pin(env, pubkey);
    uint8_t out[32];
    int rc = ufsecp_ecdh((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (const uint8_t*)pub, out);
    unpin(env, pubkey, pub);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "ecdh")) return NULL;
    return mk(env, out, 32);
}

/* ── Hashing ─────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSha256(
    JNIEnv *env, jclass clz, jbyteArray data) {
    (void)clz;
    jbyte *d = pin(env, data);
    jsize len = (*env)->GetArrayLength(env, data);
    uint8_t out[32];
    int rc = ufsecp_sha256((const uint8_t*)d, (size_t)len, out);
    unpin(env, data, d);
    if (throw_on_err(env, rc, "sha256")) return NULL;
    return mk(env, out, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeHash160(
    JNIEnv *env, jclass clz, jbyteArray data) {
    (void)clz;
    jbyte *d = pin(env, data);
    jsize len = (*env)->GetArrayLength(env, data);
    uint8_t out[20];
    int rc = ufsecp_hash160((const uint8_t*)d, (size_t)len, out);
    unpin(env, data, d);
    if (throw_on_err(env, rc, "hash160")) return NULL;
    return mk(env, out, 20);
}

/* ── Addresses ───────────────────────────────────────────────────────── */

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAddrP2pkh(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray pubkey, jint network) {
    (void)clz;
    jbyte *pk = pin(env, pubkey);
    char addr[128];
    size_t alen = 128;
    int rc = ufsecp_addr_p2pkh((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (int)network, addr, &alen);
    unpin(env, pubkey, pk);
    if (throw_on_err(env, rc, "addr_p2pkh")) return NULL;
    addr[alen] = '\0';
    return mkstr_copy(env, (const char*)addr);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAddrP2wpkh(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray pubkey, jint network) {
    (void)clz;
    jbyte *pk = pin(env, pubkey);
    char addr[128];
    size_t alen = 128;
    int rc = ufsecp_addr_p2wpkh((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (int)network, addr, &alen);
    unpin(env, pubkey, pk);
    if (throw_on_err(env, rc, "addr_p2wpkh")) return NULL;
    addr[alen] = '\0';
    return mkstr_copy(env, (const char*)addr);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAddrP2tr(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray xonly, jint network) {
    (void)clz;
    jbyte *pk = pin(env, xonly);
    char addr[128];
    size_t alen = 128;
    int rc = ufsecp_addr_p2tr((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (int)network, addr, &alen);
    unpin(env, xonly, pk);
    if (throw_on_err(env, rc, "addr_p2tr")) return NULL;
    addr[alen] = '\0';
    return mkstr_copy(env, (const char*)addr);
}

/* ── WIF ─────────────────────────────────────────────────────────────── */

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeWifEncode(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jboolean compressed, jint network) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    char wif[128];
    size_t wlen = 128;
    int rc = ufsecp_wif_encode((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk,
                               compressed ? 1 : 0, (int)network, wif, &wlen);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "wif_encode")) {
        secure_zero(wif, sizeof(wif));
        return NULL;
    }
    wif[wlen] = '\0';
    jstring out = mkstr_copy(env, (const char*)wif);
    secure_zero(wif, sizeof(wif));
    return out;
}

/* ── BIP-32 ──────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32Master(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray seed) {
    (void)clz;
    jbyte *s = pin(env, seed);
    jsize slen = (*env)->GetArrayLength(env, seed);
    ufsecp_bip32_key key;
    int rc = ufsecp_bip32_master((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)s, (size_t)slen, &key);
    unpin(env, seed, s);
    if (throw_on_err(env, rc, "bip32_master")) {
        secure_zero(&key, sizeof(key));
        return NULL;
    }
    jbyteArray out = mk(env, (const uint8_t*)&key, (int)sizeof(key));
    secure_zero(&key, sizeof(key));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32Derive(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray parent, jint index) {
    (void)clz;
    jbyte *p = pin(env, parent);
    ufsecp_bip32_key parent_key;
    ufsecp_bip32_key child;
    memcpy(&parent_key, p, sizeof(parent_key));
    int rc = ufsecp_bip32_derive((ufsecp_ctx*)(uintptr_t)ctx, &parent_key, (uint32_t)index, &child);
    unpin(env, parent, p);
    if (throw_on_err(env, rc, "bip32_derive")) {
        secure_zero(&parent_key, sizeof(parent_key));
        secure_zero(&child, sizeof(child));
        return NULL;
    }
    jbyteArray out = mk(env, (const uint8_t*)&child, (int)sizeof(child));
    secure_zero(&parent_key, sizeof(parent_key));
    secure_zero(&child, sizeof(child));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32DerivePath(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray master, jstring path) {
    (void)clz;
    jbyte *m = pin(env, master);
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    if (!p) {
        unpin(env, master, m);
        jclass cls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
        if (cls) (*env)->ThrowNew(env, cls, "GetStringUTFChars OOM");
        return NULL;
    }
    ufsecp_bip32_key master_key;
    ufsecp_bip32_key key;
    memcpy(&master_key, m, sizeof(master_key));
    int rc = ufsecp_bip32_derive_path((ufsecp_ctx*)(uintptr_t)ctx, &master_key, p, &key);
    (*env)->ReleaseStringUTFChars(env, path, p);
    unpin(env, master, m);
    if (throw_on_err(env, rc, "bip32_derive_path")) {
        secure_zero(&master_key, sizeof(master_key));
        secure_zero(&key, sizeof(key));
        return NULL;
    }
    jbyteArray out = mk(env, (const uint8_t*)&key, (int)sizeof(key));
    secure_zero(&master_key, sizeof(master_key));
    secure_zero(&key, sizeof(key));
    return out;
}

/* ── Version ─────────────────────────────────────────────────────────── */

JNIEXPORT jint JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeVersion(JNIEnv *env, jclass clz) {
    (void)env; (void)clz;
    return (jint)ufsecp_version();
}

JNIEXPORT jint JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAbiVersion(JNIEnv *env, jclass clz) {
    (void)env; (void)clz;
    return (jint)ufsecp_abi_version();
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeVersionString(JNIEnv *env, jclass clz) {
    (void)clz;
    return (*env)->NewStringUTF(env, ufsecp_version_string());
}

/* ── Context extras ──────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeClone(JNIEnv *env, jclass clz, jlong ptr) {
    (void)clz;
    ufsecp_ctx *clone = NULL;
    int rc = ufsecp_ctx_clone((const ufsecp_ctx*)(uintptr_t)ptr, &clone);
    if (throw_on_err(env, rc, "ctx_clone")) return 0;
    return (jlong)(uintptr_t)clone;
}

JNIEXPORT jint JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeLastError(JNIEnv *env, jclass clz, jlong ptr) {
    (void)env; (void)clz;
    return (jint)ufsecp_last_error((const ufsecp_ctx*)(uintptr_t)ptr);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeLastErrorMsg(JNIEnv *env, jclass clz, jlong ptr) {
    (void)clz;
    return (*env)->NewStringUTF(env, ufsecp_last_error_msg((const ufsecp_ctx*)(uintptr_t)ptr));
}

/* ── Key ops extras ──────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSeckeyNegate(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t buf[32];
    memcpy(buf, pk, 32);
    unpin(env, privkey, pk);
    int rc = ufsecp_seckey_negate((ufsecp_ctx*)(uintptr_t)ctx, buf);
    if (throw_on_err(env, rc, "seckey_negate")) {
        secure_zero(buf, sizeof(buf));
        return NULL;
    }
    jbyteArray out = mk(env, buf, 32);
    secure_zero(buf, sizeof(buf));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSeckeyTweakAdd(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray tweak) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    jbyte *tw = pin(env, tweak);
    uint8_t buf[32];
    memcpy(buf, pk, 32);
    int rc = ufsecp_seckey_tweak_add((ufsecp_ctx*)(uintptr_t)ctx, buf, (const uint8_t*)tw);
    unpin(env, tweak, tw);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "seckey_tweak_add")) {
        secure_zero(buf, sizeof(buf));
        return NULL;
    }
    jbyteArray out = mk(env, buf, 32);
    secure_zero(buf, sizeof(buf));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSeckeyTweakMul(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray tweak) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    jbyte *tw = pin(env, tweak);
    uint8_t buf[32];
    memcpy(buf, pk, 32);
    int rc = ufsecp_seckey_tweak_mul((ufsecp_ctx*)(uintptr_t)ctx, buf, (const uint8_t*)tw);
    unpin(env, tweak, tw);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "seckey_tweak_mul")) {
        secure_zero(buf, sizeof(buf));
        return NULL;
    }
    jbyteArray out = mk(env, buf, 32);
    secure_zero(buf, sizeof(buf));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativePubkeyParse(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray pubkey) {
    (void)clz;
    jbyte *pk = pin(env, pubkey);
    jsize len = (*env)->GetArrayLength(env, pubkey);
    uint8_t out[33];
    int rc = ufsecp_pubkey_parse((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (size_t)len, out);
    unpin(env, pubkey, pk);
    if (throw_on_err(env, rc, "pubkey_parse")) return NULL;
    return mk(env, out, 33);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativePubkeyXonly(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t out[32];
    int rc = ufsecp_pubkey_xonly((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, out);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "pubkey_xonly")) return NULL;
    return mk(env, out, 32);
}

/* ── ECDSA extras ────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaSigToDer(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray sig) {
    (void)clz;
    jbyte *s = pin(env, sig);
    uint8_t der[72];
    size_t dlen = 72;
    int rc = ufsecp_ecdsa_sig_to_der((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)s, der, &dlen);
    unpin(env, sig, s);
    if (throw_on_err(env, rc, "ecdsa_sig_to_der")) return NULL;
    return mk(env, der, (int)dlen);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaSigFromDer(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray der) {
    (void)clz;
    jbyte *d = pin(env, der);
    jsize dlen = (*env)->GetArrayLength(env, der);
    uint8_t sig[64];
    int rc = ufsecp_ecdsa_sig_from_der((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)d, (size_t)dlen, sig);
    unpin(env, der, d);
    if (throw_on_err(env, rc, "ecdsa_sig_from_der")) return NULL;
    return mk(env, sig, 64);
}

JNIEXPORT jobject JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaSignRecoverable(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msgHash, jbyteArray privkey) {
    (void)clz;
    jbyte *msg = pin(env, msgHash);
    jbyte *pk  = pin(env, privkey);
    uint8_t sig[64];
    int recid = 0;
    int rc = ufsecp_ecdsa_sign_recoverable((ufsecp_ctx*)(uintptr_t)ctx,
        (const uint8_t*)msg, (const uint8_t*)pk, sig, &recid);
    unpin(env, privkey, pk);
    unpin(env, msgHash, msg);
    if (throw_on_err(env, rc, "ecdsa_sign_recoverable")) return NULL;

    jbyteArray sigArr = mk(env, sig, 64);
    jclass cls = (*env)->FindClass(env, "com/ultrafast/ufsecp/RecoverableSignature");
    jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "([BI)V");
    return (*env)->NewObject(env, cls, ctor, sigArr, (jint)recid);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaRecover(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msgHash, jbyteArray sig, jint recid) {
    (void)clz;
    jbyte *msg = pin(env, msgHash);
    jbyte *s   = pin(env, sig);
    uint8_t pub[33];
    int rc = ufsecp_ecdsa_recover((ufsecp_ctx*)(uintptr_t)ctx,
        (const uint8_t*)msg, (const uint8_t*)s, (int)recid, pub);
    unpin(env, sig, s);
    unpin(env, msgHash, msg);
    if (throw_on_err(env, rc, "ecdsa_recover")) return NULL;
    return mk(env, pub, 33);
}

/* ── ECDH extras ─────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdhXonly(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray pubkey) {
    (void)clz;
    jbyte *pk  = pin(env, privkey);
    jbyte *pub = pin(env, pubkey);
    uint8_t out[32];
    int rc = ufsecp_ecdh_xonly((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (const uint8_t*)pub, out);
    unpin(env, pubkey, pub);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "ecdh_xonly")) return NULL;
    return mk(env, out, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdhRaw(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray pubkey) {
    (void)clz;
    jbyte *pk  = pin(env, privkey);
    jbyte *pub = pin(env, pubkey);
    uint8_t out[32];
    int rc = ufsecp_ecdh_raw((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (const uint8_t*)pub, out);
    unpin(env, pubkey, pub);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "ecdh_raw")) return NULL;
    return mk(env, out, 32);
}

/* ── Hash extras ─────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeTaggedHash(
    JNIEnv *env, jclass clz, jbyteArray tag, jbyteArray data) {
    (void)clz;
    const char *t = (const char*)(*env)->GetByteArrayElements(env, tag, NULL);
    jsize tlen = (*env)->GetArrayLength(env, tag);
    /* null-terminate the tag */
    char tbuf[256];
    if (tlen >= (jsize)sizeof(tbuf)) tlen = (jsize)sizeof(tbuf) - 1;
    memcpy(tbuf, t, (size_t)tlen);
    tbuf[tlen] = '\0';
    (*env)->ReleaseByteArrayElements(env, tag, (jbyte*)t, JNI_ABORT);

    jbyte *d = pin(env, data);
    jsize dlen = (*env)->GetArrayLength(env, data);
    uint8_t out[32];
    int rc = ufsecp_tagged_hash(tbuf, (const uint8_t*)d, (size_t)dlen, out);
    unpin(env, data, d);
    if (throw_on_err(env, rc, "tagged_hash")) return NULL;
    return mk(env, out, 32);
}

/* ── WIF extras ──────────────────────────────────────────────────────── */

JNIEXPORT jobject JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeWifDecode(
    JNIEnv *env, jclass clz, jlong ctx, jstring wif) {
    (void)clz;
    const char *w = (*env)->GetStringUTFChars(env, wif, NULL);
    if (!w) {
        jclass cls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
        if (cls) (*env)->ThrowNew(env, cls, "GetStringUTFChars OOM");
        return NULL;
    }
    uint8_t privkey[32];
    int comp = 0, net = 0;
    int rc = ufsecp_wif_decode((ufsecp_ctx*)(uintptr_t)ctx, w, privkey, &comp, &net);
    (*env)->ReleaseStringUTFChars(env, wif, w);
    if (throw_on_err(env, rc, "wif_decode")) {
        secure_zero(privkey, sizeof(privkey));
        return NULL;
    }

    jbyteArray keyArr = mk(env, privkey, 32);
    jclass cls = (*env)->FindClass(env, "com/ultrafast/ufsecp/WifDecoded");
    jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "([BZI)V");
    jobject out = (*env)->NewObject(env, cls, ctor, keyArr, (jboolean)(comp == 1), (jint)net);
    secure_zero(privkey, sizeof(privkey));
    return out;
}

/* ── BIP-32 extras ───────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32Privkey(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray key) {
    (void)clz;
    jbyte *k = pin(env, key);
    ufsecp_bip32_key key_in;
    memcpy(&key_in, k, sizeof(key_in));
    uint8_t priv[32];
    int rc = ufsecp_bip32_privkey((ufsecp_ctx*)(uintptr_t)ctx, &key_in, priv);
    unpin(env, key, k);
    if (throw_on_err(env, rc, "bip32_privkey")) {
        secure_zero(&key_in, sizeof(key_in));
        secure_zero(priv, sizeof(priv));
        return NULL;
    }
    jbyteArray out = mk(env, priv, 32);
    secure_zero(&key_in, sizeof(key_in));
    secure_zero(priv, sizeof(priv));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32Pubkey(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray key) {
    (void)clz;
    jbyte *k = pin(env, key);
    ufsecp_bip32_key key_in;
    memcpy(&key_in, k, sizeof(key_in));
    uint8_t pub[33];
    int rc = ufsecp_bip32_pubkey((ufsecp_ctx*)(uintptr_t)ctx, &key_in, pub);
    unpin(env, key, k);
    if (throw_on_err(env, rc, "bip32_pubkey")) {
        secure_zero(&key_in, sizeof(key_in));
        return NULL;
    }
    secure_zero(&key_in, sizeof(key_in));
    return mk(env, pub, 33);
}

/* ── Taproot ─────────────────────────────────────────────────────────── */

JNIEXPORT jobject JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeTaprootOutputKey(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray internalX, jbyteArray merkleRoot) {
    (void)clz;
    jbyte *ix = pin(env, internalX);
    jbyte *mr = merkleRoot ? pin(env, merkleRoot) : NULL;
    uint8_t outx[32];
    int parity = 0;
    int rc = ufsecp_taproot_output_key((ufsecp_ctx*)(uintptr_t)ctx,
        (const uint8_t*)ix, mr ? (const uint8_t*)mr : NULL, outx, &parity);
    if (mr) unpin(env, merkleRoot, mr);
    unpin(env, internalX, ix);
    if (throw_on_err(env, rc, "taproot_output_key")) return NULL;

    jbyteArray outArr = mk(env, outx, 32);
    jclass cls = (*env)->FindClass(env, "com/ultrafast/ufsecp/TaprootOutputKeyResult");
    jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "([BI)V");
    jobject out = (*env)->NewObject(env, cls, ctor, outArr, (jint)parity);
    secure_zero(outx, sizeof(outx));
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeTaprootTweakSeckey(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray merkleRoot) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    jbyte *mr = merkleRoot ? pin(env, merkleRoot) : NULL;
    uint8_t out[32];
    int rc = ufsecp_taproot_tweak_seckey((ufsecp_ctx*)(uintptr_t)ctx,
        (const uint8_t*)pk, mr ? (const uint8_t*)mr : NULL, out);
    if (mr) unpin(env, merkleRoot, mr);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "taproot_tweak_seckey")) {
        secure_zero(out, sizeof(out));
        return NULL;
    }
    jbyteArray result = mk(env, out, 32);
    secure_zero(out, sizeof(out));
    return result;
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeTaprootVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray outputX, jint parity,
    jbyteArray internalX, jbyteArray merkleRoot) {
    (void)clz;
    jbyte *ox = pin(env, outputX);
    jbyte *ix = pin(env, internalX);
    jbyte *mr = merkleRoot ? pin(env, merkleRoot) : NULL;
    size_t mrLen = merkleRoot ? (size_t)(*env)->GetArrayLength(env, merkleRoot) : 0;
    int rc = ufsecp_taproot_verify((ufsecp_ctx*)(uintptr_t)ctx,
        (const uint8_t*)ox, (int)parity, (const uint8_t*)ix,
        mr ? (const uint8_t*)mr : NULL, mrLen);
    if (mr) unpin(env, merkleRoot, mr);
    unpin(env, internalX, ix);
    unpin(env, outputX, ox);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}
