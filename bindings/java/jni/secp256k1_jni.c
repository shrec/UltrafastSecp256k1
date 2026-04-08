/* ============================================================================
 * UltrafastSecp256k1 — JNI Bridge
 * ============================================================================
 * Bridges Java native methods to the C API.
 * Build: compile as shared lib linking against ultrafast_secp256k1.
 * ============================================================================ */

#include "com_ultrafast_secp256k1_Secp256k1.h"
#include "ultrafast_secp256k1.h"
#include <string.h>
#include <stdlib.h>

/* ── Helpers ───────────────────────────────────────────────────────────────── */

static jbyteArray make_byte_array(JNIEnv *env, const uint8_t *data, jsize len) {
    jbyteArray arr = (*env)->NewByteArray(env, len);
    if (arr) (*env)->SetByteArrayRegion(env, arr, 0, len, (const jbyte *)data);
    return arr;
}

static void get_bytes(JNIEnv *env, jbyteArray arr, uint8_t *dst, jsize len) {
    (*env)->GetByteArrayRegion(env, arr, 0, len, (jbyte *)dst);
}

static void throw_exc(JNIEnv *env, const char *msg) {
    (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/IllegalArgumentException"), msg);
}

#define CHECK_LEN(arr, expected, name) \
    if ((*env)->GetArrayLength(env, arr) != (expected)) { \
        throw_exc(env, name " must be " #expected " bytes"); \
        return NULL; \
    }

#define CHECK_LEN_BOOL(arr, expected, name) \
    if ((*env)->GetArrayLength(env, arr) != (expected)) { \
        throw_exc(env, name " must be " #expected " bytes"); \
        return JNI_FALSE; \
    }

/* ── Init / Version ────────────────────────────────────────────────────────── */

JNIEXPORT jint JNICALL Java_com_ultrafast_secp256k1_Secp256k1_nativeInit
  (JNIEnv *env, jclass cls)
{
    (void)env; (void)cls;
    return (jint)secp256k1_init();
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_secp256k1_Secp256k1_nativeVersion
  (JNIEnv *env, jclass cls)
{
    (void)cls;
    return (*env)->NewStringUTF(env, secp256k1_version());
}

/* ── Key Operations ────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecPubkeyCreate
  (JNIEnv *env, jclass cls, jbyteArray privkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t pk[32], out[33];
    get_bytes(env, privkey, pk, 32);
    if (secp256k1_ec_pubkey_create(pk, out) != 0) {
        throw_exc(env, "Invalid private key");
        return NULL;
    }
    return make_byte_array(env, out, 33);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecPubkeyCreateUncompressed
  (JNIEnv *env, jclass cls, jbyteArray privkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t pk[32], out[65];
    get_bytes(env, privkey, pk, 32);
    if (secp256k1_ec_pubkey_create_uncompressed(pk, out) != 0) {
        throw_exc(env, "Invalid private key");
        return NULL;
    }
    return make_byte_array(env, out, 65);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecPubkeyParse
  (JNIEnv *env, jclass cls, jbyteArray input)
{
    (void)cls;
    jsize ilen = (*env)->GetArrayLength(env, input);
    if (ilen != 33 && ilen != 65) {
        throw_exc(env, "pubkey must be 33 or 65 bytes");
        return NULL;
    }
    uint8_t ibuf[65], out[33];
    get_bytes(env, input, ibuf, ilen);
    if (secp256k1_ec_pubkey_parse(ibuf, (size_t)ilen, out) != 0) {
        throw_exc(env, "Invalid public key");
        return NULL;
    }
    return make_byte_array(env, out, 33);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecSeckeyVerify
  (JNIEnv *env, jclass cls, jbyteArray privkey)
{
    (void)cls;
    CHECK_LEN_BOOL(privkey, 32, "privkey");
    uint8_t pk[32];
    get_bytes(env, privkey, pk, 32);
    return secp256k1_ec_seckey_verify(pk) == 1 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecPrivkeyNegate
  (JNIEnv *env, jclass cls, jbyteArray privkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t pk[32];
    get_bytes(env, privkey, pk, 32);
    if (secp256k1_ec_privkey_negate(pk) != 1) {
        throw_exc(env, "Privkey negate failed");
        return NULL;
    }
    return make_byte_array(env, pk, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecPrivkeyTweakAdd
  (JNIEnv *env, jclass cls, jbyteArray privkey, jbyteArray tweak)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    CHECK_LEN(tweak, 32, "tweak");
    uint8_t pk[32], tw[32];
    get_bytes(env, privkey, pk, 32);
    get_bytes(env, tweak, tw, 32);
    if (secp256k1_ec_privkey_tweak_add(pk, tw) != 0) {
        throw_exc(env, "Tweak add failed");
        return NULL;
    }
    return make_byte_array(env, pk, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecPrivkeyTweakMul
  (JNIEnv *env, jclass cls, jbyteArray privkey, jbyteArray tweak)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    CHECK_LEN(tweak, 32, "tweak");
    uint8_t pk[32], tw[32];
    get_bytes(env, privkey, pk, 32);
    get_bytes(env, tweak, tw, 32);
    if (secp256k1_ec_privkey_tweak_mul(pk, tw) != 0) {
        throw_exc(env, "Tweak mul failed");
        return NULL;
    }
    return make_byte_array(env, pk, 32);
}

/* ── ECDSA ─────────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdsaSign
  (JNIEnv *env, jclass cls, jbyteArray msgHash, jbyteArray privkey)
{
    (void)cls;
    CHECK_LEN(msgHash, 32, "msgHash");
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t mh[32], pk[32], sig[64];
    get_bytes(env, msgHash, mh, 32);
    get_bytes(env, privkey, pk, 32);
    if (secp256k1_ecdsa_sign(mh, pk, sig) != 0) {
        throw_exc(env, "ECDSA signing failed");
        return NULL;
    }
    return make_byte_array(env, sig, 64);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdsaVerify
  (JNIEnv *env, jclass cls, jbyteArray msgHash, jbyteArray sig, jbyteArray pubkey)
{
    (void)cls;
    CHECK_LEN_BOOL(msgHash, 32, "msgHash");
    CHECK_LEN_BOOL(sig, 64, "sig");
    CHECK_LEN_BOOL(pubkey, 33, "pubkey");
    uint8_t mh[32], s[64], pk[33];
    get_bytes(env, msgHash, mh, 32);
    get_bytes(env, sig, s, 64);
    get_bytes(env, pubkey, pk, 33);
    return secp256k1_ecdsa_verify(mh, s, pk) == 1 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdsaSerializeDer
  (JNIEnv *env, jclass cls, jbyteArray sig)
{
    (void)cls;
    CHECK_LEN(sig, 64, "sig");
    uint8_t s[64], der[72];
    size_t der_len = 72;
    get_bytes(env, sig, s, 64);
    if (secp256k1_ecdsa_signature_serialize_der(s, der, &der_len) != 0) {
        throw_exc(env, "DER serialization failed");
        return NULL;
    }
    return make_byte_array(env, der, (jsize)der_len);
}

/* ── Recovery ──────────────────────────────────────────────────────────────── */

JNIEXPORT jobject JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdsaSignRecoverable
  (JNIEnv *env, jclass cls, jbyteArray msgHash, jbyteArray privkey)
{
    (void)cls;
    uint8_t mh[32], pk[32], sig[64];
    int recid;
    if ((*env)->GetArrayLength(env, msgHash) != 32 || (*env)->GetArrayLength(env, privkey) != 32) {
        throw_exc(env, "msgHash and privkey must be 32 bytes");
        return NULL;
    }
    get_bytes(env, msgHash, mh, 32);
    get_bytes(env, privkey, pk, 32);
    if (secp256k1_ecdsa_sign_recoverable(mh, pk, sig, &recid) != 0) {
        throw_exc(env, "Recoverable signing failed");
        return NULL;
    }
    jbyteArray sigArr = make_byte_array(env, sig, 64);
    jclass rsCls = (*env)->FindClass(env, "com/ultrafast/secp256k1/Secp256k1$RecoverableSignature");
    jmethodID ctor = (*env)->GetMethodID(env, rsCls, "<init>", "([BI)V");
    return (*env)->NewObject(env, rsCls, ctor, sigArr, (jint)recid);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdsaRecover
  (JNIEnv *env, jclass cls, jbyteArray msgHash, jbyteArray sig, jint recid)
{
    (void)cls;
    CHECK_LEN(msgHash, 32, "msgHash");
    CHECK_LEN(sig, 64, "sig");
    uint8_t mh[32], s[64], pubkey[33];
    get_bytes(env, msgHash, mh, 32);
    get_bytes(env, sig, s, 64);
    if (secp256k1_ecdsa_recover(mh, s, (int)recid, pubkey) != 0) {
        throw_exc(env, "Recovery failed");
        return NULL;
    }
    return make_byte_array(env, pubkey, 33);
}

/* ── Schnorr ───────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_schnorrSign
  (JNIEnv *env, jclass cls, jbyteArray msg, jbyteArray privkey, jbyteArray auxRand)
{
    (void)cls;
    CHECK_LEN(msg, 32, "msg");
    CHECK_LEN(privkey, 32, "privkey");
    CHECK_LEN(auxRand, 32, "auxRand");
    uint8_t m[32], pk[32], ar[32], sig[64];
    get_bytes(env, msg, m, 32);
    get_bytes(env, privkey, pk, 32);
    get_bytes(env, auxRand, ar, 32);
    if (secp256k1_schnorr_sign(m, pk, ar, sig) != 0) {
        throw_exc(env, "Schnorr signing failed");
        return NULL;
    }
    return make_byte_array(env, sig, 64);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_secp256k1_Secp256k1_schnorrVerify
  (JNIEnv *env, jclass cls, jbyteArray msg, jbyteArray sig, jbyteArray pubkeyX)
{
    (void)cls;
    CHECK_LEN_BOOL(msg, 32, "msg");
    CHECK_LEN_BOOL(sig, 64, "sig");
    CHECK_LEN_BOOL(pubkeyX, 32, "pubkeyX");
    uint8_t m[32], s[64], px[32];
    get_bytes(env, msg, m, 32);
    get_bytes(env, sig, s, 64);
    get_bytes(env, pubkeyX, px, 32);
    return secp256k1_schnorr_verify(m, s, px) == 1 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_schnorrPubkey
  (JNIEnv *env, jclass cls, jbyteArray privkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t pk[32], out[32];
    get_bytes(env, privkey, pk, 32);
    if (secp256k1_schnorr_pubkey(pk, out) != 0) {
        throw_exc(env, "Invalid private key");
        return NULL;
    }
    return make_byte_array(env, out, 32);
}

/* ── ECDH ──────────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdh
  (JNIEnv *env, jclass cls, jbyteArray privkey, jbyteArray pubkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    CHECK_LEN(pubkey, 33, "pubkey");
    uint8_t pk[32], pub[33], out[32];
    get_bytes(env, privkey, pk, 32);
    get_bytes(env, pubkey, pub, 33);
    if (secp256k1_ecdh(pk, pub, out) != 0) {
        throw_exc(env, "ECDH failed");
        return NULL;
    }
    return make_byte_array(env, out, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdhXonly
  (JNIEnv *env, jclass cls, jbyteArray privkey, jbyteArray pubkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    CHECK_LEN(pubkey, 33, "pubkey");
    uint8_t pk[32], pub[33], out[32];
    get_bytes(env, privkey, pk, 32);
    get_bytes(env, pubkey, pub, 33);
    if (secp256k1_ecdh_xonly(pk, pub, out) != 0) {
        throw_exc(env, "ECDH xonly failed");
        return NULL;
    }
    return make_byte_array(env, out, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_ecdhRaw
  (JNIEnv *env, jclass cls, jbyteArray privkey, jbyteArray pubkey)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    CHECK_LEN(pubkey, 33, "pubkey");
    uint8_t pk[32], pub[33], out[32];
    get_bytes(env, privkey, pk, 32);
    get_bytes(env, pubkey, pub, 33);
    if (secp256k1_ecdh_raw(pk, pub, out) != 0) {
        throw_exc(env, "ECDH raw failed");
        return NULL;
    }
    return make_byte_array(env, out, 32);
}

/* ── Hashing ───────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_sha256
  (JNIEnv *env, jclass cls, jbyteArray data)
{
    (void)cls;
    jsize dlen = (*env)->GetArrayLength(env, data);
    uint8_t *dbuf = (uint8_t *)(*env)->GetByteArrayElements(env, data, NULL);
    uint8_t out[32];
    secp256k1_sha256(dbuf, (size_t)dlen, out);
    (*env)->ReleaseByteArrayElements(env, data, (jbyte *)dbuf, JNI_ABORT);
    return make_byte_array(env, out, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_hash160
  (JNIEnv *env, jclass cls, jbyteArray data)
{
    (void)cls;
    jsize dlen = (*env)->GetArrayLength(env, data);
    uint8_t *dbuf = (uint8_t *)(*env)->GetByteArrayElements(env, data, NULL);
    uint8_t out[20];
    secp256k1_hash160(dbuf, (size_t)dlen, out);
    (*env)->ReleaseByteArrayElements(env, data, (jbyte *)dbuf, JNI_ABORT);
    return make_byte_array(env, out, 20);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_taggedHash
  (JNIEnv *env, jclass cls, jstring tag, jbyteArray data)
{
    (void)cls;
    const char *ctag = (*env)->GetStringUTFChars(env, tag, NULL);
    jsize dlen = (*env)->GetArrayLength(env, data);
    uint8_t *dbuf = (uint8_t *)(*env)->GetByteArrayElements(env, data, NULL);
    uint8_t out[32];
    secp256k1_tagged_hash(ctag, dbuf, (size_t)dlen, out);
    (*env)->ReleaseByteArrayElements(env, data, (jbyte *)dbuf, JNI_ABORT);
    (*env)->ReleaseStringUTFChars(env, tag, ctag);
    return make_byte_array(env, out, 32);
}

/* ── Addresses ─────────────────────────────────────────────────────────────── */

static jstring get_address(JNIEnv *env, int (*fn)(const uint8_t*, int, char*, size_t*),
                           jbyteArray key, jint key_len, jint network)
{
    uint8_t kbuf[33];
    get_bytes(env, key, kbuf, key_len);
    char addr[128];
    size_t alen = 128;
    if (fn(kbuf, (int)network, addr, &alen) != 0) {
        throw_exc(env, "Address generation failed");
        return NULL;
    }
    addr[alen] = '\0';
    return (*env)->NewStringUTF(env, addr);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_secp256k1_Secp256k1_addressP2pkh
  (JNIEnv *env, jclass cls, jbyteArray pubkey, jint network)
{
    (void)cls;
    CHECK_LEN(pubkey, 33, "pubkey");
    return get_address(env, secp256k1_address_p2pkh, pubkey, 33, network);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_secp256k1_Secp256k1_addressP2wpkh
  (JNIEnv *env, jclass cls, jbyteArray pubkey, jint network)
{
    (void)cls;
    CHECK_LEN(pubkey, 33, "pubkey");
    return get_address(env, secp256k1_address_p2wpkh, pubkey, 33, network);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_secp256k1_Secp256k1_addressP2tr
  (JNIEnv *env, jclass cls, jbyteArray internalKeyX, jint network)
{
    (void)cls;
    CHECK_LEN(internalKeyX, 32, "internalKeyX");
    return get_address(env, (int(*)(const uint8_t*, int, char*, size_t*))secp256k1_address_p2tr,
                       internalKeyX, 32, network);
}

/* ── WIF ───────────────────────────────────────────────────────────────────── */

JNIEXPORT jstring JNICALL Java_com_ultrafast_secp256k1_Secp256k1_wifEncode
  (JNIEnv *env, jclass cls, jbyteArray privkey, jboolean compressed, jint network)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t pk[32];
    get_bytes(env, privkey, pk, 32);
    char wif[128];
    size_t wlen = 128;
    if (secp256k1_wif_encode(pk, compressed ? 1 : 0, (int)network, wif, &wlen) != 0) {
        throw_exc(env, "WIF encode failed");
        return NULL;
    }
    wif[wlen] = '\0';
    return (*env)->NewStringUTF(env, wif);
}

JNIEXPORT jobject JNICALL Java_com_ultrafast_secp256k1_Secp256k1_wifDecode
  (JNIEnv *env, jclass cls, jstring wif)
{
    (void)cls;
    const char *cwif = (*env)->GetStringUTFChars(env, wif, NULL);
    uint8_t pk[32];
    int comp, net;
    int rc = secp256k1_wif_decode(cwif, pk, &comp, &net);
    (*env)->ReleaseStringUTFChars(env, wif, cwif);
    if (rc != 0) {
        throw_exc(env, "Invalid WIF");
        return NULL;
    }
    jbyteArray pkArr = make_byte_array(env, pk, 32);
    jclass wdCls = (*env)->FindClass(env, "com/ultrafast/secp256k1/Secp256k1$WifDecodeResult");
    jmethodID ctor = (*env)->GetMethodID(env, wdCls, "<init>", "([BZI)V");
    return (*env)->NewObject(env, wdCls, ctor, pkArr, comp == 1 ? JNI_TRUE : JNI_FALSE, (jint)net);
}

/* ── BIP-32 ────────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_bip32MasterKey
  (JNIEnv *env, jclass cls, jbyteArray seed)
{
    (void)cls;
    jsize slen = (*env)->GetArrayLength(env, seed);
    if (slen < 16 || slen > 64) {
        throw_exc(env, "Seed must be 16-64 bytes");
        return NULL;
    }
    uint8_t sbuf[64];
    get_bytes(env, seed, sbuf, slen);
    secp256k1_bip32_key key;
    if (secp256k1_bip32_master_key(sbuf, (size_t)slen, &key) != 0) {
        throw_exc(env, "Master key generation failed");
        return NULL;
    }
    return make_byte_array(env, (const uint8_t *)&key, 79);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_bip32DeriveChild
  (JNIEnv *env, jclass cls, jbyteArray parentKey, jint index)
{
    (void)cls;
    if ((*env)->GetArrayLength(env, parentKey) != 79) {
        throw_exc(env, "parentKey must be 79 bytes");
        return NULL;
    }
    secp256k1_bip32_key parent, child;
    get_bytes(env, parentKey, (uint8_t *)&parent, 79);
    if (secp256k1_bip32_derive_child(&parent, (uint32_t)index, &child) != 0) {
        throw_exc(env, "Child derivation failed");
        return NULL;
    }
    return make_byte_array(env, (const uint8_t *)&child, 79);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_bip32DerivePath
  (JNIEnv *env, jclass cls, jbyteArray masterKey, jstring path)
{
    (void)cls;
    if ((*env)->GetArrayLength(env, masterKey) != 79) {
        throw_exc(env, "masterKey must be 79 bytes");
        return NULL;
    }
    secp256k1_bip32_key master, out;
    get_bytes(env, masterKey, (uint8_t *)&master, 79);
    const char *cpath = (*env)->GetStringUTFChars(env, path, NULL);
    int rc = secp256k1_bip32_derive_path(&master, cpath, &out);
    (*env)->ReleaseStringUTFChars(env, path, cpath);
    if (rc != 0) {
        throw_exc(env, "Path derivation failed");
        return NULL;
    }
    return make_byte_array(env, (const uint8_t *)&out, 79);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_bip32GetPrivkey
  (JNIEnv *env, jclass cls, jbyteArray key)
{
    (void)cls;
    if ((*env)->GetArrayLength(env, key) != 79) {
        throw_exc(env, "key must be 79 bytes");
        return NULL;
    }
    secp256k1_bip32_key k;
    get_bytes(env, key, (uint8_t *)&k, 79);
    uint8_t pk[32];
    if (secp256k1_bip32_get_privkey(&k, pk) != 0) {
        throw_exc(env, "Not a private key");
        return NULL;
    }
    return make_byte_array(env, pk, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_bip32GetPubkey
  (JNIEnv *env, jclass cls, jbyteArray key)
{
    (void)cls;
    if ((*env)->GetArrayLength(env, key) != 79) {
        throw_exc(env, "key must be 79 bytes");
        return NULL;
    }
    secp256k1_bip32_key k;
    get_bytes(env, key, (uint8_t *)&k, 79);
    uint8_t pub[33];
    if (secp256k1_bip32_get_pubkey(&k, pub) != 0) {
        throw_exc(env, "Public key extraction failed");
        return NULL;
    }
    return make_byte_array(env, pub, 33);
}

/* ── Taproot ───────────────────────────────────────────────────────────────── */

JNIEXPORT jobject JNICALL Java_com_ultrafast_secp256k1_Secp256k1_taprootOutputKey
  (JNIEnv *env, jclass cls, jbyteArray internalKeyX, jbyteArray merkleRoot)
{
    (void)cls;
    CHECK_LEN(internalKeyX, 32, "internalKeyX");
    uint8_t ik[32], out[32];
    get_bytes(env, internalKeyX, ik, 32);
    const uint8_t *mr = NULL;
    uint8_t mrbuf[32];
    if (merkleRoot != NULL) {
        get_bytes(env, merkleRoot, mrbuf, 32);
        mr = mrbuf;
    }
    int parity;
    if (secp256k1_taproot_output_key(ik, mr, out, &parity) != 0) {
        throw_exc(env, "Taproot output key failed");
        return NULL;
    }
    jbyteArray outArr = make_byte_array(env, out, 32);
    jclass tokCls = (*env)->FindClass(env, "com/ultrafast/secp256k1/Secp256k1$TaprootOutputKeyResult");
    jmethodID ctor = (*env)->GetMethodID(env, tokCls, "<init>", "([BI)V");
    return (*env)->NewObject(env, tokCls, ctor, outArr, (jint)parity);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_secp256k1_Secp256k1_taprootTweakPrivkey
  (JNIEnv *env, jclass cls, jbyteArray privkey, jbyteArray merkleRoot)
{
    (void)cls;
    CHECK_LEN(privkey, 32, "privkey");
    uint8_t pk[32], out[32];
    get_bytes(env, privkey, pk, 32);
    const uint8_t *mr = NULL;
    uint8_t mrbuf[32];
    if (merkleRoot != NULL) {
        get_bytes(env, merkleRoot, mrbuf, 32);
        mr = mrbuf;
    }
    if (secp256k1_taproot_tweak_privkey(pk, mr, out) != 0) {
        throw_exc(env, "Taproot tweak failed");
        return NULL;
    }
    return make_byte_array(env, out, 32);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_secp256k1_Secp256k1_taprootVerifyCommitment
  (JNIEnv *env, jclass cls, jbyteArray outputKeyX, jint parity,
   jbyteArray internalKeyX, jbyteArray merkleRoot)
{
    (void)cls;
    CHECK_LEN_BOOL(outputKeyX, 32, "outputKeyX");
    CHECK_LEN_BOOL(internalKeyX, 32, "internalKeyX");
    uint8_t ok[32], ik[32];
    get_bytes(env, outputKeyX, ok, 32);
    get_bytes(env, internalKeyX, ik, 32);
    const uint8_t *mr = NULL;
    size_t mr_len = 0;
    uint8_t mrbuf[32];
    if (merkleRoot != NULL) {
        mr_len = (size_t)(*env)->GetArrayLength(env, merkleRoot);
        get_bytes(env, merkleRoot, mrbuf, (jsize)mr_len);
        mr = mrbuf;
    }
    return secp256k1_taproot_verify_commitment(ok, (int)parity, ik, mr, mr_len) == 1
        ? JNI_TRUE : JNI_FALSE;
}
