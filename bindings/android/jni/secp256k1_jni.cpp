// ============================================================================
// UltrafastSecp256k1 -- JNI Bridge for Android
// ============================================================================
// Exposes core ECC operations to Java/Kotlin via JNI.
//
// Fast API: Maximum speed, no side-channel protection
// CT API:   Constant-time, side-channel resistant (for key operations)
//
// All byte arrays use big-endian encoding (standard crypto convention).
// Scalars: 32 bytes. Points (uncompressed): 65 bytes. Compressed: 33 bytes.
// ============================================================================

#include <jni.h>
#include <cstring>
#include <array>
#include <android/log.h>

// Fast API
#include <secp256k1/field.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/init.hpp>
#include <secp256k1/selftest.hpp>

// Constant-Time API
#include <secp256k1/ct/ops.hpp>
#include <secp256k1/ct/field.hpp>
#include <secp256k1/ct/scalar.hpp>
#include <secp256k1/ct/point.hpp>

#define LOG_TAG "Secp256k1JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace secp256k1;
using FE = fast::FieldElement;
using SC = fast::Scalar;
using PT = fast::Point;

// ============================================================================
// Helpers: JNI <-> C++ conversion
// ============================================================================

// Extract 32 bytes from jbyteArray into std::array
static bool get_bytes32(JNIEnv* env, jbyteArray arr, std::array<uint8_t, 32>& out) {
    if (!arr || env->GetArrayLength(arr) != 32) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Expected 32-byte array");
        return false;
    }
    env->GetByteArrayRegion(arr, 0, 32, reinterpret_cast<jbyte*>(out.data()));
    return true;
}

// Create jbyteArray from raw bytes
static jbyteArray make_jbytes(JNIEnv* env, const uint8_t* data, jsize len) {
    jbyteArray result = env->NewByteArray(len);
    if (result) {
        env->SetByteArrayRegion(result, 0, len, reinterpret_cast<const jbyte*>(data));
    }
    return result;
}

// Scalar from 32 big-endian bytes
static SC scalar_from_jbytes(JNIEnv* env, jbyteArray arr) {
    std::array<uint8_t, 32> bytes{};
    if (!get_bytes32(env, arr, bytes)) return SC::zero();
    return SC::from_bytes(bytes);
}

// Point from 65-byte uncompressed (04 || x[32] || y[32])
static PT point_from_uncompressed(JNIEnv* env, jbyteArray arr) {
    if (!arr || env->GetArrayLength(arr) != 65) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Expected 65-byte uncompressed point (04 || x || y)");
        return PT::infinity();
    }
    uint8_t buf[65];
    env->GetByteArrayRegion(arr, 0, 65, reinterpret_cast<jbyte*>(buf));

    if (buf[0] != 0x04) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Uncompressed point must start with 0x04");
        return PT::infinity();
    }

    // Big-endian x,y -> FieldElement
    std::array<uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), buf + 1,  32);
    std::memcpy(yb.data(), buf + 33, 32);

    FE x = FE::from_bytes(xb);
    FE y = FE::from_bytes(yb);
    return PT::from_affine(x, y);
}

// ============================================================================
// JNI exports -- package: com.secp256k1.native
// ============================================================================
extern "C" {

// ----------------------------------------------------------------------------
// Library lifecycle
// ----------------------------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_secp256k1_native_Secp256k1_nativeInit(JNIEnv* env, jclass) {
    LOGI("UltrafastSecp256k1 initializing...");

    // Run self-test and initialize (done once, thread-safe)
    bool ok = fast::ensure_library_integrity(false);
    LOGI("Self-test: %s", ok ? "PASS" : "FAIL");
    return static_cast<jboolean>(ok);
}

JNIEXPORT jboolean JNICALL
Java_com_secp256k1_native_Secp256k1_nativeSelfTest(JNIEnv* env, jclass) {
    return static_cast<jboolean>(fast::Selftest(false));
}

// ----------------------------------------------------------------------------
// Scalar operations
// ----------------------------------------------------------------------------

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_scalarAdd(
    JNIEnv* env, jclass, jbyteArray a_bytes, jbyteArray b_bytes)
{
    SC a = scalar_from_jbytes(env, a_bytes);
    SC b = scalar_from_jbytes(env, b_bytes);
    auto result = (a + b).to_bytes();
    return make_jbytes(env, result.data(), 32);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_scalarMul(
    JNIEnv* env, jclass, jbyteArray a_bytes, jbyteArray b_bytes)
{
    SC a = scalar_from_jbytes(env, a_bytes);
    SC b = scalar_from_jbytes(env, b_bytes);
    auto result = (a * b).to_bytes();
    return make_jbytes(env, result.data(), 32);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_scalarSub(
    JNIEnv* env, jclass, jbyteArray a_bytes, jbyteArray b_bytes)
{
    SC a = scalar_from_jbytes(env, a_bytes);
    SC b = scalar_from_jbytes(env, b_bytes);
    auto result = (a - b).to_bytes();
    return make_jbytes(env, result.data(), 32);
}

// ----------------------------------------------------------------------------
// Point operations (fast -- no side-channel protection)
// ----------------------------------------------------------------------------

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_getGenerator(JNIEnv* env, jclass)
{
    auto g = PT::generator().to_uncompressed();
    return make_jbytes(env, g.data(), 65);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_pointAdd(
    JNIEnv* env, jclass, jbyteArray p1_bytes, jbyteArray p2_bytes)
{
    PT p1 = point_from_uncompressed(env, p1_bytes);
    PT p2 = point_from_uncompressed(env, p2_bytes);
    auto result = p1.add(p2).to_uncompressed();
    return make_jbytes(env, result.data(), 65);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_pointDouble(
    JNIEnv* env, jclass, jbyteArray p_bytes)
{
    PT p = point_from_uncompressed(env, p_bytes);
    auto result = p.dbl().to_uncompressed();
    return make_jbytes(env, result.data(), 65);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_pointNegate(
    JNIEnv* env, jclass, jbyteArray p_bytes)
{
    PT p = point_from_uncompressed(env, p_bytes);
    auto result = p.negate().to_uncompressed();
    return make_jbytes(env, result.data(), 65);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_scalarMulPoint(
    JNIEnv* env, jclass, jbyteArray scalar_bytes, jbyteArray point_bytes)
{
    SC k = scalar_from_jbytes(env, scalar_bytes);
    PT p = point_from_uncompressed(env, point_bytes);
    auto result = p.scalar_mul(k).to_uncompressed();
    return make_jbytes(env, result.data(), 65);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_scalarMulGenerator(
    JNIEnv* env, jclass, jbyteArray scalar_bytes)
{
    SC k = scalar_from_jbytes(env, scalar_bytes);
    auto result = PT::generator().scalar_mul(k).to_uncompressed();
    return make_jbytes(env, result.data(), 65);
}

// Compressed point output (33 bytes)
JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_pointCompress(
    JNIEnv* env, jclass, jbyteArray p_bytes)
{
    PT p = point_from_uncompressed(env, p_bytes);
    auto result = p.to_compressed();
    return make_jbytes(env, result.data(), 33);
}

JNIEXPORT jboolean JNICALL
Java_com_secp256k1_native_Secp256k1_pointIsInfinity(
    JNIEnv* env, jclass, jbyteArray p_bytes)
{
    PT p = point_from_uncompressed(env, p_bytes);
    return static_cast<jboolean>(p.is_infinity());
}

// ----------------------------------------------------------------------------
// CT (Constant-Time) operations -- side-channel resistant
// ----------------------------------------------------------------------------

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_ctScalarMulGenerator(
    JNIEnv* env, jclass, jbyteArray scalar_bytes)
{
    SC k = scalar_from_jbytes(env, scalar_bytes);
    // Use CT generator_mul (fixed-window, complete addition)
    PT result = ct::generator_mul(k);
    auto out_bytes = result.to_uncompressed();
    return make_jbytes(env, out_bytes.data(), 65);
}

JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_ctScalarMulPoint(
    JNIEnv* env, jclass, jbyteArray scalar_bytes, jbyteArray point_bytes)
{
    SC k = scalar_from_jbytes(env, scalar_bytes);
    PT p = point_from_uncompressed(env, point_bytes);
    PT out_pt = ct::scalar_mul(p, k);
    auto out = out_pt.to_uncompressed();
    return make_jbytes(env, out.data(), 65);
}

// CT ECDH: shared_secret = k * Q (returns x-coordinate only, 32 bytes)
JNIEXPORT jbyteArray JNICALL
Java_com_secp256k1_native_Secp256k1_ctEcdh(
    JNIEnv* env, jclass, jbyteArray privkey_bytes, jbyteArray pubkey_bytes)
{
    SC k = scalar_from_jbytes(env, privkey_bytes);
    PT q = point_from_uncompressed(env, pubkey_bytes);
    
    PT result = ct::scalar_mul(q, k);
    
    // Extract x-coordinate (the shared secret) -- already affine from to_point()
    auto x_bytes = result.x().to_bytes();
    return make_jbytes(env, x_bytes.data(), 32);
}

} // extern "C"
