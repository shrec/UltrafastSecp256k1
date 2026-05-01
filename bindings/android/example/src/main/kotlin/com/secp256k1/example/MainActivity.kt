package com.secp256k1.example

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.secp256k1.native.Secp256k1

/**
 * Example Activity demonstrating UltrafastSecp256k1 usage on Android.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "Secp256k1Example"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Initialize library (precomputes generator table, runs self-test)
        val ok = Secp256k1.init()
        Log.i(TAG, "Library initialized: $ok")

        // 2) Example: Generate a keypair
        // Private key (32 bytes) — in production, use SecureRandom!
        val privkey = ByteArray(32) { (it + 1).toByte() }  // DEMO ONLY
        
        // Public key derivation (CT — side-channel safe)
        val pubkey = Secp256k1.ctScalarMulGenerator(privkey)
        Log.i(TAG, "Public key (${pubkey.size} bytes): ${pubkey.toHex()}")

        // 3) Example: Point operations (fast API)
        val g = Secp256k1.getGenerator()
        val g2 = Secp256k1.pointDouble(g)       // 2G
        val g3 = Secp256k1.pointAdd(g2, g)       // 3G
        Log.i(TAG, "3G = ${g3.toHex().take(20)}...")

        // 4) Example: ECDH (CT — side-channel safe)
        val alicePriv = ByteArray(32).also { it[31] = 42 }
        val bobPriv   = ByteArray(32).also { it[31] = 99 }
        val alicePub  = Secp256k1.ctScalarMulGenerator(alicePriv)
        val bobPub    = Secp256k1.ctScalarMulGenerator(bobPriv)

        val sharedA = Secp256k1.ctEcdh(alicePriv, bobPub)
        val sharedB = Secp256k1.ctEcdh(bobPriv, alicePub)
        
        val ecdhOk = sharedA.contentEquals(sharedB)
        Log.i(TAG, "ECDH shared secret match: $ecdhOk")
        Log.i(TAG, "Shared secret: ${sharedA.toHex().take(32)}...")

        // 5) Compressed public key
        val compressed = Secp256k1.pointCompress(pubkey)
        Log.i(TAG, "Compressed pubkey (${compressed.size} bytes): ${compressed.toHex()}")

        Log.i(TAG, "All examples completed successfully!")
    }

    private fun ByteArray.toHex(): String =
        joinToString("") { "%02x".format(it) }
}
