// ============================================================================
// UltrafastSecp256k1 — Rust Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies FFI boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.
//
// Usage:
//   cargo test --package ufsecp -- smoke
// ============================================================================

#[cfg(test)]
mod smoke {
    use ufsecp::{Context, Network};

    // ── Golden Vectors ──────────────────────────────────────────────

    fn known_privkey() -> [u8; 32] {
        let mut k = [0u8; 32];
        k[31] = 1;
        k
    }

    fn known_pubkey() -> [u8; 33] {
        hex_decode_33("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
    }

    fn known_xonly() -> [u8; 32] {
        hex_decode_32("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
    }

    fn sha256_empty() -> [u8; 32] {
        hex_decode_32("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")
    }

    fn hex_decode_32(s: &str) -> [u8; 32] {
        let bytes = hex::decode(s).unwrap();
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        out
    }

    fn hex_decode_33(s: &str) -> [u8; 33] {
        let bytes = hex::decode(s).unwrap();
        let mut out = [0u8; 33];
        out.copy_from_slice(&bytes);
        out
    }

    // ── Tests ───────────────────────────────────────────────────────

    #[test]
    fn smoke_ctx_create_abi() {
        let ctx = Context::new().expect("ctx_create failed");
        let abi = ctx.abi_version();
        assert!(abi >= 1, "ABI {} < 1", abi);
    }

    #[test]
    fn smoke_pubkey_create_golden() {
        let ctx = Context::new().unwrap();
        let pub_key = ctx.pubkey_create(&known_privkey()).unwrap();
        assert_eq!(pub_key, known_pubkey(), "compressed pubkey mismatch");
    }

    #[test]
    fn smoke_pubkey_xonly_golden() {
        let ctx = Context::new().unwrap();
        let xonly = ctx.pubkey_xonly(&known_privkey()).unwrap();
        assert_eq!(xonly, known_xonly(), "x-only pubkey mismatch");
    }

    #[test]
    fn smoke_ecdsa_sign_verify() {
        let ctx = Context::new().unwrap();
        let msg = [0u8; 32];
        let sig = ctx.ecdsa_sign(&msg, &known_privkey()).unwrap();
        assert_eq!(sig.len(), 64, "sig length");
        assert!(ctx.ecdsa_verify(&msg, &sig, &known_pubkey()), "valid sig rejected");

        // Mutated → fail
        let mut bad = sig;
        bad[0] ^= 0x01;
        assert!(!ctx.ecdsa_verify(&msg, &bad, &known_pubkey()), "mutated sig accepted");
    }

    #[test]
    fn smoke_schnorr_sign_verify() {
        let ctx = Context::new().unwrap();
        let msg = [0u8; 32];
        let aux = [0u8; 32];
        let sig = ctx.schnorr_sign(&msg, &known_privkey(), &aux).unwrap();
        assert_eq!(sig.len(), 64, "schnorr sig length");
        assert!(ctx.schnorr_verify(&msg, &sig, &known_xonly()), "valid schnorr sig rejected");
    }

    #[test]
    fn smoke_ecdsa_recover() {
        let ctx = Context::new().unwrap();
        let msg = [0u8; 32];
        let rec = ctx.ecdsa_sign_recoverable(&msg, &known_privkey()).unwrap();
        let pub_key = ctx.ecdsa_recover(&msg, &rec.signature, rec.recovery_id).unwrap();
        assert_eq!(pub_key, known_pubkey(), "recovered pubkey mismatch");
    }

    #[test]
    fn smoke_sha256_golden() {
        let digest = Context::sha256(&[]).unwrap();
        assert_eq!(digest, sha256_empty(), "SHA-256 empty mismatch");
    }

    #[test]
    fn smoke_addr_p2wpkh() {
        let ctx = Context::new().unwrap();
        let addr = ctx.addr_p2wpkh(&known_pubkey(), Network::Mainnet).unwrap();
        assert!(addr.starts_with("bc1q"), "expected bc1q..., got {}", addr);
    }

    #[test]
    fn smoke_wif_roundtrip() {
        let ctx = Context::new().unwrap();
        let wif = ctx.wif_encode(&known_privkey(), true, Network::Mainnet).unwrap();
        let decoded = ctx.wif_decode(&wif).unwrap();
        assert_eq!(decoded.privkey, known_privkey(), "WIF privkey mismatch");
        assert!(decoded.compressed, "WIF should be compressed");
    }

    #[test]
    fn smoke_ecdh_symmetric() {
        let ctx = Context::new().unwrap();
        let mut k2 = [0u8; 32];
        k2[31] = 2;
        let pub1 = ctx.pubkey_create(&known_privkey()).unwrap();
        let pub2 = ctx.pubkey_create(&k2).unwrap();
        let s12 = ctx.ecdh(&known_privkey(), &pub2).unwrap();
        let s21 = ctx.ecdh(&k2, &pub1).unwrap();
        assert_eq!(s12, s21, "ECDH asymmetric");
    }

    #[test]
    fn smoke_error_path() {
        let ctx = Context::new().unwrap();
        let zeroes = [0u8; 32];
        assert!(ctx.pubkey_create(&zeroes).is_err(), "zero key should fail");
    }

    #[test]
    fn smoke_ecdsa_deterministic() {
        let ctx = Context::new().unwrap();
        let msg = [0u8; 32];
        let sig1 = ctx.ecdsa_sign(&msg, &known_privkey()).unwrap();
        let sig2 = ctx.ecdsa_sign(&msg, &known_privkey()).unwrap();
        assert_eq!(sig1, sig2, "RFC 6979 non-deterministic");
    }
}
