
use k256::{
    ecdsa::{SigningKey, Signature, signature::hazmat::PrehashSigner},
    PublicKey, SecretKey,
    ecdh::diffie_hellman,
};
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Deserialize)]
struct Req { op: String, sk_hex: String, msg_hex: String, pk2_hex: String }

#[derive(Serialize)]
struct Resp { result_hex: String }

#[derive(Serialize)]
struct Err { error: String }

fn main() {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).unwrap();
    let req: Req = serde_json::from_str(&buf).unwrap();
    let sk_bytes = hex::decode(&req.sk_hex).unwrap();
    let sk = SigningKey::from_bytes(sk_bytes.as_slice().into()).unwrap();
    let pk = sk.verifying_key();

    match req.op.as_str() {
        "pubkey" => {
            let compressed = pk.to_encoded_point(true);
            let result_hex = hex::encode(compressed.as_bytes());
            println!("{}", serde_json::to_string(&Resp { result_hex }).unwrap());
        }
        "sign" => {
            let msg = hex::decode(&req.msg_hex).unwrap();
            let sig: Signature = sk.sign_prehash(&msg).unwrap();
            let bytes = sig.to_bytes();
            // low-S normalization
            let n_half: [u8; 32] = hex::decode(
                "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0"
            ).unwrap().try_into().unwrap();
            let s_bytes: [u8; 32] = bytes[32..].try_into().unwrap();
            let n: [u8; 32] = hex::decode(
                "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
            ).unwrap().try_into().unwrap();
            let result_hex = if s_bytes > n_half {
                // flip S = n - S
                let mut out = bytes.to_vec();
                let s_int = u256_from_be(&s_bytes);
                let n_int = u256_from_be(&n);
                let flipped = n_int - s_int;
                out[32..].copy_from_slice(&u256_to_be(flipped));
                hex::encode(&out)
            } else {
                hex::encode(bytes)
            };
            println!("{}", serde_json::to_string(&Resp { result_hex }).unwrap());
        }
        "ecdh" => {
            let pk2_bytes = hex::decode(&req.pk2_hex).unwrap();
            let pk2 = PublicKey::from_sec1_bytes(&pk2_bytes).unwrap();
            let shared = diffie_hellman(sk.as_nonzero_scalar(), pk2.as_affine());
            let hash = Sha256::digest(shared.raw_secret_bytes());
            println!("{}", serde_json::to_string(&Resp { result_hex: hex::encode(hash) }).unwrap());
        }
        _ => {}
    }
}

fn u256_from_be(bytes: &[u8; 32]) -> u128 {
    // simplified: only use low 128 bits for comparison
    u128::from_be_bytes(bytes[16..].try_into().unwrap())
}

fn u256_to_be(v: u128) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    out[16..].copy_from_slice(&v.to_be_bytes());
    out
}
