# Layer Routing Matrix

> ABI function → FAST / CT layer routing rationale

## Routing rule

Every `ufsecp_*` ABI function is **statically routed** at compile time to either
the **CT (constant-time)** layer or the **FAST** layer.  The routing decision is
based on a single criterion:

> **Does the function receive, produce, or internally manipulate a secret scalar?**
>
> - **Yes → CT layer**: branchless arithmetic, `value_barrier()`, `secure_erase()`
> - **No  → FAST layer**: optimized with branches, early exits, and SIMD

The routing is enforced in `include/ufsecp/ufsecp_impl.cpp` and recorded in the
project graph's `abi_routing` table.

## CT-routed functions (secret-handling)

These functions touch private keys, nonces, or secret blinding factors.
All use the `secp256k1::ct::` namespace internally.

| ABI Function | Internal Call | Rationale |
|---|---|---|
| `ufsecp_ecdsa_sign` | `ct::ecdsa_sign(msg, sk)` | Signs with secret key |
| `ufsecp_ecdsa_sign_recoverable` | `ct::ecdsa_sign + recovery_id` | Signs with secret key + recid |
| `ufsecp_ecdsa_sign_verified` | `ct::ecdsa_sign + ecdsa_verify` | Sign + self-verify |
| `ufsecp_schnorr_sign` | `ct::schnorr_sign(msg, sk)` | Schnorr sign with secret key |
| `ufsecp_schnorr_sign_verified` | `ct::schnorr_sign + schnorr_verify` | Sign + self-verify |
| `ufsecp_schnorr_keypair` | `generate_schnorr_keypair(sk)` | Keypair from secret |
| `ufsecp_pubkey_create` | `ct::generator_mul(sk)` | G×sk |
| `ufsecp_pubkey_create_uncompressed` | `ct::generator_mul(sk)` | G×sk uncompressed |
| `ufsecp_pubkey_xonly` | `schnorr_pubkey(sk)` | x-only from secret |
| `ufsecp_seckey_verify` | `Scalar::parse_bytes_strict_nonzero` | Secret key validation |
| `ufsecp_seckey_negate` | `Scalar::negate` | Secret key operation |
| `ufsecp_seckey_tweak_add` | `ct scalar add + validate` | Secret key tweak |
| `ufsecp_seckey_tweak_mul` | `ct scalar mul + validate` | Secret key tweak |
| `ufsecp_ecdh` | `ct::scalar_mul(pubkey, sk)` | ECDH with secret |
| `ufsecp_ecdh_raw` | `ct::scalar_mul + raw output` | Raw ECDH |
| `ufsecp_ecdh_xonly` | `ct::scalar_mul + x-only output` | x-only ECDH |
| `ufsecp_bip32_derive` | `CKD_priv or CKD_pub` | Child key derivation |
| `ufsecp_bip32_derive_path` | `multi-level CKD` | Path derivation |
| `ufsecp_bip32_master` | `HMAC-SHA512(seed)` | Master key from seed |
| `ufsecp_bip32_privkey` | `ExtendedKey::privkey()` | Private key access |
| `ufsecp_bip39_generate` | `bip39_generate(strength)` | Mnemonic generation (entropy) |
| `ufsecp_bip39_to_seed` | `PBKDF2-SHA512(mnemonic, passphrase)` | Seed derivation |
| `ufsecp_musig2_nonce_gen` | `musig2_nonce_gen(sk)` | Nonce generation with secret |
| `ufsecp_musig2_partial_sign` | `ct::musig2_partial_sign(sk)` | Partial signature |
| `ufsecp_musig2_start_sign_session` | `musig2_session_init` | Session with secret state |
| `ufsecp_frost_keygen_begin` | `frost_keygen_begin` | Key generation |
| `ufsecp_frost_sign` | `ct::frost_sign(sk, nonce)` | FROST sign |
| `ufsecp_frost_sign_nonce_gen` | `frost_sign_nonce_gen` | Nonce gen |
| `ufsecp_ecdsa_adaptor_sign` | `ct::ecdsa_adaptor_sign(sk)` | Adaptor sign |
| `ufsecp_ecdsa_adaptor_adapt` | `ecdsa_adaptor_adapt` | Adaptor adapt (reveals secret) |
| `ufsecp_schnorr_adaptor_sign` | `ct::adaptor_sign(sk)` | Adaptor sign |
| `ufsecp_schnorr_adaptor_adapt` | `adaptor_adapt(pre_sig, secret)` | Adaptor adapt |
| `ufsecp_ecies_decrypt` | `ecies_decrypt(sk, ciphertext)` | Decrypt with secret |
| `ufsecp_ecies_encrypt` | `ecies_encrypt(pubkey, msg)` | Ephemeral key inside |
| `ufsecp_eth_sign` | `ct::ecdsa_sign(keccak(msg), sk) + v` | Ethereum sign |
| `ufsecp_btc_message_sign` | `btc_message_sign(msg, sk)` | Bitcoin message sign |
| `ufsecp_coin_hd_derive` | `coin_hd_derive(coin, xprv, path)` | Coin-specific derivation |
| `ufsecp_silent_payment_create_output` | `silent_payment_create_output` | BIP-352 (secret scan key) |
| `ufsecp_silent_payment_scan` | `silent_payment_scan` | BIP-352 scan |
| `ufsecp_taproot_tweak_seckey` | `taproot_tweak_seckey(sk, merkle)` | Secret key tweak |
| `ufsecp_pedersen_blind_sum` | `blind factor sum` | Blinding factors |
| `ufsecp_zk_dleq_prove` | `dleq_prove(sk)` | ZK prove with secret |
| `ufsecp_zk_knowledge_prove` | `prove_knowledge(sk)` | ZK prove with secret |
| `ufsecp_zk_range_proof_create` | `create_range_proof` | Range proof (blinding) |

## FAST-routed functions (public-data only)

These functions operate on public keys, signatures, proofs, or addresses.
No secret data flows through them.

| ABI Function | Internal Call | Rationale |
|---|---|---|
| `ufsecp_ecdsa_verify` | `ecdsa_verify(msg, pubkey, sig)` | Public verification |
| `ufsecp_schnorr_verify` | `schnorr_verify(msg, xpub, sig)` | Public verification |
| `ufsecp_ecdsa_batch_verify` | `batch_ecdsa_verify(entries)` | Batch public verify |
| `ufsecp_schnorr_batch_verify` | `batch_schnorr_verify(entries)` | Batch public verify |
| `ufsecp_ecdsa_recover` | `ecrecover(msg, sig, v)` | Recover pubkey from sig |
| `ufsecp_eth_recover` | `ecrecover(keccak(msg), sig, v)` | ETH ecrecover |
| `ufsecp_pubkey_parse` | `point_from_compressed + on_curve` | Parse public key |
| `ufsecp_pubkey_add` | `Point::add` | Public point arithmetic |
| `ufsecp_pubkey_combine` | `multi-point add` | Public point combine |
| `ufsecp_pubkey_negate` | `Point::negate` | Public point negate |
| `ufsecp_pubkey_tweak_add` | `Point::add(gen_mul(tweak))` | Public key tweak |
| `ufsecp_pubkey_tweak_mul` | `Point::scalar_mul(tweak)` | Public key tweak |
| `ufsecp_multi_scalar_mul` | `pippenger_msm(points, scalars)` | Public MSM |
| `ufsecp_shamir_trick` | `shamir_trick(a*G + b*P)` | Public double mul |
| `ufsecp_musig2_key_agg` | `musig2_key_agg(pubkeys)` | Public key aggregation |
| `ufsecp_musig2_nonce_agg` | `musig2_nonce_agg(nonces)` | Nonce aggregation |
| `ufsecp_musig2_partial_verify` | `musig2_partial_verify` | Public verify |
| `ufsecp_musig2_aggregate` | `musig2_aggregate(partials)` | Signature aggregation |
| `ufsecp_frost_keygen_finalize` | `frost_keygen_finalize` | Public key finalize |
| `ufsecp_frost_verify_partial` | `frost_verify_partial` | Public verify |
| `ufsecp_frost_aggregate` | `frost_aggregate(partials)` | Signature aggregation |
| `ufsecp_ecdsa_adaptor_verify` | `ecdsa_adaptor_verify` | Public verify |
| `ufsecp_ecdsa_adaptor_extract` | `ecdsa_adaptor_extract` | Extract from sigs |
| `ufsecp_schnorr_adaptor_verify` | `adaptor_verify` | Public verify |
| `ufsecp_schnorr_adaptor_extract` | `adaptor_extract(sig, pre_sig)` | Extract secret |
| `ufsecp_sha256` | `SHA256::digest(data, len)` | Hash (no secrets) |
| `ufsecp_sha512` | `SHA512::digest(data, len)` | Hash |
| `ufsecp_keccak256` | `keccak256(data)` | Hash |
| `ufsecp_hash160` | `SHA256+RIPEMD160` | Hash |
| `ufsecp_tagged_hash` | `tagged_hash(tag, msg)` | BIP-340 tagged hash |
| `ufsecp_addr_p2pkh` | `hash160 + base58check` | Address encoding |
| `ufsecp_addr_p2wpkh` | `hash160 + bech32` | Address encoding |
| `ufsecp_addr_p2tr` | `taproot_output_key + bech32m` | Address encoding |
| `ufsecp_eth_address` | `keccak256(pubkey)[12:]` | ETH address |
| `ufsecp_bip32_pubkey` | `ExtendedKey::pubkey()` | Public key access |
| `ufsecp_bip39_validate` | `bip39_validate(mnemonic)` | Mnemonic validation |
| `ufsecp_bip39_to_entropy` | `bip39_to_entropy(mnemonic)` | Decode (no secret) |
| `ufsecp_pedersen_commit` | `pedersen_commit(v, r)` | Public commitment |
| `ufsecp_pedersen_verify` | `pedersen_verify(C, v, r)` | Public verify |
| `ufsecp_pedersen_verify_tally` | `verify_tally(inputs, outputs)` | Public verify |
| `ufsecp_pedersen_switch_commit` | `switch_commit(v, r)` | Public commitment |
| `ufsecp_zk_dleq_verify` | `dleq_verify(proof)` | Public verify |
| `ufsecp_zk_knowledge_verify` | `verify_knowledge(proof, P)` | Public verify |
| `ufsecp_zk_range_proof_verify` | `verify_range_proof` | Public verify |
| `ufsecp_taproot_output_key` | `taproot_output_key(xpub, merkle)` | Public key |
| `ufsecp_taproot_verify` | `taproot_verify(xpub, merkle, output)` | Public verify |
| `ufsecp_wif_encode` | `base58check(privkey)` | Encoding (no CT needed) |
| `ufsecp_wif_decode` | `base58check_decode + validate` | Decoding |
| `ufsecp_ecdsa_sig_from_der` | `DER parse` | Signature parsing |
| `ufsecp_ecdsa_sig_to_der` | `ECDSASignature::to_der` | Signature encoding |
| `ufsecp_ecdsa_batch_identify_invalid` | `bisection invalid finder` | Public utility |
| `ufsecp_schnorr_batch_identify_invalid` | `bisection invalid finder` | Public utility |
| `ufsecp_btc_message_verify` | `btc_message_verify(msg, sig, addr)` | Public verify |
| `ufsecp_ctx_clone` | `memcpy ctx` | Context copy |
| `ufsecp_last_error` | `ctx->last_err` | Error retrieval |
| `ufsecp_coin_params` | `get coin_params(coin_id)` | Static lookup |
| `ufsecp_coin_address` | `coin_address(coin, pubkey)` | Address encoding |
| `ufsecp_coin_address_validate` | `coin_address_validate(coin, addr)` | Address validation |
| `ufsecp_silent_payment_verify_label` | `silent_payment_verify_label` | Public verify |
| `ufsecp_eth_eip55_checksum` | `eip55_checksum(addr)` | Checksum encoding |
| `ufsecp_eth_typed_data_hash` | `eip712_hash(domain, msg)` | Hash |

## Special cases

| ABI Function | Layer | Note |
|---|---|---|
| `ufsecp_ctx_create` | both | Runs self-test + allocator; no secrets but initializes CT infrastructure |

## How to verify

```bash
# From project graph
sqlite3 .project_graph.db "SELECT abi_function, layer FROM abi_routing ORDER BY layer, abi_function"

# From source graph
python3 tools/source_graph_kit/source_graph.py find "abi_routing"
```
