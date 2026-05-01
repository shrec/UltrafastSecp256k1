window.BENCHMARK_DATA = {
  "lastUpdate": 1777598827286,
  "repoUrl": "https://github.com/shrec/UltrafastSecp256k1",
  "entries": {
    "UltrafastSecp256k1 Performance": [
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "edcaa4c732bc957a1baf62736024673ddb86a313",
          "message": "fix(benchmark): store data on gh-pages not dev — stops Pipeline cancellations\n\nbenchmark-action/github-action-benchmark was auto-pushing benchmark data\ncommits to the dev branch (benchmark-data-dir-path: 'dev/bench-v2' wrote\nto the current branch, not gh-pages). Each benchmark run created a new\ncommit on dev, which triggered cancel-in-progress on the Pipeline workflow\nand cancelled Preflight, Phase 1 gates, and other jobs ~10 minutes in.\n\nFix: explicit gh-pages-branch: gh-pages + benchmark-data-dir-path: 'bench-v2'.\nData now lands in gh-pages/bench-v2 — no commit to dev, no Pipeline cancel.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-30T20:37:01Z",
          "tree_id": "7d0565b58a9a0a8d1997dd8d248a340eb03dee1b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/edcaa4c732bc957a1baf62736024673ddb86a313"
        },
        "date": 1777583252692,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1142.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1331.3,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7929.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 32982.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 30783.7,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 35871.9,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1307.9,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 241.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 146.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 254.2,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1289.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 29407,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 89654,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38010.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 22877.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 24262.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 68784.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38848.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 43941,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 170575.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 42643.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=4)",
            "value": 150364.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=4)",
            "value": 37591.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 681994.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 42624.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=16)",
            "value": 600885,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=16)",
            "value": 37555.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 2816399.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 44006.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=64)",
            "value": 2488498.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=64)",
            "value": 38882.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=128)",
            "value": 5075098.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=128)",
            "value": 39649.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=128)",
            "value": 4744337.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=128)",
            "value": 37065.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=192)",
            "value": 6765467.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=192)",
            "value": 35236.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=192)",
            "value": 6437145.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=192)",
            "value": 33526.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 145048.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 581057.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2333207,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=128)",
            "value": 4667836.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=192)",
            "value": 6989114.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1863.5,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 21461.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42135.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 144.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 275.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 273.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 29420.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 89675.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 24272.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 68754.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 22835.7,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 441.7,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 449.9,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 435.2,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 30630.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 30638.5,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 48229.5,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 31125.3,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 985.9,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute (SHA256 shared secret)",
            "value": 45622.2,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute_raw (x-only shared)",
            "value": 44183,
            "unit": "ns"
          },
          {
            "name": "taproot_output_key (BIP-341 key path)",
            "value": 17774,
            "unit": "ns"
          },
          {
            "name": "taproot_tweak_privkey (BIP-341)",
            "value": 24483.1,
            "unit": "ns"
          },
          {
            "name": "bip32_master_key (64B seed)",
            "value": 1367.1,
            "unit": "ns"
          },
          {
            "name": "bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)",
            "value": 123352.1,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (BTC end-to-end)",
            "value": 148674.3,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (ETH end-to-end)",
            "value": 148508.2,
            "unit": "ns"
          },
          {
            "name": "silent_payment_create_output",
            "value": 51668.7,
            "unit": "ns"
          },
          {
            "name": "silent_payment_scan (single output set)",
            "value": 74870.9,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1)",
            "value": 41243.4,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=16)",
            "value": 40068.2,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=64)",
            "value": 39677.3,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=256)",
            "value": 39592.7,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1024)",
            "value": 39596.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1132.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1844.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1162.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 149.6,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 247,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37141.8,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17914.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19882.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 35291.2,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2531.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21387.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39476,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 389278.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 413071.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375308.5,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 65820.5,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 47211.6,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 41464.7,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 90721,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 108607.6,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24454308.2,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 2979376.2,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor sign",
            "value": 50775,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor verify",
            "value": 51252.9,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor adapt",
            "value": 22932.3,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor sign",
            "value": 89719.8,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor verify",
            "value": 64752,
            "unit": "ns"
          },
          {
            "name": "keygen_begin (DKG round 1)",
            "value": 44860.4,
            "unit": "ns"
          },
          {
            "name": "nonce_gen",
            "value": 44492.1,
            "unit": "ns"
          },
          {
            "name": "partial_sign",
            "value": 86219.1,
            "unit": "ns"
          },
          {
            "name": "partial_verify",
            "value": 158776.6,
            "unit": "ns"
          },
          {
            "name": "aggregate → Schnorr sig",
            "value": 81423,
            "unit": "ns"
          },
          {
            "name": "key_agg (BIP-327)",
            "value": 57023.8,
            "unit": "ns"
          },
          {
            "name": "sig_agg → Schnorr sig",
            "value": 1097.1,
            "unit": "ns"
          },
          {
            "name": "ECIES encrypt (256B payload)",
            "value": 101706.4,
            "unit": "ns"
          },
          {
            "name": "ECIES decrypt (256B payload)",
            "value": 83991.9,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message sign",
            "value": 31259,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message verify",
            "value": 36256.6,
            "unit": "ns"
          },
          {
            "name": "SHA-256 (32B input)",
            "value": 243.2,
            "unit": "ns"
          },
          {
            "name": "SHA-512 (32B input)",
            "value": 333.7,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (4 points)",
            "value": 90260.9,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (64 points)",
            "value": 1118505.6,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (12 words)",
            "value": 17366.3,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (24 words)",
            "value": 17623.6,
            "unit": "ns"
          },
          {
            "name": "bip39_validate (12 words)",
            "value": 1429.1,
            "unit": "ns"
          },
          {
            "name": "bip39_to_seed (PBKDF2, 12 words)",
            "value": 2683923.4,
            "unit": "ns"
          },
          {
            "name": "BIP-143 sighash (1-in/1-out)",
            "value": 1086.8,
            "unit": "ns"
          },
          {
            "name": "BIP-144 compute_wtxid",
            "value": 1370.2,
            "unit": "ns"
          },
          {
            "name": "BIP-144 witness_commitment",
            "value": 780.6,
            "unit": "ns"
          },
          {
            "name": "BIP-144 tx_weight",
            "value": 213.1,
            "unit": "ns"
          },
          {
            "name": "BIP-341 keypath_sighash",
            "value": 2474.3,
            "unit": "ns"
          },
          {
            "name": "BIP-342 tapscript_sighash",
            "value": 2703.7,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift create",
            "value": 183405.9,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift XDH (ECDH)",
            "value": 61337.3,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 extract",
            "value": 1029,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 expand",
            "value": 1012.5,
            "unit": "ns"
          },
          {
            "name": "AEAD encrypt (256B)",
            "value": 741.6,
            "unit": "ns"
          },
          {
            "name": "AEAD decrypt (256B)",
            "value": 754.9,
            "unit": "ns"
          },
          {
            "name": "Session handshake (full)",
            "value": 514464,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (256B)",
            "value": 903.3,
            "unit": "ns"
          },
          {
            "name": "Session decrypt (256B)",
            "value": 1849.7,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (1KB)",
            "value": 2717.5,
            "unit": "ns"
          },
          {
            "name": "Session roundtrip (256B)",
            "value": 1849.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "b45d6e10939f1a9aac78208294a6d5c17295281d",
          "message": "release: merge dev→main (v3.68.0)",
          "timestamp": "2026-05-01T00:52:18Z",
          "tree_id": "439771f8f26cddbaee8b8fc78cb5b5a4aced6474",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b45d6e10939f1a9aac78208294a6d5c17295281d"
        },
        "date": 1777598812016,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1055.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1331.6,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7950.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33066.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 29981.7,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 35896.7,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1309.7,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 240.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 146.3,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 254.2,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1271.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 29397.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 89480.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 37966.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 22826.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 23258.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 68689.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38859,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 43981.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 170434.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 42608.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=4)",
            "value": 150124.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=4)",
            "value": 37531.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 680868.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 42554.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=16)",
            "value": 600637.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=16)",
            "value": 37539.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 2811826.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 43934.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=64)",
            "value": 2478763.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=64)",
            "value": 38730.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=128)",
            "value": 5072442.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=128)",
            "value": 39628.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=128)",
            "value": 4741392.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=128)",
            "value": 37042.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=192)",
            "value": 6655011.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=192)",
            "value": 34661.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=192)",
            "value": 6435912.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=192)",
            "value": 33520.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 144736.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 581822.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2329433.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=128)",
            "value": 4666007.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=192)",
            "value": 6986959.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1863.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 21395.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42058.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 144.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 275.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 273.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 29407.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 89490.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 24259.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 68756.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 22850.8,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 436.9,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 446.8,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 435,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 30613.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 30651.5,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 48176.7,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 31144,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 986.7,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute (SHA256 shared secret)",
            "value": 44812,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute_raw (x-only shared)",
            "value": 43494.2,
            "unit": "ns"
          },
          {
            "name": "taproot_output_key (BIP-341 key path)",
            "value": 17777.7,
            "unit": "ns"
          },
          {
            "name": "taproot_tweak_privkey (BIP-341)",
            "value": 24498.5,
            "unit": "ns"
          },
          {
            "name": "bip32_master_key (64B seed)",
            "value": 1360.3,
            "unit": "ns"
          },
          {
            "name": "bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)",
            "value": 123969.8,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (BTC end-to-end)",
            "value": 148327.1,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (ETH end-to-end)",
            "value": 148376.2,
            "unit": "ns"
          },
          {
            "name": "silent_payment_create_output",
            "value": 51613.7,
            "unit": "ns"
          },
          {
            "name": "silent_payment_scan (single output set)",
            "value": 74757,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1)",
            "value": 41242.2,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=16)",
            "value": 40054.1,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=64)",
            "value": 39368.1,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=256)",
            "value": 39152.9,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1024)",
            "value": 39220.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1137.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1848,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1154.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 149.4,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 246.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37117.3,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17902.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19878.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34610.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2531.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 20274.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39371.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 388401,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 411101.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375852.8,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 65698.3,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 47150.9,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 41443.4,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 89112.5,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 108635.5,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24445497.1,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 2982965.9,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor sign",
            "value": 49220,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor verify",
            "value": 50899.2,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor adapt",
            "value": 22939.9,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor sign",
            "value": 89654.3,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor verify",
            "value": 64625.8,
            "unit": "ns"
          },
          {
            "name": "keygen_begin (DKG round 1)",
            "value": 44869.3,
            "unit": "ns"
          },
          {
            "name": "nonce_gen",
            "value": 44483.4,
            "unit": "ns"
          },
          {
            "name": "partial_sign",
            "value": 86243.6,
            "unit": "ns"
          },
          {
            "name": "partial_verify",
            "value": 158806.5,
            "unit": "ns"
          },
          {
            "name": "aggregate → Schnorr sig",
            "value": 81321.9,
            "unit": "ns"
          },
          {
            "name": "key_agg (BIP-327)",
            "value": 57073.5,
            "unit": "ns"
          },
          {
            "name": "sig_agg → Schnorr sig",
            "value": 1149.9,
            "unit": "ns"
          },
          {
            "name": "ECIES encrypt (256B payload)",
            "value": 101628.8,
            "unit": "ns"
          },
          {
            "name": "ECIES decrypt (256B payload)",
            "value": 83841.2,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message sign",
            "value": 31188.8,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message verify",
            "value": 36236.4,
            "unit": "ns"
          },
          {
            "name": "SHA-256 (32B input)",
            "value": 243.5,
            "unit": "ns"
          },
          {
            "name": "SHA-512 (32B input)",
            "value": 334.2,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (4 points)",
            "value": 90170.4,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (64 points)",
            "value": 1114064,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (12 words)",
            "value": 17389.5,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (24 words)",
            "value": 17611.7,
            "unit": "ns"
          },
          {
            "name": "bip39_validate (12 words)",
            "value": 1401,
            "unit": "ns"
          },
          {
            "name": "bip39_to_seed (PBKDF2, 12 words)",
            "value": 2682552.9,
            "unit": "ns"
          },
          {
            "name": "BIP-143 sighash (1-in/1-out)",
            "value": 1061.4,
            "unit": "ns"
          },
          {
            "name": "BIP-144 compute_wtxid",
            "value": 1358.1,
            "unit": "ns"
          },
          {
            "name": "BIP-144 witness_commitment",
            "value": 780.5,
            "unit": "ns"
          },
          {
            "name": "BIP-144 tx_weight",
            "value": 213.4,
            "unit": "ns"
          },
          {
            "name": "BIP-341 keypath_sighash",
            "value": 2450.1,
            "unit": "ns"
          },
          {
            "name": "BIP-342 tapscript_sighash",
            "value": 2728.2,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift create",
            "value": 181548.1,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift XDH (ECDH)",
            "value": 61397.8,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 extract",
            "value": 1028.4,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 expand",
            "value": 1014.3,
            "unit": "ns"
          },
          {
            "name": "AEAD encrypt (256B)",
            "value": 742,
            "unit": "ns"
          },
          {
            "name": "AEAD decrypt (256B)",
            "value": 759.9,
            "unit": "ns"
          },
          {
            "name": "Session handshake (full)",
            "value": 509065.1,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (256B)",
            "value": 903.9,
            "unit": "ns"
          },
          {
            "name": "Session decrypt (256B)",
            "value": 1848.4,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (1KB)",
            "value": 2720.6,
            "unit": "ns"
          },
          {
            "name": "Session roundtrip (256B)",
            "value": 1851,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}