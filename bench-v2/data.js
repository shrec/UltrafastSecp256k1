window.BENCHMARK_DATA = {
  "lastUpdate": 1777658028418,
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
          "id": "b559ba7f178e5f73978eb7a5d5a692b4a831b1f4",
          "message": "fix(ci): benchmark-windows libsecp + Security Audit MSan timeout + SonarCloud\n\nbenchmark-windows: bench_unified.exe not found because libsecp256k1\nclone was removed from Windows step. bench_unified requires LIBSECP_SRC_DIR\nto build the comparison binary. Added clone back.\n\nSecurity Audit MSan: regression_bip324_session timed out under MSan\n10-20x overhead (7200s per-test limit exceeded). Added to exclusion list\nalongside other known-slow tests.\n\nSonarCloud: unicode_nfkd.cpp excluded from coverage — platform dispatcher\nwith three mutually exclusive branches (Windows/macOS/Linux). Only Linux\nbranch runs on CI; 2/3 of the file is structurally uncoverable.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-01T07:08:24Z",
          "tree_id": "1233056516afdb8a0ad262f08ee348c5b71acc77",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b559ba7f178e5f73978eb7a5d5a692b4a831b1f4"
        },
        "date": 1777621067551,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1053.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1331.3,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7939.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 32993.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 30774.3,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 35855,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1309.6,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 240.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 145.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 253.8,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1295.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 29445.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 89596.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 37959.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 22822.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 24270.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 68731.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38831.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 43911.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 170406.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 42601.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=4)",
            "value": 150387,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=4)",
            "value": 37596.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 682554.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 42659.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=16)",
            "value": 601304,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=16)",
            "value": 37581.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 2811522.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 43930,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=64)",
            "value": 2487904.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=64)",
            "value": 38873.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=128)",
            "value": 5072560.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=128)",
            "value": 39629.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=128)",
            "value": 4744410.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=128)",
            "value": 37065.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=192)",
            "value": 6761198.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=192)",
            "value": 35214.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=192)",
            "value": 6437998.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=192)",
            "value": 33531.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 144884.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 580851.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2331949.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=128)",
            "value": 4663108.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=192)",
            "value": 6992370.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1864.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 21439.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42000.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 144.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 275.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 273,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 29478.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 89607.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 24276.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 68747.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 22866.5,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 438.4,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 446.2,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 435.8,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 30676.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 30688.3,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 48166.5,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 31183.5,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 983.6,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute (SHA256 shared secret)",
            "value": 44819.6,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute_raw (x-only shared)",
            "value": 43447,
            "unit": "ns"
          },
          {
            "name": "taproot_output_key (BIP-341 key path)",
            "value": 17780.7,
            "unit": "ns"
          },
          {
            "name": "taproot_tweak_privkey (BIP-341)",
            "value": 24508.2,
            "unit": "ns"
          },
          {
            "name": "bip32_master_key (64B seed)",
            "value": 1360.8,
            "unit": "ns"
          },
          {
            "name": "bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)",
            "value": 123168.4,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (BTC end-to-end)",
            "value": 148311.4,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (ETH end-to-end)",
            "value": 148299.1,
            "unit": "ns"
          },
          {
            "name": "silent_payment_create_output",
            "value": 51649.8,
            "unit": "ns"
          },
          {
            "name": "silent_payment_scan (single output set)",
            "value": 74834.5,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1)",
            "value": 41225.1,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=16)",
            "value": 40151.2,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=64)",
            "value": 39634.9,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=256)",
            "value": 39562.6,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1024)",
            "value": 39589.7,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1132.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1848.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1157.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 150.6,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 246.9,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37088.8,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17895.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19885.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34465.3,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2530,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21377.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39472,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 389693.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414991.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 377688.6,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 65613.5,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 47166,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 41466.7,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 90436.3,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 108508.7,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24431493.3,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 2977502,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor sign",
            "value": 50794.4,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor verify",
            "value": 51204.3,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor adapt",
            "value": 22938.3,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor sign",
            "value": 89696.9,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor verify",
            "value": 64538.2,
            "unit": "ns"
          },
          {
            "name": "keygen_begin (DKG round 1)",
            "value": 44994.7,
            "unit": "ns"
          },
          {
            "name": "nonce_gen",
            "value": 44604.3,
            "unit": "ns"
          },
          {
            "name": "partial_sign",
            "value": 86286,
            "unit": "ns"
          },
          {
            "name": "partial_verify",
            "value": 158766.5,
            "unit": "ns"
          },
          {
            "name": "aggregate → Schnorr sig",
            "value": 81500.7,
            "unit": "ns"
          },
          {
            "name": "key_agg (BIP-327)",
            "value": 57008.4,
            "unit": "ns"
          },
          {
            "name": "sig_agg → Schnorr sig",
            "value": 1105,
            "unit": "ns"
          },
          {
            "name": "ECIES encrypt (256B payload)",
            "value": 101566,
            "unit": "ns"
          },
          {
            "name": "ECIES decrypt (256B payload)",
            "value": 83818.1,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message sign",
            "value": 31246.7,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message verify",
            "value": 36263.1,
            "unit": "ns"
          },
          {
            "name": "SHA-256 (32B input)",
            "value": 245.6,
            "unit": "ns"
          },
          {
            "name": "SHA-512 (32B input)",
            "value": 332.8,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (4 points)",
            "value": 90191.2,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (64 points)",
            "value": 1115114.8,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (12 words)",
            "value": 17378,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (24 words)",
            "value": 17635.2,
            "unit": "ns"
          },
          {
            "name": "bip39_validate (12 words)",
            "value": 1450.8,
            "unit": "ns"
          },
          {
            "name": "bip39_to_seed (PBKDF2, 12 words)",
            "value": 2674901.2,
            "unit": "ns"
          },
          {
            "name": "BIP-143 sighash (1-in/1-out)",
            "value": 1093.2,
            "unit": "ns"
          },
          {
            "name": "BIP-144 compute_wtxid",
            "value": 1381,
            "unit": "ns"
          },
          {
            "name": "BIP-144 witness_commitment",
            "value": 785,
            "unit": "ns"
          },
          {
            "name": "BIP-144 tx_weight",
            "value": 220,
            "unit": "ns"
          },
          {
            "name": "BIP-341 keypath_sighash",
            "value": 2471.2,
            "unit": "ns"
          },
          {
            "name": "BIP-342 tapscript_sighash",
            "value": 2723.2,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift create",
            "value": 184826.3,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift XDH (ECDH)",
            "value": 70312.7,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 extract",
            "value": 1023.7,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 expand",
            "value": 1012.3,
            "unit": "ns"
          },
          {
            "name": "AEAD encrypt (256B)",
            "value": 743.8,
            "unit": "ns"
          },
          {
            "name": "AEAD decrypt (256B)",
            "value": 755.7,
            "unit": "ns"
          },
          {
            "name": "Session handshake (full)",
            "value": 510302.3,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (256B)",
            "value": 914.9,
            "unit": "ns"
          },
          {
            "name": "Session decrypt (256B)",
            "value": 1835.4,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (1KB)",
            "value": 2726.9,
            "unit": "ns"
          },
          {
            "name": "Session roundtrip (256B)",
            "value": 1834.5,
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
          "id": "18fae057053f5085215a92360bfbf27560abb746",
          "message": "fix(ci): benchmark-windows libsecp + Security Audit MSan timeout + SonarCloud\n\nbenchmark-windows: bench_unified.exe not found because libsecp256k1\nclone was removed from Windows step. bench_unified requires LIBSECP_SRC_DIR\nto build the comparison binary. Added clone back.\n\nSecurity Audit MSan: regression_bip324_session timed out under MSan\n10-20x overhead (7200s per-test limit exceeded). Added to exclusion list\nalongside other known-slow tests.\n\nSonarCloud: unicode_nfkd.cpp excluded from coverage — platform dispatcher\nwith three mutually exclusive branches (Windows/macOS/Linux). Only Linux\nbranch runs on CI; 2/3 of the file is structurally uncoverable.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-01T07:08:02Z",
          "tree_id": "1233056516afdb8a0ad262f08ee348c5b71acc77",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/18fae057053f5085215a92360bfbf27560abb746"
        },
        "date": 1777621115339,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1053.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1331.5,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7927.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 32999.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 30754.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 35919.3,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1309.6,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 241.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 146.1,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 254.5,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1292.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 29392.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 89569.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 37960.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 22844.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 24259.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 68757.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38874.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 43929.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 170564.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 42641.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=4)",
            "value": 150421.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=4)",
            "value": 37605.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 682223.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 42639,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=16)",
            "value": 601260.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=16)",
            "value": 37578.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 2813634.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 43963,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=64)",
            "value": 2491089.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=64)",
            "value": 38923.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=128)",
            "value": 5074716.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=128)",
            "value": 39646.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=128)",
            "value": 4746431.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=128)",
            "value": 37081.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=192)",
            "value": 6766131.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=192)",
            "value": 35240.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=192)",
            "value": 6436854.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=192)",
            "value": 33525.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 145040.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 581686.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2334203.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=128)",
            "value": 4665044.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=192)",
            "value": 6997711.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1862.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 21415.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42147,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 143.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 275.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 273.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 29442.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 89597.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 24265.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 68822.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 22848.7,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 436.6,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 446.7,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 435.2,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 30652.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 30668.4,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 48218.4,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 31101.3,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 983.8,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute (SHA256 shared secret)",
            "value": 44793.1,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute_raw (x-only shared)",
            "value": 43503.6,
            "unit": "ns"
          },
          {
            "name": "taproot_output_key (BIP-341 key path)",
            "value": 17776.3,
            "unit": "ns"
          },
          {
            "name": "taproot_tweak_privkey (BIP-341)",
            "value": 24494.4,
            "unit": "ns"
          },
          {
            "name": "bip32_master_key (64B seed)",
            "value": 1358.5,
            "unit": "ns"
          },
          {
            "name": "bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)",
            "value": 124138.7,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (BTC end-to-end)",
            "value": 148574.3,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (ETH end-to-end)",
            "value": 148455.5,
            "unit": "ns"
          },
          {
            "name": "silent_payment_create_output",
            "value": 51685.4,
            "unit": "ns"
          },
          {
            "name": "silent_payment_scan (single output set)",
            "value": 74752.3,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1)",
            "value": 41299.6,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=16)",
            "value": 40074,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=64)",
            "value": 39701.5,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=256)",
            "value": 39668.7,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1024)",
            "value": 39686.7,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1133.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1848,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1160,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 246.7,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37074.8,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17906.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19893.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34549.2,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2533.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21394.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39371.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 389048.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416201.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375230.6,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 65702.7,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 47124,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 41450.1,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 90652.5,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 108609.5,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24442076,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 2984071.6,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor sign",
            "value": 50755,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor verify",
            "value": 51252.6,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor adapt",
            "value": 22935.7,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor sign",
            "value": 89701.6,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor verify",
            "value": 64635.1,
            "unit": "ns"
          },
          {
            "name": "keygen_begin (DKG round 1)",
            "value": 44860.9,
            "unit": "ns"
          },
          {
            "name": "nonce_gen",
            "value": 44497.9,
            "unit": "ns"
          },
          {
            "name": "partial_sign",
            "value": 86329.6,
            "unit": "ns"
          },
          {
            "name": "partial_verify",
            "value": 158787.1,
            "unit": "ns"
          },
          {
            "name": "aggregate → Schnorr sig",
            "value": 81416.5,
            "unit": "ns"
          },
          {
            "name": "key_agg (BIP-327)",
            "value": 57040.3,
            "unit": "ns"
          },
          {
            "name": "sig_agg → Schnorr sig",
            "value": 1162.9,
            "unit": "ns"
          },
          {
            "name": "ECIES encrypt (256B payload)",
            "value": 101517.9,
            "unit": "ns"
          },
          {
            "name": "ECIES decrypt (256B payload)",
            "value": 83907.1,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message sign",
            "value": 31199.7,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message verify",
            "value": 36274,
            "unit": "ns"
          },
          {
            "name": "SHA-256 (32B input)",
            "value": 243.4,
            "unit": "ns"
          },
          {
            "name": "SHA-512 (32B input)",
            "value": 333.3,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (4 points)",
            "value": 90262.4,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (64 points)",
            "value": 1116595.2,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (12 words)",
            "value": 17455.3,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (24 words)",
            "value": 17727.2,
            "unit": "ns"
          },
          {
            "name": "bip39_validate (12 words)",
            "value": 1386.2,
            "unit": "ns"
          },
          {
            "name": "bip39_to_seed (PBKDF2, 12 words)",
            "value": 2689825.4,
            "unit": "ns"
          },
          {
            "name": "BIP-143 sighash (1-in/1-out)",
            "value": 1059.8,
            "unit": "ns"
          },
          {
            "name": "BIP-144 compute_wtxid",
            "value": 1366.2,
            "unit": "ns"
          },
          {
            "name": "BIP-144 witness_commitment",
            "value": 776.9,
            "unit": "ns"
          },
          {
            "name": "BIP-144 tx_weight",
            "value": 213,
            "unit": "ns"
          },
          {
            "name": "BIP-341 keypath_sighash",
            "value": 2451.5,
            "unit": "ns"
          },
          {
            "name": "BIP-342 tapscript_sighash",
            "value": 2704.3,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift create",
            "value": 189465.7,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift XDH (ECDH)",
            "value": 61277.9,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 extract",
            "value": 1017.1,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 expand",
            "value": 1001.1,
            "unit": "ns"
          },
          {
            "name": "AEAD encrypt (256B)",
            "value": 741.4,
            "unit": "ns"
          },
          {
            "name": "AEAD decrypt (256B)",
            "value": 751.1,
            "unit": "ns"
          },
          {
            "name": "Session handshake (full)",
            "value": 517855.5,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (256B)",
            "value": 915.3,
            "unit": "ns"
          },
          {
            "name": "Session decrypt (256B)",
            "value": 1832.8,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (1KB)",
            "value": 2718.7,
            "unit": "ns"
          },
          {
            "name": "Session roundtrip (256B)",
            "value": 1834.1,
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
          "id": "96cea54b9f151a6493233b95a9df725a8f4de625",
          "message": "chore: canonical data — shim_api_function_count 66→67 (pubkey_sort added)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-01T16:18:06Z",
          "tree_id": "ac93813aa71856885521d74bd35a191b112fad8b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/96cea54b9f151a6493233b95a9df725a8f4de625"
        },
        "date": 1777654104115,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1057.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1329.3,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7946.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33003.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 30809.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 35852.5,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1310.3,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 241.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 146.8,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 254,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1296.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 29434.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 89587,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 37928,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 22853,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 24258.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 68759,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38821.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 43940.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 170462.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 42615.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=4)",
            "value": 150236.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=4)",
            "value": 37559.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 681976.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 42623.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=16)",
            "value": 600965.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=16)",
            "value": 37560.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 2810696.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 43917.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=64)",
            "value": 2487034.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=64)",
            "value": 38859.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=128)",
            "value": 5074385.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=128)",
            "value": 39643.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=128)",
            "value": 4742682.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=128)",
            "value": 37052.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=192)",
            "value": 6763197.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=192)",
            "value": 35225,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=192)",
            "value": 6433759.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=192)",
            "value": 33509.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 144754.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 581058.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2332702.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=128)",
            "value": 4668088.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=192)",
            "value": 6998285.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1863.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 21413.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42077.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 143.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 399.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 275.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 273,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 29479.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 89686.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 24277.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 68793.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 22841.6,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 436.4,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 446.8,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 435.1,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 30727.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 30684.7,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 48157.4,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 31185.2,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 983.8,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute (SHA256 shared secret)",
            "value": 44889.8,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute_raw (x-only shared)",
            "value": 43409.3,
            "unit": "ns"
          },
          {
            "name": "taproot_output_key (BIP-341 key path)",
            "value": 17764.2,
            "unit": "ns"
          },
          {
            "name": "taproot_tweak_privkey (BIP-341)",
            "value": 24494.5,
            "unit": "ns"
          },
          {
            "name": "bip32_master_key (64B seed)",
            "value": 1357.6,
            "unit": "ns"
          },
          {
            "name": "bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)",
            "value": 123152.4,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (BTC end-to-end)",
            "value": 148326.4,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (ETH end-to-end)",
            "value": 148262.6,
            "unit": "ns"
          },
          {
            "name": "silent_payment_create_output",
            "value": 51669.3,
            "unit": "ns"
          },
          {
            "name": "silent_payment_scan (single output set)",
            "value": 74778.6,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1)",
            "value": 41245.5,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=16)",
            "value": 40079.6,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=64)",
            "value": 39693.7,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=256)",
            "value": 39607.1,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1024)",
            "value": 39635.4,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1131.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1844.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1164.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 148.6,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 246.2,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37117,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17921.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19881.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34555.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2533.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21388.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39494.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 388934.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414587,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 377268.8,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 65606.5,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 47152.2,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 41452.3,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 90741,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 108477.3,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24439837.8,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 2978528.1,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor sign",
            "value": 50803.1,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor verify",
            "value": 51305.9,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor adapt",
            "value": 22927.1,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor sign",
            "value": 89640.6,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor verify",
            "value": 64678.1,
            "unit": "ns"
          },
          {
            "name": "keygen_begin (DKG round 1)",
            "value": 44904.4,
            "unit": "ns"
          },
          {
            "name": "nonce_gen",
            "value": 44510.4,
            "unit": "ns"
          },
          {
            "name": "partial_sign",
            "value": 86299.5,
            "unit": "ns"
          },
          {
            "name": "partial_verify",
            "value": 158887.7,
            "unit": "ns"
          },
          {
            "name": "aggregate → Schnorr sig",
            "value": 81431.6,
            "unit": "ns"
          },
          {
            "name": "key_agg (BIP-327)",
            "value": 57082.2,
            "unit": "ns"
          },
          {
            "name": "sig_agg → Schnorr sig",
            "value": 1115.4,
            "unit": "ns"
          },
          {
            "name": "ECIES encrypt (256B payload)",
            "value": 101749.5,
            "unit": "ns"
          },
          {
            "name": "ECIES decrypt (256B payload)",
            "value": 83929.4,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message sign",
            "value": 31276,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message verify",
            "value": 36261.3,
            "unit": "ns"
          },
          {
            "name": "SHA-256 (32B input)",
            "value": 245.9,
            "unit": "ns"
          },
          {
            "name": "SHA-512 (32B input)",
            "value": 333.5,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (4 points)",
            "value": 90263.7,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (64 points)",
            "value": 1116661.5,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (12 words)",
            "value": 17269.9,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (24 words)",
            "value": 17558.7,
            "unit": "ns"
          },
          {
            "name": "bip39_validate (12 words)",
            "value": 1367.2,
            "unit": "ns"
          },
          {
            "name": "bip39_to_seed (PBKDF2, 12 words)",
            "value": 2683305.4,
            "unit": "ns"
          },
          {
            "name": "BIP-143 sighash (1-in/1-out)",
            "value": 1083.3,
            "unit": "ns"
          },
          {
            "name": "BIP-144 compute_wtxid",
            "value": 1378.3,
            "unit": "ns"
          },
          {
            "name": "BIP-144 witness_commitment",
            "value": 784.3,
            "unit": "ns"
          },
          {
            "name": "BIP-144 tx_weight",
            "value": 213.7,
            "unit": "ns"
          },
          {
            "name": "BIP-341 keypath_sighash",
            "value": 2468.7,
            "unit": "ns"
          },
          {
            "name": "BIP-342 tapscript_sighash",
            "value": 2729.3,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift create",
            "value": 180174.5,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift XDH (ECDH)",
            "value": 70489.2,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 extract",
            "value": 1035.2,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 expand",
            "value": 1027.3,
            "unit": "ns"
          },
          {
            "name": "AEAD encrypt (256B)",
            "value": 757.2,
            "unit": "ns"
          },
          {
            "name": "AEAD decrypt (256B)",
            "value": 754.4,
            "unit": "ns"
          },
          {
            "name": "Session handshake (full)",
            "value": 525183.9,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (256B)",
            "value": 902.5,
            "unit": "ns"
          },
          {
            "name": "Session decrypt (256B)",
            "value": 1833.4,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (1KB)",
            "value": 2717.9,
            "unit": "ns"
          },
          {
            "name": "Session roundtrip (256B)",
            "value": 1835.1,
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
          "id": "34e3cd50de5288f2918d29ee4f153dcda33eea4f",
          "message": "security: fix all red-team audit findings (CRIT-1/2, HIGH-1/2/3, MED-1/2, INFO-1/2)\n\nCRIT-1: MuSig2 shim secnonce reuse blocks private key leak\n- sn_unpack: parse_bytes_strict → parse_bytes_strict_nonzero (k1, k2)\n- partial_sign keypair parsing: _strict → _strict_nonzero (MED-1)\n- nonce_gen seckey and session_id32: _strict → _strict_nonzero (MED-1)\n- partial_sign: fail-closed on zero psig return\n\nCRIT-2: Guardrail #4 — zero signatures no longer serialized as success\n- ufsecp_ecdsa_sign: is_valid() check + UFSECP_ERR_INTERNAL on zero sig\n- ufsecp_ecdsa_sign_verified: same\n- ufsecp_ecdsa_sign_recoverable: same\n- ufsecp_schnorr_sign: s.is_zero() check\n- ufsecp_schnorr_sign_verified: same\n- ufsecp_ecdsa_sign_batch: degenerate sig → clear all + ERR_INTERNAL\n- ufsecp_schnorr_sign_batch: same\n\nHIGH-1: secp256k1_schnorrsig_sign32 shim — CT path\n- Replace non-CT secp256k1::schnorr_sign(sk) with ct::schnorr_keypair_create\n  + ct::schnorr_sign (branchless scalar_cneg throughout)\n- Add zero-sig fail-closed check\n\nHIGH-2: shim DER parser — negative integer check\n- parse_der_int: add `if (*p & 0x80) return 0` (BIP-66 compliance)\n\nHIGH-3: shim DER parser — trailing bytes rejected\n- secp256k1_ecdsa_signature_parse_der: `> end` → `!= end`\n\nMED-2: sign_custom CT nonce branch\n- shim_schnorr.cpp sign_custom: r_y_odd ternary → ct::scalar_cneg\n\nCAAS: new exploit PoC tests wired (237/237)\n- test_exploit_shim_der_bip66.cpp (DER66-1..8)\n- test_exploit_shim_musig_secnonce.cpp (MSN-1..6)\n- BFC-9 in test_exploit_bug004_batch_failclosed.cpp\n\nDocs: AUDIT_CHANGELOG.md, DER_PARITY_MATRIX.md, BITCOIN_CORE_PR_BLOCKERS.md,\nEXPLOIT_TEST_CATALOG.md, canonical_data.json, AUDIT_SCOPE.md all updated.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-01T17:23:59Z",
          "tree_id": "e434fb0ad85422bd316716935b8cf219606a577e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/34e3cd50de5288f2918d29ee4f153dcda33eea4f"
        },
        "date": 1777658013822,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1049.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1427,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 8492.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35849.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34675.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39742.6,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1353.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 267.6,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 163.8,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 283.4,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1344.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 32476.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 98866.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41611.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 25080.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 26659.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 74823.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41836.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 47314.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 188943.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 47235.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=4)",
            "value": 167172.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=4)",
            "value": 41793.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 754934.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 47183.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=16)",
            "value": 668190.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=16)",
            "value": 41761.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3027099.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 47298.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=64)",
            "value": 2679620.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=64)",
            "value": 41869.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=128)",
            "value": 5624030.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=128)",
            "value": 43937.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=128)",
            "value": 5278573.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=128)",
            "value": 41238.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=192)",
            "value": 7515624.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=192)",
            "value": 39143.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(cached,N=192)",
            "value": 7165379.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig cached (N=192)",
            "value": 37319.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 161627.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 647883.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2583110.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=128)",
            "value": 5167674.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=192)",
            "value": 7747354.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 2064.8,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 23825.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 46813.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 162.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 440,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 305,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 299.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 32476.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 98806.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 26643.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 75466.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 25016.3,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 478.7,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 493.4,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 468.3,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 33650.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 33646.4,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 52120.8,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 34163.5,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 1076.8,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute (SHA256 shared secret)",
            "value": 48331,
            "unit": "ns"
          },
          {
            "name": "ecdh_compute_raw (x-only shared)",
            "value": 47890.3,
            "unit": "ns"
          },
          {
            "name": "taproot_output_key (BIP-341 key path)",
            "value": 19760.6,
            "unit": "ns"
          },
          {
            "name": "taproot_tweak_privkey (BIP-341)",
            "value": 27079,
            "unit": "ns"
          },
          {
            "name": "bip32_master_key (64B seed)",
            "value": 1502.2,
            "unit": "ns"
          },
          {
            "name": "bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)",
            "value": 136370.2,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (BTC end-to-end)",
            "value": 164203.7,
            "unit": "ns"
          },
          {
            "name": "coin_address_from_seed (ETH end-to-end)",
            "value": 164153.6,
            "unit": "ns"
          },
          {
            "name": "silent_payment_create_output",
            "value": 57264.1,
            "unit": "ns"
          },
          {
            "name": "silent_payment_scan (single output set)",
            "value": 82837.3,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1)",
            "value": 46015.3,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=16)",
            "value": 44927,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=64)",
            "value": 44473.2,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=256)",
            "value": 44416.8,
            "unit": "ns"
          },
          {
            "name": "fast_scan_batch /tx (N=1024)",
            "value": 44450.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1198.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2310.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1237.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 161.6,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 281.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 41272,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 19756.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 22117.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 38741.8,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 3057.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 23681.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 43917.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 421549.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 449395.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 406129.7,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 70931.7,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 51116.2,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 46123.9,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 98329.3,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 121062.3,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 27321382.9,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 3316844.2,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor sign",
            "value": 54937.7,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor verify",
            "value": 57073.9,
            "unit": "ns"
          },
          {
            "name": "Schnorr adaptor adapt",
            "value": 25422.4,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor sign",
            "value": 99100.4,
            "unit": "ns"
          },
          {
            "name": "ECDSA adaptor verify",
            "value": 71752.8,
            "unit": "ns"
          },
          {
            "name": "keygen_begin (DKG round 1)",
            "value": 49985.2,
            "unit": "ns"
          },
          {
            "name": "nonce_gen",
            "value": 49567,
            "unit": "ns"
          },
          {
            "name": "partial_sign",
            "value": 94752.7,
            "unit": "ns"
          },
          {
            "name": "partial_verify",
            "value": 175537.9,
            "unit": "ns"
          },
          {
            "name": "aggregate → Schnorr sig",
            "value": 89523.2,
            "unit": "ns"
          },
          {
            "name": "key_agg (BIP-327)",
            "value": 63552.2,
            "unit": "ns"
          },
          {
            "name": "sig_agg → Schnorr sig",
            "value": 1171.6,
            "unit": "ns"
          },
          {
            "name": "ECIES encrypt (256B payload)",
            "value": 114462.4,
            "unit": "ns"
          },
          {
            "name": "ECIES decrypt (256B payload)",
            "value": 94772.7,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message sign",
            "value": 34267.9,
            "unit": "ns"
          },
          {
            "name": "Bitcoin message verify",
            "value": 40349.2,
            "unit": "ns"
          },
          {
            "name": "SHA-256 (32B input)",
            "value": 278.6,
            "unit": "ns"
          },
          {
            "name": "SHA-512 (32B input)",
            "value": 365.5,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (4 points)",
            "value": 100313.6,
            "unit": "ns"
          },
          {
            "name": "Multi-scalar mul (64 points)",
            "value": 1236024.7,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (12 words)",
            "value": 20159,
            "unit": "ns"
          },
          {
            "name": "bip39_generate (24 words)",
            "value": 20431.4,
            "unit": "ns"
          },
          {
            "name": "bip39_validate (12 words)",
            "value": 1463.6,
            "unit": "ns"
          },
          {
            "name": "bip39_to_seed (PBKDF2, 12 words)",
            "value": 3007524.6,
            "unit": "ns"
          },
          {
            "name": "BIP-143 sighash (1-in/1-out)",
            "value": 1225.8,
            "unit": "ns"
          },
          {
            "name": "BIP-144 compute_wtxid",
            "value": 1514.9,
            "unit": "ns"
          },
          {
            "name": "BIP-144 witness_commitment",
            "value": 889.8,
            "unit": "ns"
          },
          {
            "name": "BIP-144 tx_weight",
            "value": 155.6,
            "unit": "ns"
          },
          {
            "name": "BIP-341 keypath_sighash",
            "value": 2763,
            "unit": "ns"
          },
          {
            "name": "BIP-342 tapscript_sighash",
            "value": 3049.2,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift create",
            "value": 202133.2,
            "unit": "ns"
          },
          {
            "name": "ElligatorSwift XDH (ECDH)",
            "value": 78344,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 extract",
            "value": 1170.8,
            "unit": "ns"
          },
          {
            "name": "HKDF-SHA256 expand",
            "value": 1152.1,
            "unit": "ns"
          },
          {
            "name": "AEAD encrypt (256B)",
            "value": 851.1,
            "unit": "ns"
          },
          {
            "name": "AEAD decrypt (256B)",
            "value": 871.4,
            "unit": "ns"
          },
          {
            "name": "Session handshake (full)",
            "value": 573149.6,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (256B)",
            "value": 1013,
            "unit": "ns"
          },
          {
            "name": "Session decrypt (256B)",
            "value": 2064.8,
            "unit": "ns"
          },
          {
            "name": "Session encrypt (1KB)",
            "value": 3114.7,
            "unit": "ns"
          },
          {
            "name": "Session roundtrip (256B)",
            "value": 2087.2,
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