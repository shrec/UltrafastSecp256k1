/// UltrafastSecp256k1 -- Rust Example (CPU + GPU)
///
/// Demonstrates the safe Rust wrapper (CPU) and raw FFI (GPU):
/// key ops, ECDSA, Schnorr, ECDH, hashing, Bitcoin addresses,
/// BIP-32, Taproot, Pedersen, and GPU batch operations.
///
/// Build & Run:
///   UFSECP_LIB_DIR=../../build-linux/include/ufsecp \
///   LD_LIBRARY_PATH=../../build-linux/include/ufsecp \
///   cargo run
use ufsecp::{Context, Network};

fn hexs(data: &[u8]) -> String {
    hex::encode(data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("UltrafastSecp256k1 -- Rust Example");
    println!("ABI version: {}", Context::abi_version());
    println!("Library:     {}", Context::version_string());
    println!();

    let ctx = Context::new()?;

    let privkey: [u8; 32] = {
        let mut k = [0u8; 32];
        k[31] = 1;
        k
    };
    let privkey2: [u8; 32] = {
        let mut k = [0u8; 32];
        k[31] = 2;
        k
    };

    // ── 1. Key Generation ────────────────────────────────────────────
    println!("[1] Key Generation");
    let pub33 = ctx.pubkey_create(&privkey)?;
    let pub65 = ctx.pubkey_create_uncompressed(&privkey)?;
    let xonly = ctx.pubkey_xonly(&privkey)?;

    println!("  Private key:        {}", hexs(&privkey));
    println!("  Compressed (33B):   {}", hexs(&pub33));
    println!("  Uncompressed (65B): {}", hexs(&pub65));
    println!("  X-only (32B):       {}", hexs(&xonly));
    println!();

    // ── 2. ECDSA ─────────────────────────────────────────────────────
    println!("[2] ECDSA Sign / Verify (RFC 6979)");
    let msg = Context::sha256(b"Hello UltrafastSecp256k1!")?;
    println!("  Message hash:       {}", hexs(&msg));

    let sig = ctx.ecdsa_sign(&msg, &privkey)?;
    println!("  ECDSA signature:    {}", hexs(&sig));

    let valid = ctx.ecdsa_verify(&msg, &sig, &pub33);
    println!("  Verify:             {}", if valid { "VALID" } else { "INVALID" });

    // DER encoding
    let der = ctx.ecdsa_sig_to_der(&sig)?;
    println!("  DER length:         {} bytes", der.len());
    let sig_back = ctx.ecdsa_sig_from_der(&der)?;
    println!("  DER roundtrip:      {}", if sig == sig_back { "match" } else { "MISMATCH" });

    // Recovery
    let rsig = ctx.ecdsa_sign_recoverable(&msg, &privkey)?;
    let recovered = ctx.ecdsa_recover(&msg, &rsig.signature, rsig.recovery_id)?;
    println!("  Recovery:           recid={}, match={}", rsig.recovery_id,
             if recovered == pub33 { "YES" } else { "NO" });
    println!();

    // ── 3. Schnorr ───────────────────────────────────────────────────
    println!("[3] Schnorr Sign / Verify (BIP-340)");
    let aux = [0u8; 32];
    let schnorr_sig = ctx.schnorr_sign(&msg, &privkey, &aux)?;
    println!("  Schnorr signature:  {}", hexs(&schnorr_sig));

    let valid = ctx.schnorr_verify(&msg, &schnorr_sig, &xonly);
    println!("  Verify:             {}", if valid { "VALID" } else { "INVALID" });
    println!();

    // ── 4. ECDH ──────────────────────────────────────────────────────
    println!("[4] ECDH Key Agreement");
    let pub2 = ctx.pubkey_create(&privkey2)?;
    let secret_a = ctx.ecdh(&privkey, &pub2)?;
    let secret_b = ctx.ecdh(&privkey2, &pub33)?;
    println!("  Secret (A->B):      {}", hexs(&secret_a));
    println!("  Secret (B->A):      {}", hexs(&secret_b));
    println!("  Match:              {}", if secret_a == secret_b { "YES" } else { "NO" });
    println!();

    // ── 5. Hashing ───────────────────────────────────────────────────
    println!("[5] Hashing");
    let sha = Context::sha256(&pub33)?;
    let h160 = Context::hash160(&pub33)?;
    let tagged = Context::tagged_hash("BIP0340/challenge", &msg)?;
    println!("  SHA-256(pubkey):    {}", hexs(&sha));
    println!("  Hash160(pubkey):    {}", hexs(&h160));
    println!("  Tagged hash:        {}", hexs(&tagged));
    println!();

    // ── 6. Bitcoin Addresses ─────────────────────────────────────────
    println!("[6] Bitcoin Addresses");
    println!("  P2PKH:              {}", ctx.addr_p2pkh(&pub33, Network::Mainnet)?);
    println!("  P2WPKH:             {}", ctx.addr_p2wpkh(&pub33, Network::Mainnet)?);
    println!("  P2TR:               {}", ctx.addr_p2tr(&xonly, Network::Mainnet)?);
    println!();

    // ── 7. WIF ───────────────────────────────────────────────────────
    println!("[7] WIF Encoding");
    let wif = ctx.wif_encode(&privkey, true, Network::Mainnet)?;
    println!("  WIF:                {}", wif);
    let decoded = ctx.wif_decode(&wif)?;
    println!("  Decode roundtrip:   match={}", if decoded.privkey == privkey { "YES" } else { "NO" });
    println!();

    // ── 8. BIP-32 ────────────────────────────────────────────────────
    println!("[8] BIP-32 HD Key Derivation");
    let seed = [0x42u8; 64];
    let master = ctx.bip32_master(&seed)?;
    let child_key = ctx.bip32_derive_path(&master, "m/44'/0'/0'/0/0")?;
    let child_priv = ctx.bip32_privkey(&child_key)?;
    let child_pub = ctx.bip32_pubkey(&child_key)?;
    println!("  BIP-32 child priv:  {}", hexs(&child_priv));
    println!("  BIP-32 child pub:   {}", hexs(&child_pub));
    println!();

    // ── 9. Taproot ───────────────────────────────────────────────────
    println!("[9] Taproot (BIP-341)");
    let tap = ctx.taproot_output_key(&xonly, None)?;
    println!("  Output key:         {}", hexs(&tap.output_key_x));
    println!("  Parity:             {}", tap.parity);
    let valid = ctx.taproot_verify(&tap.output_key_x, tap.parity, &xonly, None);
    println!("  Verify:             {}", if valid { "VALID" } else { "INVALID" });
    println!();

    // ── 10. Pedersen Commitment ──────────────────────────────────────
    println!("[10] Pedersen Commitment");
    demo_pedersen(&ctx)?;
    println!();

    // ── GPU ──────────────────────────────────────────────────────────
    println!("=== GPU Operations ===");
    println!();
    demo_gpu(&ctx)?;

    println!("All examples completed successfully.");
    Ok(())
}

fn demo_pedersen(ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
    use ufsecp::ufsecp_sys::*;

    let mut value = [0u8; 32];
    value[31] = 42;
    let mut blinding = [0u8; 32];
    blinding[31] = 7;

    let mut commitment = [0u8; 33];
    let rc = unsafe {
        ufsecp_pedersen_commit(
            ctx.as_ptr(),
            value.as_ptr(),
            blinding.as_ptr(),
            commitment.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "pedersen_commit failed: {rc}");
    println!("  Commitment:         {}", hexs(&commitment));

    let rc = unsafe {
        ufsecp_pedersen_verify(
            ctx.as_ptr(),
            commitment.as_ptr(),
            value.as_ptr(),
            blinding.as_ptr(),
        )
    };
    println!("  Verify:             {}", if rc == 0 { "VALID" } else { "INVALID" });
    Ok(())
}

fn demo_gpu(ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
    use ufsecp::ufsecp_sys::*;
    use std::ffi::CStr;
    use std::os::raw::c_int;

    // 11. Backend Discovery
    println!("[11] GPU Backend Discovery");
    let mut bids = [0u32; 4];
    let n_backends = unsafe { ufsecp_gpu_backend_count(bids.as_mut_ptr(), 4) };
    println!("  Backends compiled:  {n_backends}");

    let mut use_backend = 0u32;
    for i in 0..n_backends as usize {
        let bid = bids[i];
        let name = unsafe { CStr::from_ptr(ufsecp_gpu_backend_name(bid)) }.to_str().unwrap_or("?");
        let avail = unsafe { ufsecp_gpu_is_available(bid) };
        let devs = unsafe { ufsecp_gpu_device_count(bid) };
        println!("  Backend {bid}: {name:<8} available={avail} devices={devs}");
        if avail != 0 && use_backend == 0 {
            use_backend = bid;
        }
    }

    if use_backend == 0 {
        println!("  No GPU backends available -- skipping GPU demos.");
        println!();
        return Ok(());
    }

    // Create GPU context
    let mut gpu: *mut ufsecp_gpu_ctx = std::ptr::null_mut();
    let rc = unsafe { ufsecp_gpu_ctx_create(&mut gpu, use_backend, 0) };
    if rc != 0 {
        let msg = unsafe { CStr::from_ptr(ufsecp_gpu_error_str(rc)) }.to_str().unwrap_or("?");
        println!("  GPU context creation failed: {msg}");
        return Ok(());
    }

    let gpu_err = |rc: c_int| -> String {
        unsafe { CStr::from_ptr(ufsecp_gpu_error_str(rc)) }.to_str().unwrap_or("?").to_string()
    };

    const N: usize = 4;

    // 12. Batch Key Generation
    println!();
    println!("[12] GPU Batch Key Generation ({N} keys)");
    let mut scalars = [0u8; N * 32];
    for i in 0..N {
        scalars[i * 32 + 31] = (i + 1) as u8;
    }

    let mut pubkeys = [0u8; N * 33];
    let rc = unsafe {
        ufsecp_gpu_generator_mul_batch(gpu, scalars.as_ptr(), N, pubkeys.as_mut_ptr())
    };
    if rc == 0 {
        for i in 0..N {
            println!("  GPU pubkey[{i}]:      {}", hexs(&pubkeys[i*33..(i+1)*33]));
        }
    } else {
        println!("  gpu_generator_mul_batch: {}", gpu_err(rc));
    }

    // 13. ECDSA Batch Verify
    println!();
    println!("[13] GPU ECDSA Batch Verify");
    let mut msgs = [0u8; N * 32];
    let mut sigs = [0u8; N * 64];
    let mut pubs = [0u8; N * 33];

    for i in 0..N {
        let mut msg_hash = [0u8; 32];
        let b = [i as u8];
        unsafe { ufsecp_sha256(b.as_ptr(), 1, msg_hash.as_mut_ptr()) };
        msgs[i*32..(i+1)*32].copy_from_slice(&msg_hash);

        let mut sk = [0u8; 32];
        sk[31] = (i + 1) as u8;
        let mut sig = [0u8; 64];
        unsafe { ufsecp_ecdsa_sign(ctx.as_ptr(), msg_hash.as_ptr(), sk.as_ptr(), sig.as_mut_ptr()) };
        sigs[i*64..(i+1)*64].copy_from_slice(&sig);
        pubs[i*33..(i+1)*33].copy_from_slice(&pubkeys[i*33..(i+1)*33]);
    }

    let mut results = [0u8; N];
    let rc = unsafe {
        ufsecp_gpu_ecdsa_verify_batch(gpu, msgs.as_ptr(), pubs.as_ptr(), sigs.as_ptr(), N, results.as_mut_ptr())
    };
    if rc == 0 {
        let r: Vec<String> = (0..N).map(|i| format!("[{i}]={}", if results[i] != 0 { "VALID" } else { "INVALID" })).collect();
        println!("  Results: {}", r.join(" "));
    } else {
        println!("  gpu_ecdsa_verify_batch: {}", gpu_err(rc));
    }

    // 14. Hash160 Batch
    println!();
    println!("[14] GPU Hash160 Batch");
    let mut hashes = [0u8; N * 20];
    let rc = unsafe {
        ufsecp_gpu_hash160_pubkey_batch(gpu, pubkeys.as_ptr(), N, hashes.as_mut_ptr())
    };
    if rc == 0 {
        for i in 0..N {
            println!("  Hash160[{i}]:         {}", hexs(&hashes[i*20..(i+1)*20]));
        }
    } else {
        println!("  gpu_hash160_pubkey_batch: {}", gpu_err(rc));
    }

    // 15. MSM
    println!();
    println!("[15] GPU Multi-Scalar Multiplication");
    let mut msm_result = [0u8; 33];
    let rc = unsafe {
        ufsecp_gpu_msm(gpu, scalars.as_ptr(), pubkeys.as_ptr(), N, msm_result.as_mut_ptr())
    };
    if rc == 0 {
        println!("  MSM result:         {}", hexs(&msm_result));
    } else {
        println!("  gpu_msm: {}", gpu_err(rc));
    }
    println!();

    unsafe { ufsecp_gpu_ctx_destroy(gpu) };
    Ok(())
}
