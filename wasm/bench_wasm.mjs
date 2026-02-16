#!/usr/bin/env node
// ============================================================================
// UltrafastSecp256k1 — WASM Comprehensive Benchmark
// ============================================================================
// Benchmarks all WASM-exported operations using the high-level JS wrapper.
// Matches CUDA/OpenCL/Metal benchmark format for cross-platform comparison.
//
// Build:
//   emcmake cmake -S wasm -B build-wasm -DCMAKE_BUILD_TYPE=Release
//   cmake --build build-wasm -j
//
// Run:
//   node --experimental-modules wasm/bench_wasm.mjs
//   # or from build output:
//   cd build-wasm/dist && node ../../wasm/bench_wasm.mjs
// ============================================================================

import { createRequire } from 'module';
import { performance } from 'perf_hooks';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Try multiple paths for the WASM module
async function loadModule() {
    const paths = [
        resolve(__dirname, '../build-wasm/dist/secp256k1.mjs'),
        resolve(__dirname, 'dist/secp256k1.mjs'),
        resolve(process.cwd(), 'dist/secp256k1.mjs'),
    ];

    for (const p of paths) {
        try {
            const mod = await import(p);
            const Secp256k1 = mod.Secp256k1 || mod.default;
            return await Secp256k1.create();
        } catch (_) { /* try next */ }
    }
    throw new Error(
        'Could not find secp256k1.mjs. Build first:\n' +
        '  emcmake cmake -S wasm -B build-wasm -DCMAKE_BUILD_TYPE=Release\n' +
        '  cmake --build build-wasm -j'
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function randomBytes(n) {
    const buf = new Uint8Array(n);
    for (let i = 0; i < n; i++) buf[i] = Math.random() * 256 | 0;
    return buf;
}

// Well-known test private key (non-zero, < curve order)
function testPrivkey() {
    const sk = new Uint8Array(32);
    sk[31] = 1; // scalar = 1
    return sk;
}

function formatTime(ns) {
    if (ns >= 1e6) return (ns / 1e6).toFixed(2) + ' ms';
    if (ns >= 1e3) return (ns / 1e3).toFixed(2) + ' us';
    return ns.toFixed(1) + ' ns';
}

function formatThroughput(opsPerSec) {
    if (opsPerSec >= 1e6) return (opsPerSec / 1e6).toFixed(2) + ' M/s';
    if (opsPerSec >= 1e3) return (opsPerSec / 1e3).toFixed(0) + ' K/s';
    return opsPerSec.toFixed(0) + ' /s';
}

// ── Benchmark runner ─────────────────────────────────────────────────────────

function bench(name, fn, iterations, warmup = 50) {
    // Warmup
    for (let i = 0; i < warmup; i++) fn();

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) fn();
    const t1 = performance.now();

    const total_ms = t1 - t0;
    const ns_per_op = (total_ms * 1e6) / iterations;
    const ops_per_sec = iterations / (total_ms / 1000);

    return { name, ns_per_op, ops_per_sec, iterations, total_ms };
}

function printResult(r) {
    const name = r.name.padEnd(24);
    const time = formatTime(r.ns_per_op).padStart(10);
    const throughput = formatThroughput(r.ops_per_sec).padStart(12);
    console.log(`  ${name} | ${time} | ${throughput} | n=${r.iterations}`);
}

function printSummary(results) {
    console.log('\n============================================================');
    console.log('  Performance Summary (for README)');
    console.log('============================================================');
    console.log(`Runtime: Node.js ${process.version} (WASM)\n`);
    console.log('| Operation              | Time/Op    | Throughput   |');
    console.log('|------------------------|------------|-------------:|');
    for (const r of results) {
        const name = r.name.padEnd(22);
        const time = formatTime(r.ns_per_op).padStart(8);
        const throughput = formatThroughput(r.ops_per_sec).padStart(10);
        console.log(`| ${name} | ${time} | ${throughput} |`);
    }
    console.log('');
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
    console.log('============================================================');
    console.log('  Secp256k1 WASM Benchmark');
    console.log('============================================================\n');

    console.log('Loading WASM module...');
    const lib = await loadModule();

    console.log(`Version: ${lib.version()}`);
    console.log(`Runtime: Node.js ${process.version}`);
    console.log(`Platform: ${process.platform} ${process.arch}`);
    console.log('');

    // Self-test first
    if (!lib.selftest()) {
        console.error('SELFTEST FAILED!');
        process.exit(1);
    }
    console.log('Self-test: PASSED\n');

    const results = [];

    // ── Key Generation (pubkey_create = G*k scalar mul) ─────────────────────
    console.log('=== Key Generation (G*k) ===');
    {
        const iterations = 1000;
        // Pre-generate random private keys
        const keys = Array.from({length: iterations}, () => {
            const k = randomBytes(32);
            k[0] &= 0x7F; // ensure < curve order
            k[31] |= 1;   // ensure non-zero
            return k;
        });
        let idx = 0;
        const r = bench('Pubkey Create (G*k)', () => {
            lib.pubkeyCreate(keys[idx % keys.length]);
            idx++;
        }, iterations);
        printResult(r);
        results.push(r);
    }

    // ── Point Multiplication (P*k) ──────────────────────────────────────────
    console.log('\n=== Point Operations ===');
    {
        const iterations = 500;
        // Generate a base point via pubkeyCreate
        const sk = testPrivkey();
        const { x: px, y: py } = lib.pubkeyCreate(sk);
        const scalar = randomBytes(32);
        scalar[0] &= 0x7F;
        scalar[31] |= 1;

        const r = bench('Point Mul (P*k)', () => {
            lib.pointMul(px, py, scalar);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    {
        const iterations = 2000;
        const sk1 = randomBytes(32); sk1[0] &= 0x7F; sk1[31] |= 1;
        const sk2 = randomBytes(32); sk2[0] &= 0x7F; sk2[31] |= 1;
        const p1 = lib.pubkeyCreate(sk1);
        const p2 = lib.pubkeyCreate(sk2);

        const r = bench('Point Add (P+Q)', () => {
            lib.pointAdd(p1.x, p1.y, p2.x, p2.y);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    // ── ECDSA ───────────────────────────────────────────────────────────────
    console.log('\n=== ECDSA ===');
    {
        const iterations = 500;
        const msg = randomBytes(32);
        const sk = randomBytes(32); sk[0] &= 0x7F; sk[31] |= 1;

        const r = bench('ECDSA Sign', () => {
            lib.ecdsaSign(msg, sk);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    {
        const iterations = 500;
        const msg = randomBytes(32);
        const sk = randomBytes(32); sk[0] &= 0x7F; sk[31] |= 1;
        const pub = lib.pubkeyCreate(sk);
        const sig = lib.ecdsaSign(msg, sk);

        const r = bench('ECDSA Verify', () => {
            lib.ecdsaVerify(msg, pub.x, pub.y, sig);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    // ── Schnorr BIP-340 ─────────────────────────────────────────────────────
    console.log('\n=== Schnorr (BIP-340) ===');
    {
        const iterations = 500;
        const sk = randomBytes(32); sk[0] &= 0x7F; sk[31] |= 1;
        const msg = randomBytes(32);
        const aux = randomBytes(32);

        const r = bench('Schnorr Sign', () => {
            lib.schnorrSign(sk, msg, aux);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    {
        const iterations = 500;
        const sk = randomBytes(32); sk[0] &= 0x7F; sk[31] |= 1;
        const msg = randomBytes(32);
        const aux = randomBytes(32);
        const pk = lib.schnorrPubkey(sk);
        const sig = lib.schnorrSign(sk, msg, aux);

        const r = bench('Schnorr Verify', () => {
            lib.schnorrVerify(pk, msg, sig);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    // ── SHA-256 ─────────────────────────────────────────────────────────────
    console.log('\n=== SHA-256 ===');
    {
        const data32 = randomBytes(32);
        const iterations = 50000;
        const r = bench('SHA-256 (32 bytes)', () => {
            lib.sha256(data32);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    {
        const data1k = randomBytes(1024);
        const iterations = 20000;
        const r = bench('SHA-256 (1 KB)', () => {
            lib.sha256(data1k);
        }, iterations);
        printResult(r);
        results.push(r);
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    printSummary(results);

    console.log('Benchmark complete.');
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
