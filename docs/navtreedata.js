/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "UltrafastSecp256k1", "index.html", [
    [ "What Is This?", "index.html#autotoc_md19", null ],
    [ "Quick Start", "index.html#autotoc_md21", null ],
    [ "Where to Start", "index.html#autotoc_md23", null ],
    [ "</blockquote>", "index.html#autotoc_md24", null ],
    [ "Why This Exists", "index.html#autotoc_md25", null ],
    [ "The Audit Model", "index.html#autotoc_md27", null ],
    [ "Recent Performance Milestones (March 2026)", "index.html#autotoc_md29", null ],
    [ "Why UltrafastSecp256k1? — Detail", "index.html#autotoc_md30", [
      [ "Real-world Adoption", "index.html#autotoc_md31", null ]
    ] ],
    [ "Highlights", "index.html#autotoc_md34", null ],
    [ "Engineering Quality & Self-Audit Culture", "index.html#autotoc_md36", [
      [ "By the Numbers", "index.html#autotoc_md37", null ],
      [ "CI/CD Pipeline Highlights", "index.html#autotoc_md38", null ],
      [ "What \"Self-Audit Culture\" Means in Practice", "index.html#autotoc_md39", null ],
      [ "Exploit PoC Test Suite (187 Tests, 20+ Coverage Areas)", "index.html#autotoc_md40", null ],
      [ "Self-Audit Document Index", "index.html#autotoc_md41", null ]
    ] ],
    [ "</blockquote>", "index.html#autotoc_md42", null ],
    [ "Performance", "index.html#autotoc_md43", null ],
    [ "Architecture", "index.html#autotoc_md44", null ],
    [ "Examples", "index.html#autotoc_md45", null ],
    [ "Use Cases", "index.html#autotoc_md46", null ],
    [ "</blockquote>", "index.html#autotoc_md47", null ],
    [ "Security & Vulnerability Reporting", "index.html#autotoc_md48", null ],
    [ "Seeking Sponsors – Bug Bounty & Development", "index.html#autotoc_md50", [
      [ "1. Bug Bounty Program", "index.html#autotoc_md51", null ],
      [ "2. Open Audit Infrastructure", "index.html#autotoc_md52", null ],
      [ "3. Ongoing Development", "index.html#autotoc_md53", null ],
      [ "How to Sponsor", "index.html#autotoc_md54", null ]
    ] ],
    [ "secp256k1 Feature Overview", "index.html#autotoc_md56", [
      [ "BIP-340 Strict Encoding", "index.html#autotoc_md57", null ]
    ] ],
    [ "BIP-352 Silent Payments Scanning Benchmark", "index.html#autotoc_md59", [
      [ "GPU Pipeline (CUDA, RTX 5060 Ti)", "index.html#autotoc_md60", null ],
      [ "GPU vs CPU Comparison", "index.html#autotoc_md61", null ],
      [ "Community & Contributor Benchmarks", "index.html#autotoc_md62", null ],
      [ "Real-world scanning performance (Frigate / Sparrow Wallet)", "index.html#autotoc_md63", null ],
      [ "CPU vs libsecp256k1 (standalone external benchmark)", "index.html#autotoc_md64", null ]
    ] ],
    [ "</blockquote>", "index.html#autotoc_md65", null ],
    [ "60-Second Quickstart", "index.html#autotoc_md66", null ],
    [ "Platform Support Matrix", "index.html#autotoc_md68", null ],
    [ "Installation", "index.html#autotoc_md70", [
      [ "Linux (APT – Debian / Ubuntu)", "index.html#autotoc_md71", null ],
      [ "Linux (RPM – Fedora / RHEL)", "index.html#autotoc_md72", null ],
      [ "Arch Linux (AUR)", "index.html#autotoc_md73", null ],
      [ "From source (any platform)", "index.html#autotoc_md74", null ],
      [ "Use in your CMake project", "index.html#autotoc_md75", null ],
      [ "Use with pkg-config", "index.html#autotoc_md76", null ]
    ] ],
    [ "secp256k1 GPU Acceleration (CUDA / OpenCL / Metal / ROCm)", "index.html#autotoc_md78", [
      [ "CUDA Core ECC Operations (Kernel-Only Throughput)", "index.html#autotoc_md79", null ],
      [ "GPU Signature Operations (ECDSA + Schnorr)", "index.html#autotoc_md80", null ],
      [ "CUDA vs OpenCL Comparison (RTX 5060 Ti)", "index.html#autotoc_md81", null ],
      [ "Apple Metal (M3 Pro) – Kernel-Only", "index.html#autotoc_md82", null ]
    ] ],
    [ "secp256k1 ECDSA & Schnorr Signatures (BIP-340, RFC 6979)", "index.html#autotoc_md84", [
      [ "CPU Signature Benchmarks (x86-64, Clang 19, AVX2, Release)", "index.html#autotoc_md85", null ]
    ] ],
    [ "Constant-Time secp256k1 (Side-Channel Resistance)", "index.html#autotoc_md87", [
      [ "CT Evidence & Methodology", "index.html#autotoc_md88", null ]
    ] ],
    [ "Zero-Knowledge Proofs (Schnorr Sigma, DLEQ, Bulletproofs)", "index.html#autotoc_md90", null ],
    [ "secp256k1 Benchmarks – Cross-Platform Comparison", "index.html#autotoc_md92", [
      [ "CPU: x86-64 vs ARM64 vs RISC-V", "index.html#autotoc_md93", null ],
      [ "GPU: CUDA vs OpenCL vs Metal", "index.html#autotoc_md94", null ],
      [ "Embedded: ESP32-S3 vs ESP32 vs STM32", "index.html#autotoc_md95", null ],
      [ "Field Representation: 5x52 vs 4x64", "index.html#autotoc_md96", null ]
    ] ],
    [ "secp256k1 on Embedded (ESP32 / STM32 / ARM Cortex-M)", "index.html#autotoc_md98", [
      [ "Porting to New Platforms", "index.html#autotoc_md99", null ]
    ] ],
    [ "WASM secp256k1 (Browser & Node.js)", "index.html#autotoc_md101", null ],
    [ "secp256k1 Batch Modular Inverse (Montgomery Trick)", "index.html#autotoc_md103", [
      [ "Mixed Addition (Jacobian + Affine)", "index.html#autotoc_md104", null ],
      [ "GPU Pattern: H-Product Serial Inversion", "index.html#autotoc_md105", null ]
    ] ],
    [ "</blockquote>", "index.html#autotoc_md106", null ],
    [ "secp256k1 Stable C ABI (<tt>ufsecp</tt>) – FFI Bindings", "index.html#autotoc_md107", [
      [ "Quick Start (C)", "index.html#autotoc_md108", null ],
      [ "GPU C ABI (<tt>ufsecp_gpu</tt>)", "index.html#autotoc_md109", null ],
      [ "CPU C ABI Coverage", "index.html#autotoc_md110", null ]
    ] ],
    [ "secp256k1 Use Cases", "index.html#autotoc_md112", null ],
    [ "</blockquote>", "index.html#autotoc_md113", null ],
    [ "Building secp256k1 from Source (CMake)", "index.html#autotoc_md114", [
      [ "Prerequisites", "index.html#autotoc_md115", null ],
      [ "CPU-Only Build", "index.html#autotoc_md116", null ],
      [ "With CUDA GPU Support", "index.html#autotoc_md117", null ],
      [ "WebAssembly (Emscripten)", "index.html#autotoc_md118", null ],
      [ "iOS (XCFramework)", "index.html#autotoc_md119", null ],
      [ "Local ARM64 / RISC-V QEMU Smoke", "index.html#autotoc_md120", null ],
      [ "Build Options", "index.html#autotoc_md121", null ]
    ] ],
    [ "secp256k1 Quick Start (C++ Examples)", "index.html#autotoc_md123", [
      [ "Basic Point Operations", "index.html#autotoc_md124", null ],
      [ "GPU Batch Multiplication", "index.html#autotoc_md125", null ]
    ] ],
    [ "secp256k1 Security Model (FAST vs CT)", "index.html#autotoc_md127", [
      [ "FAST Profile (Default)", "index.html#autotoc_md128", null ],
      [ "CT / Hardened Profile (<tt>ct::</tt> namespace)", "index.html#autotoc_md129", null ]
    ] ],
    [ "secp256k1 Supported Coins (27 Blockchains)", "index.html#autotoc_md131", null ],
    [ "secp256k1 Architecture", "index.html#autotoc_md133", [
      [ "Library Stack", "index.html#autotoc_md134", null ],
      [ "Hardware Compatibility", "index.html#autotoc_md135", null ],
      [ "Embedded Targets", "index.html#autotoc_md136", null ],
      [ "Source Directory", "index.html#autotoc_md137", null ]
    ] ],
    [ "secp256k1 Testing & Verification", "index.html#autotoc_md139", [
      [ "Built-in Selftest", "index.html#autotoc_md140", null ],
      [ "Sanitizer Builds", "index.html#autotoc_md141", null ],
      [ "Fuzz Testing", "index.html#autotoc_md142", null ],
      [ "Platform CI Coverage", "index.html#autotoc_md143", null ],
      [ "Cross-Platform Audit Results", "index.html#autotoc_md144", null ]
    ] ],
    [ "</blockquote>", "index.html#autotoc_md145", null ],
    [ "secp256k1 Benchmark Targets", "index.html#autotoc_md146", null ],
    [ "Research Statement", "index.html#autotoc_md148", null ],
    [ "API Stability", "index.html#autotoc_md150", null ],
    [ "Release Signing & Verification", "index.html#autotoc_md152", [
      [ "Verify checksums", "index.html#autotoc_md153", null ],
      [ "Verify signature (cosign)", "index.html#autotoc_md154", null ]
    ] ],
    [ "FAQ", "index.html#autotoc_md156", null ],
    [ "</blockquote>", "index.html#autotoc_md157", null ],
    [ "Documentation", "index.html#autotoc_md158", null ],
    [ "Contributing", "index.html#autotoc_md160", null ],
    [ "License", "index.html#autotoc_md162", null ],
    [ "Contact & Community", "index.html#autotoc_md164", null ],
    [ "Acknowledgements", "index.html#autotoc_md166", null ],
    [ "Support the Project", "index.html#autotoc_md168", [
      [ "What Your Sponsorship Funds", "index.html#autotoc_md169", null ]
    ] ],
    [ "Supported Guarantees – <tt>ufsecp</tt> C ABI", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html", [
      [ "Tier 1 – Stable (ABI >= 1)", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md2", [
        [ "Thread safety", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md3", null ],
        [ "Memory", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md4", null ]
      ] ],
      [ "Tier 2 – Protocol Security Advisory", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md6", null ],
      [ "Tier 3 – Internal (never exposed)", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md8", null ],
      [ "Constant-Time Architecture (Dual-Layer Model)", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md10", [
        [ "Verification tools", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md11", null ]
      ] ],
      [ "</blockquote>", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md12", null ],
      [ "Versioning Rules", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md13", null ],
      [ "What This Library Does NOT Do", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md15", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", "globals_func" ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Enumerator", "globals_eval.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"UltrafastSecp256k1_8hpp.html",
"classsecp256k1_1_1fast_1_1CombGenContext.html#a39f7b0f5e9c3e2e2ee4fa78fde132e83",
"classsecp256k1_1_1fast_1_1Scalar.html#acc52e92477f518388b7e0827e3a3ce86",
"field_8hpp.html#a2f78be3f271012542db24ae54e1b5a8e",
"globals_s.html",
"namespacemembers_func_r.html",
"namespacesecp256k1_1_1coins.html#a8afd0775b34eec6206fbe1d3b782c7df",
"namespacesecp256k1_1_1fast.html#a8a85eef57b722a52c17ac36a28c6fc10",
"precompute_8hpp.html#a4e14e3dabefd70e6b0c5ec91f5926756",
"structsecp256k1_1_1ExtendedKey.html#a99769644a1833f5642dcade2e8ce1f65",
"structsecp256k1_1_1TxOut.html#acdae5bc2ac3db1274a6f071439aa6cb3",
"structsecp256k1_1_1fast_1_1FieldElement52.html#a2b67481d515dfabef3bc79b32e048af1",
"structsecp256k1_1_1test_1_1TestCounters.html#af61094071ee42c1027a95c403445d6ad",
"ufsecp_8h.html#a020107f8cf408e67204d4459a45228bb",
"ufsecp__gpu__impl_8cpp.html#a03ab5f225d6d26dfd3324a67b705c803"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';