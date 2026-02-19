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
    [ "⚠️ Security Notice", "index.html#autotoc_md18", null ],
    [ "🚀 Features", "index.html#autotoc_md20", [
      [ "Feature Coverage (v3.4.0)", "index.html#autotoc_md21", null ]
    ] ],
    [ "� Batch Modular Inverse (Montgomery Trick)", "index.html#autotoc_md22", null ],
    [ "⚡ Mixed Addition (Jacobian + Affine)", "index.html#autotoc_md23", [
      [ "Usage Example (CPU)", "index.html#autotoc_md24", null ],
      [ "Mixed Add + Batch Inverse: Collecting Z Values for Cheap Jacobian→Affine", "index.html#autotoc_md25", null ],
      [ "GPU Pattern: H-Product Serial Inversion (<tt>jacobian_add_mixed_h</tt>)", "index.html#autotoc_md26", null ],
      [ "Other Batch Inverse Use Cases", "index.html#autotoc_md27", [
        [ "1. Full Point Conversion: Jacobian → Affine (X + Y)", "index.html#autotoc_md28", null ],
        [ "2. X-Only Coordinate Extraction", "index.html#autotoc_md29", null ],
        [ "3. CUDA: Z Extraction → batch_inverse_kernel → Affine X", "index.html#autotoc_md30", null ],
        [ "4. Batch Modular Division: a[i] / b[i]", "index.html#autotoc_md31", null ],
        [ "5. Scratch Buffer Reuse", "index.html#autotoc_md32", null ]
      ] ],
      [ "Montgomery Trick — Full Algorithm Explanation", "index.html#autotoc_md33", null ]
    ] ],
    [ "�📦 Use Cases", "index.html#autotoc_md34", null ],
    [ "🔐 Security Model", "index.html#autotoc_md35", [
      [ "FAST Profile (Default)", "index.html#autotoc_md36", null ],
      [ "CT / HARDENED Profile (Implemented)", "index.html#autotoc_md37", null ]
    ] ],
    [ "� Stable C ABI (<tt>ufsecp</tt>)", "index.html#autotoc_md38", [
      [ "Architecture", "index.html#autotoc_md39", null ],
      [ "Quick Start (C)", "index.html#autotoc_md40", null ],
      [ "API Coverage", "index.html#autotoc_md41", null ],
      [ "Building ufsecp", "index.html#autotoc_md42", null ]
    ] ],
    [ "�🛠️ Building", "index.html#autotoc_md43", [
      [ "Prerequisites", "index.html#autotoc_md44", null ],
      [ "CPU-Only Build", "index.html#autotoc_md45", null ],
      [ "With CUDA Support", "index.html#autotoc_md46", null ],
      [ "WebAssembly (Emscripten)", "index.html#autotoc_md47", null ],
      [ "iOS (XCFramework)", "index.html#autotoc_md48", null ],
      [ "Build Options", "index.html#autotoc_md49", null ],
      [ "Build Profiles", "index.html#autotoc_md50", [
        [ "1️⃣ FAST (Performance Research Mode)", "index.html#autotoc_md51", null ],
        [ "2️⃣ CT (Constant-Time Hardened Mode)", "index.html#autotoc_md52", null ]
      ] ]
    ] ],
    [ "🎯 Quick Start", "index.html#autotoc_md53", [
      [ "Basic CPU Usage", "index.html#autotoc_md54", null ],
      [ "Advanced: Batch Signature Verification", "index.html#autotoc_md55", null ],
      [ "CUDA GPU Acceleration", "index.html#autotoc_md56", null ],
      [ "CUDA: Batch Address Generation", "index.html#autotoc_md57", null ],
      [ "Performance Tuning Example", "index.html#autotoc_md58", null ]
    ] ],
    [ "📊 Performance", "index.html#autotoc_md59", [
      [ "x86_64 / Windows (Clang 21.1.0, AVX2, BMI2/ADX, Release)", "index.html#autotoc_md60", [
        [ "Signature Performance Summary", "index.html#autotoc_md61", null ],
        [ "Scalar Multiplication Breakdown", "index.html#autotoc_md62", null ],
        [ "Field Representation Comparison (5×52 vs 4×64)", "index.html#autotoc_md63", null ],
        [ "Constant-Time (CT) Layer Overhead", "index.html#autotoc_md64", null ]
      ] ],
      [ "x86_64 / Linux (i5, Clang 19.1.7, AVX2, Release)", "index.html#autotoc_md65", null ],
      [ "RISC-V 64-bit / Linux (Milk-V Mars, RVV, Clang 21.1.8, Release)", "index.html#autotoc_md66", null ],
      [ "ESP32-S3 / Embedded (Xtensa LX7 @ 240 MHz, ESP-IDF v5.5.1, -O3)", "index.html#autotoc_md67", null ],
      [ "ESP32-PICO-D4 / Embedded (Xtensa LX6 Dual Core @ 240 MHz, ESP-IDF v5.5.1, -O3)", "index.html#autotoc_md68", null ],
      [ "STM32F103ZET6 / Embedded (ARM Cortex-M3 @ 72 MHz, GCC 13.3.1, -O3)", "index.html#autotoc_md69", null ],
      [ "Android ARM64 (RK3588, Cortex-A55/A76 @ 2.4 GHz, NDK r27 Clang 18, -O3)", "index.html#autotoc_md70", null ],
      [ "Embedded Cross-Platform Comparison", "index.html#autotoc_md71", null ],
      [ "CUDA (NVIDIA RTX 5060 Ti) — Kernel-Only", "index.html#autotoc_md72", [
        [ "Core ECC Operations", "index.html#autotoc_md73", null ],
        [ "GPU Signature Operations (ECDSA + Schnorr)", "index.html#autotoc_md74", null ]
      ] ],
      [ "OpenCL (NVIDIA RTX 5060 Ti) — Kernel-Only", "index.html#autotoc_md75", null ],
      [ "CUDA vs OpenCL — Kernel-Only Comparison (RTX 5060 Ti)", "index.html#autotoc_md76", null ],
      [ "Apple Metal (Apple M3 Pro) — Kernel-Only", "index.html#autotoc_md77", null ],
      [ "Available Benchmark Targets", "index.html#autotoc_md78", null ]
    ] ],
    [ "🏗️ Architecture", "index.html#autotoc_md79", null ],
    [ "🔬 Research Statement", "index.html#autotoc_md80", null ],
    [ "📚 Variant Overview", "index.html#autotoc_md81", null ],
    [ "🪙 Supported Coins", "index.html#autotoc_md82", null ],
    [ "🚫 Scope", "index.html#autotoc_md83", null ],
    [ "⚠️ API Stability", "index.html#autotoc_md84", null ],
    [ "📚 Documentation", "index.html#autotoc_md85", null ],
    [ "🧪 Testing", "index.html#autotoc_md86", [
      [ "Built-in Selftest", "index.html#autotoc_md87", null ],
      [ "Three Modes", "index.html#autotoc_md88", null ],
      [ "Repro Bundle", "index.html#autotoc_md89", null ],
      [ "Sanitizer Builds", "index.html#autotoc_md90", null ],
      [ "Running Tests", "index.html#autotoc_md91", null ],
      [ "Platform Coverage Dashboard", "index.html#autotoc_md92", null ],
      [ "Fuzz Testing", "index.html#autotoc_md93", null ]
    ] ],
    [ "🤝 Contributing", "index.html#autotoc_md94", [
      [ "Development Setup", "index.html#autotoc_md95", null ]
    ] ],
    [ "📄 License", "index.html#autotoc_md96", [
      [ "Open Source License", "index.html#autotoc_md97", null ],
      [ "Commercial License", "index.html#autotoc_md98", null ]
    ] ],
    [ "🙏 Acknowledgments", "index.html#autotoc_md99", null ],
    [ "📧 Contact & Community", "index.html#autotoc_md100", null ],
    [ "☕ Support the Project", "index.html#autotoc_md101", null ],
    [ "Supported Guarantees — <tt>ufsecp</tt> C ABI", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html", [
      [ "Tier 1 — Stable (ABI ≥ 1)", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md2", [
        [ "Thread safety", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md3", null ],
        [ "Memory", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md4", null ]
      ] ],
      [ "Tier 2 — Experimental (no ABI promise)", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md6", null ],
      [ "Tier 3 — Internal (never exposed)", "md_include_2ufsecp_2SUPPORTED__GUARANTEES.html#autotoc_md8", null ],
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
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"UltrafastSecp256k1_8hpp.html",
"classsecp256k1_1_1fast_1_1Point.html#abff5c00ffb3074c4cd1b502e447b33c7",
"field_8hpp.html#a2a0daebfdeafbf5c13da255f371d6673",
"hash__accel_8hpp.html#aa9baa48a38bbe07a5b3a664e5767bc92",
"namespacesecp256k1.html#aa822de77c4daffb35bf7581982912615",
"namespacesecp256k1_1_1fast.html#aa1ade8639ae8e331731389af48483662",
"precompute_8hpp.html#aec97c3e9d6fb0e5fff4dcf590c97f463",
"structsecp256k1_1_1PedersenCommitment.html#aacbd8dbfab914b46ad93b52d44ad6788",
"structsecp256k1_1_1fast_1_1FixedBaseConfig.html#a49ec39077d6e057d10ec3df4308eeabc",
"ufsecp_8h.html#ab2ce2db2666d413953cd8cd4c1f41604"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';