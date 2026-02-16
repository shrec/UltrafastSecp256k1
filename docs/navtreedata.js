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
    [ "⚠️ Security Notice", "index.html#autotoc_md1", null ],
    [ "🚀 Features", "index.html#autotoc_md3", [
      [ "Feature Coverage (v3.3.0)", "index.html#autotoc_md4", null ]
    ] ],
    [ "� Batch Modular Inverse (Montgomery Trick)", "index.html#autotoc_md5", null ],
    [ "⚡ Mixed Addition (Jacobian + Affine)", "index.html#autotoc_md6", [
      [ "Usage Example (CPU)", "index.html#autotoc_md7", null ],
      [ "Mixed Add + Batch Inverse: Collecting Z Values for Cheap Jacobian→Affine", "index.html#autotoc_md8", null ],
      [ "GPU Pattern: H-Product Serial Inversion (<tt>jacobian_add_mixed_h</tt>)", "index.html#autotoc_md9", null ],
      [ "Other Batch Inverse Use Cases", "index.html#autotoc_md10", [
        [ "1. Full Point Conversion: Jacobian → Affine (X + Y)", "index.html#autotoc_md11", null ],
        [ "2. X-Only Coordinate Extraction", "index.html#autotoc_md12", null ],
        [ "3. CUDA: Z Extraction → batch_inverse_kernel → Affine X", "index.html#autotoc_md13", null ],
        [ "4. Batch Modular Division: a[i] / b[i]", "index.html#autotoc_md14", null ],
        [ "5. Scratch Buffer Reuse", "index.html#autotoc_md15", null ]
      ] ],
      [ "Montgomery Trick — Full Algorithm Explanation", "index.html#autotoc_md16", null ]
    ] ],
    [ "�📦 Use Cases", "index.html#autotoc_md17", null ],
    [ "🔐 Security Model", "index.html#autotoc_md18", [
      [ "FAST Profile (Default)", "index.html#autotoc_md19", null ],
      [ "CT / HARDENED Profile (Implemented)", "index.html#autotoc_md20", null ]
    ] ],
    [ "🛠️ Building", "index.html#autotoc_md21", [
      [ "Prerequisites", "index.html#autotoc_md22", null ],
      [ "CPU-Only Build", "index.html#autotoc_md23", null ],
      [ "With CUDA Support", "index.html#autotoc_md24", null ],
      [ "WebAssembly (Emscripten)", "index.html#autotoc_md25", null ],
      [ "iOS (XCFramework)", "index.html#autotoc_md26", null ],
      [ "Build Options", "index.html#autotoc_md27", null ],
      [ "Build Profiles", "index.html#autotoc_md28", [
        [ "1️⃣ FAST (Performance Research Mode)", "index.html#autotoc_md29", null ],
        [ "2️⃣ CT (Constant-Time Hardened Mode)", "index.html#autotoc_md30", null ]
      ] ]
    ] ],
    [ "🎯 Quick Start", "index.html#autotoc_md31", [
      [ "Basic CPU Usage", "index.html#autotoc_md32", null ],
      [ "Advanced: Batch Signature Verification", "index.html#autotoc_md33", null ],
      [ "CUDA GPU Acceleration", "index.html#autotoc_md34", null ],
      [ "CUDA: Batch Address Generation", "index.html#autotoc_md35", null ],
      [ "Performance Tuning Example", "index.html#autotoc_md36", null ]
    ] ],
    [ "📊 Performance", "index.html#autotoc_md37", [
      [ "x86_64 / Windows (Clang 21.1.0, Release)", "index.html#autotoc_md38", null ],
      [ "x86_64 / Linux (i5, Clang 19.1.7, AVX2, Release)", "index.html#autotoc_md39", null ],
      [ "RISC-V 64-bit / Linux (Milk-V Mars, RVV, Clang 21.1.8, Release)", "index.html#autotoc_md40", null ],
      [ "ESP32-S3 / Embedded (Xtensa LX7 @ 240 MHz, ESP-IDF v5.5.1, -O3)", "index.html#autotoc_md41", null ],
      [ "ESP32-PICO-D4 / Embedded (Xtensa LX6 Dual Core @ 240 MHz, ESP-IDF v5.5.1, -O3)", "index.html#autotoc_md42", null ],
      [ "STM32F103ZET6 / Embedded (ARM Cortex-M3 @ 72 MHz, GCC 13.3.1, -O3)", "index.html#autotoc_md43", null ],
      [ "Android ARM64 (RK3588, Cortex-A55/A76 @ 2.4 GHz, NDK r27 Clang 18, -O3)", "index.html#autotoc_md44", null ],
      [ "Embedded Cross-Platform Comparison", "index.html#autotoc_md45", null ],
      [ "CUDA (NVIDIA RTX 5060 Ti) — Kernel-Only", "index.html#autotoc_md46", null ],
      [ "OpenCL (NVIDIA RTX 5060 Ti) — Kernel-Only", "index.html#autotoc_md47", null ],
      [ "CUDA vs OpenCL — Kernel-Only Comparison (RTX 5060 Ti)", "index.html#autotoc_md48", null ],
      [ "Apple Metal (Apple M3 Pro) — Kernel-Only", "index.html#autotoc_md49", null ]
    ] ],
    [ "🏗️ Architecture", "index.html#autotoc_md50", null ],
    [ "🔬 Research Statement", "index.html#autotoc_md51", null ],
    [ "📚 Variant Overview", "index.html#autotoc_md52", null ],
    [ "🚫 Scope", "index.html#autotoc_md53", null ],
    [ "⚠️ API Stability", "index.html#autotoc_md54", null ],
    [ "📚 Documentation", "index.html#autotoc_md55", null ],
    [ "🧪 Testing", "index.html#autotoc_md56", [
      [ "Built-in Selftest", "index.html#autotoc_md57", null ],
      [ "Three Modes", "index.html#autotoc_md58", null ],
      [ "Repro Bundle", "index.html#autotoc_md59", null ],
      [ "Sanitizer Builds", "index.html#autotoc_md60", null ],
      [ "Running Tests", "index.html#autotoc_md61", null ],
      [ "Platform Coverage Dashboard", "index.html#autotoc_md62", null ],
      [ "Fuzz Testing", "index.html#autotoc_md63", null ]
    ] ],
    [ "🤝 Contributing", "index.html#autotoc_md64", [
      [ "Development Setup", "index.html#autotoc_md65", null ]
    ] ],
    [ "📄 License", "index.html#autotoc_md66", [
      [ "Open Source License", "index.html#autotoc_md67", null ],
      [ "Commercial License", "index.html#autotoc_md68", null ]
    ] ],
    [ "🙏 Acknowledgments", "index.html#autotoc_md69", null ],
    [ "📧 Contact & Community", "index.html#autotoc_md70", null ],
    [ "☕ Support the Project", "index.html#autotoc_md71", null ],
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
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"UltrafastSecp256k1_8hpp.html",
"classsecp256k1_1_1fast_1_1Scalar.html#a47e9c6952c4339c3070d779d3fdafcc8",
"field_8hpp_source.html",
"namespacemembers_func_f.html",
"namespacesecp256k1_1_1fast.html#a0272f9fbabc1a3d20ceaa8bdb5aba8fd",
"recovery_8hpp.html",
"structsecp256k1_1_1PedersenCommitment.html#aae14e7ec3b3c8014f078cf53b155a446",
"structsecp256k1_1_1fast_1_1PrecomputedScalar.html#ae83be6d9b05c9b0feae26a062e4d8474"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';