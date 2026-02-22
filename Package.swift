// swift-tools-version:5.9
// ============================================================================
// UltrafastSecp256k1 — Swift Package Manager
// ============================================================================
// Ultra high-performance secp256k1 elliptic curve cryptography library
// Exposes C++20 headers directly (no Swift wrapper)
//
// Usage in Package.swift:
//   .package(url: "https://github.com/shrec/Secp256K1fast.git", from: "3.0.0")
//
// Usage in target:
//   .target(name: "MyApp", dependencies: ["UltrafastSecp256k1"])
//
// In C++ code:
//   #include <secp256k1/field.hpp>
//   #include <secp256k1/ecdsa.hpp>
//   #include <secp256k1/schnorr.hpp>
// ============================================================================

import PackageDescription

let package = Package(
    name: "UltrafastSecp256k1",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .watchOS(.v10),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "UltrafastSecp256k1",
            targets: ["UltrafastSecp256k1"]
        ),
    ],
    targets: [
        .target(
            name: "UltrafastSecp256k1",
            path: "cpu",
            exclude: [
                // ── x86_64 assembly (not needed on Apple ARM64) ──
                "src/field_asm_x64.asm",
                "src/field_asm_x64.cpp",
                "src/field_asm_x64_gas.S",
                // ── RISC-V assembly ──
                "src/field_asm_riscv64.S",
                "src/field_asm_riscv64.cpp",
                // ── Non-source files in src/ ──
                "src/platform_compat.h",
                "src/decomposition_optimized.hpp",
                // ── Non-library directories ──
                "bench",
                "tests",
                "fuzz",
                // ── CMake build file ──
                "CMakeLists.txt",
            ],
            sources: ["src"],
            publicHeadersPath: "include",
            cxxSettings: [
                // Shared types header lives in root include/
                .headerSearchPath("../include"),
                // Performance defines (match CMake defaults)
                .define("SECP256K1_FAST_NO_SECURITY_CHECKS", to: "1"),
                .define("SECP256K1_ULTRA_SPEED", to: "1"),
                .define("NDEBUG"),
                // ARM64 inline assembly (MUL/UMULH — available on all Apple ARM64)
                .define("SECP256K1_HAS_ARM64_ASM", to: "1",
                        .when(platforms: [.iOS, .macOS, .tvOS, .watchOS, .visionOS])),
                .define("SECP256K1_HAS_ASM", to: "1",
                        .when(platforms: [.iOS, .macOS, .tvOS, .watchOS, .visionOS])),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx20
)
