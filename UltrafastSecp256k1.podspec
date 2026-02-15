Pod::Spec.new do |s|
  s.name         = "UltrafastSecp256k1"
  s.version      = "2.2.0"
  s.summary      = "Ultra high-performance secp256k1 elliptic curve cryptography"
  s.description  = <<-DESC
    Hardware-accelerated secp256k1 library with ARM64 inline assembly,
    ECDSA (RFC 6979), Schnorr (BIP-340), and constant-time operations.
    Optimized for Apple Silicon with MUL/UMULH instructions.
  DESC

  s.homepage     = "https://github.com/shrec/UltrafastSecp256k1"
  s.license      = { :type => "AGPL-3.0", :file => "LICENSE" }
  s.author       = { "shrec" => "https://github.com/shrec" }

  s.ios.deployment_target     = "17.0"
  s.osx.deployment_target     = "14.0"
  s.watchos.deployment_target = "10.0"
  s.tvos.deployment_target    = "17.0"
  s.visionos.deployment_target = "1.0"

  s.source = {
    :git => "https://github.com/shrec/UltrafastSecp256k1.git",
    :tag => s.version.to_s
  }

  # ── Source files ──────────────────────────────────────────────────────────
  s.source_files = [
    "cpu/src/**/*.{cpp,c}",
    "cpu/include/**/*.{hpp,h}",
    "include/**/*.{hpp,h}",
  ]

  # Exclude platform-specific assembly not for Apple ARM64
  s.exclude_files = [
    "cpu/src/field_asm_x64.asm",
    "cpu/src/field_asm_x64.cpp",
    "cpu/src/field_asm_x64_gas.S",
    "cpu/src/field_asm_riscv64.S",
    "cpu/src/field_asm_riscv64.cpp",
    "cpu/src/platform_compat.h",
    "cpu/src/decomposition_optimized.hpp",
  ]

  # ── Public headers ────────────────────────────────────────────────────────
  s.public_header_files = [
    "cpu/include/**/*.{hpp,h}",
    "include/**/*.{hpp,h}",
  ]

  s.header_mappings_dir = "cpu/include"

  # ── Build settings ────────────────────────────────────────────────────────
  s.pod_target_xcconfig = {
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20",
    "HEADER_SEARCH_PATHS"         => [
      "$(PODS_TARGET_SRCROOT)/cpu/include",
      "$(PODS_TARGET_SRCROOT)/include",
    ].join(" "),
    "GCC_PREPROCESSOR_DEFINITIONS" => [
      "SECP256K1_FAST_NO_SECURITY_CHECKS=1",
      "SECP256K1_ULTRA_SPEED=1",
      "SECP256K1_HAS_ARM64_ASM=1",
      "SECP256K1_HAS_ASM=1",
      "NDEBUG=1",
    ].join(" "),
    "OTHER_CPLUSPLUSFLAGS" => "-O3 -fno-math-errno -fno-trapping-math -funroll-loops",
  }

  s.user_target_xcconfig = {
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20",
  }

  s.libraries = "c++"
  s.requires_arc = false
  s.static_framework = true

  # ── Module ────────────────────────────────────────────────────────────────
  s.module_name = "UltrafastSecp256k1"
  s.module_map  = "cpu/include/module.modulemap"
end
