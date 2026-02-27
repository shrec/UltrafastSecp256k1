# ============================================================================
# audit/CMakeLists.txt -- Audit Infrastructure
# ============================================================================
#
# This directory contains everything needed for the library audit:
#   - Unified Audit Runner (unified execution + JSON/TXT report)
#   - Standalone CTest targets (CT, differential, fault injection, ...)
#   - Protocol tests (MuSig2, FROST, KAT)
#   - Fuzz / adversarial tests
#   - Cross-library differential tests (vs bitcoin-core/libsecp256k1)
#
# Core library tests (run_selftest) remain in cpu/tests/.
# ============================================================================

if(NOT BUILD_TESTING)
    return()
endif()

# Shorthand for cpu/tests/ -- core library test sources reused by unified runner
set(CPU_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/tests)

# -- Helper: common link + stack options ------------------------------------
macro(audit_target_defaults target_name)
    target_link_libraries(${target_name} PRIVATE fastsecp256k1)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(${target_name} PRIVATE "LINKER:/STACK:8388608")
    endif()
endmacro()

# ===========================================================================
# Standalone CTest targets
# ===========================================================================

# -- dudect side-channel timing test ---------------------------------------
add_executable(test_ct_sidechannel_standalone test_ct_sidechannel.cpp)
audit_target_defaults(test_ct_sidechannel_standalone)
target_compile_definitions(test_ct_sidechannel_standalone PRIVATE STANDALONE_TEST)
add_test(NAME ct_sidechannel COMMAND test_ct_sidechannel_standalone)
set_tests_properties(ct_sidechannel PROPERTIES TIMEOUT 300)

# Smoke version of dudect (short run, relaxed threshold -- safe for CI)
add_executable(test_ct_sidechannel_smoke test_ct_sidechannel.cpp)
audit_target_defaults(test_ct_sidechannel_smoke)
target_compile_definitions(test_ct_sidechannel_smoke PRIVATE STANDALONE_TEST DUDECT_SMOKE)
add_test(NAME ct_sidechannel_smoke COMMAND test_ct_sidechannel_smoke)
set_tests_properties(ct_sidechannel_smoke PROPERTIES TIMEOUT 120)

# -- Differential/self-consistency test ------------------------------------
add_executable(test_differential_standalone differential_test.cpp)
audit_target_defaults(test_differential_standalone)
add_test(NAME differential COMMAND test_differential_standalone)
set_tests_properties(differential PROPERTIES TIMEOUT 600)

# -- FAST==CT equivalence test ---------------------------------------------
add_executable(test_ct_equivalence_standalone ${CPU_TESTS_DIR}/test_ct_equivalence.cpp)
audit_target_defaults(test_ct_equivalence_standalone)
target_compile_definitions(test_ct_equivalence_standalone PRIVATE STANDALONE_TEST)
add_test(NAME ct_equivalence COMMAND test_ct_equivalence_standalone)

# -- Fault injection simulation --------------------------------------------
add_executable(test_fault_injection test_fault_injection.cpp)
audit_target_defaults(test_fault_injection)
target_compile_definitions(test_fault_injection PRIVATE STANDALONE_TEST)
add_test(NAME fault_injection COMMAND test_fault_injection)
set_tests_properties(fault_injection PROPERTIES TIMEOUT 900)

# -- Debug invariant assertions --------------------------------------------
add_executable(test_debug_invariants test_debug_invariants.cpp)
audit_target_defaults(test_debug_invariants)
add_test(NAME debug_invariants COMMAND test_debug_invariants)
set_tests_properties(debug_invariants PROPERTIES TIMEOUT 600)

# -- Fiat-Crypto comparison vectors ----------------------------------------
add_executable(test_fiat_crypto_vectors test_fiat_crypto_vectors.cpp)
audit_target_defaults(test_fiat_crypto_vectors)
target_compile_definitions(test_fiat_crypto_vectors PRIVATE STANDALONE_TEST)
add_test(NAME fiat_crypto_vectors COMMAND test_fiat_crypto_vectors)
set_tests_properties(fiat_crypto_vectors PROPERTIES TIMEOUT 900)

# -- Carry propagation stress test -----------------------------------------
add_executable(test_carry_propagation test_carry_propagation.cpp)
audit_target_defaults(test_carry_propagation)
target_compile_definitions(test_carry_propagation PRIVATE STANDALONE_TEST)
add_test(NAME carry_propagation COMMAND test_carry_propagation)
set_tests_properties(carry_propagation PROPERTIES TIMEOUT 900)

# -- Cross-platform KAT equivalence ---------------------------------------
add_executable(test_cross_platform_kat test_cross_platform_kat.cpp)
audit_target_defaults(test_cross_platform_kat)
target_compile_definitions(test_cross_platform_kat PRIVATE STANDALONE_TEST)
add_test(NAME cross_platform_kat COMMAND test_cross_platform_kat)
set_tests_properties(cross_platform_kat PROPERTIES TIMEOUT 900)

# -- ABI version gate (compile-time check) ---------------------------------
add_executable(test_abi_gate test_abi_gate.cpp)
target_include_directories(test_abi_gate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_compile_definitions(test_abi_gate PRIVATE STANDALONE_TEST)
add_test(NAME abi_gate COMMAND test_abi_gate)
set_tests_properties(abi_gate PROPERTIES TIMEOUT 30)

# -- Standalone audit_fuzz test --------------------------------------------
add_executable(test_audit_fuzz_standalone audit_fuzz.cpp)
audit_target_defaults(test_audit_fuzz_standalone)
add_test(NAME audit_fuzz COMMAND test_audit_fuzz_standalone)
set_tests_properties(audit_fuzz PROPERTIES TIMEOUT 600)

# -- Diagnostic: ct::scalar_mul step-by-step comparison --------------------
add_executable(diag_scalar_mul ${CPU_TESTS_DIR}/diag_scalar_mul.cpp)
audit_target_defaults(diag_scalar_mul)
target_compile_definitions(diag_scalar_mul PRIVATE STANDALONE_TEST)
add_test(NAME diag_scalar_mul COMMAND diag_scalar_mul)

# ===========================================================================
# Cross-library differential test (vs bitcoin-core/libsecp256k1)
# ===========================================================================
option(SECP256K1_BUILD_CROSS_TESTS
       "Build in-process differential tests against bitcoin-core/libsecp256k1" OFF)

if(SECP256K1_BUILD_CROSS_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        libsecp256k1_ref
        GIT_REPOSITORY https://github.com/bitcoin-core/secp256k1.git
        GIT_TAG        v0.6.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(libsecp256k1_ref)
    if(NOT libsecp256k1_ref_POPULATED)
        FetchContent_Populate(libsecp256k1_ref)
        set(BUILD_SHARED_LIBS                  OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_SHARED             OFF CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_SCHNORRSIG ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_EXTRAKEYS  ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_RECOVERY   ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_ECDH       ON  CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_BENCHMARK          OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_TESTS              OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_EXHAUSTIVE_TESTS   OFF CACHE BOOL "" FORCE)
        add_subdirectory(
            ${libsecp256k1_ref_SOURCE_DIR}
            ${libsecp256k1_ref_BINARY_DIR}
            EXCLUDE_FROM_ALL
        )
    endif()

    add_executable(test_cross_libsecp256k1 test_cross_libsecp256k1.cpp)
    target_link_libraries(test_cross_libsecp256k1 PRIVATE fastsecp256k1 secp256k1)
    target_include_directories(test_cross_libsecp256k1
        PRIVATE ${libsecp256k1_ref_SOURCE_DIR}/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_cross_libsecp256k1 PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME cross_libsecp256k1 COMMAND test_cross_libsecp256k1)
    set_tests_properties(cross_libsecp256k1 PROPERTIES TIMEOUT 900)
    message(STATUS "  Cross-test vs libsecp256k1: ON (ref: v0.6.0)")
endif()

# ===========================================================================
# Parser fuzz tests (deterministic pseudo-fuzz)
# ===========================================================================
option(SECP256K1_BUILD_FUZZ_TESTS
       "Build deterministic fuzz tests for parsers (DER, Schnorr, Pubkey)" OFF)

if(SECP256K1_BUILD_FUZZ_TESTS AND TARGET ufsecp_static)
    add_executable(test_fuzz_parsers test_fuzz_parsers.cpp)
    target_link_libraries(test_fuzz_parsers PRIVATE ufsecp_static fastsecp256k1)
    target_include_directories(test_fuzz_parsers PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    target_compile_definitions(test_fuzz_parsers PRIVATE "UFSECP_API=")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_fuzz_parsers PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME fuzz_parsers COMMAND test_fuzz_parsers)
    set_tests_properties(fuzz_parsers PROPERTIES TIMEOUT 900)
    message(STATUS "  Parser fuzz tests: ON")

    # Address + BIP32 + FFI boundary fuzz
    add_executable(test_fuzz_address_bip32_ffi test_fuzz_address_bip32_ffi.cpp)
    target_link_libraries(test_fuzz_address_bip32_ffi PRIVATE ufsecp_static fastsecp256k1)
    target_include_directories(test_fuzz_address_bip32_ffi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/include/ufsecp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    target_compile_definitions(test_fuzz_address_bip32_ffi PRIVATE "UFSECP_API=")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_fuzz_address_bip32_ffi PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME fuzz_address_bip32_ffi COMMAND test_fuzz_address_bip32_ffi)
    set_tests_properties(fuzz_address_bip32_ffi PROPERTIES TIMEOUT 900)
    message(STATUS "  Address + BIP32 + FFI fuzz tests: ON")
endif()

# ===========================================================================
# MuSig2 + FROST protocol tests
# ===========================================================================
option(SECP256K1_BUILD_PROTOCOL_TESTS
       "Build MuSig2 + FROST protocol tests" OFF)

if(SECP256K1_BUILD_PROTOCOL_TESTS)
    add_executable(test_musig2_frost test_musig2_frost.cpp)
    target_link_libraries(test_musig2_frost PRIVATE fastsecp256k1)
    target_include_directories(test_musig2_frost PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_musig2_frost PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME musig2_frost COMMAND test_musig2_frost)
    set_tests_properties(musig2_frost PROPERTIES TIMEOUT 600)
    message(STATUS "  MuSig2 + FROST protocol tests: ON")

    # Advanced protocol tests
    add_executable(test_musig2_frost_advanced test_musig2_frost_advanced.cpp)
    target_link_libraries(test_musig2_frost_advanced PRIVATE fastsecp256k1)
    target_include_directories(test_musig2_frost_advanced PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_musig2_frost_advanced PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME musig2_frost_advanced COMMAND test_musig2_frost_advanced)
    set_tests_properties(musig2_frost_advanced PROPERTIES TIMEOUT 600)
    message(STATUS "  MuSig2 + FROST advanced tests: ON")

    # FROST reference KAT vectors
    add_executable(test_frost_kat test_frost_kat.cpp)
    target_link_libraries(test_frost_kat PRIVATE fastsecp256k1)
    target_include_directories(test_frost_kat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_frost_kat PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME frost_kat COMMAND test_frost_kat)
    set_tests_properties(frost_kat PROPERTIES TIMEOUT 600)
    message(STATUS "  FROST reference KAT vectors: ON")
endif()

# ===========================================================================
# Unified Audit Runner -- Unified Audit Binary
# ===========================================================================
# Single binary that runs ALL test modules + generates JSON/TXT reports.
# Build once, run on any platform. Self-audit artifact.
add_executable(unified_audit_runner
    unified_audit_runner.cpp
    # -- selftest modules (from cpu/tests/) --
    ${CPU_TESTS_DIR}/test_large_scalar_multiplication.cpp
    ${CPU_TESTS_DIR}/test_mul.cpp
    ${CPU_TESTS_DIR}/test_arithmetic_correctness.cpp
    ${CPU_TESTS_DIR}/test_ct.cpp
    ${CPU_TESTS_DIR}/test_ct_equivalence.cpp
    ${CPU_TESTS_DIR}/test_ecdsa_schnorr.cpp
    ${CPU_TESTS_DIR}/test_multiscalar_batch.cpp
    ${CPU_TESTS_DIR}/test_bip32.cpp
    ${CPU_TESTS_DIR}/test_bip32_vectors.cpp
    ${CPU_TESTS_DIR}/test_musig2.cpp
    ${CPU_TESTS_DIR}/test_ecdh_recovery_taproot.cpp
    ${CPU_TESTS_DIR}/test_simd_batch.cpp
    ${CPU_TESTS_DIR}/test_v4_features.cpp
    ${CPU_TESTS_DIR}/test_coins.cpp
    ${CPU_TESTS_DIR}/test_batch_add_affine.cpp
    ${CPU_TESTS_DIR}/test_hash_accel.cpp
    ${CPU_TESTS_DIR}/test_exhaustive.cpp
    ${CPU_TESTS_DIR}/test_comprehensive.cpp
    ${CPU_TESTS_DIR}/test_bip340_vectors.cpp
    ${CPU_TESTS_DIR}/test_rfc6979_vectors.cpp
    ${CPU_TESTS_DIR}/test_ecc_properties.cpp
    # -- standalone audit modules (in this directory) --
    test_carry_propagation.cpp
    test_fault_injection.cpp
    test_fiat_crypto_vectors.cpp
    test_cross_platform_kat.cpp
    test_debug_invariants.cpp
    test_abi_gate.cpp
    test_ct_sidechannel.cpp
    differential_test.cpp
    # -- MuSig2 / FROST / adversarial / fuzz --
    test_musig2_frost.cpp
    test_musig2_frost_advanced.cpp
    test_frost_kat.cpp
    audit_fuzz.cpp
    test_fuzz_parsers.cpp
    test_fuzz_address_bip32_ffi.cpp
    # -- Deep audit modules --
    audit_field.cpp
    audit_scalar.cpp
    audit_point.cpp
    audit_ct.cpp
    audit_integration.cpp
    audit_security.cpp
    audit_perf.cpp
    # -- ufsecp FFI implementation (needed by fuzz_parsers + fuzz_address) --
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ufsecp/ufsecp_impl.cpp
    # -- field representation tests --
    ${CPU_TESTS_DIR}/test_field_26.cpp
    # -- diagnostics --
    ${CPU_TESTS_DIR}/diag_scalar_mul.cpp
)
# Conditionally add 5x52 field test (requires __uint128_t; skip on MSVC)
if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    target_sources(unified_audit_runner PRIVATE ${CPU_TESTS_DIR}/test_field_52.cpp)
endif()
target_link_libraries(unified_audit_runner PRIVATE fastsecp256k1)
target_include_directories(unified_audit_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include   # for generated version.hpp
    ${CMAKE_BINARY_DIR}/include              # fallback for version.hpp
)
# Inject git hash at compile time (short hash, 8 chars)
execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short=8 HEAD
    OUTPUT_VARIABLE _git_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_rc
)
if(_git_rc)
    set(_git_hash "unknown")
endif()
target_compile_definitions(unified_audit_runner PRIVATE
    UNIFIED_AUDIT_RUNNER
    DUDECT_SMOKE
    UFSECP_BUILDING
    GIT_HASH="${_git_hash}"
)
if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
    target_link_options(unified_audit_runner PRIVATE "LINKER:/STACK:8388608")
endif()
add_test(NAME unified_audit COMMAND unified_audit_runner)
set_tests_properties(unified_audit PROPERTIES TIMEOUT 1200)

# ===========================================================================
# Full Audit Orchestrator (custom target)
# ===========================================================================
# "cmake --build <dir> --target run_full_audit" runs the orchestrator script.
# On Windows, runs the PowerShell version; on Linux/macOS, runs the bash version.
if(WIN32)
    add_custom_target(run_full_audit
        COMMAND ${CMAKE_COMMAND} -E echo "Running full audit orchestrator (A-M)..."
        COMMAND pwsh -NoProfile -File "${CMAKE_CURRENT_SOURCE_DIR}/run_full_audit.ps1"
            -BuildDir "${CMAKE_BINARY_DIR}"
            -OutputDir "${CMAKE_BINARY_DIR}/audit-output"
            -SkipBuild
        DEPENDS unified_audit_runner
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        COMMENT "Full audit orchestrator (categories A-M)"
        VERBATIM
    )
else()
    add_custom_target(run_full_audit
        COMMAND ${CMAKE_COMMAND} -E echo "Running full audit orchestrator (A-M)..."
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/run_full_audit.sh"
        DEPENDS unified_audit_runner
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        COMMENT "Full audit orchestrator (categories A-M)"
        VERBATIM
    )
    # Pass build dir via environment
    set_property(TARGET run_full_audit APPEND PROPERTY
        ENVIRONMENT "BUILD_DIR=${CMAKE_BINARY_DIR}" "SKIP_BUILD=1"
    )
endif()

# -- CTest labels for grouping ---------------------------------------------
# Label all audit tests so they can be run as a group:
#   ctest --test-dir <build> -L audit
set_tests_properties(
    ct_sidechannel ct_sidechannel_smoke
    differential ct_equivalence
    fault_injection debug_invariants
    fiat_crypto_vectors carry_propagation
    cross_platform_kat abi_gate
    audit_fuzz diag_scalar_mul
    unified_audit
    PROPERTIES LABELS "audit"
)

message(STATUS "  Audit test plan: audit/AUDIT_TEST_PLAN.md")
message(STATUS "  Full audit script (Windows): audit/run_full_audit.ps1")
message(STATUS "  Full audit script (Linux):   audit/run_full_audit.sh")
