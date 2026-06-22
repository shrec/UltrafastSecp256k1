cmake_minimum_required(VERSION 3.18)
project(secp256k1_shim LANGUAGES CXX)

# -- Shim library --------------------------------------------------------------
# Two compilation modes:
#
# A) Normal (SECP256K1_BUILD_SHIM OFF — default):
#    Shim sources are compiled here into a separate secp256k1_shim.a that
#    PRIVATELY links fastsecp256k1.  This is the standalone / developer mode.
#
# B) Engine-integrated (SECP256K1_BUILD_SHIM ON — used by Bitcoin Core backend):
#    Shim sources are already compiled INTO fastsecp256k1 (no LTO boundary).
#    secp256k1_shim is a stub that just exports the headers and PRIVATELY links
#    the engine.  DO NOT recompile the sources here — that causes ODR violations.

if(SECP256K1_BUILD_SHIM)
    # ---- Mode B: stub wrapper only -------------------------------------------
    # Sources already in fastsecp256k1.  Create a minimal stub so that the
    # `secp256k1 ALIAS secp256k1_shim` in bitcoin-core-dev/cmake/secp256k1.cmake
    # has a concrete target to alias, and callers get the public headers.
    set(_STUB "${CMAKE_CURRENT_BINARY_DIR}/_secp256k1_shim_stub.cpp")
    file(WRITE "${_STUB}" "/* stub: shim sources compiled into fastsecp256k1 */\n")
    add_library(secp256k1_shim STATIC "${_STUB}")

    # Public headers (secp256k1.h, secp256k1_schnorrsig.h, etc.).
    # Also expose the CPU C++ headers, because secp256k1.h pulls in
    # `secp256k1/ecdsa.hpp` under `#ifdef __cplusplus` (for the inlined
    # `secp256k1_pubkey_precomp` definition).  C consumers are unaffected.
    target_include_directories(secp256k1_shim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include  # ufsecp/ufsecp_cancel.h (secp256k1_batch.h dep)
    )

    # The stub must link the engine so callers transitively get the symbols.
    # PRIVATE keeps fastsecp256k1's compile-definitions (NDEBUG, SECP256K1_HAS_ASM=1, …)
    # from leaking into Bitcoin Core's own compilation units.
    if(TARGET fastsecp256k1)
        target_link_libraries(secp256k1_shim PRIVATE fastsecp256k1)
    elseif(TARGET secp256k1_fast)
        target_link_libraries(secp256k1_shim PRIVATE secp256k1_fast)
    else()
        message(FATAL_ERROR
            "secp256k1_shim stub: engine target 'fastsecp256k1' not found.\n"
            "Add UltrafastSecp256k1 via add_subdirectory before this shim.")
    endif()

else()
    # ---- Mode A: normal shim library -----------------------------------------
    add_library(secp256k1_shim STATIC
        src/shim_context.cpp
        src/shim_pubkey.cpp
        src/shim_ecdsa.cpp
        src/shim_schnorr.cpp
        src/shim_extrakeys.cpp
        src/shim_seckey.cpp
        src/shim_tagged_hash.cpp
        src/shim_recovery.cpp
        src/shim_ellswift.cpp
        src/shim_ecdh.cpp
        src/shim_musig.cpp
        src/shim_batch_verify.cpp
    )

    # C++20 is needed to compile the shim (it includes UF C++ headers),
    # but must not leak to consumers — PRIVATE keeps it shim-internal.
    target_compile_features(secp256k1_shim PRIVATE cxx_std_20)

    # Public includes -- exposes libsecp256k1-compatible headers.
    # Also expose the CPU C++ headers: secp256k1.h pulls in `secp256k1/ecdsa.hpp`
    # under `#ifdef __cplusplus` (inlined secp256k1_pubkey_precomp). C consumers are
    # unaffected; C++ consumers (the shim conformance tests) need it to compile.
    target_include_directories(secp256k1_shim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include  # ufsecp/ufsecp_cancel.h (secp256k1_batch.h dep)
    )

    # Link against UltrafastSecp256k1 CPU library.
    if(TARGET fastsecp256k1)
        target_link_libraries(secp256k1_shim PRIVATE fastsecp256k1)
    elseif(TARGET secp256k1_fast)
        target_link_libraries(secp256k1_shim PRIVATE secp256k1_fast)
    elseif(TARGET secp256k1::fast)
        target_link_libraries(secp256k1_shim PRIVATE secp256k1::fast)
    else()
        message(FATAL_ERROR
            "UltrafastSecp256k1 library target not found.\n"
            "Add UltrafastSecp256k1 via add_subdirectory before this shim.")
    endif()

endif() # SECP256K1_BUILD_SHIM

# -- Optional: self-contained shared ultrafast_secp256k1 (DLL / .so / .dylib) --
# Builds ONE shared library that exports the libsecp256k1 C ABI (the secp256k1_*
# symbols + <secp256k1.h>), but under OUR OWN file name so it never silently
# substitutes the system libsecp256k1: ultrafast_secp256k1.dll /
# libultrafast_secp256k1.so / .dylib (+ import lib on Windows). A consumer opts in
# explicitly by linking ultrafast_secp256k1 (e.g. under WITH_ULTRAFAST); the ABI
# (headers + symbols) is unchanged, so no source changes are needed.
# The ABI symbols are compiled into fastsecp256k1 (the shim sources are part of
# the engine under SECP256K1_BUILD_SHIM=ON), so the shared lib whole-archives
# fastsecp256k1 to pull every object in, and on MSVC auto-exports all symbols.
option(SECP256K1_SHIM_BUILD_SHARED
    "Also build a self-contained shared ultrafast_secp256k1 (DLL/.so) exporting the libsecp256k1 ABI" OFF)
if(SECP256K1_SHIM_BUILD_SHARED)
    if(NOT SECP256K1_BUILD_SHIM)
        message(FATAL_ERROR
            "SECP256K1_SHIM_BUILD_SHARED requires SECP256K1_BUILD_SHIM=ON "
            "(the secp256k1 ABI symbols are compiled into fastsecp256k1).")
    endif()
    if(NOT TARGET fastsecp256k1)
        message(FATAL_ERROR "SECP256K1_SHIM_BUILD_SHARED: fastsecp256k1 target not found.")
    endif()
    set(_SHARED_STUB "${CMAKE_CURRENT_BINARY_DIR}/_ultrafast_secp256k1_shared_stub.cpp")
    file(WRITE "${_SHARED_STUB}"
        "/* ultrafast_secp256k1 shared: libsecp256k1 ABI pulled from fastsecp256k1 via whole-archive */\n")
    add_library(ultrafast_secp256k1_shared SHARED "${_SHARED_STUB}")
    set_target_properties(ultrafast_secp256k1_shared PROPERTIES
        OUTPUT_NAME ultrafast_secp256k1
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_include_directories(ultrafast_secp256k1_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)  # ufsecp/ufsecp_cancel.h (secp256k1_batch.h dep)
    # WHOLE_ARCHIVE (CMake >= 3.24) forces every fastsecp256k1 object into the
    # shared lib so the secp256k1_* ABI is present (and thus exported).
    target_link_libraries(ultrafast_secp256k1_shared PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,fastsecp256k1>")
    message(STATUS "  ultrafast_secp256k1 shared (libsecp256k1 ABI, our file name): ON")

    # ---------------------------------------------------------------------
    # Install under OUR OWN name (ultrafast_secp256k1) — NEVER as libsecp256k1.
    #
    # The shared lib exports the libsecp256k1 C ABI (secp256k1_* symbols +
    # secp256k1*.h headers), but every installed FILE carries our name:
    # libultrafast_secp256k1.so + ultrafast_secp256k1.pc + the secp256k1*.h ABI
    # headers under an ultrafast_secp256k1/ subdir. We deliberately do NOT ship a
    # `libsecp256k1.pc`, nor drop `secp256k1*.h` at the bare include root, because
    # that collides with / overwrites a system libsecp256k1 and leaves pkg-config +
    # the loader unable to tell which is the real one. To use this as a Bitcoin
    # Core/Knots backend, the integrator aliases `secp256k1` EXPLICITLY —
    # bundled-tree `add_library(secp256k1 ALIAS secp256k1_shim)`, or their own
    # libsecp256k1 alias/symlink over the installed ultrafast_secp256k1, at their
    # own risk. See docs/BITCOIN_KNOTS_INTEGRATION.md.
    # ---------------------------------------------------------------------
    option(SECP256K1_SHIM_INSTALL
        "Install the shim shared lib + secp256k1*.h ABI headers + ultrafast_secp256k1.pc (our-name pkg-config; integrator aliases secp256k1 explicitly)"
        ON)
    if(SECP256K1_SHIM_INSTALL)
        include(GNUInstallDirs)
        # libsecp256k1-ABI-compat level advertised by ultrafast_secp256k1.pc; matches
        # the Knots/Core libsecp256k1 0.6.x subtree so an integrator's own `secp256k1`
        # alias resolves against the same ABI. This is NOT a libsecp256k1 package.
        set(UFSECP_SHIM_PC_VERSION "0.6.0")
        install(TARGETS ultrafast_secp256k1_shared
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ultrafast_secp256k1
                FILES_MATCHING PATTERN "secp256k1*.h")
        # secp256k1_batch.h includes "ufsecp/ufsecp_cancel.h" — ship it under the
        # same include root so the installed ABI headers stay self-contained.
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ufsecp/ufsecp_cancel.h
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ultrafast_secp256k1/ufsecp)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/ultrafast_secp256k1.pc.in
            ${CMAKE_CURRENT_BINARY_DIR}/ultrafast_secp256k1.pc @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ultrafast_secp256k1.pc
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
        message(STATUS "  shim install (ultrafast_secp256k1.pc + secp256k1*.h ABI headers + libultrafast_secp256k1.so): ON "
                       "-> integrator aliases secp256k1 explicitly (add_library(secp256k1 ALIAS secp256k1_shim))")
    endif()
endif()

# -- Optional: RFC 6979 libsecp-compatible nonce mode -------------------------
# When ON, secp256k1_ecdsa_sign / secp256k1_ecdsa_sign_recoverable use
# rfc6979_nonce_libsecp_compat, which appends the "ECDSA" algo16 tag and
# passes ndata directly — byte-identical to upstream libsecp256k1.
# Trade-off: disables the hedged nonce's OS-CSPRNG fault-attack resistance.
option(SECP256K1_SHIM_RFC6979_COMPAT
    "Match upstream libsecp256k1 nonce bytes exactly (includes ECDSA algo16 tag). Disables fault-attack resistance of hedged nonce."
    OFF)
if(SECP256K1_SHIM_RFC6979_COMPAT)
    target_compile_definitions(secp256k1_shim PRIVATE SECP256K1_SHIM_RFC6979_COMPAT=1)
endif()

# -- Optional: shim API test ---------------------------------------------------

option(SECP256K1_SHIM_BUILD_TESTS "Build shim compatibility test" OFF)

if(SECP256K1_SHIM_BUILD_TESTS AND NOT SECP256K1_BUILD_SHIM)
    enable_testing()
    add_executable(shim_test tests/shim_test.cpp)
    target_link_libraries(shim_test PRIVATE secp256k1_shim)
    add_test(NAME secp256k1_shim_test COMMAND shim_test)
    set_tests_properties(secp256k1_shim_test PROPERTIES LABELS "shim")

    # Official BIP-340 Schnorr test vectors (bitcoin/bips bip-0340/test-vectors.csv)
    add_executable(test_bip340_vectors tests/test_bip340_vectors.cpp)
    target_link_libraries(test_bip340_vectors PRIVATE secp256k1_shim)
    add_test(NAME bip340_vectors COMMAND test_bip340_vectors)
    set_tests_properties(bip340_vectors PROPERTIES LABELS "shim")

    # Official BIP-341 Taproot key-path tweak vectors (bip-0341/wallet-test-vectors.json)
    add_executable(test_bip341_vectors tests/test_bip341_vectors.cpp)
    target_link_libraries(test_bip341_vectors PRIVATE secp256k1_shim)
    add_test(NAME bip341_vectors COMMAND test_bip341_vectors)
    set_tests_properties(bip341_vectors PROPERTIES LABELS "shim")

    # Official BIP-327 MuSig2 key-aggregation vectors (bip-0327/vectors/key_agg_vectors.json)
    add_executable(test_bip327_keyagg_vectors tests/test_bip327_keyagg_vectors.cpp)
    target_link_libraries(test_bip327_keyagg_vectors PRIVATE secp256k1_shim)
    add_test(NAME bip327_keyagg_vectors COMMAND test_bip327_keyagg_vectors)
    set_tests_properties(bip327_keyagg_vectors PROPERTIES LABELS "shim")

    # BIP-327 MuSig2 tweak correctness: output KATs (reference.py oracle) + sign/verify
    # roundtrips. Guards the gacc/tacc tweaked-signing fix.
    add_executable(test_bip327_tweak_sign tests/test_bip327_tweak_sign.cpp)
    target_link_libraries(test_bip327_tweak_sign PRIVATE secp256k1_shim)
    add_test(NAME bip327_tweak_sign COMMAND test_bip327_tweak_sign)
    set_tests_properties(bip327_tweak_sign PROPERTIES LABELS "shim")

    # BIP-327 sign_verify_vectors.json (verify side): every external (reference-impl)
    # partial signature must verify under the engine. Guards the binding-factor tag
    # ("MuSig/noncecoef") — regression for the cross-implementation interop bug.
    add_executable(test_bip327_sign_verify_vectors tests/test_bip327_sign_verify_vectors.cpp)
    target_link_libraries(test_bip327_sign_verify_vectors PRIVATE secp256k1_shim)
    add_test(NAME bip327_sign_verify_vectors COMMAND test_bip327_sign_verify_vectors)
    set_tests_properties(bip327_sign_verify_vectors PROPERTIES LABELS "shim")

    # Multi-threaded + per-row batch verify (bridge-free integration standard):
    # MT==single across thread counts, per-row results pinpoint invalid rows.
    add_executable(test_shim_batch_mt tests/test_shim_batch_mt.cpp)
    target_link_libraries(test_shim_batch_mt PRIVATE secp256k1_shim)
    add_test(NAME shim_batch_mt COMMAND test_shim_batch_mt)
    set_tests_properties(shim_batch_mt PROPERTIES LABELS "shim")

    # External cancellation token on all batch-verify functions: NULL parity,
    # immediate + mid-batch cancel, throwing callback fail-closed, per-row partial.
    add_executable(test_shim_batch_cancel tests/test_shim_batch_cancel.cpp)
    target_link_libraries(test_shim_batch_cancel PRIVATE secp256k1_shim)
    add_test(NAME shim_batch_cancel COMMAND test_shim_batch_cancel)
    set_tests_properties(shim_batch_cancel PROPERTIES LABELS "shim")
endif()
