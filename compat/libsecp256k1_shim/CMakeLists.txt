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

    # Public headers (secp256k1.h, secp256k1_schnorrsig.h, etc.)
    target_include_directories(secp256k1_shim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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

    # Public includes -- exposes libsecp256k1-compatible headers
    target_include_directories(secp256k1_shim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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

# -- Optional: shim API test ---------------------------------------------------

option(SECP256K1_SHIM_BUILD_TESTS "Build shim compatibility test" OFF)

if(SECP256K1_SHIM_BUILD_TESTS AND NOT SECP256K1_BUILD_SHIM)
    enable_testing()
    add_executable(shim_test tests/shim_test.cpp)
    target_link_libraries(shim_test PRIVATE secp256k1_shim)
    add_test(NAME secp256k1_shim_test COMMAND shim_test)
    set_tests_properties(secp256k1_shim_test PROPERTIES LABELS "shim")
endif()
