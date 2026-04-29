cmake_minimum_required(VERSION 3.18)
project(secp256k1_shim LANGUAGES CXX)

# -- Shim library --------------------------------------------------------------
# The shim implementation uses C++20 internally (it includes UF C++ headers),
# but exposes a pure C API.  Use PRIVATE so the C++20 requirement does NOT
# propagate to consumers (e.g. Bitcoin Core which targets C++17).

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
)

# C++20 is needed to compile the shim (it includes UF C++ headers),
# but must not leak to consumers — PRIVATE keeps it shim-internal.
target_compile_features(secp256k1_shim PRIVATE cxx_std_20)

# Public includes -- exposes libsecp256k1-compatible headers
target_include_directories(secp256k1_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against UltrafastSecp256k1 CPU library.
# The internal target is 'fastsecp256k1'; 'secp256k1_fast' is an older alias
# kept for backward compatibility. Try both names.
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

# -- Optional: shim API test ---------------------------------------------------

option(SECP256K1_SHIM_BUILD_TESTS "Build shim compatibility test" OFF)

if(SECP256K1_SHIM_BUILD_TESTS)
    enable_testing()
    add_executable(shim_test tests/shim_test.cpp)
    target_link_libraries(shim_test PRIVATE secp256k1_shim)
    add_test(NAME secp256k1_shim_test COMMAND shim_test)
    set_tests_properties(secp256k1_shim_test PROPERTIES LABELS "shim")
endif()
