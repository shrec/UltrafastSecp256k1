cmake_minimum_required(VERSION 3.18)
project(secp256k1_bchn_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -- Shim library (Bitcoin Cash Node / BCHN secp256k1 API) ---------------------
# Reuses the core libsecp256k1_shim source files for context, pubkeys, ECDSA,
# secret keys, tagged hash, recovery, and ECDH.
# Adds BCH-specific legacy Schnorr (secp256k1_schnorr.h, not BIP-340).

set(CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../libsecp256k1_shim/src")

add_library(secp256k1_bchn_shim STATIC
    ${CORE_SRC}/shim_context.cpp
    ${CORE_SRC}/shim_pubkey.cpp
    ${CORE_SRC}/shim_ecdsa.cpp
    ${CORE_SRC}/shim_seckey.cpp
    ${CORE_SRC}/shim_tagged_hash.cpp
    ${CORE_SRC}/shim_recovery.cpp
    ${CORE_SRC}/shim_ecdh.cpp
    src/shim_schnorr_bch.cpp
)

# BCH shim public headers. The BCH shim owns secp256k1_schnorr.h and reuses
# the shared shim's secp256k1.h, recovery, and ECDH compatibility headers.
target_include_directories(secp256k1_bchn_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../libsecp256k1_shim/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu/include
)

if(TARGET fastsecp256k1)
    target_link_libraries(secp256k1_bchn_shim PRIVATE fastsecp256k1)
elseif(TARGET secp256k1_fast)
    target_link_libraries(secp256k1_bchn_shim PRIVATE secp256k1_fast)
else()
    message(WARNING "fastsecp256k1 target not found -- add UltrafastSecp256k1 via add_subdirectory first")
endif()

option(SECP256K1_BCHN_SHIM_BUILD_TESTS "Build BCHN shim compatibility tests" OFF)

if(SECP256K1_BCHN_SHIM_BUILD_TESTS)
    enable_testing()
    add_executable(test_bchn_schnorr_fail_clear tests/test_bchn_schnorr_fail_clear.cpp)
    target_link_libraries(test_bchn_schnorr_fail_clear PRIVATE secp256k1_bchn_shim)
    add_test(NAME bchn_schnorr_fail_clear COMMAND test_bchn_schnorr_fail_clear)
    set_tests_properties(bchn_schnorr_fail_clear PROPERTIES LABELS "shim;bchn")
endif()
