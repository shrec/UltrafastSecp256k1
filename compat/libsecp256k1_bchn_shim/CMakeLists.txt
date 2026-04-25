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

# BCH shim public headers (secp256k1.h, secp256k1_recovery.h,
# secp256k1_ecdh.h, secp256k1_schnorr.h)
target_include_directories(secp256k1_bchn_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TARGET secp256k1_fast)
    target_link_libraries(secp256k1_bchn_shim PRIVATE secp256k1_fast)
else()
    message(WARNING "secp256k1_fast target not found -- add UltrafastSecp256k1 via add_subdirectory first")
endif()
