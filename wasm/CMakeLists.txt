# ============================================================================
# UltrafastSecp256k1 -- WebAssembly (Emscripten) Build
# ============================================================================
# Usage (standalone):
#   emcmake cmake -S wasm -B build-wasm -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-wasm -j
#
# Or use the build script: ./scripts/build_wasm.sh
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# Read version from VERSION.txt (single source of truth)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION.txt" _version_raw)
string(STRIP "${_version_raw}" UFSECP_VERSION)

project(UltrafastSecp256k1Wasm
    VERSION ${UFSECP_VERSION}
    LANGUAGES CXX
    DESCRIPTION "UltrafastSecp256k1 WASM bindings"
)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR
        "This CMakeLists.txt is for Emscripten/WebAssembly builds only.\n"
        "Use: emcmake cmake -S wasm -B build-wasm"
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "======================================")
message(STATUS "UltrafastSecp256k1 WASM Build")
message(STATUS "  Emscripten:  ${EMSCRIPTEN_VERSION}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "======================================")

# -- CPU library (portable mode, no ASM) --------------------------------------
set(SECP256K1_USE_ASM OFF CACHE BOOL "" FORCE)
set(SECP256K1_USE_FAST_REDUCTION OFF CACHE BOOL "" FORCE)
set(SECP256K1_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpu ${CMAKE_CURRENT_BINARY_DIR}/cpu)

# -- WASM module --------------------------------------------------------------

# Exported C functions for JS interop
set(WASM_EXPORTED_FUNCTIONS
    "_secp256k1_wasm_selftest"
    "_secp256k1_wasm_version"
    "_secp256k1_wasm_pubkey_create"
    "_secp256k1_wasm_point_mul"
    "_secp256k1_wasm_point_add"
    "_secp256k1_wasm_ecdsa_sign"
    "_secp256k1_wasm_ecdsa_verify"
    "_secp256k1_wasm_schnorr_sign"
    "_secp256k1_wasm_schnorr_verify"
    "_secp256k1_wasm_schnorr_pubkey"
    "_secp256k1_wasm_sha256"
    "_malloc"
    "_free"
)

# Join into comma-separated string
list(JOIN WASM_EXPORTED_FUNCTIONS "," EXPORTED_FUNCTIONS_STR)

add_executable(secp256k1_wasm secp256k1_wasm.cpp)

target_link_libraries(secp256k1_wasm PRIVATE fastsecp256k1)

target_include_directories(secp256k1_wasm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Emscripten link flags
target_link_options(secp256k1_wasm PRIVATE
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_ES6=1"
    "SHELL:-s EXPORT_NAME=createSecp256k1"
    "SHELL:-s EXPORTED_FUNCTIONS=[${EXPORTED_FUNCTIONS_STR}]"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,getValue,setValue,UTF8ToString,stackAlloc,stackSave,stackRestore]"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s INITIAL_MEMORY=4194304"      # 4 MB initial (selftest uses vectors)
    "SHELL:-s STACK_SIZE=524288"            # 512 KB stack
    "SHELL:-s WASM=1"
    "SHELL:-s NO_EXIT_RUNTIME=1"
    "SHELL:-s FILESYSTEM=0"               # No FS needed
    "SHELL:-s SINGLE_FILE=0"              # Separate .wasm file
    "SHELL:-s ENVIRONMENT=web,node,worker"
)

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(secp256k1_wasm PRIVATE -O3 -flto -fno-exceptions)
    target_link_options(secp256k1_wasm PRIVATE
        -O3 -flto
        "SHELL:-s ASSERTIONS=0"
        # NOTE: --closure 1 removed -- it breaks atexit/exception runtime
    )
else()
    target_link_options(secp256k1_wasm PRIVATE
        "SHELL:-s ASSERTIONS=2"
        "SHELL:-s SAFE_HEAP=1"
        "SHELL:-s STACK_OVERFLOW_CHECK=2"
    )
endif()

# Output: secp256k1_wasm.js + secp256k1_wasm.wasm
set_target_properties(secp256k1_wasm PROPERTIES
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
)

# Copy JS wrapper and TypeScript declarations to dist/
add_custom_command(TARGET secp256k1_wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/secp256k1.mjs"
        "${CMAKE_BINARY_DIR}/dist/secp256k1.mjs"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/secp256k1.d.ts"
        "${CMAKE_BINARY_DIR}/dist/secp256k1.d.ts"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/package.json"
        "${CMAKE_BINARY_DIR}/dist/package.json"
    COMMENT "Copying JS/TS wrappers to dist/"
)

message(STATUS "")
message(STATUS "WASM build configured. Output will be in build-wasm/dist/")
message(STATUS "  secp256k1_wasm.js    -- Emscripten loader (ES6 module)")
message(STATUS "  secp256k1_wasm.wasm  -- WebAssembly binary")
message(STATUS "  secp256k1.mjs        -- High-level JS wrapper")
message(STATUS "  secp256k1.d.ts       -- TypeScript declarations")
message(STATUS "")
