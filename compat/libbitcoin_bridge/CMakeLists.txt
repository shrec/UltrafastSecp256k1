cmake_minimum_required(VERSION 3.18)
project(ufsecp_libbitcoin_bridge LANGUAGES CXX)

# UltrafastSecp256k1 ⇄ libbitcoin acceleration bridge.
#
# Two build modes:
#   A) Integrated: added via add_subdirectory() under the engine build, where the
#      `fastsecp256k1` target exists. Links it directly.
#   B) Standalone: built on its own; locates an installed/!prebuilt `ufsecp`
#      library via UFSECP_LIBRARY / find_library and the engine headers via
#      UFSECP_INCLUDE_DIR.
#
# GPU acceleration is OFF by default (CPU-only consensus reference). Turn on with
# -DUFSECP_LBTC_WITH_GPU=ON when linking an engine library that exports the GPU
# ABI (ufsecp_gpu_*).

option(UFSECP_LBTC_WITH_GPU "Enable GPU dispatch (requires engine GPU ABI)" OFF)
option(UFSECP_LBTC_BUILD_TESTS "Build the bridge correctness test" ON)
option(UFSECP_LBTC_BUILD_EXAMPLE "Build the example harness" ON)
option(UFSECP_LBTC_BUILD_BENCH "Build the bridge throughput benchmark" ON)

add_library(ufsecp_lbtc_bridge STATIC src/ufsecp_libbitcoin.cpp)
target_compile_features(ufsecp_lbtc_bridge PUBLIC cxx_std_17)
target_include_directories(ufsecp_lbtc_bridge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Engine headers (ufsecp.h, ufsecp_gpu.h). Override with -DUFSECP_INCLUDE_DIR=...
if(NOT UFSECP_INCLUDE_DIR)
    set(UFSECP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ufsecp)
endif()
target_include_directories(ufsecp_lbtc_bridge PUBLIC ${UFSECP_INCLUDE_DIR})

if(UFSECP_LBTC_WITH_GPU)
    target_compile_definitions(ufsecp_lbtc_bridge PUBLIC UFSECP_LBTC_WITH_GPU)
endif()

# Link the engine onto an arbitrary target (used by the bridge lib and by the
# smallchunk test, which recompiles the bridge source with a tiny kChunk).
function(ufsecp_lbtc_link_engine tgt access)
    if(TARGET fastsecp256k1)
        target_link_libraries(${tgt} ${access} fastsecp256k1)             # mode A
    elseif(TARGET secp256k1_fast)
        target_link_libraries(${tgt} ${access} secp256k1_fast)            # mode A
    else()
        find_library(UFSECP_LIBRARY NAMES ufsecp                          # mode B
            HINTS ${UFSECP_LIB_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../out)
        if(UFSECP_LIBRARY)
            target_link_libraries(${tgt} ${access} ${UFSECP_LIBRARY})
        else()
            message(WARNING
                "ufsecp_lbtc_bridge: no engine target/library found. "
                "Set -DUFSECP_LIBRARY=/path/to/libufsecp.so or build under the engine.")
        endif()
    endif()
endfunction()

ufsecp_lbtc_link_engine(ufsecp_lbtc_bridge PUBLIC)

if(UFSECP_LBTC_BUILD_TESTS)
    enable_testing()
    add_executable(test_lbtc_bridge tests/test_lbtc_bridge.cpp)
    # C++20 so the test can exercise the typed-span overloads (std::span) the way a
    # libbitcoin consumer does; the bridge header guards them behind __cplusplus>=202002L.
    target_compile_features(test_lbtc_bridge PRIVATE cxx_std_20)
    target_link_libraries(test_lbtc_bridge PRIVATE ufsecp_lbtc_bridge)
    add_test(NAME lbtc_bridge COMMAND test_lbtc_bridge)

    # GPU-vs-CPU consensus differential for the script-signature batch. The GPU path
    # is consensus-bearing, so it must agree bit-for-bit with the CPU reference on
    # every per-row accept/reject verdict (the CPU path is itself gated bit-for-bit
    # against libsecp256k1, so GPU==CPU here means GPU==libsecp transitively).
    # Builds STANDALONE_TEST so main() is provided. When no GPU backend is present
    # the test advisory-skips (exit 77); SKIP_RETURN_CODE makes CTest report that as
    # "skipped" rather than "failed" (GPU is local-only; absent on CI runners).
    add_executable(test_lbtc_consensus_diff tests/test_lbtc_consensus_diff.cpp)
    target_compile_definitions(test_lbtc_consensus_diff PRIVATE STANDALONE_TEST)
    target_link_libraries(test_lbtc_consensus_diff PRIVATE ufsecp_lbtc_bridge)
    # Direct libsecp256k1 leg (optional): proves GPU==CPU==libsecp directly rather
    # than transitively. Falls back to GPU==CPU when the shim target is absent.
    if(TARGET secp256k1_shim)
        target_link_libraries(test_lbtc_consensus_diff PRIVATE secp256k1_shim)
        target_include_directories(test_lbtc_consensus_diff PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../libsecp256k1_shim/include)
        target_compile_definitions(test_lbtc_consensus_diff PRIVATE UFSECP_LBTC_HAVE_LIBSECP)
    endif()
    add_test(NAME lbtc_consensus_diff COMMAND test_lbtc_consensus_diff)
    set_tests_properties(lbtc_consensus_diff PROPERTIES SKIP_RETURN_CODE 77)

    # In-place "collect" verify: verdict collapsed into each row's key cell.
    add_executable(test_lbtc_collect tests/test_lbtc_collect.cpp)
    target_compile_features(test_lbtc_collect PRIVATE cxx_std_20) # std::span collect overload
    target_link_libraries(test_lbtc_collect PRIVATE ufsecp_lbtc_bridge)
    add_test(NAME lbtc_collect COMMAND test_lbtc_collect)

    # 4-table batching model (libbitcoin): multisig (ECDSA m-of-n) + threshold
    # (Schnorr/tapscript m-of-n) verify/collect through the ECDSA/Schnorr cores
    # with a 6-byte m|n+group+block-fk tail; proves tail-width independence and
    # the canonical MultisigRow/ThresholdRow typed-span overloads.
    add_executable(test_lbtc_multisig_threshold tests/test_lbtc_multisig_threshold.cpp)
    target_compile_features(test_lbtc_multisig_threshold PRIVATE cxx_std_20)
    target_link_libraries(test_lbtc_multisig_threshold PRIVATE ufsecp_lbtc_bridge)
    add_test(NAME lbtc_multisig_threshold COMMAND test_lbtc_multisig_threshold)

    # Same source, but the bridge is RECOMPILED with a tiny kChunk so the corpus
    # straddles chunk boundaries — exercises CollectSink's (base+i)*stride global
    # indexing across chunks deterministically (no 262k-row corpus needed).
    add_executable(test_lbtc_collect_smallchunk
        tests/test_lbtc_collect.cpp src/ufsecp_libbitcoin.cpp)
    target_compile_features(test_lbtc_collect_smallchunk PRIVATE cxx_std_20)
    target_compile_definitions(test_lbtc_collect_smallchunk
        PRIVATE UFSECP_LBTC_CHUNK_OVERRIDE=8)
    if(UFSECP_LBTC_WITH_GPU)
        target_compile_definitions(test_lbtc_collect_smallchunk PRIVATE UFSECP_LBTC_WITH_GPU)
    endif()
    target_include_directories(test_lbtc_collect_smallchunk
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${UFSECP_INCLUDE_DIR})
    ufsecp_lbtc_link_engine(test_lbtc_collect_smallchunk PRIVATE)
    add_test(NAME lbtc_collect_smallchunk COMMAND test_lbtc_collect_smallchunk)
endif()

if(UFSECP_LBTC_BUILD_EXAMPLE)
    add_executable(lbtc_batch_demo example/lbtc_batch_demo.cpp)
    target_link_libraries(lbtc_batch_demo PRIVATE ufsecp_lbtc_bridge)
endif()

if(UFSECP_LBTC_BUILD_BENCH)
    add_executable(bench_lbtc_batch bench/bench_lbtc_batch.cpp)
    target_link_libraries(bench_lbtc_batch PRIVATE ufsecp_lbtc_bridge)
endif()
