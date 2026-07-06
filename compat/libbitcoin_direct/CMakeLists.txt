# ============================================================================
# libbitcoin direct integration — canonical, bridge-free C++ surface
# ============================================================================
# Header-only: <ufsecp/libbitcoin.hpp> (ufsecp::lbtc::* inline functions) calling
# straight into the engine. This is the ONE target a libbitcoin consumer links:
# it carries the include dir for <ufsecp/libbitcoin.hpp>, C++20, HAVE_ULTRAFAST,
# and the engine (secp256k1::fastsecp256k1) as transitive usage requirements.
# No libsecp256k1 shim, no ufsecp C ABI, no ufsecp_lbtc bridge are involved.
#
# Added from the top-level CMake under SECP256K1_BUILD_LIBBITCOIN, after the
# engine target (fastsecp256k1) already exists.

# Resolve the in-tree engine target. The canonical libbitcoin profile builds a
# package-friendly static engine target; OBJECT mode also works for embedders
# that explicitly opt into it outside the installed package path.
if(TARGET fastsecp256k1)
    set(_lbtc_engine fastsecp256k1)
elseif(TARGET secp256k1::fastsecp256k1)
    set(_lbtc_engine secp256k1::fastsecp256k1)
else()
    message(FATAL_ERROR
        "libbitcoin_direct: engine target 'fastsecp256k1' not found "
        "(add_subdirectory(src/cpu) must run first)")
endif()

# Canonical libbitcoin integration target (header-only INTERFACE library).
add_library(fastsecp256k1_libbitcoin INTERFACE)
add_library(secp256k1::fastsecp256k1_libbitcoin ALIAS fastsecp256k1_libbitcoin)

target_include_directories(fastsecp256k1_libbitcoin INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(fastsecp256k1_libbitcoin INTERFACE cxx_std_20)
target_compile_definitions(fastsecp256k1_libbitcoin INTERFACE HAVE_ULTRAFAST)
target_link_libraries(fastsecp256k1_libbitcoin INTERFACE ${_lbtc_engine})

# GPU accelerator host library: when SECP256K1_BUILD_LIBBITCOIN_GPU=ON and at least one
# GPU backend (CUDA/OpenCL/Metal) is compiled, the self-installing GpuColumnsVerifyHook
# lives in secp256k1_gpu_host. Linking it into the test/bench executables transparently
# accelerates the ufsecp::lbtc::ecdsa_verify_columns / schnorr_verify_columns calls via
# the engine's GpuColumnsVerifyHook mechanism — no caller-visible GPU API or status
# change. The hook falls back to CPU when no GPU backend is available at runtime.
# GPU is an internal math-engine tool, not a caller-facing feature.
if(SECP256K1_BUILD_LIBBITCOIN_GPU)
    if(TARGET secp256k1_gpu_host)
        set(_lbtc_gpu_host secp256k1_gpu_host)
        # Retain the self-installing hook object (gpu_engine_hook.cpp) at link. It installs
        # the engine's GpuColumnsVerifyHook purely through a file-scope static initializer
        # (EngineGpuColumnsInstaller) and exports no symbol these executables reference, so a
        # normal static link of secp256k1_gpu_host discards the whole object and the hook
        # never installs — "transparent GPU column verify" then silently stays CPU-only
        # (verified: nm shows the installer absent from the executable without this).
        #
        # Retention is done by declaring the provider's exported anchor symbol as UNDEFINED
        # at the executable link, so the linker pulls exactly that one archive member (and its
        # genuine transitive deps). A blanket $<LINK_LIBRARY:WHOLE_ARCHIVE,secp256k1_gpu_host>
        # is deliberately NOT used: it would also drag in gpu_backend_fallback.o, whose
        # snark-witness CPU fallback references secp256k1::zk::* symbols that are not part of
        # the minimal libbitcoin engine, breaking the link. Targeted -u retention pulls only
        # the hook. (The ufsecp C ABI path retains the same object via a source-level anchor
        # reference in ufsecp_gpu_impl.cpp — see src/gpu/src/gpu_engine_hook.cpp and
        # src/gpu/CMakeLists.txt "LBTC-GPU-SELFINSTALL-DROP".)
        set(_lbtc_gpu_retain "LINKER:--undefined=secp256k1_gpu_columns_provider_anchor")
        # Enable the seven libbitcoin public-data batch-op GPU trampolines in
        # gpu_engine_hook.cpp (guarded by SECP256K1_LBTC_GPU_OPS) and give that TU
        # the include path for <ufsecp/lbtc_gpu_ops.hpp>. Both are target-level
        # properties on the already-created secp256k1_gpu_host target; CMake
        # collects them across all subdirectories before generation, so they take
        # effect even though compat/ is add_subdirectory'd after src/gpu. The guard
        # stays off for non-libbitcoin GPU builds (no include-path dependency there).
        # No new backend source, no new anchor: the self-installing
        # EngineLbtcOpsInstaller rides the same TU/anchor as the column provider.
        target_include_directories(secp256k1_gpu_host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_compile_definitions(secp256k1_gpu_host PRIVATE SECP256K1_LBTC_GPU_OPS)
        message(STATUS "  libbitcoin direct GPU: ON (SECP256K1_BUILD_LIBBITCOIN_GPU=ON; secp256k1_gpu_host linked, hook retained via -u anchor — transparent GPU column verify + public-data batch ops)")
    else()
        set(_lbtc_gpu_host "")
        set(_lbtc_gpu_retain "")
        message(STATUS "  libbitcoin direct GPU: requested but no GPU backend compiled (CPU-only column verify)")
    endif()
else()
    set(_lbtc_gpu_host "")
    set(_lbtc_gpu_retain "")
    message(STATUS "  libbitcoin direct GPU: OFF (SECP256K1_BUILD_LIBBITCOIN_GPU=OFF — CPU-only column verify)")
endif()

# Canonical libbitcoin integration test: direct ECDSA + Schnorr verify
# (single + batch + columns + fail-closed). Registered as `lbtc_direct_verify`.
if(SECP256K1_BUILD_LIBBITCOIN_TESTS)
    add_executable(test_lbtc_direct_verify tests/test_direct_verify.cpp)
    # Link the interface target (headers + defines + C++20) and the engine
    # directly so its objects/archive resolve at link time. When the GPU host
    # is available, link it too so the self-installing hook accelerates column
    # verify transparently (CPU fallback when no GPU runtime).
    target_link_libraries(test_lbtc_direct_verify PRIVATE
        fastsecp256k1_libbitcoin
        ${_lbtc_gpu_host}
        ${_lbtc_engine})
    if(_lbtc_gpu_retain)
        target_link_options(test_lbtc_direct_verify PRIVATE ${_lbtc_gpu_retain})
    endif()
    find_package(Threads QUIET)
    if(TARGET Threads::Threads)
        target_link_libraries(test_lbtc_direct_verify PRIVATE Threads::Threads)
    endif()
    add_test(NAME lbtc_direct_verify COMMAND test_lbtc_direct_verify)
    set_tests_properties(lbtc_direct_verify PROPERTIES LABELS "libbitcoin;direct;verify")

    # libbitcoin direct integration test: signing, key operations, recovery, utilities.
    add_executable(test_lbtc_direct_operations tests/test_direct_operations.cpp)
    target_link_libraries(test_lbtc_direct_operations PRIVATE
        fastsecp256k1_libbitcoin
        ${_lbtc_gpu_host}
        ${_lbtc_engine})
    if(_lbtc_gpu_retain)
        target_link_options(test_lbtc_direct_operations PRIVATE ${_lbtc_gpu_retain})
    endif()
    find_package(Threads QUIET)
    if(TARGET Threads::Threads)
        target_link_libraries(test_lbtc_direct_operations PRIVATE Threads::Threads)
    endif()
    add_test(NAME lbtc_direct_operations COMMAND test_lbtc_direct_operations)
    set_tests_properties(lbtc_direct_operations PROPERTIES LABELS "libbitcoin;direct;operations")

    # GPU column-verify provider/hook runtime smoke — built ONLY in the direct-GPU
    # opt-in profile (SECP256K1_BUILD_LIBBITCOIN_GPU=ON with a GPU backend compiled,
    # i.e. _lbtc_gpu_host resolved above). Asserts the engine's GpuColumnsVerifyHook
    # is self-installed at startup by secp256k1_gpu_host (retained here via the same
    # -u anchor) BEFORE any test double clears it, then exercises the transparent
    # column path + fail-closed. Registered as `lbtc_direct_gpu_columns_hook` so the
    # direct profile's own `ctest -N` / target help carries the acceptance smoke —
    # NOT the audit/test_gpu_lbtc_columns_diff.cpp or C ABI path.
    if(_lbtc_gpu_host)
        add_executable(test_lbtc_direct_gpu_columns_hook tests/test_direct_gpu_columns_hook.cpp)
        target_link_libraries(test_lbtc_direct_gpu_columns_hook PRIVATE
            fastsecp256k1_libbitcoin
            ${_lbtc_gpu_host}
            ${_lbtc_engine})
        if(_lbtc_gpu_retain)
            target_link_options(test_lbtc_direct_gpu_columns_hook PRIVATE ${_lbtc_gpu_retain})
        endif()
        find_package(Threads QUIET)
        if(TARGET Threads::Threads)
            target_link_libraries(test_lbtc_direct_gpu_columns_hook PRIVATE Threads::Threads)
        endif()
        add_test(NAME lbtc_direct_gpu_columns_hook COMMAND test_lbtc_direct_gpu_columns_hook)
        set_tests_properties(lbtc_direct_gpu_columns_hook PROPERTIES
            LABELS "libbitcoin;direct;gpu;hook;columns")
        message(STATUS "  libbitcoin direct GPU hook smoke: ON (lbtc_direct_gpu_columns_hook)")
    endif()
    message(STATUS "  libbitcoin direct test: ON (lbtc_direct_verify, lbtc_direct_operations)")
endif()

# Canonical direct C++ benchmark. This intentionally links only the direct
# interface target and the engine: no ufsecp C ABI, no shim, no ufsecp_lbtc bridge.
if(SECP256K1_BUILD_LIBBITCOIN_BENCH)
    add_executable(bench_lbtc_direct_batch bench/bench_direct_batch.cpp)
    target_link_libraries(bench_lbtc_direct_batch PRIVATE
        fastsecp256k1_libbitcoin
        ${_lbtc_gpu_host}
        ${_lbtc_engine})
    if(_lbtc_gpu_retain)
        target_link_options(bench_lbtc_direct_batch PRIVATE ${_lbtc_gpu_retain})
    endif()
    find_package(Threads QUIET)
    if(TARGET Threads::Threads)
        target_link_libraries(bench_lbtc_direct_batch PRIVATE Threads::Threads)
    endif()

    add_executable(bench_lbtc_hash256_var bench/bench_hash256_var.cpp)
    target_link_libraries(bench_lbtc_hash256_var PRIVATE
        fastsecp256k1_libbitcoin
        ${_lbtc_gpu_host}
        ${_lbtc_engine})
    if(_lbtc_gpu_retain)
        target_link_options(bench_lbtc_hash256_var PRIVATE ${_lbtc_gpu_retain})
    endif()
    if(TARGET Threads::Threads)
        target_link_libraries(bench_lbtc_hash256_var PRIVATE Threads::Threads)
    endif()
    message(STATUS "  libbitcoin direct bench: ON (bench_lbtc_direct_batch, bench_lbtc_hash256_var, no C ABI/shim/bridge)")
endif()
