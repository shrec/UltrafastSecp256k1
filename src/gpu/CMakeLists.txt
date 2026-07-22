# ============================================================================
# UltrafastSecp256k1 -- GPU Host Ops Layer (Layer 2)
# ============================================================================
# Builds the backend-neutral GPU host operations library.
# Links against available GPU backends (CUDA, OpenCL, Metal) depending on
# which are compiled.
#
# This directory provides:
#   - gpu_backend.hpp      (abstract C++ interface for backends)
#   - gpu_registry.cpp     (backend factory / discovery)
#   - gpu_backend_*.cpp/cu (one per backend)
#
# The C ABI (ufsecp_gpu_impl.cpp) lives in include/ufsecp/ alongside the
# existing CPU C ABI, and links against this target.
# ============================================================================

cmake_minimum_required(VERSION 3.18)

set(GPU_REGISTRY_SRC
    src/gpu_registry.cpp
    src/gpu_engine_hook.cpp
)

# gpu_backend_fallback.cpp provides the Schnorr SNARK-witness host CPU fallback,
# which calls secp256k1::zk::schnorr_snark_witness — a symbol that exists ONLY when
# the ZK module is compiled into the engine. Under SECP256K1_BUILD_LIBBITCOIN the
# top-level CMake forces SECP256K1_BUILD_ZK=OFF, so compiling this TU would leave an
# undefined zk reference that a provider-retention link (-u anchor, whole-archive)
# drags in and fails on (LBTC-GPU-DIRECT-ZK-BLOCKER). The matching default in
# gpu_backend.hpp is #if SECP256K1_HAS_ZK-guarded, so with ZK off nothing references
# the fallback and the TU is simply omitted; with ZK on it is compiled and the
# default calls it. SECP256K1_BUILD_ZK is read from the same top-level scope as
# src/cpu (both add_subdirectory'd after the LIBBITCOIN profile sets it), so this
# stays byte-consistent with whether the engine actually defines the zk symbol.
if(SECP256K1_BUILD_ZK)
    list(APPEND GPU_REGISTRY_SRC src/gpu_backend_fallback.cpp)
endif()

set(GPU_BACKEND_SOURCES "")
set(GPU_BACKEND_DEFS "")
set(GPU_BACKEND_LIBS "")

# -- CUDA backend -----------------------------------------------------------
if(SECP256K1_BUILD_CUDA AND TARGET secp256k1_cuda_lib)
    list(APPEND GPU_BACKEND_SOURCES src/gpu_backend_cuda.cu)
    list(APPEND GPU_BACKEND_DEFS SECP256K1_HAVE_CUDA=1)
    list(APPEND GPU_BACKEND_LIBS secp256k1_cuda_lib)
    message(STATUS "  GPU API: CUDA backend enabled")
endif()

# -- OpenCL backend ---------------------------------------------------------
if(SECP256K1_BUILD_OPENCL AND TARGET secp256k1_opencl)
    list(APPEND GPU_BACKEND_SOURCES src/gpu_backend_opencl.cpp)
    list(APPEND GPU_BACKEND_DEFS SECP256K1_HAVE_OPENCL=1)
    list(APPEND GPU_BACKEND_LIBS secp256k1_opencl)
    message(STATUS "  GPU API: OpenCL backend enabled")
endif()

# -- Metal backend ----------------------------------------------------------
if(SECP256K1_BUILD_METAL AND TARGET secp256k1_metal_lib)
    list(APPEND GPU_BACKEND_SOURCES src/gpu_backend_metal.mm)
    list(APPEND GPU_BACKEND_DEFS SECP256K1_HAVE_METAL=1)
    list(APPEND GPU_BACKEND_LIBS secp256k1_metal_lib)
    message(STATUS "  GPU API: Metal backend enabled")
endif()

# Only build if at least one backend is available
list(LENGTH GPU_BACKEND_SOURCES _num_backends)
if(_num_backends EQUAL 0)
    message(STATUS "  GPU API: no backends available -- skipping gpu/ build")
    return()
endif()

# -- Static library target --------------------------------------------------
add_library(secp256k1_gpu_host STATIC
    ${GPU_REGISTRY_SRC}
    ${GPU_BACKEND_SOURCES}
)

# ---------------------------------------------------------------------------
# Retaining the self-installing GPU column-verify hook (LBTC-GPU-SELFINSTALL-DROP)
# ---------------------------------------------------------------------------
# gpu_engine_hook.cpp installs the engine's GpuColumnsVerifyHook purely through a
# file-scope static initializer (EngineGpuColumnsInstaller); it exports no symbol
# any consumer references, so a normal static link discards the whole object and the
# hook never installs — "transparent GPU column verify" silently stays CPU-only.
#
# A target-level `INTERFACE $<LINK_LIBRARY:WHOLE_ARCHIVE,secp256k1_gpu_host>` does NOT
# work: CMake silently drops the self-referential entry (self-link cycle avoidance),
# so consumers still link the archive without --whole-archive. Retention is therefore
# handled at the two consumer topologies instead:
#   * ufsecp C ABI / audit: gpu_engine_hook.cpp exports a retention anchor referenced
#     by the always-linked C ABI TU (src/cpu/src/ufsecp_gpu_impl.cpp), which drags the
#     provider object into libufsecp — see that file. (Source-level, so it also covers
#     the non-writable include/ufsecp link line.)
#   * libbitcoin-direct: it references no gpu_host symbol at all, so its executables
#     force the provider object in with a linker -u (--undefined) on the anchor at
#     their own link (compat/libbitcoin_direct/CMakeLists.txt). Targeted -u retention
#     is preferred over a blanket WHOLE_ARCHIVE because it pulls only the provider
#     object, not every backend TU. (Historically WHOLE_ARCHIVE also broke the link by
#     dragging in gpu_backend_fallback.o and its undefined secp256k1::zk:: reference;
#     that is now independently prevented — the fallback TU is #if SECP256K1_HAS_ZK-
#     gated and dropped from GPU_REGISTRY_SRC when ZK is off, see above and
#     gpu_backend.hpp. LBTC-GPU-DIRECT-ZK-BLOCKER.)

# Issue #335 acceptance repair (round 11): this directory was PUBLIC, an
# unwrapped source-tree-absolute path -- CMake refuses to export a target
# whose PUBLIC/INTERFACE include dirs are not BUILD_INTERFACE/INSTALL_INTERFACE-
# wrapped. Checked every current consumer before demoting to PRIVATE (round 10's
# theorized fix would have been to wrap it instead, which also would have
# worked but would additionally require installing this header pair into the
# public package tree for no real consumer):
#   - include/ufsecp/CMakeLists.txt's ufsecp_configure() (the only in-tree
#     consumer that links secp256k1_gpu_host into an installed target) already
#     explicitly adds this exact directory itself, PRIVATE, one level up
#     ("Link GPU host ops layer if available" block) -- this PUBLIC copy was
#     entirely redundant for it.
#   - compat/libbitcoin_direct/CMakeLists.txt links secp256k1_gpu_host into
#     several executables but only for its self-installing static-initializer
#     retention anchor (LBTC-GPU-SELFINSTALL-DROP, `-u` linker retention) --
#     none of its headers/tests #include gpu_backend.hpp or
#     gpu_cuda_host_helpers.h, so none of them compile against this path.
# PRIVATE is therefore correct: it keeps this target's own two TUs compiling
# unchanged and removes the export-interface requirement entirely (no
# INSTALL_INTERFACE / install(DIRECTORY) needed for this internal host-side
# abstraction header -- the STABLE public GPU surface is ufsecp_gpu.h,
# installed separately by the root CMakeLists.txt `if(TARGET secp256k1_gpu_host)`).
target_include_directories(secp256k1_gpu_host
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Pass backend availability defines
target_compile_definitions(secp256k1_gpu_host PRIVATE ${GPU_BACKEND_DEFS})

# Issue #335 acceptance repair (round 9; DEMOTED round 10): bake the
# compile-time install prefix for OpenCL kernel sources into
# secp256k1_gpu_host itself. This used to be gpu_backend_opencl.cpp's
# resolve_opencl_kernel() PRIMARY production/installed-package discovery
# strategy; round 10 replaced that role with a genuinely RELOCATABLE
# runtime discovery mechanism (dladdr()/GetModuleHandleExA on the loaded
# module's own current on-disk path -- resolve_opencl_kernel() Strategy 2)
# because this compile-time-baked path is NOT relocatable: it silently
# breaks the moment an installed package is moved to a different prefix
# after `cmake --install` (round 10's acceptance bar -- proven with a real
# `cmake --install` to prefix A, `mv` to prefix B, prefix A deleted, then a
# fresh consumer linked against prefix B succeeds with this macro's baked
# value pointing at a prefix that no longer exists). This macro is now
# kept ONLY as defense-in-depth Strategy 3, tried after the relocatable
# strategy fails, for the narrower case of an installed-but-never-moved
# package. CMAKE_INSTALL_PREFIX is always defined by CMake (defaults to
# /usr/local or platform equivalent) even without an install() rule in
# THIS file. The path intentionally matches, byte for byte,
# src/opencl/CMakeLists.txt's own
# `install(DIRECTORY kernels/ DESTINATION share/secp256k1/opencl ...)` rule
# -- the two must stay in sync so this fallback stays valid for an
# un-relocated install/packaged build. This is a fallback discovery path,
# not a user override: resolve_opencl_kernel() still checks the explicit
# UFSECP_OPENCL_KERNEL_DIR environment-variable override FIRST (round 9
# fixed that override to fail closed rather than silently falling through
# to weaker strategies).
if(SECP256K1_BUILD_OPENCL AND TARGET secp256k1_opencl)
    target_compile_definitions(secp256k1_gpu_host PRIVATE
        SECP256K1_GPU_OPENCL_INSTALL_DIR="${CMAKE_INSTALL_PREFIX}/share/secp256k1/opencl")
endif()

# Link CPU library (needed for host-side type conversions)
if(TARGET fastsecp256k1)
    target_link_libraries(secp256k1_gpu_host PRIVATE fastsecp256k1)
endif()

# Link GPU backend libraries
foreach(_lib ${GPU_BACKEND_LIBS})
    target_link_libraries(secp256k1_gpu_host PRIVATE ${_lib})
endforeach()

# CUDA-specific settings
if(SECP256K1_BUILD_CUDA AND TARGET secp256k1_cuda_lib)
    target_include_directories(secp256k1_gpu_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
    )
    set_source_files_properties(src/gpu_backend_cuda.cu PROPERTIES LANGUAGE CUDA)
    set_target_properties(secp256k1_gpu_host PROPERTIES
        CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
endif()

# OpenCL-specific includes
if(SECP256K1_BUILD_OPENCL AND TARGET secp256k1_opencl)
    target_include_directories(secp256k1_gpu_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../opencl/include
    )
endif()

# Metal-specific includes and frameworks
if(SECP256K1_BUILD_METAL AND TARGET secp256k1_metal_lib)
    target_include_directories(secp256k1_gpu_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../metal/include
    )
    if(APPLE)
        target_link_libraries(secp256k1_gpu_host PRIVATE
            "-framework Metal"
            "-framework Foundation"
        )
    endif()
    # Compile .mm as Objective-C++
    set_source_files_properties(src/gpu_backend_metal.mm PROPERTIES
        LANGUAGE OBJCXX
    )
endif()

set_target_properties(secp256k1_gpu_host PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Issue #335 acceptance repair (round 11) -- FIXED.
#
# History: SECP256K1_INSTALL_CABI=ON combined with any GPU backend
# (SECP256K1_BUILD_OPENCL/CUDA/METAL) never configured successfully (rounds
# 9-10). secp256k1_gpu_host was never installed/exported, even though
# ufsecp_static/ufsecp_shared (include/ufsecp/CMakeLists.txt) link it
# PRIVATE whenever a GPU backend is enabled -- CMake's
# `install(EXPORT ufsecpTargets ...)` (root CMakeLists.txt) refused to
# generate ("includes target ufsecp_static which requires target
# secp256k1_gpu_host that is not in any export set"), a hard configure-time
# error. Round 9/10 diagnosed two compounding causes but could not fix them
# (both files were outside their allowed_writes):
#   1. secp256k1_gpu_host had no `install(TARGETS ...)` rule at all -- it
#      could not join any export set.
#   2. Its own PUBLIC include dir (immediately above) was an unwrapped
#      source-tree-absolute path; CMake refuses to export a target whose
#      PUBLIC/INTERFACE include dirs are not BUILD_INTERFACE/INSTALL_INTERFACE-
#      wrapped. Round 11 resolved this by demoting it to PRIVATE instead
#      (see the comment on target_include_directories(secp256k1_gpu_host ...)
#      above) after confirming no consumer needs the PUBLIC propagation --
#      simpler than wrapping + installing the header pair, and removes the
#      export-interface requirement for this internal host-side header
#      entirely.
#   3. secp256k1_gpu_host itself PRIVATE-links secp256k1_opencl, which
#      joined a dead-code export set ("secp256k1_opencl-targets", never
#      finalized by a matching `install(EXPORT ...)` anywhere in the repo)
#      instead of `ufsecpTargets` -- fixed in src/opencl/CMakeLists.txt
#      (one-line EXPORT-name fix; its include dirs were already correctly
#      wrapped).
#   4. secp256k1_cuda_lib had the same two gaps (no install rule, unwrapped
#      PUBLIC include dirs) -- fixed in src/cuda/CMakeLists.txt with the
#      same install(TARGETS ... EXPORT ufsecpTargets ...) + PRIVATE-include
#      treatment (verified: nothing outside src/cuda/CMakeLists.txt's own
#      directory-scope `include_directories()` sibling targets relies on the
#      PUBLIC propagation).
#   5. secp256k1_metal_lib got the same structural fix for parity
#      (src/metal/CMakeLists.txt) -- NOT runtime-verified on this Linux host
#      (Metal GPU code only compiles under APPLE); configure-level parity
#      only. See METAL_RUNTIME_CONFIRMATION_PENDING convention.
#
# The install(TARGETS secp256k1_gpu_host ...) rule below joins the SAME
# `ufsecpTargets` export set the root CMakeLists.txt finalizes for the
# ufsecp C ABI package, closing the export-set closure for
# `SECP256K1_INSTALL_CABI=ON` + any GPU backend. Verified with a real
# `cmake --install` (including ufsecp_static, not just ufsecp_shared -- the
# `-DUFSECP_BUILD_STATIC=OFF` workaround from round 10 is no longer needed)
# and a real prefix-move relocation test; see
# ci/check_release_package_contents.py --gpu-export-closure.
include(GNUInstallDirs)
install(TARGETS secp256k1_gpu_host
    EXPORT ufsecpTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
