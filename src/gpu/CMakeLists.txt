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

target_include_directories(secp256k1_gpu_host
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Pass backend availability defines
target_compile_definitions(secp256k1_gpu_host PRIVATE ${GPU_BACKEND_DEFS})

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
