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
)

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

target_include_directories(secp256k1_gpu_host
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
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
