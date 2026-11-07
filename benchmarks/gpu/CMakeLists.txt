# ============================================================================
# End-to-end GPU C ABI batch benchmark (bench_gpu_abi_batch)
# ============================================================================
# Added from include/ufsecp/CMakeLists.txt right after the ufsecp C ABI
# targets are created, when SECP256K1_BUILD_BENCH is on and the GPU host layer
# (secp256k1_gpu_host) is in the build.
#
# Not registered with CTest: it is a measurement tool. Run it directly, e.g.
#   ./bench_gpu_abi_batch --backend cuda --json cuda.json
# ============================================================================

# Same choice as the audit GPU executables (audit/CMakeLists.txt): prefer the
# already device-linked shared C ABI, which avoids duplicate CUDA static
# archive / device-link ownership; fall back to the static library.
if(TARGET ufsecp_shared)
    set(_bench_gpu_abi_lib ufsecp_shared)
elseif(TARGET ufsecp_static)
    set(_bench_gpu_abi_lib ufsecp_static)
else()
    message(STATUS "  GPU API: no ufsecp C ABI target -- bench_gpu_abi_batch not built")
    return()
endif()

add_executable(bench_gpu_abi_batch bench_gpu_abi_batch.cpp)
target_link_libraries(bench_gpu_abi_batch PRIVATE ${_bench_gpu_abi_lib})
target_include_directories(bench_gpu_abi_batch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_BINARY_DIR}/include/ufsecp
)
set_target_properties(bench_gpu_abi_batch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
message(STATUS "  GPU API: bench_gpu_abi_batch enabled (links ${_bench_gpu_abi_lib})")
