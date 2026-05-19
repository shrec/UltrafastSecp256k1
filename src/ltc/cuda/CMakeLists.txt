if(NOT SECP256K1_BUILD_CUDA)
    return()
endif()

add_executable(bench_ltcsp bench_ltcsp.cu)
target_include_directories(bench_ltcsp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpu/include)
target_link_libraries(bench_ltcsp PRIVATE secp256k1_cuda fastsecp256k1)
set_target_properties(bench_ltcsp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
