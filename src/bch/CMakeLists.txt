if(NOT SECP256K1_BUILD_BCH)
    return()
endif()
message(STATUS "Secp256k1: BCH module: ON — CashAddr, RPA, scan pipeline")
target_sources(fastsecp256k1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cashaddr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rpa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bch_scan.cpp)
target_compile_definitions(fastsecp256k1 PUBLIC SECP256K1_BUILD_BCH=1)
target_include_directories(fastsecp256k1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include)
