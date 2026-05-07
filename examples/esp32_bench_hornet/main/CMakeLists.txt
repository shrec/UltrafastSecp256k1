idf_component_register(
    SRCS "main.cpp"
         "libsecp_bench.c"
         "../../../../../_research_repos/secp256k1/src/precomputed_ecmult_gen.c"
         "../../../../../_research_repos/secp256k1/src/precomputed_ecmult.c"
         "../../../src/cpu/src/field.cpp"
         "../../../src/cpu/src/field_26.cpp"
         "../../../src/cpu/src/field_52.cpp"
         "../../../src/cpu/src/scalar.cpp"
         "../../../src/cpu/src/point.cpp"
         "../../../src/cpu/src/glv.cpp"
         "../../../src/cpu/src/selftest.cpp"
         "../../../src/cpu/src/ct_field.cpp"
         "../../../src/cpu/src/ct_scalar.cpp"
         "../../../src/cpu/src/ct_point.cpp"
         "../../../src/cpu/src/ct_sign.cpp"
         "../../../src/cpu/src/ecdsa.cpp"
         "../../../src/cpu/src/schnorr.cpp"
         "../../../src/cpu/src/multiscalar.cpp"
         "../../../src/cpu/src/batch_verify.cpp"
         "../../../src/cpu/src/pippenger.cpp"
         "../../../src/cpu/src/hash_accel.cpp"
    INCLUDE_DIRS "."
                 "../../../src/cpu/include"
                 "../../../include"
                 "../../../../../_research_repos/secp256k1"
                 "../../../../../_research_repos/secp256k1/src"
                 "../../../../../_research_repos/secp256k1/include"
    REQUIRES esp_timer
)

# ESP32-specific defines
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    SECP256K1_PLATFORM_ESP32=1
    SECP256K1_NO_INT128=1
    SECP256K1_NO_ASM=1
    NDEBUG=1
)

# libsecp256k1 precomputed tables
set(LIBSECP_PRECOMP_GEN "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../_research_repos/secp256k1/src/precomputed_ecmult_gen.c")
set(LIBSECP_PRECOMP     "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../_research_repos/secp256k1/src/precomputed_ecmult.c")
set_source_files_properties(
    ${LIBSECP_PRECOMP_GEN} ${LIBSECP_PRECOMP}
    PROPERTIES COMPILE_DEFINITIONS "ECMULT_WINDOW_SIZE=2;COMB_BLOCKS=11;COMB_TEETH=6"
)

# Release flags
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -fno-exceptions
    -fno-rtti
    -fomit-frame-pointer
    -fno-stack-protector
    -Wno-error=return-type
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-sign-compare
    -Wno-cast-function-type
)
