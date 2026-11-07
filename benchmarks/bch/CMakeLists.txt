# ============================================================================
# bench_bch_schnorr -- BCH 2019 Schnorr verification, single and batch,
# against the libsecp256k1 fork Knuth vendors (issue #466)
# ============================================================================
# Added from src/cpu/CMakeLists.txt next to bench_unified, so it is built
# whenever SECP256K1_BUILD_BENCH is on, independent of SECP256K1_BUILD_BCH:
# src/cpu/src/bch_schnorr.cpp is a core source of the engine.
#
# Reference rows. Set both cache variables, or neither:
#   UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR  include/ directory of the reference
#                                         (secp256k1.h and its BCH Schnorr header)
#   UFSECP_BCH_REF_SECP256K1_LIBRARY      the reference's static library, built
#                                         with -DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON
# The pinned reference is k-nuth/kth d1b9a8ac50cce15843c6925374709e2391e7aba4,
# src/secp256k1 (Knuth's vendored libsecp256k1 v0.8.0). Build steps:
# docs/BCH_SCHNORR_BATCH_BENCH.md.
#
# Never link compat/libsecp256k1_bchn_shim into this target: it exports the
# same secp256k1_schnorr_* symbols as the reference.
#
# Portable flags only: nothing here adds -march, /arch or any other ISA flag.
# The target inherits the project's compile settings like every other bench,
# and the binary records them in its JSON (bench_bch_schnorr_build_info.h).
# ============================================================================

set(UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR "" CACHE PATH
    "bench_bch_schnorr: include directory of a libsecp256k1 built with the BCH Schnorr module (empty: no reference rows)")
set(UFSECP_BCH_REF_SECP256K1_LIBRARY "" CACHE FILEPATH
    "bench_bch_schnorr: static libsecp256k1 built with -DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON (empty: no reference rows)")

add_executable(bench_bch_schnorr bench_bch_schnorr.cpp)
target_link_libraries(bench_bch_schnorr PRIVATE ${SECP256K1_LIB_NAME})
target_include_directories(bench_bch_schnorr PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# The C ABI row. include/ufsecp is added after src/cpu, so ufsecp_static does
# not exist yet while this file is read: both expressions below resolve at
# generate time. Without the static C ABI library the row is left out.
target_link_libraries(bench_bch_schnorr PRIVATE $<TARGET_NAME_IF_EXISTS:ufsecp_static>)
target_compile_definitions(bench_bch_schnorr PRIVATE
    $<$<TARGET_EXISTS:ufsecp_static>:UFSECP_BCH_BENCH_HAVE_CABI=1>
    "UFSECP_BCH_BENCH_CONFIG=$<CONFIG>")

if(MSVC)
    # fopen / getenv / gmtime: the CRT's "use the _s variant" deprecation warnings.
    target_compile_definitions(bench_bch_schnorr PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# -- Reference ------------------------------------------------------------------
set(_bch_ref_inc "${UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR}")
set(_bch_ref_lib "${UFSECP_BCH_REF_SECP256K1_LIBRARY}")
set(_bch_ref_header "")
set(_bch_ref_git "")
find_package(Git QUIET)
if(NOT "${_bch_ref_inc}" STREQUAL "" OR NOT "${_bch_ref_lib}" STREQUAL "")
    if("${_bch_ref_inc}" STREQUAL "" OR "${_bch_ref_lib}" STREQUAL "")
        message(FATAL_ERROR
            "bench_bch_schnorr: set both UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR and "
            "UFSECP_BCH_REF_SECP256K1_LIBRARY, or neither")
    endif()
    file(TO_CMAKE_PATH "${_bch_ref_inc}" _bch_ref_inc)
    file(TO_CMAKE_PATH "${_bch_ref_lib}" _bch_ref_lib)
    if(NOT EXISTS "${_bch_ref_inc}/secp256k1.h")
        message(FATAL_ERROR
            "bench_bch_schnorr: ${_bch_ref_inc}/secp256k1.h not found "
            "(UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR)")
    endif()
    if(NOT EXISTS "${_bch_ref_lib}" OR IS_DIRECTORY "${_bch_ref_lib}")
        message(FATAL_ERROR
            "bench_bch_schnorr: library ${_bch_ref_lib} not found "
            "(UFSECP_BCH_REF_SECP256K1_LIBRARY)")
    endif()
    # The BCH Schnorr API sits in its own header: take the one that declares
    # the batch call of Knuth PR #826.
    file(GLOB _bch_ref_candidates RELATIVE "${_bch_ref_inc}" "${_bch_ref_inc}/secp256k1_*.h")
    list(SORT _bch_ref_candidates)
    foreach(_bch_h IN LISTS _bch_ref_candidates)
        file(STRINGS "${_bch_ref_inc}/${_bch_h}" _bch_hits REGEX "secp256k1_schnorr_verify_batch")
        if(_bch_hits)
            set(_bch_ref_header "${_bch_h}")
            break()
        endif()
    endforeach()
    if("${_bch_ref_header}" STREQUAL "")
        message(FATAL_ERROR
            "bench_bch_schnorr: no secp256k1_*.h in ${_bch_ref_inc} declares "
            "secp256k1_schnorr_verify_batch. Build the pinned reference (k-nuth/kth "
            "d1b9a8ac50cce15843c6925374709e2391e7aba4, src/secp256k1) with "
            "-DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON; see docs/BCH_SCHNORR_BATCH_BENCH.md")
    endif()
    # The reference commit, recorded only when the include dir is
    # src/secp256k1/include of a git checkout of its own (a kth clone). An
    # include dir copied under some other repository would otherwise report
    # that repository's commit.
    if(GIT_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --show-prefix
            WORKING_DIRECTORY "${_bch_ref_inc}"
            OUTPUT_VARIABLE _bch_prefix RESULT_VARIABLE _bch_rc
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_bch_rc EQUAL 0 AND "${_bch_prefix}" STREQUAL "src/secp256k1/include/")
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                WORKING_DIRECTORY "${_bch_ref_inc}"
                OUTPUT_VARIABLE _bch_out RESULT_VARIABLE _bch_rc
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_bch_rc EQUAL 0)
                set(_bch_ref_git "${_bch_out}")
            endif()
        endif()
    endif()
    # bench_bch_schnorr.cpp includes this, not a hard-coded header name.
    file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ufsecp_bch_ref_schnorr.h"
        CONTENT "/* Generated by benchmarks/bch/CMakeLists.txt. */\n#include \"@_bch_ref_header@\"\n"
        @ONLY)
    # BEFORE: the reference's secp256k1.h must win over any other on the path.
    target_include_directories(bench_bch_schnorr BEFORE PRIVATE "${_bch_ref_inc}")
    target_link_libraries(bench_bch_schnorr PRIVATE "${_bch_ref_lib}")
    # SECP256K1_STATIC: on Windows the reference's secp256k1.h declares its
    # API dllimport unless the consumer says the library is static.
    target_compile_definitions(bench_bch_schnorr PRIVATE UFSECP_BCH_BENCH_HAVE_REF=1 SECP256K1_STATIC)
    message(STATUS "bench_bch_schnorr: reference rows ON (${_bch_ref_lib}, ${_bch_ref_header})")
else()
    message(STATUS "bench_bch_schnorr: reference rows OFF (set UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR "
                   "and UFSECP_BCH_REF_SECP256K1_LIBRARY)")
endif()

# -- Build record ---------------------------------------------------------------
# A generated header the binary copies into its JSON. Every value is taken at
# configure time (reconfigure after pulling to refresh the git fields).
function(ufsecp_bch_bench_c_string out_var value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

set(_bch_git_describe "unknown")
set(_bch_git_commit "unknown")
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --always --dirty --abbrev=12
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _bch_out RESULT_VARIABLE _bch_rc
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_bch_rc EQUAL 0 AND NOT "${_bch_out}" STREQUAL "")
        set(_bch_git_describe "${_bch_out}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _bch_out RESULT_VARIABLE _bch_rc
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_bch_rc EQUAL 0 AND NOT "${_bch_out}" STREQUAL "")
        set(_bch_git_commit "${_bch_out}")
    endif()
endif()

set(_bch_version "${UltrafastSecp256k1_VERSION}")
if("${_bch_version}" STREQUAL "")
    set(_bch_version "unknown")
endif()
set(_bch_system "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
set(_bch_compiler "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(_bch_build_type "${CMAKE_BUILD_TYPE}")

set(_bch_flags "CMAKE_CXX_FLAGS=[${CMAKE_CXX_FLAGS}]")
if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _bch_cfg)
    string(APPEND _bch_flags " CMAKE_CXX_FLAGS_${_bch_cfg}=[${CMAKE_CXX_FLAGS_${_bch_cfg}}]")
else()
    string(APPEND _bch_flags " CMAKE_CXX_FLAGS_RELEASE=[${CMAKE_CXX_FLAGS_RELEASE}]")
endif()
if(DEFINED MARCH_FLAG)
    # The -march the top-level CMakeLists.txt adds to every target (GCC/Clang).
    string(APPEND _bch_flags " project_march=[${MARCH_FLAG}]")
endif()
foreach(_bch_knob SECP256K1_MARCH SECP256K1_MSVC_ARCH SECP256K1_MSVC_OB3 SECP256K1_MSVC_WPO
                  SECP256K1_USE_ASM SECP256K1_USE_LTO SECP256K1_BUILD_PIPPENGER)
    if(DEFINED ${_bch_knob})
        string(APPEND _bch_flags " ${_bch_knob}=${${_bch_knob}}")
    endif()
endforeach()

foreach(_bch_var _bch_git_describe _bch_git_commit _bch_version _bch_system _bch_compiler
                 _bch_build_type _bch_flags _bch_ref_inc _bch_ref_lib _bch_ref_header _bch_ref_git)
    ufsecp_bch_bench_c_string(${_bch_var}_c "${${_bch_var}}")
endforeach()

file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bench_bch_schnorr_build_info.h"
    CONTENT [=[
/* Generated by benchmarks/bch/CMakeLists.txt at configure time. Do not edit. */
#ifndef BENCH_BCH_SCHNORR_BUILD_INFO_H
#define BENCH_BCH_SCHNORR_BUILD_INFO_H
#define UFSECP_BCH_BENCH_GIT_DESCRIBE "@_bch_git_describe_c@"
#define UFSECP_BCH_BENCH_GIT_COMMIT "@_bch_git_commit_c@"
#define UFSECP_BCH_BENCH_VERSION "@_bch_version_c@"
#define UFSECP_BCH_BENCH_SYSTEM "@_bch_system_c@"
#define UFSECP_BCH_BENCH_CMAKE_COMPILER "@_bch_compiler_c@"
#define UFSECP_BCH_BENCH_BUILD_TYPE "@_bch_build_type_c@"
#define UFSECP_BCH_BENCH_FLAGS "@_bch_flags_c@"
#define UFSECP_BCH_BENCH_REF_INCLUDE_DIR "@_bch_ref_inc_c@"
#define UFSECP_BCH_BENCH_REF_LIBRARY "@_bch_ref_lib_c@"
#define UFSECP_BCH_BENCH_REF_HEADER "@_bch_ref_header_c@"
#define UFSECP_BCH_BENCH_REF_GIT_COMMIT "@_bch_ref_git_c@"
#endif
]=]
    @ONLY)
