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

# Canonical libbitcoin integration test: direct ECDSA + Schnorr verify
# (single + batch + columns + fail-closed). Registered as `lbtc_direct_verify`.
if(SECP256K1_BUILD_LIBBITCOIN_TESTS)
    add_executable(test_lbtc_direct_verify tests/test_direct_verify.cpp)
    # Link the interface target (headers + defines + C++20) and the engine
    # directly so its objects/archive resolve at link time.
    target_link_libraries(test_lbtc_direct_verify PRIVATE
        fastsecp256k1_libbitcoin
        ${_lbtc_engine})
    find_package(Threads QUIET)
    if(TARGET Threads::Threads)
        target_link_libraries(test_lbtc_direct_verify PRIVATE Threads::Threads)
    endif()
    add_test(NAME lbtc_direct_verify COMMAND test_lbtc_direct_verify)
    set_tests_properties(lbtc_direct_verify PROPERTIES LABELS "libbitcoin;direct;verify")
    message(STATUS "  libbitcoin direct test: ON (lbtc_direct_verify)")
endif()
