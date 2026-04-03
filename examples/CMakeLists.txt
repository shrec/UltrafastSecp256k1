# Examples CMakeLists.txt

# ESP32 / STM32 examples are separate projects (ESP-IDF / STM32CubeMX)
# See examples/esp32_test/ and examples/stm32_test/

# -- Desktop examples ---------------------------------------------------------
if(NOT _SECP_TARGET_ARCH MATCHES "xtensa|Xtensa" AND NOT SECP256K1_PLATFORM_STM32)

    # Basic usage: key gen, point ops, serialization
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/basic_usage/main.cpp")
        add_executable(example_basic_usage basic_usage/main.cpp)
        target_link_libraries(example_basic_usage PRIVATE fastsecp256k1)
        message(STATUS "Examples: basic_usage enabled")
    else()
        message(STATUS "Examples: basic_usage skipped (main.cpp not found)")
    endif()

    # Signing demo: ECDSA + Schnorr sign/verify
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/signing_demo/main.cpp")
        add_executable(example_signing_demo signing_demo/main.cpp)
        target_link_libraries(example_signing_demo PRIVATE fastsecp256k1)
        message(STATUS "Examples: signing_demo enabled")
    else()
        message(STATUS "Examples: signing_demo skipped (main.cpp not found)")
    endif()

    # Threshold demo: FROST 2-of-3 DKG + signing ceremony
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/threshold_demo/main.cpp")
        add_executable(example_threshold_demo threshold_demo/main.cpp)
        target_link_libraries(example_threshold_demo PRIVATE fastsecp256k1)
        message(STATUS "Examples: threshold_demo enabled")
    else()
        message(STATUS "Examples: threshold_demo skipped (main.cpp not found)")
    endif()

    # Ethereum ZK-layer demo: BIP-32 HD wallet, EIP-191 personal_sign,
    # ecrecover, Pedersen commitments, ZK proof bundle, batch ECDSA verify
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eth_zk_layer/main.cpp")
        add_executable(example_eth_zk_layer eth_zk_layer/main.cpp)
        # Link only against the public C ABI wrapper — no internal C++ headers used.
        # This avoids inheriting the -flto INTERFACE flag from fastsecp256k1, which
        # would misoptimise calls into the non-LTO ufsecp_static object.
        target_link_libraries(example_eth_zk_layer PRIVATE ufsecp::ufsecp)
        message(STATUS "Examples: eth_zk_layer enabled")
    else()
        message(STATUS "Examples: eth_zk_layer skipped (main.cpp not found)")
    endif()

endif()
