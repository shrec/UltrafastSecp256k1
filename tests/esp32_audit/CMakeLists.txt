# ESP-IDF project for UltrafastSecp256k1 cross-platform audit
cmake_minimum_required(VERSION 3.16)

# Include UltrafastSecp256k1 as extra component
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/components"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(secp256k1_esp32_audit)
