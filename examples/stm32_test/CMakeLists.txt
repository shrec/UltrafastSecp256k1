cmake_minimum_required(VERSION 3.20)

# Cross-compile toolchain
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/arm-none-eabi-gcc.cmake")

project(stm32_secp256k1_test C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Library source directory
set(LIB_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../cpu/src")
set(LIB_INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../cpu/include")
set(LIB_GLOBAL_INC "${CMAKE_CURRENT_LIST_DIR}/../../include")

# Linker script
set(LINKER_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/STM32F103ZET6.ld")

# Sources
set(SOURCES
    main.cpp
    startup_stm32f103ze.cpp
    syscalls.cpp
    ${LIB_SRC_DIR}/field.cpp
    ${LIB_SRC_DIR}/scalar.cpp
    ${LIB_SRC_DIR}/point.cpp
    ${LIB_SRC_DIR}/glv.cpp
    ${LIB_SRC_DIR}/selftest.cpp
)

add_executable(${PROJECT_NAME}.elf ${SOURCES})

# Include paths
target_include_directories(${PROJECT_NAME}.elf PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${LIB_INC_DIR}
    ${LIB_GLOBAL_INC}
)

# Platform definitions (same as ESP32 but for STM32)
target_compile_definitions(${PROJECT_NAME}.elf PUBLIC
    SECP256K1_PLATFORM_STM32=1
    SECP256K1_NO_INT128=1
    SECP256K1_NO_ASM=1
    STM32F103xE
)

# Compile flags: -O3 aggressively, no exceptions/RTTI (save Flash/RAM)
target_compile_options(${PROJECT_NAME}.elf PRIVATE
    -O3
    -fno-exceptions
    -fno-rtti
    -ffunction-sections
    -fdata-sections
    -Wall
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-return-type
)

# Linker flags
target_link_options(${PROJECT_NAME}.elf PRIVATE
    -T${LINKER_SCRIPT}
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    -Wl,-Map=${PROJECT_NAME}.map
)

# Post-build: generate .bin and .hex, print sizes
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
    COMMAND ${CMAKE_SIZE} --format=berkeley $<TARGET_FILE:${PROJECT_NAME}.elf>
    COMMENT "Generating .bin and .hex, printing sizes"
)
