idf_component_register(
    SRCS "audit_main.cpp"
    INCLUDE_DIRS "."
)

# Add UltrafastSecp256k1 include paths
target_include_directories(${COMPONENT_LIB} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpu/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpu/src"
)

# C++20 for the component
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_20)

# ESP32 is 32-bit, no __int128, no 64-bit ASM
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    SECP256K1_FIELD_26BIT=1
)
