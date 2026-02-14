idf_component_register(
    SRCS "main.cpp"
           "../../../cpu/src/field.cpp"
           "../../../cpu/src/scalar.cpp"
           "../../../cpu/src/point.cpp"
           "../../../cpu/src/glv.cpp"
           "../../../cpu/src/selftest.cpp"
    INCLUDE_DIRS "."
                 "../../../cpu/include"
                 "../../../include"
    REQUIRES esp_timer
)

# ESP32-specific compiler definitions (PUBLIC so all sources get them)
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    SECP256K1_PLATFORM_ESP32=1
    SECP256K1_NO_INT128=1
    SECP256K1_NO_ASM=1
)

# ESP32-specific compiler flags
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -fno-exceptions
    -fno-rtti
    -Wno-error=return-type
    -Wno-unused-variable
    -Wno-unused-function
)

