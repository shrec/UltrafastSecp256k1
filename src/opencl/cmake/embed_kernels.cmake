# embed_kernels.cmake
# Converts OpenCL kernel files into a C++ header with embedded strings
#
# Input variables:
#   KERNEL_FILES  - pipe-separated list of .cl files
#   OUTPUT_FILE   - path to the output .hpp header

if(NOT DEFINED KERNEL_FILES)
    message(FATAL_ERROR "KERNEL_FILES not defined")
endif()
if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE not defined")
endif()

# Split pipe-separated list
string(REPLACE "|" ";" FILE_LIST "${KERNEL_FILES}")

# Ensure output directory exists
get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# Header guard
set(HEADER_CONTENT "#pragma once\n")
string(APPEND HEADER_CONTENT "// Auto-generated - do not edit\n")
string(APPEND HEADER_CONTENT "// Embedded OpenCL kernel sources\n\n")
string(APPEND HEADER_CONTENT "#include <string>\n\n")
string(APPEND HEADER_CONTENT "namespace secp256k1 {\nnamespace opencl {\nnamespace kernels {\n\n")

foreach(KERNEL_FILE ${FILE_LIST})
    if(NOT EXISTS "${KERNEL_FILE}")
        message(FATAL_ERROR "Kernel file not found: ${KERNEL_FILE}")
    endif()

    # Read file content
    file(READ "${KERNEL_FILE}" FILE_CONTENT)

    # Get filename without path/extension for variable name
    get_filename_component(FNAME "${KERNEL_FILE}" NAME_WE)

    # Escape backslashes and quotes for C++ raw string
    string(APPEND HEADER_CONTENT "static const std::string ${FNAME}_source = R\"__CL__(\n")
    string(APPEND HEADER_CONTENT "${FILE_CONTENT}")
    string(APPEND HEADER_CONTENT ")__CL__\";\n\n")
endforeach()

string(APPEND HEADER_CONTENT "} // namespace kernels\n} // namespace opencl\n} // namespace secp256k1\n")

file(WRITE "${OUTPUT_FILE}" "${HEADER_CONTENT}")

message(STATUS "Embedded ${FILE_LIST} -> ${OUTPUT_FILE}")
