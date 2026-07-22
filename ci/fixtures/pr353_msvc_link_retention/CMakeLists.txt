# ============================================================================
# pr353_msvc_link_retention -- deterministic MSVC/GNU-like link-retention
# fixture (PR #353, Eric Voskuil / evoskuil, acceptance item 6)
# ============================================================================
# Standalone CMake project -- NOT add_subdirectory()'d into the main build
# (the top-level CMakeLists.txt is out of this fix's allowed_writes). CI
# configures and builds this directory on its own: fast, CUDA-free,
# engine-free, no GPU hardware required.
#
# Reproduces in miniature the exact retention pattern used for the real
# secp256k1_gpu_columns_provider_anchor in
# compat/libbitcoin_direct/CMakeLists.txt (`_lbtc_gpu_retain`):
#   - pr353_anchor_lib: a static archive whose only payload is an exported
#     anchor symbol + a self-installing static initializer that NOTHING in
#     main.cpp references directly (see anchor.cpp) -- structurally
#     identical to EngineGpuColumnsInstaller / secp256k1_gpu_columns_provider_anchor
#     in src/gpu/src/gpu_engine_hook.cpp.
#   - pr353_link_retention_baseline: links the archive with NO retention
#     option -- the linker is free to drop the whole archive member, so
#     neither the anchor symbol nor its static-init side effect may survive
#     (negative control -- proves the positive result below is not vacuous).
#   - pr353_link_retention_retained: links the SAME archive with the
#     platform-specific targeted retention option under test:
#     LINKER:/INCLUDE:<anchor> on MSVC, LINKER:--undefined=<anchor> on
#     GNU-like linkers -- the same if(MSVC)/else() branch shape as
#     _lbtc_gpu_retain in compat/libbitcoin_direct/CMakeLists.txt.
#
# Two independent proofs, per executable:
#   1. Runtime behavior (portable, no external tool required): does the
#      static initializer's "PR353_ANCHOR_LINKED" line print (did the
#      archive member's code actually execute)?
#   2. Symbol-table evidence (dumpbin on MSVC, llvm-nm/nm on GNU-like),
#      best-effort at CMake-configure time via find_program: is the anchor
#      symbol literally present in the linked binary? The AUTHORITATIVE
#      Windows proof for acceptance item 6 is the explicit dumpbin step in
#      the `windows` job of .github/workflows/ci.yml, which locates
#      dumpbin.exe via vswhere independent of this configure-time PATH
#      detection -- this CTest-embedded check is a portable convenience that
#      also lets the mechanism be validated today on any GNU-like host.
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(pr353_msvc_link_retention_fixture CXX)

set(PR353_ANCHOR_SYMBOL "pr353_link_retention_anchor")

add_library(pr353_anchor_lib STATIC anchor.cpp)

# Baseline: same archive, NO retention option -- anchor must be dead-stripped.
add_executable(pr353_link_retention_baseline main.cpp)
target_link_libraries(pr353_link_retention_baseline PRIVATE pr353_anchor_lib)

# Retained: same archive, WITH the platform-specific targeted retention
# option under test -- exact branch shape reproduced from
# compat/libbitcoin_direct/CMakeLists.txt's `_lbtc_gpu_retain`.
if(MSVC)
    set(PR353_RETAIN_OPTION "LINKER:/INCLUDE:${PR353_ANCHOR_SYMBOL}")
else()
    set(PR353_RETAIN_OPTION "LINKER:--undefined=${PR353_ANCHOR_SYMBOL}")
endif()

add_executable(pr353_link_retention_retained main.cpp)
target_link_libraries(pr353_link_retention_retained PRIVATE pr353_anchor_lib)
target_link_options(pr353_link_retention_retained PRIVATE ${PR353_RETAIN_OPTION})

enable_testing()

# -- 1. Runtime-behavior proof (no external tool required, always runs) -----
add_test(NAME pr353_msvc_link_retention_runtime_baseline_absent
    COMMAND pr353_link_retention_baseline)
set_tests_properties(pr353_msvc_link_retention_runtime_baseline_absent PROPERTIES
    FAIL_REGULAR_EXPRESSION "PR353_ANCHOR_LINKED")

add_test(NAME pr353_msvc_link_retention_runtime_retained_present
    COMMAND pr353_link_retention_retained)
set_tests_properties(pr353_msvc_link_retention_runtime_retained_present PROPERTIES
    PASS_REGULAR_EXPRESSION "PR353_ANCHOR_LINKED")

# -- 2. Symbol-table proof, best-effort at configure time --------------------
if(MSVC)
    find_program(PR353_SYMBOL_TOOL NAMES dumpbin)
else()
    find_program(PR353_SYMBOL_TOOL NAMES llvm-nm nm)
endif()

if(PR353_SYMBOL_TOOL)
    if(MSVC)
        add_test(NAME pr353_msvc_link_retention_symtab_baseline_absent
            COMMAND ${PR353_SYMBOL_TOOL} /symbols "$<TARGET_FILE:pr353_link_retention_baseline>")
        add_test(NAME pr353_msvc_link_retention_symtab_retained_present
            COMMAND ${PR353_SYMBOL_TOOL} /symbols "$<TARGET_FILE:pr353_link_retention_retained>")
    else()
        add_test(NAME pr353_msvc_link_retention_symtab_baseline_absent
            COMMAND ${PR353_SYMBOL_TOOL} "$<TARGET_FILE:pr353_link_retention_baseline>")
        add_test(NAME pr353_msvc_link_retention_symtab_retained_present
            COMMAND ${PR353_SYMBOL_TOOL} "$<TARGET_FILE:pr353_link_retention_retained>")
    endif()
    set_tests_properties(pr353_msvc_link_retention_symtab_baseline_absent PROPERTIES
        FAIL_REGULAR_EXPRESSION "${PR353_ANCHOR_SYMBOL}")
    set_tests_properties(pr353_msvc_link_retention_symtab_retained_present PROPERTIES
        PASS_REGULAR_EXPRESSION "${PR353_ANCHOR_SYMBOL}")
else()
    message(WARNING "pr353_msvc_link_retention: no symbol-table tool found at configure "
        "time (dumpbin/llvm-nm/nm) -- relying on the runtime-behavior CTest cases only "
        "here; the Windows CI job (.github/workflows/ci.yml, `windows` job) runs an "
        "explicit dumpbin symbol assertion independent of this configure-time PATH "
        "detection, which is the authoritative proof for acceptance item 6.")
endif()
