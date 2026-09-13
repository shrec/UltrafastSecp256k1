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
#   2. Symbol-level evidence: on MSVC the LINKER MAP (a linked MSVC image
#      carries no COFF symbol table, so `dumpbin /symbols <exe>` finds the
#      anchor in neither executable and cannot distinguish them); on GNU-like
#      hosts nm on the executable, which does carry one. The `windows` job in
#      .github/workflows/ci.yml asserts the same property against the same map
#      files, independent of this project's CTest wiring.
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

# -- Link map, so the symbol-level proof has an artifact that can answer -----
#
# `dumpbin /symbols <exe>` cannot answer "was this archive member retained?".
# A linked MSVC image carries no COFF symbol table -- the symbols live in the
# PDB -- so that query returns nothing for ANY executable, retained or not.
# The CI step that used it therefore reported
#
#   PR353 CONTRACT VIOLATION: pr353_link_retention_anchor missing from
#   ...pr353_link_retention_retained.exe -- LINKER:/INCLUDE: retention is
#   broken on MSVC
#
# on the first run that ever reached it, while the runtime CTests immediately
# above it PASSED -- i.e. the member demonstrably WAS retained and its static
# initializer demonstrably DID run. The assertion was wrong, not the mechanism.
#
# The linker map is the artifact that records which archive members and which
# symbols were pulled into the image, so it is what the symbol-level proof
# reads. Emitted for both executables so the negative control keeps working.
if(MSVC)
    target_link_options(pr353_link_retention_baseline PRIVATE "LINKER:/MAP")
    target_link_options(pr353_link_retention_retained PRIVATE "LINKER:/MAP")
endif()

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

# -- 2. Symbol-level proof ---------------------------------------------------
#
# MSVC: read the LINKER MAP. `dumpbin /symbols <exe>` is the wrong query for
# this question -- a linked MSVC image has no COFF symbol table (symbols live
# in the PDB), so it finds the anchor in NEITHER executable and the "retained"
# assertion can never pass while the "baseline" one passes vacuously. The map
# records what the linker actually pulled in, which is the property under test.
# `cmake -E cat` keeps this a plain CTest with no external tool.
#
# GNU-like: nm on the executable is correct -- an ELF image does carry a
# symbol table -- so that branch is unchanged.
if(MSVC)
    add_test(NAME pr353_msvc_link_retention_symtab_baseline_absent
        COMMAND ${CMAKE_COMMAND} -E cat
            "$<TARGET_FILE_DIR:pr353_link_retention_baseline>/$<TARGET_FILE_BASE_NAME:pr353_link_retention_baseline>.map")
    add_test(NAME pr353_msvc_link_retention_symtab_retained_present
        COMMAND ${CMAKE_COMMAND} -E cat
            "$<TARGET_FILE_DIR:pr353_link_retention_retained>/$<TARGET_FILE_BASE_NAME:pr353_link_retention_retained>.map")
    set_tests_properties(pr353_msvc_link_retention_symtab_baseline_absent PROPERTIES
        FAIL_REGULAR_EXPRESSION "${PR353_ANCHOR_SYMBOL}")
    set_tests_properties(pr353_msvc_link_retention_symtab_retained_present PROPERTIES
        PASS_REGULAR_EXPRESSION "${PR353_ANCHOR_SYMBOL}")
else()
    find_program(PR353_SYMBOL_TOOL NAMES llvm-nm nm)
    if(PR353_SYMBOL_TOOL)
        add_test(NAME pr353_msvc_link_retention_symtab_baseline_absent
            COMMAND ${PR353_SYMBOL_TOOL} "$<TARGET_FILE:pr353_link_retention_baseline>")
        add_test(NAME pr353_msvc_link_retention_symtab_retained_present
            COMMAND ${PR353_SYMBOL_TOOL} "$<TARGET_FILE:pr353_link_retention_retained>")
        set_tests_properties(pr353_msvc_link_retention_symtab_baseline_absent PROPERTIES
            FAIL_REGULAR_EXPRESSION "${PR353_ANCHOR_SYMBOL}")
        set_tests_properties(pr353_msvc_link_retention_symtab_retained_present PROPERTIES
            PASS_REGULAR_EXPRESSION "${PR353_ANCHOR_SYMBOL}")
    else()
        message(WARNING "pr353_msvc_link_retention: no nm/llvm-nm found at configure time "
            "-- relying on the runtime-behavior CTest cases here. Those are the stronger "
            "proof anyway: they show the retained archive member's static initializer "
            "actually RAN, not merely that a name appears in a table.")
    endif()
endif()
