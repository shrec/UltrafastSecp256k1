/* ============================================================================
 * pr353_msvc_link_retention fixture -- anchor.cpp
 * ============================================================================
 * Deterministic regression fixture for PR #353 / acceptance item 6 (Eric
 * Voskuil / evoskuil): proves that the CMake
 *   if(MSVC) LINKER:/INCLUDE:<anchor> else() LINKER:--undefined=<anchor> endif()
 * pattern used by compat/libbitcoin_direct/CMakeLists.txt's
 * `_lbtc_gpu_retain` actually forces retention of an otherwise dead-stripped
 * static-archive member at link time.
 *
 * This TU is built into a static archive (pr353_anchor_lib in
 * CMakeLists.txt). Its only payload is the exported anchor symbol below plus
 * a namespace-scope static initializer with an observable runtime side
 * effect -- structurally identical to the real production pattern:
 * EngineGpuColumnsInstaller in src/gpu/src/gpu_engine_hook.cpp exports no
 * symbol of its own either, and is retained ONLY because
 * secp256k1_gpu_columns_provider_anchor forces the linker to pull this
 * translation unit's archive member in.
 *
 * The anchor symbol name here (pr353_link_retention_anchor) is deliberately
 * DISTINCT from the real production anchor
 * (secp256k1_gpu_columns_provider_anchor) -- this fixture tests the generic
 * CMake/linker retention MECHANISM; cross-file name consistency for the real
 * production anchor is checked separately, structurally, by
 * ci/check_windows_cuda_contract.py.
 *
 * Nothing in main.cpp references either symbol below directly -- a normal
 * static link of pr353_anchor_lib into an executable that does not use it
 * discards this whole .o, and the "PR353_ANCHOR_LINKED" line below never
 * prints. See CMakeLists.txt for the two executables (baseline: no
 * retention option; retained: the option under test) and the CTest
 * assertions (runtime-behavior + best-effort symbol-table proof).
 * ============================================================================ */
#include <cstdio>

extern "C" int pr353_link_retention_anchor = 1;

namespace {

struct AnchorSideEffect {
    AnchorSideEffect() noexcept {
        std::fputs("PR353_ANCHOR_LINKED\n", stderr);
    }
};
AnchorSideEffect g_anchor_side_effect;

}  // namespace
