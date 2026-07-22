/* ============================================================================
 * pr353_msvc_link_retention fixture -- main.cpp
 * ============================================================================
 * Deliberately does NOT reference pr353_link_retention_anchor,
 * AnchorSideEffect, or anything else in anchor.cpp/pr353_anchor_lib. Retention
 * of that archive member at link time must come ONLY from the linker option
 * under test (see CMakeLists.txt). This mirrors the real
 * compat/libbitcoin_direct executables, which reference no gpu_host symbol
 * either and rely entirely on the linker-option retention mechanism. If this
 * file referenced pr353_link_retention_anchor directly, that reference alone
 * would force retention and the fixture would test nothing.
 * ============================================================================ */
#include <cstdio>

int main() {
    std::puts("pr353_msvc_link_retention: fixture executable ran");
    return 0;
}
