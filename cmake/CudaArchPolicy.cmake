# CudaArchPolicy.cmake
#
# Shared CUDA architecture-selection policy for UltrafastSecp256k1.
#
# This module is the SINGLE place in the whole build graph (outer Secp256K1fast
# suite + standalone UltrafastSecp256k1 library) that ever chooses a CUDA
# architecture list on the caller's behalf. It is only reached when the caller
# has already confirmed BOTH of the following are absent:
#
#   1. An explicit -DCMAKE_CUDA_ARCHITECTURES=... on the command line.
#   2. A CUDAARCHS environment variable.
#
# i.e. this module owns precedence tier #3 (named repository profile) and
# tier #4 (fail-fast). It NEVER runs ahead of, or overwrites, an explicit
# caller value — see libs/UltrafastSecp256k1/CMakeLists.txt and the outer
# suite's CMakeLists.txt for where tiers #1/#2 are detected before this file
# is even included.
#
# Full per-architecture coverage rationale (measured this session's RTX 5060 Ti
# / nvcc 13.2 + nvcc 12.0 evidence, re-cited from
# workingdocs/secondary_invariant_constants/cuda_arch_fatbin_policy_claude_v7.json):
# see libs/UltrafastSecp256k1/cmake/CUDA_ARCHITECTURE_POLICY.md
#
# Must be include()'d AFTER enable_language(CUDA) so CMAKE_CUDA_COMPILER_VERSION
# is populated for the version guards below, and BEFORE any CUDA target
# (add_library/add_executable with .cu sources) is created.

function(ufsecp_apply_cuda_arch_profile)
    set(_menu
"  -DCMAKE_CUDA_ARCHITECTURES=<explicit list>      you know your target GPU/driver\n\
  CUDAARCHS=<list>                                 env var, same effect\n\
  -DSECP256K1_CUDA_ARCH_PROFILE=local-native       build for exactly this machine's GPU (CMake >= 3.24 required)\n\
  -DSECP256K1_CUDA_ARCH_PROFILE=ci-bounded-recent  89-real;89-virtual;120-real;120-virtual (nvcc >= 13.0 required)\n\
  -DSECP256K1_CUDA_ARCH_PROFILE=legacy-compat      52-virtual;89-real;89-virtual (nvcc 12.x only, >=12.0 and <13.0)\n\
  -DSECP256K1_CUDA_ARCH_PROFILE=redistributable    75-virtual;86-real;89-real;120-real;120-virtual (nvcc >= 13.0 required)")

    if(NOT DEFINED SECP256K1_CUDA_ARCH_PROFILE OR SECP256K1_CUDA_ARCH_PROFILE STREQUAL "")
        message(FATAL_ERROR
            "SECP256K1_BUILD_CUDA=ON but no trustworthy CUDA architecture choice was "
            "supplied. This project never silently guesses an architecture list -- "
            "every bounded profile leaves some GPUs with zero execution path and "
            "reaches others only via driver-conditional PTX JIT (see "
            "libs/UltrafastSecp256k1/cmake/CUDA_ARCHITECTURE_POLICY.md for the full "
            "per-architecture coverage tables). Pick ONE of:\n${_menu}")
    elseif(SECP256K1_CUDA_ARCH_PROFILE STREQUAL "local-native")
        if(CMAKE_VERSION VERSION_LESS 3.24)
            message(FATAL_ERROR
                "SECP256K1_CUDA_ARCH_PROFILE=local-native requires CMake >= 3.24 "
                "(CMAKE_CUDA_ARCHITECTURES=native support was added in CMake 3.24). "
                "Installed CMake is ${CMAKE_VERSION}. Use an explicit "
                "-DCMAKE_CUDA_ARCHITECTURES=<list> instead, or upgrade CMake.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "local-native profile" FORCE)
        message(STATUS
            "CUDA arch profile: local-native (CMAKE_CUDA_ARCHITECTURES=native). "
            "If the resolved CUDA toolkit cannot compile for the detected device's "
            "compute capability, the CUDA compile step will fail with an explicit "
            "nvcc diagnostic naming the unsupported --gpu-architecture value.")
    elseif(SECP256K1_CUDA_ARCH_PROFILE STREQUAL "ci-bounded-recent")
        if(NOT CMAKE_CUDA_COMPILER_VERSION OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
            message(FATAL_ERROR
                "SECP256K1_CUDA_ARCH_PROFILE=ci-bounded-recent requires nvcc >= 13.0 "
                "(compute_120/sm_120 real-target codegen support). Resolved CUDA "
                "compiler version: '${CMAKE_CUDA_COMPILER_VERSION}'.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "89-real;89-virtual;120-real;120-virtual" CACHE STRING "ci-bounded-recent profile" FORCE)
        message(STATUS
            "CUDA arch profile: ci-bounded-recent (89-real;89-virtual;120-real;120-virtual). "
            "Deterministic real (SASS) coverage: sm_89 and sm_120 ONLY. sm_75/80/86/87 "
            "have ZERO execution path in this fatbin (89 > those numbers). "
            "sm_90/100/103/110/121 are reachable only via driver-conditional PTX JIT -- "
            "not guaranteed on every consumer driver. See CUDA_ARCHITECTURE_POLICY.md.")
    elseif(SECP256K1_CUDA_ARCH_PROFILE STREQUAL "legacy-compat")
        if(NOT CMAKE_CUDA_COMPILER_VERSION)
            message(FATAL_ERROR
                "SECP256K1_CUDA_ARCH_PROFILE=legacy-compat: could not determine "
                "CMAKE_CUDA_COMPILER_VERSION (CUDA language not fully enabled).")
        endif()
        # Lower bound: this profile requires nvcc >= 12.0. nvcc 11.x and older are
        # never an accepted toolchain for this repository (no compute_52 + real
        # sm_89 codegen combination this project relies on was validated below
        # nvcc 12.0, and accepting an older, unvetted toolchain silently would be
        # exactly the kind of unverified-default this policy exists to prevent).
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
            message(FATAL_ERROR
                "SECP256K1_CUDA_ARCH_PROFILE=legacy-compat requires nvcc >= 12.0 "
                "(this profile is never accepted with an nvcc 11.x or older "
                "toolchain). Resolved compiler is nvcc "
                "${CMAKE_CUDA_COMPILER_VERSION}. Use an nvcc >= 12.0, < 13.0 "
                "toolkit for this profile (set CMAKE_CUDA_COMPILER to it), or "
                "upgrade to ci-bounded-recent / redistributable / local-native.")
        endif()
        # Upper bound: nvcc 13.x dropped compute_50-61 virtual-architecture
        # support, so the '52-virtual' PTX entry this profile targets is no
        # longer a valid build target under nvcc 13. Never silently keep
        # claiming compute_52 support under an nvcc that can't produce it.
        if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
            message(FATAL_ERROR
                "SECP256K1_CUDA_ARCH_PROFILE=legacy-compat targets a '52-virtual' "
                "(compute_52) PTX entry that nvcc 13.x no longer accepts as a build "
                "target -- nvcc 13 dropped compute_50 through compute_61 virtual-"
                "architecture support. Resolved compiler is nvcc "
                "${CMAKE_CUDA_COMPILER_VERSION}. Use an nvcc >= 12.0, < 13.0 "
                "toolkit for this profile (set CMAKE_CUDA_COMPILER to it), or "
                "choose ci-bounded-recent / redistributable / local-native instead.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "52-virtual;89-real;89-virtual" CACHE STRING "legacy-compat profile" FORCE)
        message(STATUS
            "CUDA arch profile: legacy-compat (52-virtual;89-real;89-virtual, "
            "nvcc >= 12.0 and < 13.0 required). PTX-JIT floor at compute_52 "
            "(sm_52 and newer); real/SASS deterministic coverage only for sm_89.")
    elseif(SECP256K1_CUDA_ARCH_PROFILE STREQUAL "redistributable")
        if(NOT CMAKE_CUDA_COMPILER_VERSION OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
            message(FATAL_ERROR
                "SECP256K1_CUDA_ARCH_PROFILE=redistributable requires nvcc >= 13.0 "
                "(compute_120/sm_120 real-target codegen support). Resolved CUDA "
                "compiler version: '${CMAKE_CUDA_COMPILER_VERSION}'.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;86-real;89-real;120-real;120-virtual" CACHE STRING "redistributable profile" FORCE)
        message(STATUS
            "CUDA arch profile: redistributable (75-virtual;86-real;89-real;120-real;"
            "120-virtual, nvcc 13.x). Real (SASS) coverage: sm_86, sm_89, sm_120. "
            "PTX-JIT floor at compute_75 for everything else, driver-conditional. "
            "A packager shipping this fatbin must publish its own per-architecture "
            "coverage matrix -- see CUDA_ARCHITECTURE_POLICY.md, this is broader than "
            "ci-bounded-recent but still not universal.")
    else()
        message(FATAL_ERROR
            "Unknown SECP256K1_CUDA_ARCH_PROFILE='${SECP256K1_CUDA_ARCH_PROFILE}'. "
            "Valid values:\n${_menu}")
    endif()
endfunction()
