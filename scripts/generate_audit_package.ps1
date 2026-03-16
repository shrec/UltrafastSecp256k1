#!/usr/bin/env pwsh
# ============================================================================
# generate_audit_package.ps1 -- Comprehensive Audit Evidence Generator
# ============================================================================
#
# Builds the unified_audit_runner, runs it, and aggregates all evidence
# into a single dated directory ready for auditor review.
#
# Output structure:
#   audit-evidence-YYYYMMDD-HHMMSS/
#     audit_report.json            -- Machine-readable test results (8 sections)
#     audit_report.txt             -- Human-readable summary
#     build_info.json              -- Build environment details
#     tool_evidence/
#       ctest_results.xml          -- CTest XML output (if JUnit available)
#       compiler_version.txt       -- Compiler version string
#       cmake_cache_summary.txt    -- Build options used
#     README.txt                   -- Guide for auditor
#
# Usage:
#   pwsh scripts/generate_audit_package.ps1
#   pwsh scripts/generate_audit_package.ps1 -BuildDir build-audit
#   pwsh scripts/generate_audit_package.ps1 -Section math_invariants
# ============================================================================
param(
    [string]$BuildDir = "build-audit",
    [string]$Section = "",
    [switch]$SkipBuild,
    [switch]$Help
)

$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

if ($Help) {
    Write-Host @"
generate_audit_package.ps1 -- Comprehensive Audit Evidence Generator

PARAMETERS:
  -BuildDir <dir>     Build directory (default: build-audit)
  -Section <id>       Run only one section (default: all 8)
  -SkipBuild          Skip cmake configure + build
  -Help               Show this message

SECTIONS:
  math_invariants     Mathematical Invariants (Fp, Zn, Group Laws)
  ct_analysis         Constant-Time & Side-Channel Analysis
  differential        Differential & Cross-Library Testing
  standard_vectors    Standard Test Vectors (BIP-340, RFC-6979, BIP-32)
  fuzzing             Fuzzing & Adversarial Attack Resilience
  protocol_security   Protocol Security (ECDSA, Schnorr, MuSig2, FROST)
  memory_safety       ABI & Memory Safety (zeroization, hardening)
  performance         Performance Validation & Regression
"@
    exit 0
}

# -- Locate project root (one level up from scripts/) ----------------------
$ScriptDir = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path "$ScriptDir/CMakeLists.txt")) {
    # Try from repo root directly
    $ScriptDir = $PSScriptRoot
    if (-not (Test-Path "$ScriptDir/../CMakeLists.txt")) {
        Write-Error "Cannot find project root CMakeLists.txt"
        exit 1
    }
    $ScriptDir = Resolve-Path "$ScriptDir/.."
}
$ProjectRoot = $ScriptDir

# -- Timestamp & output directory ------------------------------------------
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$OutputDir = Join-Path $ProjectRoot "audit-evidence-$ts"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir/tool_evidence" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir/ct_evidence" | Out-Null

Write-Host "================================================================"
Write-Host "  Audit Evidence Package Generator"
Write-Host "  Output: $OutputDir"
Write-Host "  Timestamp: $ts"
Write-Host "================================================================"
Write-Host ""

# -- Step 1: Build (unless skipped) ----------------------------------------
$FullBuildDir = Join-Path $ProjectRoot $BuildDir

if (-not $SkipBuild) {
    Write-Host "[1/4] Configuring + building..."

    $cmakeArgs = @(
        "-S", $ProjectRoot,
        "-B", $FullBuildDir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=ON",
        "-DSECP256K1_BUILD_PROTOCOL_TESTS=ON",
        "-DSECP256K1_BUILD_FUZZ_TESTS=ON"
    )

    # Prefer Ninja if available
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $cmakeArgs += @("-G", "Ninja")
    }

    cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }

    cmake --build $FullBuildDir --config Release -j
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }
} else {
    Write-Host "[1/4] Build skipped (--SkipBuild)"
    if (-not (Test-Path $FullBuildDir)) {
        Write-Error "Build directory $FullBuildDir does not exist"
        exit 1
    }
}

# -- Step 2: Find the runner binary ----------------------------------------
$runner = $null
$candidates = @(
    "$FullBuildDir/audit/unified_audit_runner",
    "$FullBuildDir/audit/unified_audit_runner.exe",
    "$FullBuildDir/audit/Release/unified_audit_runner.exe",
    "$FullBuildDir/audit/RelWithDebInfo/unified_audit_runner.exe"
)
foreach ($c in $candidates) {
    if (Test-Path $c) { $runner = $c; break }
}
if (-not $runner) {
    Write-Error "Cannot find unified_audit_runner binary in $FullBuildDir"
    exit 1
}
Write-Host "[2/4] Found runner: $runner"

# -- Step 3: Run unified audit runner --------------------------------------
Write-Host "[3/4] Running unified audit runner..."

$runnerArgs = @("--report-dir", $OutputDir)
if ($Section) {
    $runnerArgs += @("--section", $Section)
}

& $runner @runnerArgs
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[3/4] Audit runner: ALL PASSED"
} else {
    Write-Host "[3/4] Audit runner: FAILURES DETECTED (exit code $exitCode)"
}

# -- Step 4: Collect additional evidence ------------------------------------
Write-Host "[4/4] Collecting tool evidence..."

# 4a. Build info JSON
$buildInfo = @{
    timestamp = $ts
    project_root = $ProjectRoot
    build_dir = $FullBuildDir
    runner_path = $runner
    runner_exit_code = $exitCode
    section_filter = if ($Section) { $Section } else { "all" }
    powershell_version = $PSVersionTable.PSVersion.ToString()
    os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
}
$buildInfo | ConvertTo-Json -Depth 4 | Set-Content "$OutputDir/build_info.json"

# 4b. Compiler version
$compilerInfo = ""
foreach ($cc in @("cl", "g++-13", "g++", "clang++-17", "clang++")) {
    try {
        $ver = & $cc --version 2>&1 | Select-Object -First 3
        if ($ver) {
            $compilerInfo += "=== $cc ===`n$($ver -join "`n")`n`n"
        }
    } catch { }
}
if ($compilerInfo) {
    $compilerInfo | Set-Content "$OutputDir/tool_evidence/compiler_version.txt"
}

# 4c. CMake cache summary (key build options)
$cacheFile = "$FullBuildDir/CMakeCache.txt"
if (Test-Path $cacheFile) {
    $patterns = @(
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_STANDARD",
        "SECP256K1_BUILD_",
        "BUILD_TESTING",
        "CMAKE_SYSTEM_NAME",
        "CMAKE_SYSTEM_PROCESSOR"
    )
    $cacheLines = Get-Content $cacheFile | Where-Object {
        $line = $_
        ($patterns | ForEach-Object { $line -match $_ }) -contains $true
    }
    $cacheLines | Set-Content "$OutputDir/tool_evidence/cmake_cache_summary.txt"
}

# 4d. CTest results (if available)
$ctestXml = "$FullBuildDir/Testing/*/Test.xml"
$xmlFiles = Get-ChildItem $ctestXml -ErrorAction SilentlyContinue
if ($xmlFiles) {
    Copy-Item ($xmlFiles | Select-Object -Last 1).FullName "$OutputDir/tool_evidence/ctest_results.xml"
}

# 4e. Git info
$gitHash = git -C $ProjectRoot rev-parse --short=8 HEAD 2>$null
$gitBranch = git -C $ProjectRoot rev-parse --abbrev-ref HEAD 2>$null
$gitInfo = "Branch: $gitBranch`nCommit: $gitHash`n"
$gitInfo | Set-Content "$OutputDir/tool_evidence/git_info.txt"

# 4f. CT evidence collection
Write-Host "  Collecting CT evidence artifacts..."
$ctDir = "$OutputDir/ct_evidence"
$ctArtifacts = @(
    @{ Name = "ct_verif.log";             Candidates = @("artifacts/ct/ct_verif.log", "ct_verif.log") },
    @{ Name = "ct_verif_summary.json";    Candidates = @("artifacts/ct/ct_verif_summary.json", "ct_verif_summary.json") },
    @{ Name = "valgrind_ct.log";          Candidates = @("artifacts/ct/valgrind_ct.log", "valgrind_ct.log") },
    @{ Name = "valgrind_ct_report.json";  Candidates = @("artifacts/ct/valgrind_ct_report.json", "valgrind_ct_report.json") },
    @{ Name = "disasm_branch_scan.json";  Candidates = @("artifacts/ct/disasm_branch_scan.json", "artifacts/disasm/disasm_branch_scan.json") },
    @{ Name = "dudect_smoke.log";         Candidates = @("artifacts/ct/dudect_smoke.log", "dudect_smoke.log") },
    @{ Name = "dudect_full.log";          Candidates = @("artifacts/ct/dudect_full.log", "dudect_full.log") }
)
$ctCount = 0
foreach ($art in $ctArtifacts) {
    foreach ($cand in $art.Candidates) {
        $full = Join-Path $ProjectRoot $cand
        $buildCand = Join-Path $FullBuildDir $art.Name
        if (Test-Path $full) {
            Copy-Item $full "$ctDir/$($art.Name)"
            Write-Host "  [+] $($art.Name)"
            $ctCount++
            break
        } elseif (Test-Path $buildCand) {
            Copy-Item $buildCand "$ctDir/$($art.Name)"
            Write-Host "  [+] $($art.Name)"
            $ctCount++
            break
        }
    }
}
if ($ctCount -gt 0) {
    Write-Host "  CT evidence: $ctCount artifact(s) collected"
} else {
    Write-Host "  CT evidence: none found locally (check CI artifacts)"
}

# 4g. Auditor README
@"
================================================================
  UltrafastSecp256k1 -- Audit Evidence Package
  Generated: $ts
================================================================

This directory contains the complete self-audit evidence for the
UltrafastSecp256k1 cryptographic library.

CONTENTS:
  audit_report.json          Machine-readable test results (8 sections)
  audit_report.txt           Human-readable audit summary
  build_info.json            Build environment metadata
  tool_evidence/
    compiler_version.txt     Compiler version used
    cmake_cache_summary.txt  CMake build options
    ctest_results.xml        CTest XML results (if run)
    git_info.txt             Git branch + commit
  ct_evidence/
    ct_verif.log             Deterministic CT verification output (ct-verif)
    ct_verif_summary.json    CT verification structured results
    valgrind_ct.log          Valgrind CT memcheck output (ctgrind mode)
    valgrind_ct_report.json  Valgrind CT structured results
    disasm_branch_scan.json  Disassembly branch analysis of CT functions
    dudect_smoke.log         Statistical timing test (dudect, smoke run)
    dudect_full.log          Statistical timing test (dudect, full run)

Note: ct_evidence/ files are present when available from local or CI runs.
If empty, check CI artifacts at the URLs listed below.

AUDIT SECTIONS (8 categories):
  1. Mathematical Invariants    -- Fp, Zn, group laws (13 modules)
  2. CT & Side-Channel          -- dudect, FAST==CT, timing (5 modules)
  3. Differential Testing       -- cross-library, Fiat-Crypto (3 modules)
  4. Standard Test Vectors      -- BIP-340, RFC-6979, BIP-32, FROST (4 modules)
  5. Fuzzing & Adversarial      -- parser fuzz, fault injection (4 modules)
  6. Protocol Security          -- ECDSA, Schnorr, MuSig2, FROST (9 modules)
  7. ABI & Memory Safety        -- zeroization, ABI gate (3 modules)
  8. Performance Validation     -- SIMD, hash accel, multi-scalar (4 modules)

TOTAL: 47 test modules + 1 library selftest = 48 checks

CT VERIFICATION STACK:
  Layer 1: Statistical timing leakage testing (dudect)
    - Welch t-test, |t| > 4.5 = leak. Advisory on shared runners.
  Layer 2: Deterministic CT verification (ct-verif + valgrind-ct)
    - ct-verif: LLVM-based taint tracking, blocking in CI
    - valgrind-ct: ctgrind-mode memcheck, blocking in CI
  Layer 3: Disassembly branch scan
    - Checks compiled CT functions for conditional branches on secrets
  Layer 4: Machine-checked proof frameworks
    - Not currently applied (Vale/Jasmin/Coq/Fiat-Crypto)

HOW TO VERIFY:
  1. Review audit_report.json for structured pass/fail data
  2. Confirm all 8 sections show "status": "PASS"
  3. Check ct_evidence/ for deterministic CT verification results
  4. Verify platform/compiler info matches expected target
  5. Check build_info.json for reproducible build parameters

EXTERNAL TOOL EVIDENCE (collected separately by CI):
  - ct-verif:    .github/workflows/ct-verif.yml    -> Actions artifacts
  - valgrind-ct: .github/workflows/ct-verif.yml    -> Actions artifacts
  - CodeQL:      .github/workflows/codeql.yml      -> Security tab
  - Cppcheck:    .github/workflows/cppcheck.yml    -> Security tab
  - Scorecard:   .github/workflows/scorecard.yml   -> Security tab
  - SonarCloud:  .github/workflows/sonarcloud.yml  -> sonarcloud.io
  - ClusterFuzz: .github/workflows/cflite.yml      -> PR checks
  - Mutation:    .github/workflows/mutation.yml     -> Weekly run
  - Clang-Tidy:  .github/workflows/clang-tidy.yml  -> Security tab
  - GPU audit:   .github/workflows/gpu-selfhosted.yml -> Self-hosted runner
================================================================
"@ | Set-Content "$OutputDir/README.txt"

# -- Summary ---------------------------------------------------------------
Write-Host ""
Write-Host "================================================================"
Write-Host "  Audit Evidence Package: COMPLETE"
Write-Host "  Location: $OutputDir"
Write-Host ""

$files = Get-ChildItem $OutputDir -Recurse -File
Write-Host "  Files:"
foreach ($f in $files) {
    $rel = $f.FullName.Substring($OutputDir.Length + 1)
    $size = "{0:N0}" -f $f.Length
    Write-Host ("    {0,-45} {1,8} bytes" -f $rel, $size)
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "  VERDICT: AUDIT-READY (all modules passed)"
} else {
    Write-Host "  VERDICT: AUDIT-BLOCKED (failures detected)"
}
Write-Host "================================================================"

exit $exitCode
