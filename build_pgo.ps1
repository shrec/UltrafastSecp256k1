# ============================================================================
# PGO (Profile-Guided Optimization) Build Script — Windows (MSVC / Clang-CL)
# ============================================================================
# Three-phase build: Instrument → Profile → Optimize
# Expected improvement: 10-25% on scalar multiplication hot paths.
#
# Usage: .\build_pgo.ps1 [-Compiler msvc|clang] [-Jobs 4]
# ============================================================================

param(
    [ValidateSet("msvc", "clang")]
    [string]$Compiler = "msvc",
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir  = Join-Path $ScriptDir "build/pgo"
$PGODir    = Join-Path $BuildDir "pgo_profiles"

# ── Phase 1: Instrumentation ──────────────────────────────────────────────

Write-Host "`n=============================================="
Write-Host "  PGO Build — Phase 1: Instrumentation"
Write-Host "  Compiler: $Compiler"
Write-Host "==============================================`n"

if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
New-Item -ItemType Directory -Path $PGODir   -Force | Out-Null

$cmakeArgs = @(
    "-S", $ScriptDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_TESTING=ON",
    "-DSECP256K1_USE_PGO_GEN=ON",
    "-DSECP256K1_PGO_PROFILE_DIR=$PGODir"
)

if ($Compiler -eq "clang") {
    $cmakeArgs += @("-T", "ClangCL")
}

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

cmake --build $BuildDir --config Release -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build (instrumented) failed" }

# ── Phase 2: Profiling ────────────────────────────────────────────────────

Write-Host "`n=============================================="
Write-Host "  PGO Build — Phase 2: Profiling"
Write-Host "==============================================`n"

# Run CTest to exercise hot paths
ctest --test-dir $BuildDir -C Release --output-on-failure 2>$null
Write-Host "  [OK] Test suite completed"

# Run any benchmark executables
Get-ChildItem -Path $BuildDir -Recurse -Filter "*bench*" |
    Where-Object { $_.Extension -in ".exe", "" } |
    ForEach-Object {
        Write-Host "  Running: $($_.FullName)"
        & $_.FullName 2>$null
    }

# ── Phase 3: Merge & Optimize ────────────────────────────────────────────

Write-Host "`n=============================================="
Write-Host "  PGO Build — Phase 3: Optimize"
Write-Host "==============================================`n"

if ($Compiler -eq "clang") {
    $profrawFiles = Get-ChildItem -Path $PGODir -Filter "*.profraw" -ErrorAction SilentlyContinue
    if ($profrawFiles.Count -gt 0) {
        Write-Host "  Merging $($profrawFiles.Count) profile(s)..."
        llvm-profdata merge -o (Join-Path $PGODir "default.profdata") @($profrawFiles | ForEach-Object { $_.FullName })
    }
}

# Reconfigure with PGO-USE
Remove-Item (Join-Path $BuildDir "CMakeCache.txt") -ErrorAction SilentlyContinue

$cmakeArgs = @(
    "-S", $ScriptDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_TESTING=ON",
    "-DSECP256K1_USE_PGO_GEN=OFF",
    "-DSECP256K1_USE_PGO_USE=ON",
    "-DSECP256K1_PGO_PROFILE_DIR=$PGODir"
)

if ($Compiler -eq "clang") {
    $cmakeArgs += @("-T", "ClangCL")
}

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure (PGO-USE) failed" }

cmake --build $BuildDir --config Release -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build (PGO-optimized) failed" }

# ── Verification ──────────────────────────────────────────────────────────

Write-Host "`n=============================================="
Write-Host "  PGO Build — Verification"
Write-Host "==============================================`n"

ctest --test-dir $BuildDir -C Release --output-on-failure
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] All tests pass with PGO build"
} else {
    Write-Host "  [WARN] Some tests failed"
}

Write-Host "`n=============================================="
Write-Host "  PGO Build — Complete!"
Write-Host "=============================================="
Write-Host ""
Write-Host "  Library: $BuildDir\libs\UltrafastSecp256k1\cpu\Release\fastsecp256k1.lib"
Write-Host "  Profile: $PGODir"
Write-Host ""
Write-Host "  Expected improvements on hot paths:"
Write-Host "    - Scalar multiplication: 10-20%"
Write-Host "    - Point addition:         5-15%"
Write-Host "    - Schnorr/ECDSA sign:    10-15%"
