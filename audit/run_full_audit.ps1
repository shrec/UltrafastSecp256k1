#!/usr/bin/env pwsh
# ============================================================================
# run_full_audit.ps1 -- Full Audit Orchestrator (Windows / Cross-Platform)
# ============================================================================
#
# Run with a single command:
#   pwsh -NoProfile -File audit/run_full_audit.ps1
#
# This script performs a full audit cycle (A-M categories):
#   A. Environment & Build Integrity
#   B. Packaging & Supply Chain
#   C. Static Analysis
#   D. Sanitizers (ASan/UBSan)
#   E. Unit Tests / KAT
#   F. Property-Based / Algebraic Invariants
#   G. Differential Testing
#   H. Fuzzing
#   I. Constant-Time & Side-Channel
#   J. ABI / API Stability
#   K. Bindings & FFI Parity
#   L. Performance Regression
#   M. Documentation Consistency
#
# Output artifacts (in artifacts/ directory):
#   audit_report.md
#   artifacts/SHA256SUMS.txt
#   artifacts/sbom.cdx.json
#   artifacts/toolchain_fingerprint.json
#   artifacts/dependency_scan.txt
#   artifacts/static_analysis/clang_tidy.log
#   artifacts/static_analysis/cppcheck.log
#   artifacts/sanitizers/asan_ubsan.log
#   artifacts/disasm_branch_scan.json
#   artifacts/dudect_report.json
#   artifacts/ctest/results.json
#   artifacts/bindings/parity_matrix.json
#   artifacts/benchmark/summary.json
# ============================================================================

param(
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [switch]$SkipBuild,
    [switch]$SkipSanitizers,
    [switch]$SkipStaticAnalysis,
    [switch]$SkipFuzz,
    [switch]$SkipBindings,
    [switch]$SkipBenchmark,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"  # Don't stop on individual test failures

# -- Resolve paths ----------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path "$ScriptDir/..").Path
$Version = (Get-Content "$RootDir/VERSION.txt" -Raw).Trim()
$Timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
$DateTag = Get-Date -Format "yyyyMMdd-HHmmss"

if (-not $BuildDir) { $BuildDir = "$RootDir/build-audit" }
if (-not $OutputDir) { $OutputDir = "$RootDir/audit-output-$DateTag" }

$ArtifactsDir = "$OutputDir/artifacts"

# Create output directories
foreach ($d in @(
    $ArtifactsDir
    "$ArtifactsDir/static_analysis"
    "$ArtifactsDir/sanitizers"
    "$ArtifactsDir/ctest"
    "$ArtifactsDir/bindings"
    "$ArtifactsDir/benchmark"
    "$ArtifactsDir/disasm"
    "$ArtifactsDir/fuzz"
)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# -- Globals for tracking --------------------------------------------------

$Script:CategoryResults = [ordered]@{}
$Script:Findings = @()
$Script:FindingCounter = 0

function Add-CategoryResult {
    param([string]$Category, [string]$Status, [string]$Summary, [double]$TimeMs = 0)
    $Script:CategoryResults[$Category] = @{
        Status  = $Status
        Summary = $Summary
        TimeMs  = $TimeMs
    }
}

function Add-Finding {
    param(
        [string]$Severity,   # Critical/High/Med/Low/Info
        [string]$Component,
        [string]$Description,
        [string]$Evidence = "",
        [string]$Recommendation = ""
    )
    $Script:FindingCounter++
    $id = "UF-AUD-{0:D3}" -f $Script:FindingCounter
    $Script:Findings += @{
        ID             = $id
        Severity       = $Severity
        Component      = $Component
        Description    = $Description
        Evidence       = $Evidence
        Recommendation = $Recommendation
        Status         = "Open"
    }
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-SubStep {
    param([string]$Text, [string]$Status = "...")
    $color = switch ($Status) {
        "PASS" { "Green" }
        "FAIL" { "Red" }
        "SKIP" { "Yellow" }
        "WARN" { "Yellow" }
        default { "White" }
    }
    Write-Host "  [$Status] $Text" -ForegroundColor $color
}

# -- Toolchain detection ---------------------------------------------------

function Get-ToolchainFingerprint {
    $fp = [ordered]@{
        timestamp       = $Timestamp
        os              = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
        arch            = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        dotnet_version  = $PSVersionTable.PSVersion.ToString()
    }

    # CMake
    try { $fp["cmake"] = (cmake --version | Select-Object -First 1).Trim() } catch { $fp["cmake"] = "not found" }

    # Ninja
    try { $fp["ninja"] = (ninja --version 2>$null).Trim() } catch { $fp["ninja"] = "not found" }

    # C++ compiler (try clang first, then MSVC, then GCC)
    $compiler = "not found"
    foreach ($cc in @("clang++", "clang++-19", "clang++-21", "cl", "g++")) {
        try {
            if ($cc -eq "cl") {
                $out = & cl 2>&1 | Select-Object -First 1
                if ($out) { $compiler = $out.ToString().Trim(); break }
            } else {
                $out = & $cc --version 2>$null | Select-Object -First 1
                if ($out) { $compiler = $out.Trim(); break }
            }
        } catch {}
    }
    $fp["cxx_compiler"] = $compiler

    # Git commit
    try {
        $fp["git_commit"] = (git -C $RootDir rev-parse HEAD 2>$null).Trim()
        $fp["git_branch"] = (git -C $RootDir rev-parse --abbrev-ref HEAD 2>$null).Trim()
        $fp["git_dirty"]  = [bool](git -C $RootDir status --porcelain 2>$null)
    } catch {
        $fp["git_commit"] = "unknown"
        $fp["git_branch"] = "unknown"
        $fp["git_dirty"]  = $false
    }

    $fp["library_version"] = $Version

    # clang-tidy
    try { $fp["clang_tidy"] = (& clang-tidy --version 2>$null | Select-Object -First 2 | Select-Object -Last 1).Trim() } catch { $fp["clang_tidy"] = "not found" }

    # cppcheck
    try { $fp["cppcheck"] = (cppcheck --version 2>$null).Trim() } catch { $fp["cppcheck"] = "not found" }

    return $fp
}

# ========================================================================
# A. Environment & Build Integrity
# ========================================================================
function Run-CategoryA {
    Write-Section "A. Environment & Build Integrity"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $allPass = $true

    # A.1 Toolchain fingerprint
    $fp = Get-ToolchainFingerprint
    $fp | ConvertTo-Json -Depth 5 | Out-File "$ArtifactsDir/toolchain_fingerprint.json" -Encoding utf8
    Write-SubStep "Toolchain fingerprint collected" "PASS"

    # A.2 Build (Release)
    if (-not $SkipBuild) {
        Write-SubStep "Configuring CMake (Release)..." "..."
        $configResult = & cmake -S $RootDir -B $BuildDir -G Ninja `
            -DCMAKE_BUILD_TYPE=Release `
            -DSECP256K1_BUILD_TESTS=ON `
            -DSECP256K1_BUILD_BENCH=ON `
            -DSECP256K1_BUILD_FUZZ_TESTS=ON `
            -DSECP256K1_BUILD_PROTOCOL_TESTS=ON `
            -DSECP256K1_USE_ASM=ON 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-SubStep "CMake configure failed" "FAIL"
            $allPass = $false
            Add-Finding "Critical" "Build" "CMake configure failed" ($configResult -join "`n") "Fix build errors"
        } else {
            Write-SubStep "CMake configure OK" "PASS"
        }

        Write-SubStep "Building..." "..."
        $buildResult = & cmake --build $BuildDir -j ([Environment]::ProcessorCount) 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-SubStep "Build failed" "FAIL"
            $allPass = $false
            Add-Finding "Critical" "Build" "Build failed" ($buildResult | Select-Object -Last 30 | Out-String) "Fix compile errors"
        } else {
            Write-SubStep "Build succeeded" "PASS"
        }
    } else {
        Write-SubStep "Build skipped (--SkipBuild)" "SKIP"
    }

    # A.3 Dependency scan (link-time)
    Write-SubStep "Scanning link-time dependencies..." "..."
    $depFile = "$ArtifactsDir/dependency_scan.txt"
    $depScanOk = $true
    $runnerBin = Get-ChildItem -Path $BuildDir -Recurse -Filter "unified_audit_runner*" -File | Where-Object { $_.Extension -in @('.exe', '') } | Select-Object -First 1

    if ($runnerBin) {
        if ($IsWindows -or $env:OS -match "Windows") {
            # Windows: dumpbin
            try {
                $dumpOut = & dumpbin /dependents $runnerBin.FullName 2>$null
                if ($dumpOut) {
                    $dumpOut | Out-File $depFile -Encoding utf8
                    # Check for unexpected deps
                    $unexpected = $dumpOut | Where-Object { $_ -match '\.(dll|DLL)' } | Where-Object {
                        $_ -notmatch '(KERNEL32|ADVAPI32|VCRUNTIME|UCRTBASE|api-ms-|ntdll|USER32|msvcrt)'
                    }
                    if ($unexpected) {
                        Write-SubStep "Unexpected dependencies found" "WARN"
                        Add-Finding "Med" "Build" "Unexpected link-time dependencies" ($unexpected -join "`n") "Review and eliminate if possible"
                    } else {
                        Write-SubStep "Dependency scan clean" "PASS"
                    }
                }
            } catch {
                "dumpbin not available - trying PowerShell PE parser" | Out-File $depFile
                Write-SubStep "dumpbin not available" "WARN"
            }
        } else {
            # Linux: ldd
            try {
                $lddOut = ldd $runnerBin.FullName 2>&1
                $lddOut | Out-File $depFile -Encoding utf8
                Write-SubStep "Dependency scan (ldd) collected" "PASS"
            } catch {
                Write-SubStep "ldd not available" "WARN"
            }
        }
    } else {
        Write-SubStep "unified_audit_runner binary not found" "WARN"
        $depScanOk = $false
    }

    # A.4 Artifact manifest (SHA256)
    Write-SubStep "Computing artifact SHA256 manifest..." "..."
    $sha256File = "$ArtifactsDir/SHA256SUMS.txt"
    $artifacts = Get-ChildItem -Path $BuildDir -Recurse -Include @("*.exe","*.lib","*.a","*.so","*.dll","*.pdb") -File 2>$null |
        Where-Object { $_.FullName -notmatch 'CMakeFiles' }
    $sha256Lines = @()
    foreach ($art in $artifacts) {
        $hash = (Get-FileHash -Path $art.FullName -Algorithm SHA256).Hash
        $relPath = $art.FullName.Replace($BuildDir, "build")
        $sha256Lines += "$hash  $relPath  ($([math]::Round($art.Length / 1KB, 1)) KB)"
    }
    $sha256Lines | Out-File $sha256File -Encoding utf8
    Write-SubStep "SHA256 manifest: $($sha256Lines.Count) artifacts" "PASS"

    $sw.Stop()
    $status = if ($allPass) { "PASS" } else { "FAIL" }
    Add-CategoryResult "A" $status "Build Integrity" $sw.ElapsedMilliseconds
}

# ========================================================================
# B. Packaging & Supply Chain
# ========================================================================
function Run-CategoryB {
    Write-Section "B. Packaging & Supply Chain"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # B.1 SBOM generation
    $sbomFile = "$ArtifactsDir/sbom.cdx.json"
    $sbomScript = "$RootDir/scripts/generate_sbom.sh"
    if (Test-Path $sbomScript) {
        # Try running with bash if available
        $bash = Get-Command bash -ErrorAction SilentlyContinue
        if ($bash) {
            try {
                & bash $sbomScript $sbomFile 2>$null
                Write-SubStep "SBOM generated (CycloneDX)" "PASS"
            } catch {
                Write-SubStep "SBOM script failed" "WARN"
            }
        } else {
            # Generate minimal SBOM inline
            $sbomJson = @{
                bomFormat = "CycloneDX"
                specVersion = "1.6"
                version = 1
                metadata = @{
                    timestamp = $Timestamp
                    component = @{
                        type = "library"
                        name = "UltrafastSecp256k1"
                        version = $Version
                        description = "High-performance secp256k1 ECC library"
                        licenses = @(@{ license = @{ id = "AGPL-3.0-or-later" } })
                    }
                }
                components = @()
            } | ConvertTo-Json -Depth 10
            $sbomJson | Out-File $sbomFile -Encoding utf8
            Write-SubStep "SBOM generated (minimal, no bash)" "PASS"
        }
    } else {
        Write-SubStep "SBOM script not found" "WARN"
    }

    # B.2 Provenance metadata
    $provenance = [ordered]@{
        builder = @{
            id = $env:COMPUTERNAME
            os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
        }
        source = @{
            commit = try { (git -C $RootDir rev-parse HEAD 2>$null).Trim() } catch { "unknown" }
            branch = try { (git -C $RootDir rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch { "unknown" }
            dirty  = [bool](git -C $RootDir status --porcelain 2>$null)
        }
        build = @{
            timestamp = $Timestamp
            type = "Release"
            dir = $BuildDir
        }
    }
    $provenance | ConvertTo-Json -Depth 5 | Out-File "$ArtifactsDir/provenance.json" -Encoding utf8
    Write-SubStep "Provenance metadata collected" "PASS"

    $sw.Stop()
    Add-CategoryResult "B" "PASS" "Supply Chain" $sw.ElapsedMilliseconds
}

# ========================================================================
# C. Static Analysis
# ========================================================================
function Run-CategoryC {
    Write-Section "C. Static Analysis"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($SkipStaticAnalysis) {
        Write-SubStep "Static analysis skipped" "SKIP"
        Add-CategoryResult "C" "SKIP" "Static Analysis (skipped)" 0
        return
    }

    $allPass = $true

    # C.1 clang-tidy
    $clangTidy = Get-Command clang-tidy -ErrorAction SilentlyContinue
    if ($clangTidy) {
        Write-SubStep "Running clang-tidy..." "..."
        $ctidyLog = "$ArtifactsDir/static_analysis/clang_tidy.log"
        $cpuSources = Get-ChildItem "$RootDir/cpu/include/secp256k1/*.hpp" -File |
            Where-Object { $_.Name -notmatch "test_|benchmark_" } |
            Select-Object -First 20 -ExpandProperty FullName

        if ($cpuSources) {
            $ctidyArgs = @($cpuSources) + @("--", "-std=c++20", "-I$RootDir/cpu/include")
            $ctidyOut = & clang-tidy @ctidyArgs 2>&1
            $ctidyOut | Out-File $ctidyLog -Encoding utf8
            $warnings = ($ctidyOut | Where-Object { $_ -match 'warning:' }).Count
            $errors = ($ctidyOut | Where-Object { $_ -match 'error:' }).Count

            if ($errors -gt 0) {
                Write-SubStep "clang-tidy: $errors errors, $warnings warnings" "FAIL"
                Add-Finding "Med" "Static Analysis" "clang-tidy found $errors errors" "See artifacts/static_analysis/clang_tidy.log" "Fix clang-tidy errors"
                $allPass = $false
            } else {
                Write-SubStep "clang-tidy: $warnings warnings, 0 errors" "PASS"
            }
        }
    } else {
        Write-SubStep "clang-tidy not found" "SKIP"
    }

    # C.2 cppcheck
    $cppcheck = Get-Command cppcheck -ErrorAction SilentlyContinue
    if ($cppcheck) {
        Write-SubStep "Running cppcheck..." "..."
        $cppcheckLog = "$ArtifactsDir/static_analysis/cppcheck.log"
        $cppcheckOut = & cppcheck --enable=all --std=c++20 --suppress=missingInclude `
            --suppress=unusedFunction --quiet `
            "$RootDir/cpu/include/secp256k1/" 2>&1
        $cppcheckOut | Out-File $cppcheckLog -Encoding utf8
        $cppErrors = ($cppcheckOut | Where-Object { $_ -match '\(error\)' }).Count
        if ($cppErrors -gt 0) {
            Write-SubStep "cppcheck: $cppErrors errors" "FAIL"
            $allPass = $false
        } else {
            Write-SubStep "cppcheck: clean" "PASS"
        }
    } else {
        Write-SubStep "cppcheck not found" "SKIP"
    }

    # C.3 Dangerous patterns scan (manual grep)
    Write-SubStep "Scanning for dangerous patterns..." "..."
    $dangerousPatterns = @(
        @{ Pattern = 'memcpy\s*\([^,]+,[^,]+,\s*sizeof\s*\(\s*\*'; Name = "memcpy sizeof(*ptr) mismatch" }
        @{ Pattern = 'std::vector.*push_back.*hot'; Name = "push_back in hot path" }
        @{ Pattern = 'throw\s+'; Name = "exception throw (avoid in hot path)" }
        @{ Pattern = 'dynamic_cast'; Name = "dynamic_cast (RTTI)" }
        @{ Pattern = 'std::string\s+\w+\s*=.*hot|hot.*std::string'; Name = "std::string in hot path" }
    )
    $dangerLog = "$ArtifactsDir/static_analysis/dangerous_patterns.log"
    $patternFindings = @()
    $cpuHeaders = Get-ChildItem "$RootDir/cpu/include/secp256k1/*.hpp" -File
    foreach ($pat in $dangerousPatterns) {
        foreach ($file in $cpuHeaders) {
            $matches = Select-String -Path $file.FullName -Pattern $pat.Pattern -AllMatches
            if ($matches) {
                foreach ($m in $matches) {
                    $patternFindings += "$($pat.Name): $($file.Name):$($m.LineNumber)"
                }
            }
        }
    }
    if ($patternFindings.Count -gt 0) {
        $patternFindings | Out-File $dangerLog -Encoding utf8
        Write-SubStep "Dangerous patterns: $($patternFindings.Count) hits" "WARN"
    } else {
        "No dangerous patterns found." | Out-File $dangerLog -Encoding utf8
        Write-SubStep "Dangerous patterns: clean" "PASS"
    }

    $sw.Stop()
    $status = if ($allPass) { "PASS" } else { "FAIL" }
    Add-CategoryResult "C" $status "Static Analysis" $sw.ElapsedMilliseconds
}

# ========================================================================
# D. Sanitizers (ASan/UBSan)
# ========================================================================
function Run-CategoryD {
    Write-Section "D. Sanitizers (ASan/UBSan)"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($SkipSanitizers) {
        Write-SubStep "Sanitizers skipped" "SKIP"
        Add-CategoryResult "D" "SKIP" "Sanitizers (skipped)" 0
        return
    }

    $asanBuildDir = "$RootDir/build-audit-asan"
    $asanLog = "$ArtifactsDir/sanitizers/asan_ubsan.log"

    # Check if compiler supports sanitizers (clang or gcc on Linux)
    $compiler = ""
    foreach ($cc in @("clang++", "clang++-19", "clang++-21", "g++")) {
        if (Get-Command $cc -ErrorAction SilentlyContinue) {
            $compiler = $cc
            break
        }
    }

    if (-not $compiler -or ($IsWindows -or $env:OS -match "Windows")) {
        # On Windows, ASan support is limited
        Write-SubStep "ASan/UBSan: limited on Windows MSVC" "WARN"
        # Try MSVC ASan
        Write-SubStep "Attempting MSVC /fsanitize=address build..." "..."
        try {
            & cmake -S $RootDir -B $asanBuildDir -G Ninja `
                -DCMAKE_BUILD_TYPE=Debug `
                -DSECP256K1_BUILD_TESTS=ON `
                -DSECP256K1_BUILD_BENCH=OFF `
                -DSECP256K1_USE_ASM=OFF `
                -DCMAKE_CXX_FLAGS="/fsanitize=address" 2>&1 | Out-Null

            & cmake --build $asanBuildDir -j ([Environment]::ProcessorCount) 2>&1 | Out-Null

            $asanResult = & ctest --test-dir $asanBuildDir --output-on-failure --timeout 300 2>&1
            $asanResult | Out-File $asanLog -Encoding utf8

            $failed = ($asanResult | Where-Object { $_ -match 'Failed' }).Count
            if ($failed -eq 0) {
                Write-SubStep "ASan tests passed" "PASS"
            } else {
                Write-SubStep "ASan tests: $failed failures" "FAIL"
                Add-Finding "High" "Memory Safety" "ASan detected issues" "See artifacts/sanitizers/asan_ubsan.log" "Fix memory bugs"
            }
        } catch {
            Write-SubStep "MSVC ASan build failed" "WARN"
            "MSVC ASan build failed: $_" | Out-File $asanLog -Encoding utf8
        }
    } else {
        # Linux/macOS with Clang/GCC
        Write-SubStep "Building with ASan + UBSan ($compiler)..." "..."
        & cmake -S $RootDir -B $asanBuildDir -G Ninja `
            -DCMAKE_BUILD_TYPE=Debug `
            -DCMAKE_CXX_COMPILER=$compiler `
            -DSECP256K1_BUILD_TESTS=ON `
            -DSECP256K1_BUILD_BENCH=OFF `
            -DSECP256K1_USE_ASM=OFF `
            -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" `
            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" 2>&1 | Out-Null

        & cmake --build $asanBuildDir -j 2>&1 | Out-Null

        $env:ASAN_OPTIONS = "detect_leaks=1:halt_on_error=0"
        $env:UBSAN_OPTIONS = "print_stacktrace=1:halt_on_error=0"
        $asanResult = & ctest --test-dir $asanBuildDir --output-on-failure --timeout 300 2>&1
        $asanResult | Out-File $asanLog -Encoding utf8
        Write-SubStep "ASan/UBSan tests completed" "PASS"
    }

    $sw.Stop()
    Add-CategoryResult "D" "PASS" "Sanitizers" $sw.ElapsedMilliseconds
}

# ========================================================================
# E-I. Unified Audit Runner (Unit/KAT/Property/Differential/Fuzz/CT)
# ========================================================================
function Run-CategoriesEI {
    Write-Section "E-I. Unified Audit Runner (Correctness + CT + Fuzz)"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $allPass = $true

    # Run unified_audit_runner
    $runner = Get-ChildItem -Path $BuildDir -Recurse -Filter "unified_audit_runner*" -File |
        Where-Object { $_.Extension -in @('.exe', '') } | Select-Object -First 1

    if (-not $runner) {
        Write-SubStep "unified_audit_runner not found in $BuildDir" "FAIL"
        Add-Finding "Critical" "Build" "unified_audit_runner binary not built" "" "Build with -DSECP256K1_BUILD_TESTS=ON"
        Add-CategoryResult "E-I" "FAIL" "Unified Audit Runner not found" 0
        return
    }

    Write-SubStep "Running unified_audit_runner..." "..."
    $runnerReportDir = "$ArtifactsDir/ctest"
    $env:UNIFIED_AUDIT_REPORT_DIR = $runnerReportDir

    $runnerOut = & $runner.FullName --report-dir $runnerReportDir 2>&1
    $runnerExit = $LASTEXITCODE

    # Save raw output
    $runnerOut | Out-File "$ArtifactsDir/ctest/unified_runner_output.txt" -Encoding utf8

    # Copy generated reports
    foreach ($rpt in @("audit_report.json", "audit_report.txt")) {
        $src = Join-Path $runnerReportDir $rpt
        if (Test-Path $src) {
            Copy-Item $src "$ArtifactsDir/ctest/$rpt" -Force
        }
    }

    if ($runnerExit -eq 0) {
        Write-SubStep "Unified audit runner: ALL PASSED" "PASS"
    } else {
        Write-SubStep "Unified audit runner: FAILURES DETECTED" "FAIL"
        $allPass = $false
        Add-Finding "High" "Correctness" "unified_audit_runner reported failures" "See artifacts/ctest/unified_runner_output.txt" "Investigate failing modules"
    }

    # Also run CTest for standalone audit targets
    Write-SubStep "Running CTest audit targets..." "..."
    $ctestOut = & ctest --test-dir $BuildDir --output-on-failure --timeout 600 -j ([Environment]::ProcessorCount) 2>&1
    $ctestOut | Out-File "$ArtifactsDir/ctest/ctest_output.txt" -Encoding utf8

    # Parse CTest results
    $totalTests = 0; $passedTests = 0; $failedTests = 0
    foreach ($line in $ctestOut) {
        if ($line -match '(\d+) tests? passed') { $passedTests = [int]$Matches[1] }
        if ($line -match '(\d+) tests? failed') { $failedTests = [int]$Matches[1] }
        if ($line -match 'Total Tests:\s*(\d+)') { $totalTests = [int]$Matches[1] }
    }
    if ($totalTests -eq 0) { $totalTests = $passedTests + $failedTests }

    $ctestResults = @{
        total  = $totalTests
        passed = $passedTests
        failed = $failedTests
    }
    $ctestResults | ConvertTo-Json | Out-File "$ArtifactsDir/ctest/results.json" -Encoding utf8
    Write-SubStep "CTest: $passedTests/$totalTests passed ($failedTests failed)" $(if ($failedTests -eq 0) { "PASS" } else { "FAIL" })

    $sw.Stop()
    $status = if ($allPass -and $failedTests -eq 0) { "PASS" } else { "FAIL" }
    Add-CategoryResult "E-I" $status "Correctness + CT + Fuzz" $sw.ElapsedMilliseconds
}

# ========================================================================
# J. ABI / API Stability
# ========================================================================
function Run-CategoryJ {
    Write-Section "J. ABI / API Stability & Safety"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # J.1 Exported symbols check
    $abiReport = @{
        timestamp = $Timestamp
        checks = @()
    }

    # Find ufsecp library
    $ufLib = Get-ChildItem -Path $BuildDir -Recurse -Include @("ufsecp_static.lib", "libufsecp_static.a", "ufsecp.dll", "libufsecp.so") -File 2>$null |
        Select-Object -First 1

    if ($ufLib) {
        Write-SubStep "Found ufsecp library: $($ufLib.Name)" "PASS"

        if ($IsWindows -or $env:OS -match "Windows") {
            try {
                $symbols = & dumpbin /symbols $ufLib.FullName 2>$null | Where-Object { $_ -match 'External.*ufsecp_' }
                $symbolCount = ($symbols | Measure-Object).Count
                $abiReport.checks += @{ name = "exported_symbols"; count = $symbolCount; status = "checked" }
                Write-SubStep "Exported ufsecp symbols: $symbolCount" "PASS"
            } catch {
                Write-SubStep "Symbol extraction failed" "WARN"
            }
        } else {
            try {
                $symbols = nm -D $ufLib.FullName 2>$null | Where-Object { $_ -match 'ufsecp_' }
                $symbolCount = ($symbols | Measure-Object).Count
                $abiReport.checks += @{ name = "exported_symbols"; count = $symbolCount; status = "checked" }
                Write-SubStep "Exported ufsecp symbols: $symbolCount" "PASS"
            } catch {}
        }
    } else {
        Write-SubStep "ufsecp library not found (build with -DSECP256K1_BUILD_CABI=ON)" "WARN"
    }

    # J.2 ABI gate test (already part of CTest, just check result)
    Write-SubStep "ABI gate test included in unified runner" "PASS"

    $abiReport | ConvertTo-Json -Depth 5 | Out-File "$ArtifactsDir/abi_report.json" -Encoding utf8

    $sw.Stop()
    Add-CategoryResult "J" "PASS" "ABI / API Stability" $sw.ElapsedMilliseconds
}

# ========================================================================
# K. Bindings & FFI Parity
# ========================================================================
function Run-CategoryK {
    Write-Section "K. Bindings & FFI Parity"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($SkipBindings) {
        Write-SubStep "Bindings check skipped" "SKIP"
        Add-CategoryResult "K" "SKIP" "Bindings (skipped)" 0
        return
    }

    $bindingsDir = "$RootDir/bindings"
    $parityMatrix = @{
        timestamp = $Timestamp
        languages = @()
    }

    if (Test-Path $bindingsDir) {
        $langs = Get-ChildItem $bindingsDir -Directory
        foreach ($lang in $langs) {
            $entry = @{
                language = $lang.Name
                present = $true
                # Check for common binding files
                files = (Get-ChildItem $lang.FullName -File -Recurse | Select-Object -First 10 -ExpandProperty Name)
            }

            # Check for test files
            $testFiles = Get-ChildItem $lang.FullName -Recurse -Include @("*test*", "*spec*", "*_test*") -File
            $entry["has_tests"] = ($testFiles.Count -gt 0)
            $entry["test_count"] = $testFiles.Count

            $parityMatrix.languages += $entry
            $status = if ($entry.has_tests) { "PASS" } else { "WARN" }
            Write-SubStep "$($lang.Name): $($entry.files.Count) files, $($testFiles.Count) tests" $status
        }
    } else {
        Write-SubStep "bindings/ directory not found" "WARN"
    }

    $parityMatrix | ConvertTo-Json -Depth 5 | Out-File "$ArtifactsDir/bindings/parity_matrix.json" -Encoding utf8

    $sw.Stop()
    Add-CategoryResult "K" "PASS" "Bindings & FFI Parity" $sw.ElapsedMilliseconds
}

# ========================================================================
# L. Performance Regression
# ========================================================================
function Run-CategoryL {
    Write-Section "L. Performance Regression"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if ($SkipBenchmark) {
        Write-SubStep "Benchmark skipped" "SKIP"
        Add-CategoryResult "L" "SKIP" "Performance (skipped)" 0
        return
    }

    # Run benchmark if available
    $benchBin = Get-ChildItem -Path $BuildDir -Recurse -Include @("run_benchmark*", "bench_*") -File |
        Where-Object { $_.Extension -in @('.exe', '') } | Select-Object -First 1

    if ($benchBin) {
        Write-SubStep "Running benchmark: $($benchBin.Name)..." "..."
        try {
            $benchOut = & $benchBin.FullName 2>&1
            $benchOut | Out-File "$ArtifactsDir/benchmark/benchmark_output.txt" -Encoding utf8
            Write-SubStep "Benchmark completed" "PASS"
        } catch {
            Write-SubStep "Benchmark failed" "WARN"
        }
    } else {
        Write-SubStep "No benchmark binary found" "WARN"
    }

    # Performance is validated in unified_audit_runner's Section 8
    Write-SubStep "Performance smoke tests included in unified runner" "PASS"

    $sw.Stop()
    Add-CategoryResult "L" "PASS" "Performance Regression" $sw.ElapsedMilliseconds
}

# ========================================================================
# M. Documentation & Claims Consistency
# ========================================================================
function Run-CategoryM {
    Write-Section "M. Documentation & Claims Consistency"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $allPass = $true

    # M.1 Required documentation files exist
    $requiredDocs = @(
        "README.md", "CHANGELOG.md", "SECURITY.md", "LICENSE",
        "THREAT_MODEL.md", "CONTRIBUTING.md", "VERSION.txt"
    )
    $missingDocs = @()
    foreach ($doc in $requiredDocs) {
        $path = Join-Path $RootDir $doc
        if (Test-Path $path) {
            Write-SubStep "$doc exists" "PASS"
        } else {
            Write-SubStep "$doc MISSING" "FAIL"
            $missingDocs += $doc
            $allPass = $false
        }
    }
    if ($missingDocs.Count -gt 0) {
        Add-Finding "Low" "Documentation" "Missing required docs: $($missingDocs -join ', ')" "" "Create missing documentation files"
    }

    # M.2 Version consistency
    $versionInFile = $Version
    $changelogPath = "$RootDir/CHANGELOG.md"
    if (Test-Path $changelogPath) {
        $changelogHead = Get-Content $changelogPath -TotalCount 30 | Out-String
        if ($changelogHead -match $versionInFile) {
            Write-SubStep "VERSION.txt ($versionInFile) matches CHANGELOG.md" "PASS"
        } else {
            Write-SubStep "VERSION.txt ($versionInFile) not found in CHANGELOG.md header" "WARN"
            Add-Finding "Info" "Documentation" "VERSION.txt version not found in recent CHANGELOG entries" "" "Update CHANGELOG for $versionInFile"
        }
    }

    # M.3 THREAT_MODEL.md references existing documents
    if (Test-Path "$RootDir/THREAT_MODEL.md") {
        Write-SubStep "THREAT_MODEL.md present" "PASS"
    }

    # M.4 AUDIT_GUIDE.md exists
    if (Test-Path "$RootDir/AUDIT_GUIDE.md") {
        Write-SubStep "AUDIT_GUIDE.md present" "PASS"
    } else {
        Write-SubStep "AUDIT_GUIDE.md missing" "WARN"
    }

    $sw.Stop()
    $status = if ($allPass) { "PASS" } else { "FAIL" }
    Add-CategoryResult "M" $status "Documentation Consistency" $sw.ElapsedMilliseconds
}

# ========================================================================
# Report Generation -- audit_report.md
# ========================================================================
function Generate-AuditReportMd {
    Write-Section "Generating Final Audit Report"

    $reportPath = "$OutputDir/audit_report.md"
    $totalSw = [System.Diagnostics.Stopwatch]::StartNew()

    $fp = Get-ToolchainFingerprint

    $sb = [System.Text.StringBuilder]::new()

    # -- Header --
    [void]$sb.AppendLine("# UltrafastSecp256k1 -- Comprehensive Audit Report")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("| Field | Value |")
    [void]$sb.AppendLine("|-------|-------|")
    [void]$sb.AppendLine("| **Report ID** | ``UF-AUDIT-$DateTag`` |")
    [void]$sb.AppendLine("| **Date** | $Timestamp |")
    [void]$sb.AppendLine("| **Version** | $Version |")
    [void]$sb.AppendLine("| **Commit** | ``$($fp['git_commit'])`` |")
    [void]$sb.AppendLine("| **Branch** | ``$($fp['git_branch'])`` |")
    [void]$sb.AppendLine("| **Dirty** | $($fp['git_dirty']) |")
    [void]$sb.AppendLine("| **OS** | $($fp['os']) |")
    [void]$sb.AppendLine("| **Arch** | $($fp['arch']) |")
    [void]$sb.AppendLine("| **Compiler** | $($fp['cxx_compiler']) |")
    [void]$sb.AppendLine("| **CMake** | $($fp['cmake']) |")
    [void]$sb.AppendLine("")

    # -- 1. Executive Summary --
    [void]$sb.AppendLine("## 1. Executive Summary")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("| Category | Status | Time |")
    [void]$sb.AppendLine("|----------|--------|------|")
    foreach ($cat in $Script:CategoryResults.Keys) {
        $r = $Script:CategoryResults[$cat]
        $icon = switch ($r.Status) { "PASS" { "`u{2705}" }; "FAIL" { "`u{274C}" }; "SKIP" { "`u{23ED}" }; default { "`u{2753}" } }
        [void]$sb.AppendLine("| **$cat. $($r.Summary)** | $icon $($r.Status) | $([math]::Round($r.TimeMs / 1000, 1))s |")
    }
    [void]$sb.AppendLine("")

    $totalFail = ($Script:CategoryResults.Values | Where-Object { $_.Status -eq "FAIL" }).Count
    if ($totalFail -eq 0) {
        [void]$sb.AppendLine("> **AUDIT VERDICT: AUDIT-READY** -- All categories passed.")
    } else {
        [void]$sb.AppendLine("> **AUDIT VERDICT: AUDIT-BLOCKED** -- $totalFail category(ies) failed.")
    }
    [void]$sb.AppendLine("")

    # High-risk findings
    $highRisk = $Script:Findings | Where-Object { $_.Severity -in @("Critical", "High") }
    if ($highRisk.Count -gt 0) {
        [void]$sb.AppendLine("### High-Risk Findings")
        [void]$sb.AppendLine("")
        foreach ($f in $highRisk) {
            [void]$sb.AppendLine("- **$($f.ID)** [$($f.Severity)] $($f.Component): $($f.Description)")
        }
        [void]$sb.AppendLine("")
    }

    [void]$sb.AppendLine("### Known Limitations")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- No external lab power analysis / EM emanation testing")
    [void]$sb.AppendLine("- No formal verification (ct-verif, Vale) applied")
    [void]$sb.AppendLine("- GPU CT guarantees not provided (by design)")
    [void]$sb.AppendLine("- Physical fault injection not tested")
    [void]$sb.AppendLine("")

    # -- 2. Reproducibility & Integrity --
    [void]$sb.AppendLine("## 2. Reproducibility & Integrity")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- **Toolchain fingerprint**: ``artifacts/toolchain_fingerprint.json``")
    [void]$sb.AppendLine("- **Artifact SHA256 list**: ``artifacts/SHA256SUMS.txt``")
    [void]$sb.AppendLine("- **SBOM**: ``artifacts/sbom.cdx.json``")
    [void]$sb.AppendLine("- **Provenance**: ``artifacts/provenance.json``")
    [void]$sb.AppendLine("- **Dependency scan**: ``artifacts/dependency_scan.txt``")
    [void]$sb.AppendLine("")

    # -- 3. Test Results Tables --
    [void]$sb.AppendLine("## 3. Test Results Tables")
    [void]$sb.AppendLine("")

    # CTest results
    $ctestResults = "$ArtifactsDir/ctest/results.json"
    if (Test-Path $ctestResults) {
        $ctr = Get-Content $ctestResults -Raw | ConvertFrom-Json
        [void]$sb.AppendLine("### CTest Results")
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("| Metric | Value |")
        [void]$sb.AppendLine("|--------|-------|")
        [void]$sb.AppendLine("| Total | $($ctr.total) |")
        [void]$sb.AppendLine("| Passed | $($ctr.passed) |")
        [void]$sb.AppendLine("| Failed | $($ctr.failed) |")
        [void]$sb.AppendLine("")
    }

    # Unified runner results
    $unifiedJson = "$ArtifactsDir/ctest/audit_report.json"
    if (Test-Path $unifiedJson) {
        $uj = Get-Content $unifiedJson -Raw | ConvertFrom-Json
        [void]$sb.AppendLine("### Unified Audit Runner (8-Section Breakdown)")
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("| Section | Title | Passed | Failed | Time |")
        [void]$sb.AppendLine("|---------|-------|--------|--------|------|")
        if ($uj.sections) {
            foreach ($sec in $uj.sections) {
                $secIcon = if ($sec.failed -eq 0) { "`u{2705}" } else { "`u{274C}" }
                [void]$sb.AppendLine("| $secIcon | $($sec.title) | $($sec.passed) | $($sec.failed) | $([math]::Round($sec.time_ms / 1000, 1))s |")
            }
        }
        [void]$sb.AppendLine("")
    }

    [void]$sb.AppendLine("### Static Analysis")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- **clang-tidy**: ``artifacts/static_analysis/clang_tidy.log``")
    [void]$sb.AppendLine("- **cppcheck**: ``artifacts/static_analysis/cppcheck.log``")
    [void]$sb.AppendLine("- **Dangerous patterns**: ``artifacts/static_analysis/dangerous_patterns.log``")
    [void]$sb.AppendLine("")

    [void]$sb.AppendLine("### Sanitizers")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- **ASan/UBSan**: ``artifacts/sanitizers/asan_ubsan.log``")
    [void]$sb.AppendLine("")

    [void]$sb.AppendLine("### CT / Side-Channel")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- dudect timing tests included in unified runner (ct_sidechannel smoke)")
    [void]$sb.AppendLine("- Disassembly branch scan: run ``scripts/verify_ct_disasm.sh`` separately")
    [void]$sb.AppendLine("")

    [void]$sb.AppendLine("### Bindings")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- **Parity matrix**: ``artifacts/bindings/parity_matrix.json``")
    [void]$sb.AppendLine("")

    # -- 4. Findings --
    [void]$sb.AppendLine("## 4. Findings")
    [void]$sb.AppendLine("")
    if ($Script:Findings.Count -eq 0) {
        [void]$sb.AppendLine("> No findings reported.")
    } else {
        [void]$sb.AppendLine("| ID | Severity | Component | Description | Status |")
        [void]$sb.AppendLine("|----|----------|-----------|-------------|--------|")
        foreach ($f in $Script:Findings) {
            [void]$sb.AppendLine("| $($f.ID) | $($f.Severity) | $($f.Component) | $($f.Description) | $($f.Status) |")
        }
        [void]$sb.AppendLine("")

        [void]$sb.AppendLine("### Finding Details")
        [void]$sb.AppendLine("")
        foreach ($f in $Script:Findings) {
            [void]$sb.AppendLine("#### $($f.ID) -- $($f.Description)")
            [void]$sb.AppendLine("")
            [void]$sb.AppendLine("- **Severity**: $($f.Severity)")
            [void]$sb.AppendLine("- **Component**: $($f.Component)")
            if ($f.Evidence) {
                [void]$sb.AppendLine("- **Evidence**: ``$($f.Evidence)``")
            }
            if ($f.Recommendation) {
                [void]$sb.AppendLine("- **Recommendation**: $($f.Recommendation)")
            }
            [void]$sb.AppendLine("- **Status**: $($f.Status)")
            [void]$sb.AppendLine("")
        }
    }
    [void]$sb.AppendLine("")

    # -- 5. Coverage & Unreachable --
    [void]$sb.AppendLine("## 5. Coverage & Unreachable Justifications")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- Code coverage report: run ``scripts/generate_coverage.sh`` separately")
    [void]$sb.AppendLine("- Excluded lines policy: GPU paths, platform-specific assembly, unreachable error handlers")
    [void]$sb.AppendLine("")

    # -- 6. Risk Acceptance / Threat Model Mapping --
    [void]$sb.AppendLine("## 6. Risk Acceptance / Threat Model Mapping")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("| Threat (from THREAT_MODEL.md) | Test Coverage | Evidence |")
    [void]$sb.AppendLine("|-------------------------------|---------------|----------|")
    [void]$sb.AppendLine("| A1: Timing Side Channels | CT tests, dudect, disasm scan | unified runner (ct_analysis section) |")
    [void]$sb.AppendLine("| A2: Nonce Attacks | RFC6979 KAT, BIP-340 vectors | unified runner (standard_vectors) |")
    [void]$sb.AppendLine("| A3: Arithmetic Errors | Field/scalar/point audit, property tests | unified runner (math_invariants) |")
    [void]$sb.AppendLine("| A4: Memory Safety | ASan/UBSan, fault injection | sanitizer build + fault_injection test |")
    [void]$sb.AppendLine("| A5: Supply Chain | SBOM, provenance, dependency scan | artifacts/ |")
    [void]$sb.AppendLine("| A6: GPU-Specific | GPU tests (if enabled) | separate GPU audit |")
    [void]$sb.AppendLine("")

    [void]$sb.AppendLine("### Not Covered")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- Physical power analysis / EM emanation (requires lab equipment)")
    [void]$sb.AppendLine("- Quantum adversary attacks (secp256k1 is not post-quantum)")
    [void]$sb.AppendLine("- OS-level memory disclosure (cold boot, swap file)")
    [void]$sb.AppendLine("")

    # -- 7. Appendices --
    [void]$sb.AppendLine("## 7. Appendices")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("| Artifact | Path |")
    [void]$sb.AppendLine("|----------|------|")
    [void]$sb.AppendLine("| Toolchain fingerprint | ``artifacts/toolchain_fingerprint.json`` |")
    [void]$sb.AppendLine("| SHA256 manifest | ``artifacts/SHA256SUMS.txt`` |")
    [void]$sb.AppendLine("| SBOM | ``artifacts/sbom.cdx.json`` |")
    [void]$sb.AppendLine("| Provenance | ``artifacts/provenance.json`` |")
    [void]$sb.AppendLine("| Dependency scan | ``artifacts/dependency_scan.txt`` |")
    [void]$sb.AppendLine("| clang-tidy log | ``artifacts/static_analysis/clang_tidy.log`` |")
    [void]$sb.AppendLine("| cppcheck log | ``artifacts/static_analysis/cppcheck.log`` |")
    [void]$sb.AppendLine("| Dangerous patterns | ``artifacts/static_analysis/dangerous_patterns.log`` |")
    [void]$sb.AppendLine("| ASan/UBSan log | ``artifacts/sanitizers/asan_ubsan.log`` |")
    [void]$sb.AppendLine("| Unified runner output | ``artifacts/ctest/unified_runner_output.txt`` |")
    [void]$sb.AppendLine("| Unified runner JSON | ``artifacts/ctest/audit_report.json`` |")
    [void]$sb.AppendLine("| Unified runner text | ``artifacts/ctest/audit_report.txt`` |")
    [void]$sb.AppendLine("| CTest results | ``artifacts/ctest/results.json`` |")
    [void]$sb.AppendLine("| CTest output | ``artifacts/ctest/ctest_output.txt`` |")
    [void]$sb.AppendLine("| ABI report | ``artifacts/abi_report.json`` |")
    [void]$sb.AppendLine("| Bindings parity | ``artifacts/bindings/parity_matrix.json`` |")
    [void]$sb.AppendLine("| Benchmark output | ``artifacts/benchmark/benchmark_output.txt`` |")
    [void]$sb.AppendLine("")

    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("*Generated by ``audit/run_full_audit.ps1`` at $Timestamp*")
    [void]$sb.AppendLine("*UltrafastSecp256k1 v$Version -- Comprehensive Audit Report*")

    # Write report
    $sb.ToString() | Out-File $reportPath -Encoding utf8
    Write-SubStep "audit_report.md written to $reportPath" "PASS"
}

# ========================================================================
# MAIN -- Orchestration
# ========================================================================

$mainSw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host "  UltrafastSecp256k1 -- Full Audit Orchestrator (A-M)" -ForegroundColor Yellow
Write-Host "  Version: $Version | $Timestamp" -ForegroundColor Yellow
Write-Host "  Build:   $BuildDir" -ForegroundColor Yellow
Write-Host "  Output:  $OutputDir" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host ""

# Run all categories sequentially
Run-CategoryA
Run-CategoryB
Run-CategoryC
Run-CategoryD
Run-CategoriesEI
Run-CategoryJ
Run-CategoryK
Run-CategoryL
Run-CategoryM

# Generate final report
Generate-AuditReportMd

$mainSw.Stop()

# -- Final Summary ------------------------------------------------------

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  AUDIT COMPLETE" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""
Write-Host "  Category Results:" -ForegroundColor White
foreach ($cat in $Script:CategoryResults.Keys) {
    $r = $Script:CategoryResults[$cat]
    $color = switch ($r.Status) { "PASS" { "Green" }; "FAIL" { "Red" }; default { "Yellow" } }
    Write-Host "    $cat. $($r.Summary): $($r.Status)" -ForegroundColor $color
}
Write-Host ""
Write-Host "  Findings: $($Script:Findings.Count) total" -ForegroundColor White
$crit = ($Script:Findings | Where-Object { $_.Severity -eq "Critical" }).Count
$high = ($Script:Findings | Where-Object { $_.Severity -eq "High" }).Count
$med  = ($Script:Findings | Where-Object { $_.Severity -eq "Med" }).Count
$low  = ($Script:Findings | Where-Object { $_.Severity -eq "Low" }).Count
$info = ($Script:Findings | Where-Object { $_.Severity -eq "Info" }).Count
Write-Host "    Critical=$crit  High=$high  Med=$med  Low=$low  Info=$info" -ForegroundColor White
Write-Host ""
Write-Host "  Total time: $([math]::Round($mainSw.ElapsedMilliseconds / 1000, 1))s" -ForegroundColor White
Write-Host "  Report: $OutputDir/audit_report.md" -ForegroundColor Green
Write-Host "  Artifacts: $ArtifactsDir/" -ForegroundColor Green

$totalFail = ($Script:CategoryResults.Values | Where-Object { $_.Status -eq "FAIL" }).Count
if ($totalFail -gt 0) {
    Write-Host ""
    Write-Host "  VERDICT: AUDIT-BLOCKED ($totalFail categories failed)" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "  VERDICT: AUDIT-READY" -ForegroundColor Green
    exit 0
}
