<#
.SYNOPSIS
    Pre-push CI gate -- run locally before pushing to GitHub.

.DESCRIPTION
    Runs the same tests that GitHub Actions CI executes, inside Docker.
    Catches 95%+ of CI failures locally in ~5 minutes instead of waiting
    30+ minutes for GitHub runners.

    Jobs executed:
      1. -Werror strict warnings (GCC-13, Release)
      2. Full test suite (GCC-13 Release)
      3. Full test suite (Clang-17 Release)
      4. ASan + UBSan (Clang-17 Debug)
      5. Unified audit runner (GCC-13 + Clang-17)

.PARAMETER Full
    Run ALL CI jobs (~8-12 min) instead of pre-push subset (~5 min)

.PARAMETER Job
    Run a specific job: linux-gcc, linux-clang, asan, tsan, valgrind,
    wasm, arm64, clang-tidy, coverage, warnings, audit

.PARAMETER Quick
    Quick smoke test: GCC Release + WASM KAT only (~2 min)

.PARAMETER NoBuild
    Skip rebuilding the Docker image (use existing)

.EXAMPLE
    .\scripts\pre-push-ci.ps1                    # Pre-push gate (~5 min)
    .\scripts\pre-push-ci.ps1 -Full              # All CI jobs (~10 min)
    .\scripts\pre-push-ci.ps1 -Quick             # Quick smoke (~2 min)
    .\scripts\pre-push-ci.ps1 -Job audit         # Audit only
    .\scripts\pre-push-ci.ps1 -Job asan,coverage # ASan + coverage
#>

[CmdletBinding()]
param(
    [string[]]$Job,
    [switch]$Full,
    [switch]$Quick,
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot  # one level up from scripts/

Push-Location $RepoRoot
try {
    # -- Verify Docker is available ------------------------------------------
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker not found. Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
        return
    }

    # -- Determine target ----------------------------------------------------
    if ($Quick) {
        $target = 'quick'
    } elseif ($Full) {
        $target = 'all'
    } elseif ($Job -and $Job.Count -gt 0) {
        # Run individual jobs sequentially
        foreach ($j in $Job) {
            Write-Host "`n=== Running job: $j ===" -ForegroundColor Cyan
            docker compose -f docker-compose.ci.yml run --rm $j
            if ($LASTEXITCODE -ne 0) {
                Write-Host "`n[FAIL] Job '$j' failed" -ForegroundColor Red
                exit 1
            }
            Write-Host "[PASS] Job '$j'" -ForegroundColor Green
        }
        Write-Host "`nAll requested jobs passed!" -ForegroundColor Green
        exit 0
    } else {
        $target = 'pre-push'
    }

    # -- Build image (if needed) ---------------------------------------------
    if (-not $NoBuild) {
        Write-Host "`n=== Building CI Docker image ===" -ForegroundColor Cyan
        docker compose -f docker-compose.ci.yml build ci-base
        if ($LASTEXITCODE -ne 0) {
            # Fallback: build via docker build directly
            docker build -f docker/Dockerfile.ci -t ufsecp-ci docker/
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Docker image build failed"
                return
            }
        }
    }

    # -- Run the target ------------------------------------------------------
    $startTime = Get-Date
    Write-Host "`n=== Running local CI: $target ===" -ForegroundColor Cyan

    docker compose -f docker-compose.ci.yml run --rm $target
    $exitCode = $LASTEXITCODE

    $elapsed = (Get-Date) - $startTime
    $mins = [math]::Floor($elapsed.TotalMinutes)
    $secs = $elapsed.Seconds

    # -- Report --------------------------------------------------------------
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "=== ALL PASSED ($mins min $secs sec) ===" -ForegroundColor Green
        Write-Host "Safe to push." -ForegroundColor Green
    } else {
        Write-Host "=== FAILED ($mins min $secs sec) ===" -ForegroundColor Red
        Write-Host "Fix failures before pushing to GitHub." -ForegroundColor Red
    }

    exit $exitCode
}
finally {
    Pop-Location
}
