<#
.SYNOPSIS
    Run local CI via Docker (Windows PowerShell wrapper)

.DESCRIPTION
    Builds the Docker image and runs the full CI suite locally.
    Equivalent to GitHub Actions security-audit.yml + coverage + clang-tidy.

.PARAMETER Job
    Run a specific job: werror, asan, valgrind, dudect, coverage, clang-tidy, ci

.PARAMETER Full
    Run all 7 jobs (security + coverage + clang-tidy + ci matrix)

.PARAMETER NoBuild
    Skip rebuilding the Docker image (use existing)

.EXAMPLE
    .\scripts\run-local-ci.ps1                          # 4 security-audit jobs
    .\scripts\run-local-ci.ps1 -Full                    # all 7 jobs
    .\scripts\run-local-ci.ps1 -Job coverage            # only coverage
    .\scripts\run-local-ci.ps1 -Job asan -NoBuild       # only ASan, skip image rebuild
    .\scripts\run-local-ci.ps1 -Job coverage,clang-tidy # coverage + clang-tidy
#>

[CmdletBinding()]
param(
    [string[]]$Job,
    [switch]$Full,
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

$ImageName = 'uf-local-ci'
$CcacheVolume = 'uf-ccache'  # persistent ccache across runs
$RepoRoot = Split-Path -Parent $PSScriptRoot  # one level up from scripts/

Push-Location $RepoRoot
try {
    # ── Verify Docker is available ──────────────────────────────────────
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker not found. Install Docker Desktop first: https://docs.docker.com/desktop/install/windows-install/"
        return
    }

    # ── Ensure BuildKit for layer caching ────────────────────────────
    $env:DOCKER_BUILDKIT = '1'

    # ── Build image ─────────────────────────────────────────────────────
    if (-not $NoBuild) {
        Write-Host "`n=== Building Docker image: $ImageName (BuildKit) ===" -ForegroundColor Cyan
        docker build -f Dockerfile.local-ci -t $ImageName .
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker build failed"
            return
        }
    }

    # ── Compose run arguments ───────────────────────────────────────────
    $ciArgs = @()
    if ($Full) {
        $ciArgs = @('bash', '/src/scripts/local-ci.sh', '--full')
    }
    elseif ($Job -and $Job.Count -gt 0) {
        $ciArgs = @('bash', '/src/scripts/local-ci.sh')
        foreach ($j in $Job) {
            $ciArgs += '--job'
            $ciArgs += $j
        }
    }
    # else: default CMD from Dockerfile (--all)

    # ── Run container (with ccache volume for fast rebuilds) ──────────
    Write-Host "`n=== Running local CI (ccache volume: $CcacheVolume) ===" -ForegroundColor Cyan
    $runArgs = @(
        'run', '--rm',
        '-v', "${RepoRoot}:/src",
        '-v', "${CcacheVolume}:/ccache",
        $ImageName
    )
    if ($ciArgs.Count -gt 0) {
        $runArgs += $ciArgs
    }

    & docker @runArgs
    $exitCode = $LASTEXITCODE

    # ── Report ──────────────────────────────────────────────────────────
    if ($exitCode -eq 0) {
        Write-Host "`nAll local CI jobs passed!" -ForegroundColor Green
    }
    else {
        Write-Host "`nSome local CI jobs failed (exit code: $exitCode)" -ForegroundColor Red
    }

    # Check if coverage HTML was generated
    $covHtml = Join-Path $RepoRoot 'build-local-ci-coverage/html/index.html'
    if (Test-Path $covHtml) {
        Write-Host "`nCoverage report: $covHtml" -ForegroundColor Yellow
        Write-Host "Open in browser:  start $covHtml" -ForegroundColor Yellow
    }

    exit $exitCode
}
finally {
    Pop-Location
}
