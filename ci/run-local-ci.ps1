<#
.SYNOPSIS
    Run local CI via Docker (Windows PowerShell wrapper)

.DESCRIPTION
    Builds the Docker image and runs the full CI suite locally.
    Mirrors GitHub Actions: security-audit.yml + ci.yml + audit-report.yml + clang-tidy.yml + cppcheck.yml

.PARAMETER Job
    Run specific job(s): werror, ci, asan, tsan, valgrind, dudect, audit,
                         clang-tidy, cppcheck, coverage, valgrind-ct
    Comma-separated list accepted: -Job asan,tsan

.PARAMETER Quick
    Fast pre-commit gate (~5 min): werror + ci (Release+Debug)

.PARAMETER Full
    Release-quality check (~45-60 min): all jobs including valgrind + dudect

.PARAMETER NoBuild
    Skip rebuilding the Docker image (use existing uf-local-ci image)

.PARAMETER List
    List all available jobs and presets, then exit

.EXAMPLE
    .\scripts\run-local-ci.ps1                          # Standard: 7 jobs (~20-25 min)
    .\scripts\run-local-ci.ps1 -Quick                   # Fast gate: werror + ci (~5 min)
    .\scripts\run-local-ci.ps1 -Full                    # All jobs (~45-60 min)
    .\scripts\run-local-ci.ps1 -Job audit               # Only unified_audit_runner
    .\scripts\run-local-ci.ps1 -Job asan,tsan           # Only ASan + TSan
    .\scripts\run-local-ci.ps1 -Job asan -NoBuild       # Skip image rebuild
    .\scripts\run-local-ci.ps1 -List                    # Show available jobs
#>

[CmdletBinding()]
param(
    [string[]]$Job,
    [switch]$Quick,
    [switch]$Full,
    [switch]$NoBuild,
    [switch]$List
)

$ErrorActionPreference = 'Stop'

$ImageName = 'uf-local-ci'
$CcacheVolume = 'uf-ccache'  # persistent ccache across runs
$RepoRoot = Split-Path -Parent $PSScriptRoot  # one level up from scripts/

Push-Location $RepoRoot
try {
    # -- Verify Docker is available ----------------------------------------
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker not found. Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
        return
    }

    # -- --List: delegate to local-ci.sh --list ----------------------------
    if ($List) {
        $env:DOCKER_BUILDKIT = '1'
        docker run --rm -v "${RepoRoot}:/src" $ImageName `
            bash /src/ci/ci_local.sh --list
        return
    }

    # -- Ensure BuildKit for layer caching ---------------------------------
    $env:DOCKER_BUILDKIT = '1'

    # -- Build image -------------------------------------------------------
    if (-not $NoBuild) {
        Write-Host "`n=== Building Docker image: $ImageName ===" -ForegroundColor Cyan
        docker build -f Dockerfile.local-ci -t $ImageName .
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker build failed"
            return
        }
    }

    # -- Compose local-ci.sh arguments ------------------------------------
    $ciArgs = @('bash', '/src/ci/ci_local.sh')

    if ($Quick) {
        $ciArgs += '--quick'
    }
    elseif ($Full) {
        $ciArgs += '--full'
    }
    elseif ($Job -and $Job.Count -gt 0) {
        # Support comma-separated: -Job asan,tsan  OR  -Job asan -Job tsan
        foreach ($j in $Job) {
            foreach ($single in ($j -split ',')) {
                $ciArgs += '--job'
                $ciArgs += $single.Trim()
            }
        }
    }
    # else: no flag → local-ci.sh defaults to --all

    # -- Run container (ccache volume for fast incremental builds) --------
    Write-Host "`n=== Running local CI (ccache: $CcacheVolume) ===" -ForegroundColor Cyan
    $runArgs = @(
        'run', '--rm',
        '-v', "${RepoRoot}:/src",
        '-v', "${CcacheVolume}:/ccache",
        $ImageName
    ) + $ciArgs

    & docker @runArgs
    $exitCode = $LASTEXITCODE

    # -- Report ------------------------------------------------------------
    if ($exitCode -eq 0) {
        Write-Host "`n✓ All local CI jobs passed!" -ForegroundColor Green
    }
    else {
        Write-Host "`n✗ Some local CI jobs failed (exit: $exitCode)" -ForegroundColor Red
    }

    # Surface artifacts written back to /src/local-ci-output/
    $outDir = Join-Path $RepoRoot 'local-ci-output'
    if (Test-Path $outDir) {
        Write-Host "`nArtifacts in: $outDir" -ForegroundColor Yellow

        $covHtml = Join-Path $outDir 'coverage-html\index.html'
        if (Test-Path $covHtml) {
            Write-Host "  Coverage HTML: $covHtml" -ForegroundColor Yellow
            Write-Host "  Open:  start `"$covHtml`"" -ForegroundColor DarkYellow
        }
        $auditTxt = Join-Path $outDir 'audit\audit_report.txt'
        if (Test-Path $auditTxt) {
            Write-Host "  Audit report:  $auditTxt" -ForegroundColor Yellow
        }
    }

    exit $exitCode
}
finally {
    Pop-Location
}
