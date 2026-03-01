<#
.SYNOPSIS
    Local CI runner for UltrafastSecp256k1 (Windows Docker wrapper)
.DESCRIPTION
    Builds the Docker CI image and runs CI jobs locally.
    Mirrors ALL GitHub Actions jobs -- no more waiting 30-40 min.
.EXAMPLE
    .\docker\local_ci.ps1 -Build              # Build Docker image (first time)
    .\docker\local_ci.ps1 -Job wasm            # Run WASM tests only
    .\docker\local_ci.ps1 -Job quick           # GCC Release + WASM (~2 min)
    .\docker\local_ci.ps1 -Job all             # Everything (~5-8 min)
    .\docker\local_ci.ps1 -Job sanitizers      # ASan + UBSan
    .\docker\local_ci.ps1 -Job linux-gcc,wasm  # Multiple jobs
#>
param(
    [switch]$Build,
    [string]$Job = "",
    [switch]$Rebuild,
    [switch]$Shell,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$IMAGE_NAME = "ufsecp-ci"
$DOCKER_FILE = "docker/Dockerfile.ci"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR

# Ensure we run from project root
Push-Location $PROJECT_DIR
try {

# ---- Help --------------------------------------------------------------------
if ($Help -or (-not $Build -and -not $Rebuild -and -not $Shell -and $Job -eq "")) {
    Write-Host @"

UltrafastSecp256k1 Local CI Runner
==================================

Commands:
  -Build              Build the Docker image (first time, ~3-5 min)
  -Rebuild            Force rebuild Docker image (no cache)
  -Job <target>       Run CI job(s). Comma-separated for multiple.
  -Shell              Open interactive shell in the container
  -Help               This message

Available jobs:
  all           Run ALL CI jobs (~5-8 min)
  quick         GCC Release + WASM KAT (~2 min)
  linux-gcc     GCC 13 Release build + tests
  linux-clang   Clang 17 Release build + tests
  linux-debug   GCC 13 Debug build + tests
  sanitizers    ASan + UBSan (Clang Debug)
  tsan          ThreadSanitizer (Clang Debug)
  valgrind      Valgrind memcheck
  wasm          WASM (Emscripten 3.1.51) + KAT
  arm64         ARM64 cross-compile check
  clang-tidy    Static analysis
  coverage      Code coverage (LLVM)
  warnings      -Werror strict warnings

Examples:
  .\docker\local_ci.ps1 -Build
  .\docker\local_ci.ps1 -Job wasm
  .\docker\local_ci.ps1 -Job quick
  .\docker\local_ci.ps1 -Job all
  .\docker\local_ci.ps1 -Job sanitizers,wasm
  .\docker\local_ci.ps1 -Shell

"@
    return
}

# ---- Check Docker ------------------------------------------------------------
function Test-Docker {
    try {
        $null = docker info 2>&1
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Docker)) {
    Write-Host "[!] Docker is not running. Start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# ---- Build Docker image ------------------------------------------------------
function Build-CIImage([bool]$NoCache = $false) {
    Write-Host "Building Docker CI image '$IMAGE_NAME' ..." -ForegroundColor Cyan
    $args_ = @("build", "-f", $DOCKER_FILE, "-t", $IMAGE_NAME)
    if ($NoCache) { $args_ += "--no-cache" }
    $args_ += "docker/"
    & docker @args_
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Docker build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Docker image '$IMAGE_NAME' built successfully" -ForegroundColor Green
}

if ($Build) {
    Build-CIImage $false
    if ($Job -eq "") { return }
}

if ($Rebuild) {
    Build-CIImage $true
    if ($Job -eq "") { return }
}

# ---- Check image exists ------------------------------------------------------
function Test-ImageExists {
    $images = docker images -q $IMAGE_NAME 2>&1
    return ($images -and $images.Trim() -ne "")
}

if (-not (Test-ImageExists)) {
    Write-Host "[!] Docker image '$IMAGE_NAME' not found. Building..." -ForegroundColor Yellow
    Build-CIImage $false
}

# ---- Interactive shell -------------------------------------------------------
if ($Shell) {
    Write-Host "Opening shell in $IMAGE_NAME container..." -ForegroundColor Cyan
    $srcPath = (Get-Location).Path -replace '\\', '/'
    docker run --rm -it -v "${srcPath}:/src" -w /src $IMAGE_NAME bash
    return
}

# ---- Run CI job(s) -----------------------------------------------------------
if ($Job -ne "") {
    $srcPath = (Get-Location).Path -replace '\\', '/'
    $jobs = $Job -split ','

    foreach ($j in $jobs) {
        $j = $j.Trim()
        Write-Host ""
        Write-Host "Running job: $j" -ForegroundColor Cyan
        Write-Host ("=" * 60) -ForegroundColor Cyan

        docker run --rm `
            -v "${srcPath}:/src" `
            -w /src `
            $IMAGE_NAME `
            bash ./docker/run_ci.sh $j

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Job '$j' failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit $LASTEXITCODE
        }
        Write-Host "[OK] Job '$j' passed" -ForegroundColor Green
    }
}

} finally {
    Pop-Location
}
