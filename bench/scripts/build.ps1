# bench/scripts/build.ps1 -- Configure + build bench_compare (Windows)
# Usage: .\bench\scripts\build.ps1 [-BuildType Release]
param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path "$ScriptDir\..\..").Path
$BuildDir  = "$RepoRoot\build\bench-compare"

Write-Host "=== bench_compare build ==="
Write-Host "  Repo root  : $RepoRoot"
Write-Host "  Build dir  : $BuildDir"
Write-Host "  Build type : $BuildType"
Write-Host ""

cmake -S "$RepoRoot" -B "$BuildDir" `
    -DSECP256K1_BUILD_BENCH_COMPARE=ON `
    -DSECP256K1_BUILD_TESTS=OFF `
    -DSECP256K1_BUILD_EXAMPLES=OFF `
    -DSECP256K1_BUILD_BENCH=OFF

cmake --build "$BuildDir" --config $BuildType --target bench_compare -j

Write-Host ""
Write-Host "[OK] bench_compare built in $BuildDir"
Write-Host "     Run: $BuildDir\bench\$BuildType\bench_compare.exe --help"
