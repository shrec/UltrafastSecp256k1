# bench/scripts/run.ps1 -- Run bench_compare with sane defaults (Windows)
# Usage: .\bench\scripts\run.ps1 [-PinCore 2] [-ExtraArgs @("--case=ecdsa")]
param(
    [int]$PinCore = 2,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path "$ScriptDir\..\..").Path
$BuildDir  = "$RepoRoot\build\bench-compare"

# Try Release, then Debug
$Bin = "$BuildDir\bench\Release\bench_compare.exe"
if (-not (Test-Path $Bin)) {
    $Bin = "$BuildDir\bench\Debug\bench_compare.exe"
}
if (-not (Test-Path $Bin)) {
    # Single-config generators (Ninja)
    $Bin = "$BuildDir\bench\bench_compare.exe"
}
if (-not (Test-Path $Bin)) {
    Write-Error "[!] bench_compare.exe not found. Run build.ps1 first."
    exit 1
}

# Set high priority for current process
$proc = Get-Process -Id $PID
$proc.PriorityClass = "High"

Write-Host "=== bench_compare run ==="
Write-Host "  Binary     : $Bin"
Write-Host "  Pin core   : $PinCore"
Write-Host "  Extra args : $($ExtraArgs -join ' ')"
Write-Host ""

& $Bin `
    --pin-core=$PinCore `
    --n=100000 `
    --warmup=500 `
    --measure=3000 `
    --json=report.json `
    @ExtraArgs

Write-Host ""
Write-Host "[OK] Report written to: report.json"
