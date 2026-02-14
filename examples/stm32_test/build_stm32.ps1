# STM32F103ZET6 Build Script
# Builds secp256k1 test firmware for ARM Cortex-M3

param(
    [switch]$Clean,
    [switch]$Flash,
    [string]$Port = "COM4"
)

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$BuildDir = Join-Path $ProjectDir "build"
$ArmGcc = "D:\Dev\arm-gnu-toolchain\bin"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  STM32F103ZET6 secp256k1 Build" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Verify toolchain
if (-not (Test-Path "$ArmGcc\arm-none-eabi-gcc.exe")) {
    Write-Host "ERROR: ARM GCC not found at $ArmGcc" -ForegroundColor Red
    exit 1
}
$version = & "$ArmGcc\arm-none-eabi-gcc.exe" --version | Select-Object -First 1
Write-Host "Toolchain: $version" -ForegroundColor Green

# Clean
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item -Recurse -Force $BuildDir
}

# Configure
if (-not (Test-Path "$BuildDir\CMakeCache.txt")) {
    Write-Host "`nConfiguring CMake..." -ForegroundColor Yellow
    cmake -S $ProjectDir -B $BuildDir -G "Ninja" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_TOOLCHAIN_FILE="$ProjectDir\cmake\arm-none-eabi-gcc.cmake"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configure failed!" -ForegroundColor Red
        exit 1
    }
}

# Build
Write-Host "`nBuilding..." -ForegroundColor Yellow
cmake --build $BuildDir -j
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nBuild successful!" -ForegroundColor Green

# Show binary info
$elfFile = Get-ChildItem "$BuildDir\*.elf" | Select-Object -First 1
$binFile = Get-ChildItem "$BuildDir\*.bin" | Select-Object -First 1
if ($binFile) {
    $size = [math]::Round($binFile.Length / 1024, 1)
    Write-Host "  Binary: $($binFile.Name) ($size KB)" -ForegroundColor Green
}

# Flash
if ($Flash) {
    Write-Host "`n============================================" -ForegroundColor Cyan
    Write-Host "  Flashing to STM32 via $Port" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANT: Set BOOT0=HIGH, press RESET, then press Enter" -ForegroundColor Yellow
    Read-Host "Press Enter when ready"
    
    $stm32flash = Get-Command stm32flash -ErrorAction SilentlyContinue
    if ($stm32flash) {
        stm32flash -w "$($binFile.FullName)" -v -g 0x08000000 $Port
    } else {
        Write-Host "stm32flash not found. Install it or use STM32CubeProgrammer." -ForegroundColor Red
        Write-Host "  Manual flash: stm32flash -w $($binFile.Name) -v -g 0x08000000 $Port" -ForegroundColor Yellow
    }
}

Write-Host "`nDone." -ForegroundColor Green
