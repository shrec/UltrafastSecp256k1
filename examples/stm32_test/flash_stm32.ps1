# STM32F103ZET6 Flash + Monitor Script
# Flashes .bin via UART bootloader (stm32flash) and opens serial monitor

param(
    [string]$Port = "COM4",
    [int]$Baud = 115200
)

$ErrorActionPreference = "Stop"

$BuildDir = Join-Path $PSScriptRoot "build"
$BinFile = Get-ChildItem "$BuildDir\*.bin" -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $BinFile) {
    Write-Host "ERROR: No .bin file found in $BuildDir. Run build_stm32.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Flash STM32 via UART Bootloader" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Binary: $($BinFile.Name) ($([math]::Round($BinFile.Length/1024, 1)) KB)"
Write-Host "Port:   $Port"
Write-Host ""
Write-Host "Steps:" -ForegroundColor Yellow
Write-Host "  1. Set BOOT0 jumper to HIGH (3.3V)" -ForegroundColor Yellow
Write-Host "  2. Press RESET button on board" -ForegroundColor Yellow
Write-Host "  3. Press Enter here to flash" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter when ready"

# Try stm32flash
$stm32flash = Get-Command stm32flash.exe -ErrorAction SilentlyContinue
if (-not $stm32flash) {
    # Check common locations
    $candidates = @(
        "$env:USERPROFILE\scoop\apps\stm32flash\current\stm32flash.exe",
        "C:\Program Files\stm32flash\stm32flash.exe",
        "C:\Program Files (x86)\stm32flash\stm32flash.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $stm32flash = @{Source=$c}; break }
    }
}

if ($stm32flash) {
    $exe = if ($stm32flash.Source) { $stm32flash.Source } else { "stm32flash" }
    Write-Host "Flashing with stm32flash..." -ForegroundColor Green
    & $exe -w $BinFile.FullName -v -g 0x08000000 $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Flash failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "stm32flash not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install stm32flash:" -ForegroundColor Yellow
    Write-Host '  Download from: https://sourceforge.net/projects/stm32flash/' -ForegroundColor Yellow
    Write-Host '  Or use Python: pip install stm32loader' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Manual command:" -ForegroundColor Cyan
    Write-Host "  stm32flash -w $($BinFile.FullName) -v -g 0x08000000 $Port"
    exit 1
}

Write-Host ""
Write-Host "Flash complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Post-flash:" -ForegroundColor Yellow
Write-Host "  1. Set BOOT0 jumper back to LOW (GND)" -ForegroundColor Yellow
Write-Host "  2. Press RESET" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening serial monitor on $Port @ $Baud baud..." -ForegroundColor Cyan
Write-Host "(Press Ctrl+C to exit)" -ForegroundColor Gray
Write-Host ""

# Simple serial monitor using .NET
try {
    $serial = New-Object System.IO.Ports.SerialPort $Port, $Baud, "None", 8, "One"
    $serial.ReadTimeout = 1000
    $serial.Open()
    
    while ($true) {
        try {
            $line = $serial.ReadLine()
            Write-Host $line
        } catch [System.TimeoutException] {
            # Normal timeout, continue
        }
    }
} catch {
    Write-Host "Serial error: $_" -ForegroundColor Red
} finally {
    if ($serial -and $serial.IsOpen) { $serial.Close() }
}
