# Release COM3 Port - PowerShell Helper Script
# Use this if you get "Port is busy" errors

Write-Host "`n[TOOL] COM3 Port Release Utility" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Kill all background PowerShell processes (except current)
$currentPID = $PID
$killed = 0

Get-Process powershell -ErrorAction SilentlyContinue | Where-Object {$_.Id -ne $currentPID} | ForEach-Object {
    Write-Host "Stopping PowerShell process PID: $($_.Id)" -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    $killed++
}

if ($killed -gt 0) {
    Write-Host "`n[OK] Stopped $killed background PowerShell process(es)" -ForegroundColor Green
} else {
    Write-Host "`n[OK] No background processes found" -ForegroundColor Green
}

# Try to open and close COM3 to verify it's free
Start-Sleep -Seconds 1
Write-Host "`nVerifying COM3 is free..." -ForegroundColor Cyan

try {
    $port = New-Object System.IO.Ports.SerialPort("COM3", 115200)
    $port.Open()
    Start-Sleep -Milliseconds 200
    $port.Close()
    $port.Dispose()
    Write-Host "COM3 is now FREE and ready to use!" -ForegroundColor Green
    Write-Host "`nYou can now run: Build -> ESP32_Flash`n" -ForegroundColor Yellow
}
catch {
    Write-Host "[!] COM3 may still be in use: $_" -ForegroundColor Red
    Write-Host "`nTry:" -ForegroundColor Yellow
    Write-Host "1. Unplug/replug USB cable" -ForegroundColor Yellow
    Write-Host "2. Close Arduino IDE, PuTTY, or other serial programs" -ForegroundColor Yellow
    Write-Host "3. Press Reset button on ESP32`n" -ForegroundColor Yellow
}

