# ============================================================================
# UltrafastSecp256k1 -- Android Build Script (PowerShell)
# ============================================================================
# Windows variant for building Android native libraries.
#
# Prerequisites:
#   - Android NDK (r25+ recommended)
#   - Set $env:ANDROID_NDK_HOME or install via Android Studio SDK Manager
#   - CMake 3.18+ and Ninja
#
# Usage:
#   .\build_android.ps1                      # Build all ABIs
#   .\build_android.ps1 -ABIs arm64-v8a      # ARM64 only
#   .\build_android.ps1 -MinSdk 21           # Override min SDK
# ============================================================================

param(
    [string[]]$ABIs = @("arm64-v8a", "armeabi-v7a", "x86_64", "x86"),
    [int]$MinSdk = 24,
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

# Resolve NDK path
$NdkPath = $env:ANDROID_NDK_HOME
if (-not $NdkPath -or -not (Test-Path $NdkPath)) {
    $NdkPath = $env:ANDROID_NDK
}
if (-not $NdkPath -or -not (Test-Path $NdkPath)) {
    # Search common locations
    $SearchPaths = @(
        "$env:LOCALAPPDATA\Android\Sdk\ndk",
        "$env:USERPROFILE\AppData\Local\Android\Sdk\ndk",
        "$env:ANDROID_HOME\ndk"
    )
    foreach ($search in $SearchPaths) {
        if (Test-Path $search) {
            $latest = Get-ChildItem $search -Directory | Sort-Object Name -Descending | Select-Object -First 1
            if ($latest -and (Test-Path "$($latest.FullName)\build\cmake\android.toolchain.cmake")) {
                $NdkPath = $latest.FullName
                break
            }
        }
    }
}

$Toolchain = Join-Path $NdkPath "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $Toolchain)) {
    Write-Error @"
Android NDK not found.
Set `$env:ANDROID_NDK_HOME to your NDK installation directory.
  `$env:ANDROID_NDK_HOME = 'C:\Users\$env:USERNAME\AppData\Local\Android\Sdk\ndk\26.1.10909125'
"@
    exit 1
}

Write-Host "Using NDK: $NdkPath" -ForegroundColor Green

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$NumCores = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors

foreach ($ABI in $ABIs) {
    $BuildDir = Join-Path $ScriptDir "build-android-$ABI"

    Write-Host "`n======================================" -ForegroundColor Cyan
    Write-Host "Building: $ABI (API $MinSdk, $BuildType)" -ForegroundColor Cyan
    Write-Host "  Output: $BuildDir" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan

    cmake -S $ScriptDir -B $BuildDir `
        -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
        -DANDROID_ABI="$ABI" `
        -DANDROID_PLATFORM="android-$MinSdk" `
        -DANDROID_STL=c++_static `
        -DCMAKE_BUILD_TYPE="$BuildType" `
        -G Ninja

    cmake --build $BuildDir -j $NumCores

    Write-Host "`nLibraries for ${ABI}:" -ForegroundColor Green
    Get-ChildItem $BuildDir -Recurse -Include "*.so","*.a" | Select-Object -First 10 -ExpandProperty FullName
}

# Collect output
$OutputDir = Join-Path $ScriptDir "output\jniLibs"
Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Collecting libraries to: $OutputDir" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

foreach ($ABI in $ABIs) {
    $BuildDir = Join-Path $ScriptDir "build-android-$ABI"
    $AbiOut = Join-Path $OutputDir $ABI
    New-Item -ItemType Directory -Force -Path $AbiOut | Out-Null

    $JniSo = Get-ChildItem $BuildDir -Recurse -Filter "libsecp256k1_jni.so" | Select-Object -First 1
    if ($JniSo) {
        Copy-Item $JniSo.FullName -Destination $AbiOut -Force
        $size = [math]::Round($JniSo.Length / 1KB, 1)
        Write-Host "  ${ABI}: ${size} KB"
    } else {
        Write-Warning "  ${ABI}: libsecp256k1_jni.so not found"
    }
}

Write-Host "`nDone! Copy output\jniLibs\ into your Android project's app\src\main\ directory." -ForegroundColor Green
Write-Host "Or use Gradle CMake integration (see ANDROID_BUILD.md).`n" -ForegroundColor Green
