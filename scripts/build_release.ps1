# ============================================================================
# UltrafastSecp256k1 -- Release Build Script (Windows)
# ============================================================================
# Builds release binaries + creates distribution archive + NuGet layout.
#
# Usage:
#   .\scripts\build_release.ps1                    # default: Release
#   .\scripts\build_release.ps1 -BuildType Debug
#   .\scripts\build_release.ps1 -SkipTests
#
# Output:
#   release\UltrafastSecp256k1-<version>-win-x64\
#   release\UltrafastSecp256k1-<version>-win-x64.zip
#   release\nuget\  (NuGet runtime layout)
# ============================================================================

param(
    [string]$BuildType = "Release",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = Split-Path -Parent $ScriptDir
$BuildDir  = Join-Path $RootDir "build-release-pkg"
$ReleaseDir = Join-Path $RootDir "release"

# -- Read version from CMakeLists.txt --
$CMakeContent = Get-Content (Join-Path $RootDir "CMakeLists.txt") -Raw
if ($CMakeContent -match 'VERSION\s+(\d+\.\d+\.\d+)') {
    $Version = $Matches[1]
} else {
    Write-Error "Cannot read version from CMakeLists.txt"
    exit 1
}

$Arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
$PkgName = "UltrafastSecp256k1-v${Version}-win-${Arch}"

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  UltrafastSecp256k1 Release Build" -ForegroundColor Cyan
Write-Host "  Version:    $Version"
Write-Host "  Platform:   win-${Arch}"
Write-Host "  Build Type: $BuildType"
Write-Host "  Output:     $ReleaseDir\$PkgName"
Write-Host "===============================================================" -ForegroundColor Cyan

# -- Configure --
Write-Host "`n>>> Configuring..." -ForegroundColor Yellow
cmake -S $RootDir -B $BuildDir `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DSECP256K1_BUILD_TESTS=ON `
    -DSECP256K1_BUILD_BENCH=OFF `
    -DSECP256K1_BUILD_EXAMPLES=OFF
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# -- Build --
Write-Host "`n>>> Building..." -ForegroundColor Yellow
cmake --build $BuildDir -j
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# -- Test --
if (-not $SkipTests) {
    Write-Host "`n>>> Running tests..." -ForegroundColor Yellow
    ctest --test-dir $BuildDir --output-on-failure -j4
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# -- Install to staging --
Write-Host "`n>>> Installing to staging..." -ForegroundColor Yellow
$Staging = Join-Path $BuildDir "staging"
cmake --install $BuildDir --prefix $Staging

# -- Collect release artifacts --
Write-Host "`n>>> Packaging $PkgName..." -ForegroundColor Yellow
$PkgDir = Join-Path $ReleaseDir $PkgName
if (Test-Path $PkgDir) { Remove-Item -Recurse -Force $PkgDir }
New-Item -ItemType Directory -Force -Path "$PkgDir\lib"        | Out-Null
New-Item -ItemType Directory -Force -Path "$PkgDir\include\ufsecp"    | Out-Null
New-Item -ItemType Directory -Force -Path "$PkgDir\include\secp256k1" | Out-Null

# ufsecp headers
Copy-Item "$RootDir\include\ufsecp\ufsecp.h"         "$PkgDir\include\ufsecp\"
Copy-Item "$RootDir\include\ufsecp\ufsecp_version.h"  "$PkgDir\include\ufsecp\"
Copy-Item "$RootDir\include\ufsecp\ufsecp_error.h"    "$PkgDir\include\ufsecp\"

# C++ public headers
$StagingHeaders = Join-Path $Staging "include\secp256k1"
if (Test-Path $StagingHeaders) {
    Copy-Item "$StagingHeaders\*" "$PkgDir\include\secp256k1\" -Recurse
}

# Libraries (from staging + build)
$LibSearchPaths = @(
    (Join-Path $Staging "lib"),
    (Join-Path $Staging "bin"),
    (Join-Path $BuildDir "include\ufsecp")
)
foreach ($SearchPath in $LibSearchPaths) {
    if (Test-Path $SearchPath) {
        Get-ChildItem $SearchPath -File -Recurse |
            Where-Object { $_.Extension -in ".dll", ".lib", ".a", ".so", ".dylib" } |
            ForEach-Object { Copy-Item $_.FullName "$PkgDir\lib\" -ErrorAction SilentlyContinue }
    }
}

# pkg-config + CMake config
if (Test-Path "$Staging\lib\pkgconfig") {
    Copy-Item "$Staging\lib\pkgconfig" "$PkgDir\lib\" -Recurse
}
if (Test-Path "$Staging\lib\cmake") {
    Copy-Item "$Staging\lib\cmake" "$PkgDir\lib\" -Recurse
}

# Docs
@("LICENSE", "README.md", "CHANGELOG.md") | ForEach-Object {
    $src = Join-Path $RootDir $_
    if (Test-Path $src) { Copy-Item $src "$PkgDir\" }
}
$guarantees = Join-Path $RootDir "include\ufsecp\SUPPORTED_GUARANTEES.md"
if (Test-Path $guarantees) { Copy-Item $guarantees "$PkgDir\" }

# -- Create ZIP archive --
Write-Host "`n>>> Creating archive..." -ForegroundColor Yellow
$ZipPath = "$ReleaseDir\$PkgName.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath }
Compress-Archive -Path $PkgDir -DestinationPath $ZipPath
Write-Host "  Archive: $ZipPath"

# -- Populate NuGet runtime layout --
Write-Host "`n>>> Setting up NuGet layout..." -ForegroundColor Yellow
$NugetRoot = Join-Path $ReleaseDir "nuget"
$NugetRuntime = Join-Path $NugetRoot "runtimes\win-${Arch}\native"
New-Item -ItemType Directory -Force -Path $NugetRuntime | Out-Null

# Copy native libs
Get-ChildItem "$PkgDir\lib" -File | Where-Object { $_.Name -like "ufsecp*" } | ForEach-Object {
    Copy-Item $_.FullName $NugetRuntime
}

# Copy headers
$NugetInclude = Join-Path $NugetRoot "include\ufsecp"
New-Item -ItemType Directory -Force -Path $NugetInclude | Out-Null
Copy-Item "$RootDir\include\ufsecp\ufsecp.h"         $NugetInclude
Copy-Item "$RootDir\include\ufsecp\ufsecp_version.h"  $NugetInclude
Copy-Item "$RootDir\include\ufsecp\ufsecp_error.h"    $NugetInclude

Write-Host "  NuGet runtimes: $NugetRuntime"

# -- Summary --
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Green
Write-Host "  Release build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Package: $PkgDir\"
Write-Host "  Archive: $ZipPath"
Write-Host "  NuGet:   $NugetRoot\"
Write-Host ""
Write-Host "  Contents:" -ForegroundColor Yellow
Get-ChildItem "$PkgDir\lib" | Format-Table Name, Length -AutoSize
Write-Host ""
Write-Host "  To create NuGet package:" -ForegroundColor Yellow
Write-Host "    Copy-Item 'nuget\*' '$NugetRoot\' -Recurse"
Write-Host "    cd $NugetRoot; nuget pack UltrafastSecp256k1.Native.nuspec"
Write-Host "===============================================================" -ForegroundColor Green
