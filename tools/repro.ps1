#!/usr/bin/env pwsh
# ============================================================================
# repro.ps1 -- Reproducible Environment Report Generator
# ============================================================================
# Collects system/compiler/build info for bug reports and benchmarks.
# Usage: pwsh tools/repro.ps1 [-OutputFile repro.txt]
# ============================================================================

param(
    [string]$OutputFile = "",
    [string]$BuildDir = ""
)

function Write-Section($title) {
    $sep = "-" * 60
    Write-Output ""
    Write-Output $sep
    Write-Output "  $title"
    Write-Output $sep
}

$report = @()

function Add-Line($text) {
    $script:report += $text
}

function Add-Section($title) {
    $script:report += ""
    $script:report += ("-" * 60)
    $script:report += "  $title"
    $script:report += ("-" * 60)
}

# -- Header --------------------------------------------------------------------

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
Add-Line "UltrafastSecp256k1 -- Environment Report"
Add-Line "Generated: $timestamp"
Add-Line ""

# -- Git Info ------------------------------------------------------------------

Add-Section "Git"

try {
    $commit = git rev-parse --short HEAD 2>$null
    $branch = git branch --show-current 2>$null
    $dirty  = git diff --quiet 2>$null; if ($LASTEXITCODE -ne 0) { "YES" } else { "NO" }
    $tag    = git describe --tags --exact-match HEAD 2>$null
    if (-not $tag) { $tag = "(no tag)" }

    Add-Line "  Commit:  $commit"
    Add-Line "  Branch:  $branch"
    Add-Line "  Tag:     $tag"
    Add-Line "  Dirty:   $dirty"
} catch {
    Add-Line "  (git not available)"
}

# -- OS Info -------------------------------------------------------------------

Add-Section "Operating System"

if ($IsWindows -or $env:OS -eq "Windows_NT") {
    $os = Get-CimInstance Win32_OperatingSystem
    Add-Line "  OS:      $($os.Caption) $($os.Version)"
    Add-Line "  Arch:    $env:PROCESSOR_ARCHITECTURE"
} elseif ($IsLinux) {
    $osrel = Get-Content /etc/os-release -ErrorAction SilentlyContinue | Where-Object { $_ -match "^PRETTY_NAME=" }
    $osname = if ($osrel) { ($osrel -split '=')[1].Trim('"') } else { "Linux" }
    $arch   = uname -m
    $kernel = uname -r
    Add-Line "  OS:      $osname"
    Add-Line "  Kernel:  $kernel"
    Add-Line "  Arch:    $arch"
} elseif ($IsMacOS) {
    $ver = sw_vers -productVersion
    $arch = uname -m
    Add-Line "  OS:      macOS $ver"
    Add-Line "  Arch:    $arch"
}

# -- CPU Info ------------------------------------------------------------------

Add-Section "CPU"

if ($IsWindows -or $env:OS -eq "Windows_NT") {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    Add-Line "  Model:   $($cpu.Name.Trim())"
    Add-Line "  Cores:   $($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical"
    Add-Line "  MHz:     $($cpu.MaxClockSpeed)"
} elseif ($IsLinux) {
    $model = Get-Content /proc/cpuinfo | Where-Object { $_ -match "^model name" } | Select-Object -First 1
    $model = if ($model) { ($model -split ':')[1].Trim() } else { "unknown" }
    $cores = (Get-Content /proc/cpuinfo | Where-Object { $_ -match "^processor" }).Count
    Add-Line "  Model:   $model"
    Add-Line "  Cores:   $cores"
}

# -- Memory --------------------------------------------------------------------

Add-Section "Memory"

if ($IsWindows -or $env:OS -eq "Windows_NT") {
    $mem = Get-CimInstance Win32_ComputerSystem
    $totalGB = [math]::Round($mem.TotalPhysicalMemory / 1GB, 1)
    Add-Line "  Total:   ${totalGB} GB"
} elseif ($IsLinux) {
    $meminfo = Get-Content /proc/meminfo | Where-Object { $_ -match "^MemTotal:" }
    if ($meminfo) {
        $kb = [int64](($meminfo -split '\s+')[1])
        $gb = [math]::Round($kb / 1MB, 1)
        Add-Line "  Total:   ${gb} GB"
    }
}

# -- Compilers -----------------------------------------------------------------

Add-Section "Compilers"

$compilers = @("gcc", "g++", "clang", "clang++", "cl", "nvcc")
foreach ($cc in $compilers) {
    $found = Get-Command $cc -ErrorAction SilentlyContinue
    if ($found) {
        try {
            $ver = & $cc --version 2>&1 | Select-Object -First 1
            Add-Line "  ${cc}: $ver"
        } catch {
            Add-Line "  ${cc}: (found but version unknown)"
        }
    }
}

# -- CMake ---------------------------------------------------------------------

Add-Section "CMake"

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) {
    $cmakeVer = cmake --version | Select-Object -First 1
    Add-Line "  $cmakeVer"
} else {
    Add-Line "  (not found)"
}

$ninja = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninja) {
    $ninjaVer = ninja --version
    Add-Line "  Ninja: $ninjaVer"
}

# -- Build Config (if build dir exists) ----------------------------------------

if ($BuildDir -and (Test-Path "$BuildDir/CMakeCache.txt")) {
    Add-Section "Build Configuration ($BuildDir)"

    $cache = Get-Content "$BuildDir/CMakeCache.txt"
    $interesting = @(
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "SECP256K1_BUILD_CPU",
        "SECP256K1_BUILD_CUDA",
        "SECP256K1_USE_ASM",
        "SECP256K1_SPEED_FIRST",
        "CMAKE_CUDA_ARCHITECTURES"
    )

    foreach ($key in $interesting) {
        $line = $cache | Where-Object { $_ -match "^${key}:" }
        if ($line) {
            $val = ($line -split '=', 2)[1]
            Add-Line "  ${key} = $val"
        }
    }
}

# -- GPU (if available) --------------------------------------------------------

Add-Section "GPU"

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $gpuName = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        $gpuMem  = nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>$null | Select-Object -First 1
        $gpuDriver = nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        Add-Line "  GPU:     $gpuName"
        Add-Line "  VRAM:    $gpuMem"
        Add-Line "  Driver:  $gpuDriver"
    } catch {
        Add-Line "  nvidia-smi found but query failed"
    }
} else {
    Add-Line "  (no NVIDIA GPU detected)"
}

# -- Output --------------------------------------------------------------------

Add-Line ""
Add-Line ("-" * 60)
Add-Line "  End of Report"
Add-Line ("-" * 60)

$output = $report -join "`n"

if ($OutputFile) {
    $output | Out-File -FilePath $OutputFile -Encoding UTF8
    Write-Host "Report saved to: $OutputFile"
} else {
    Write-Output $output
}
