<#
.SYNOPSIS
  Install the toolchain needed to run the FULL local audit (ci/ci_local.sh) on
  Windows, so it behaves the same as Linux CI.

.DESCRIPTION
  ci_local.sh on Windows needs the same external tools the Linux audit uses:
    - python3 + z3-solver   -> Z3 SMT formal proofs (REQUIRED gate)
    - Lean 4 (elan/lake)     -> Lean SafeGCD proofs (REQUIRED gate)
    - Cryptol (+ solvers)    -> GF(p)/EC property specs (ADVISORY — skips if absent)
    - ufsecp.dll             -> determinism gate + 22 lib-loading audit scripts
                                (discovered automatically by ci/_ufsecp.py find_lib)

  The determinism gate and every _ufsecp-importing script already work on Windows
  once a ufsecp.dll exists (ci/_ufsecp.py is cross-platform). This script installs
  the formal-proof tools and a python3 shim so `./ci/ci_local.sh` runs end-to-end.

  Idempotent: re-running skips anything already present. Nothing here is committed
  to the repo — these are local developer-environment installs.

.NOTES
  Cryptol's bundled specs (audit/formal/cryptol/*.cry) currently target an older
  Cryptol syntax and do not parse under Cryptol 3.5; until they are updated the
  Cryptol gate stays ADVISORY-SKIP (the same state as Linux CI, where Cryptol is
  typically absent). Installing Cryptol is therefore optional (-WithCryptol).
#>
param(
    [switch]$WithCryptol,           # also download Cryptol (advisory tool; specs need updating)
    [string]$PythonExe = 'C:\Python312\python.exe'
)

$ErrorActionPreference = 'Continue'
function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }

# --- 1) Z3 (python module) — REQUIRED formal gate ---------------------------
Step 'Z3 SMT (pip install z3-solver)'
if (Test-Path $PythonExe) {
    & $PythonExe -c "import z3" 2>$null
    if ($LASTEXITCODE -ne 0) { & $PythonExe -m pip install z3-solver --quiet }
    & $PythonExe -c "import z3; print('  z3', z3.get_version_string())"
} else {
    Write-Host "  ! $PythonExe not found — set -PythonExe to your CPython 3.x" -ForegroundColor Yellow
}

# --- 2) Lean 4 via elan — REQUIRED formal gate ------------------------------
Step 'Lean 4 (elan + lake)'
$elanBin = Join-Path $env:USERPROFILE '.elan\bin'
if (-not (Test-Path (Join-Path $elanBin 'lake.exe'))) {
    $tmp = Join-Path $env:TEMP 'elan'; New-Item -ItemType Directory -Force $tmp | Out-Null
    $zip = Join-Path $tmp 'elan.zip'
    Invoke-WebRequest -Uri 'https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-pc-windows-msvc.zip' -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $init = Get-ChildItem -Path $tmp -Filter 'elan-init.exe' -Recurse | Select-Object -First 1
    & $init.FullName -y --default-toolchain none | Out-Null
}
Write-Host ("  lake: " + (Join-Path $elanBin 'lake.exe'))
# Toolchain (lean-toolchain pins the version) is fetched on first `lake build`.

# --- 3) Cryptol (optional, advisory) ----------------------------------------
if ($WithCryptol) {
    Step 'Cryptol 3.5 (with bundled solvers)'
    $dest = Join-Path $env:USERPROFILE '.local\cryptol'; New-Item -ItemType Directory -Force $dest | Out-Null
    if (-not (Get-ChildItem -Path $dest -Filter 'cryptol.exe' -Recurse -ErrorAction SilentlyContinue)) {
        $tgz = Join-Path $env:TEMP 'cryptol.tar.gz'
        Invoke-WebRequest -Uri 'https://github.com/GaloisInc/cryptol/releases/download/3.5.0/cryptol-3.5.0-windows-2022-X64-with-solvers.tar.gz' -OutFile $tgz -UseBasicParsing
        tar -xzf $tgz -C $dest
    }
    $cry = Get-ChildItem -Path $dest -Filter 'cryptol.exe' -Recurse | Select-Object -First 1
    Write-Host ("  cryptol: " + $cry.FullName)
    Write-Host "  NOTE: audit/formal/cryptol/*.cry need a syntax update for Cryptol 3.5;" -ForegroundColor Yellow
    Write-Host "        until then keep cryptol OFF the audit PATH so the gate ADVISORY-SKIPs." -ForegroundColor Yellow
}

# --- 4) python3 shim + PATH (Git Bash) --------------------------------------
Step 'python3 shim + PATH (~/.bashrc)'
$bashHome = (& bash -lc 'echo $HOME') 2>$null
if ($bashHome) {
    & bash -lc "mkdir -p ~/.local/bin && printf '#!/bin/sh\nexec '$(cygpath -u '$PythonExe' 2>/dev/null || echo /c/Python312/python.exe)' \`\"\`\$@\`\"\n' > ~/.local/bin/python3 && chmod +x ~/.local/bin/python3" 2>$null
    & bash -lc "grep -q 'UFSECP audit toolchain' ~/.bashrc 2>/dev/null || printf '\n# --- UFSECP audit toolchain ---\nexport PATH=\"\$HOME/.local/bin:\$HOME/.elan/bin:\$PATH\"\nexport PYTHONUTF8=1\n' >> ~/.bashrc"
    Write-Host "  ~/.bashrc updated (python3 -> $PythonExe, lake on PATH, PYTHONUTF8=1)"
}

Write-Host ""
Step 'Done. Build the ABI DLL then run the audit:'
Write-Host '  cmake --build out/<profile> --target ufsecp_shared   # emits ufsecp.dll'
Write-Host '  bash -lc "./ci/ci_local.sh"                          # full local audit'
Write-Host ''
Write-Host 'Determinism + formal (Z3+Lean) now run on Windows; Cryptol stays advisory-skip.'
