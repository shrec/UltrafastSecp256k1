@echo off
setlocal

cd /d %~dp0

echo ============================================ > build_log.txt
echo UltrafastSecp256k1 OpenCL Build Log >> build_log.txt
echo ============================================ >> build_log.txt
echo. >> build_log.txt

echo [1] Checking OpenCL.dll... >> build_log.txt
if exist C:\Windows\System32\OpenCL.dll (
    echo    OpenCL.dll found >> build_log.txt
) else (
    echo    WARNING: OpenCL.dll not found >> build_log.txt
)

echo. >> build_log.txt
echo [2] Removing old build directory... >> build_log.txt
if exist build rmdir /s /q build

echo. >> build_log.txt
echo [3] Running CMake... >> build_log.txt
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >> build_log.txt 2>&1

if %ERRORLEVEL% neq 0 (
    echo    CMake FAILED with error %ERRORLEVEL% >> build_log.txt
    goto end
)

echo. >> build_log.txt
echo [4] Building... >> build_log.txt
cmake --build build >> build_log.txt 2>&1

if %ERRORLEVEL% neq 0 (
    echo    Build FAILED with error %ERRORLEVEL% >> build_log.txt
    goto end
)

echo. >> build_log.txt
echo [5] Running Tests... >> build_log.txt
if exist build\opencl_test.exe (
    build\opencl_test.exe >> build_log.txt 2>&1
) else (
    echo    opencl_test.exe not found >> build_log.txt
)

:end
echo. >> build_log.txt
echo ============================================ >> build_log.txt
echo Build process complete >> build_log.txt
echo ============================================ >> build_log.txt

type build_log.txt

