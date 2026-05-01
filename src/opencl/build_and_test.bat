@echo off
echo ============================================
echo Building UltrafastSecp256k1 OpenCL
echo ============================================

cd /d %~dp0

if exist build rmdir /s /q build

echo.
echo [1/3] Running CMake...
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Building...
cmake --build build
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Running Tests...
if exist build\opencl_test.exe (
    build\opencl_test.exe
) else (
    echo Test executable not found!
)

echo.
echo ============================================
echo Build Complete
echo ============================================
pause

