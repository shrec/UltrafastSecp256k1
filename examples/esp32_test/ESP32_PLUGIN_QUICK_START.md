# CLion ESP-IDF Plugin - Quick Setup

## [OK] Configuration Complete

All files are configured for ESP-IDF Plugin:

- [OK] `.clion-esp-idf.json` - Python 3.12, ESP-IDF 5.5.1, COM3
- [OK] `CMakePresets.json` - Correct toolchain paths
- [OK] `sdkconfig` - ESP32-S3 configuration
- [OK] `CMakeLists.txt` - Project structure

---

## 🚀 Quick Start (3 Steps)

### 1. Open Project

```
File -> Open -> D:\Dev\Secp256K1\libs\UltrafastSecp256k1\examples\esp32_test
```

### 2. Wait for ESP-IDF Plugin

CLion will automatically:
- Detect `.clion-esp-idf.json`
- Configure ESP-IDF 5.5.1
- Show toolbar: 🔨 Build | 📤 Flash | 📡 Monitor

### 3. Build & Flash

Click toolbar button:
- 🔨 **Build** - Compile firmware
- 📤 **Flash** - Upload to COM3
- 📡 **Monitor** - View serial output

Or use Run Configuration dropdown:
- `flash & monitor` <- **Recommended** (one-click)

---

## 📊 Expected Output

After flashing, press **Reset** on ESP32:

```
============================================================
   UltrafastSecp256k1 - ESP32 Comprehensive Benchmark
============================================================

Platform Information:
  Chip Model:   ESP32-S3
  Cores:        2
  Free Heap:    393736 bytes

==============================================
  Results: 29/29 tests passed [OK]
  [SUCCESS] ALL TESTS PASSED
==============================================
```

---

## ⚙ Configuration Details

**Python Environment:**
- Version: 3.12.4
- Path: `C:/Espressif/python_env/idf5.5_py3.12_env`

**ESP-IDF:**
- Version: 5.5.1
- Path: `C:/Espressif/frameworks/esp-idf-v5.5.1`

**Target:**
- Board: ESP32-S3
- Port: COM3
- Baud: 115200 (monitor), 460800 (flash)

**Toolchain:**
- xtensa-esp-elf: 14.2.0_20241119
- Ninja: 1.12.1
- CMake: 3.30.7

---

## [TOOL] Troubleshooting

### Plugin toolbar not showing?

1. Close project
2. Delete `.idea` folder (if exists)
3. Reopen: `File -> Open -> esp32_test`
4. Wait for indexing (bottom progress bar)

### Build error: "Python 3.8"?

Fixed! `.clion-esp-idf.json` already uses Python 3.12.
If error persists:
```
Settings -> ESP-IDF -> Python Path
-> Set to: C:\Espressif\python_env\idf5.5_py3.12_env\Scripts\python.exe
```

### COM3 busy?

Close other programs (Arduino IDE, PuTTY, monitor scripts).

### Can't see test output?

Press **Reset button** on ESP32 board after flashing.

---

## 📚 Resources

- [ESP-IDF Plugin GitHub](https://github.com/espressif/idf-eclipse-plugin)
- [ESP-IDF v5.5.1 Docs](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/)
- [ESP32-S3 Datasheet](https://www.espressif.com/en/products/socs/esp32-s3)

---

**Status:** [OK] ESP-IDF Plugin configured and ready!  
**Updated:** 2026-02-12  
**Device:** ESP32-S3 on COM3

