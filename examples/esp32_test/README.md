# ESP32 secp256k1 Test

This example tests portable secp256k1 field arithmetic on ESP32.

## Requirements

- ESP-IDF v5.4+ installed
- ESP32-S3 board (recommended) or ESP32/ESP32-C3/C6

## Supported Boards

| Chip | Architecture | Clock | Cores | Status |
|------|--------------|-------|-------|--------|
| **ESP32-S3** | Xtensa LX7 | 240 MHz | 2 | ✅ Recommended |
| ESP32 | Xtensa LX6 | 240 MHz | 2 | ✅ Supported |
| ESP32-S2 | Xtensa LX7 | 240 MHz | 1 | ✅ Supported |
| ESP32-C3 | RISC-V | 160 MHz | 1 | ✅ Supported |
| ESP32-C6 | RISC-V | 160 MHz | 1 | ✅ Supported |

## Build & Flash

### Option 1: CLion with ESP-IDF plugin

1. Open this folder in CLion
2. Configure ESP-IDF path in Settings
3. Select target: **esp32s3** (or your chip)
4. Build and Flash

### Option 2: Command Line

```bash
# Set up ESP-IDF environment
# Windows:
C:\Espressif\frameworks\esp-idf-v5.5.1\export.bat

# Linux/Mac:
. ~/esp/esp-idf/export.sh

# Set target to ESP32-S3
idf.py set-target esp32s3

# Build
idf.py build

# Flash (replace COM3 with your port)
idf.py -p COM3 flash

# Monitor output
idf.py -p COM3 monitor
```

## Expected Output

```
I (xxx) secp256k1: ╔══════════════════════════════════════════════════════════╗
I (xxx) secp256k1: ║   UltrafastSecp256k1 - ESP32 Benchmark                   ║
I (xxx) secp256k1: ╚══════════════════════════════════════════════════════════╝
I (xxx) secp256k1: 
I (xxx) secp256k1: === Benchmark Results ===
I (xxx) secp256k1: Field Mul: XXXX ns/op (10000 iterations)
I (xxx) secp256k1: Field Sqr: XXXX ns/op (10000 iterations)
I (xxx) secp256k1: Field Add: XXX ns/op (10000 iterations)
```

## ESP32-S3 Specific Features

- **Dual-core** Xtensa LX7 @ 240 MHz
- **Vector instructions** for DSP (potential optimization)
- **8MB PSRAM** on most boards
- **USB OTG** for easy flashing

## Notes

- This uses portable C++ code (no assembly)
- ESP32-S3 is ~30% faster than ESP32 for integer math
- Future: May add Xtensa-optimized assembly

