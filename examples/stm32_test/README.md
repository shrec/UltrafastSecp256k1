# UltrafastSecp256k1 - STM32F103ZET6 Port

## Hardware
- **MCU**: STM32F103ZET6 (ARM Cortex-M3 @ 72MHz)
- **Flash**: 512KB
- **SRAM**: 64KB
- **Connection**: CH340 USB-UART on COM4
- **UART**: USART1 (PA9=TX, PA10=RX) @ 115200 baud

## Build Requirements
- ARM GCC Toolchain: `D:\Dev\arm-gnu-toolchain\` (13.3.1)
- CMake 3.20+
- Ninja build system

## Quick Start

### Build
```powershell
cd examples/stm32_test
.\build_stm32.ps1
```

### Flash & Monitor
```powershell
.\flash_stm32.ps1 -Port COM4
```

**Flash procedure:**
1. Set BOOT0 jumper → HIGH (3.3V)
2. Press RESET on board
3. Run `flash_stm32.ps1`
4. After flashing, set BOOT0 → LOW (GND)
5. Press RESET — output appears on COM4

### Manual Build
```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Manual Flash
```bash
stm32flash -w build/stm32_secp256k1_test.bin -v -g 0x08000000 COM4
```

## Memory Budget

| Section | Size | Limit |
|---------|------|-------|
| Flash (.text + .rodata) | ~180KB est. | 512KB |
| SRAM (.data + .bss + stack) | ~20KB est. | 64KB |
| Stack | 8KB reserved | - |
| Heap | 2KB reserved | - |

**Note**: Generator fixed-base table (30KB) is **disabled** for STM32
due to 64KB SRAM constraint. Uses GLV+Shamir instead.

## Expected Performance (72MHz, no cache)

| Operation | Estimated |
|-----------|-----------|
| Field Mul | ~18 μs |
| Field Square | ~14 μs |
| Field Inversion | ~5 ms |
| Scalar*G (GLV+Shamir) | ~35 ms |

## Architecture Notes

Uses the same optimized code paths as ESP32:
- Fully unrolled 32-bit Comba multiplication (64 products, zero loops)
- Fully unrolled Comba squaring (36 products, branch-free)
- Optimized point doubling (5S+2M formula)
- GLV decomposition + Shamir's trick for scalar multiplication
- No exceptions, no RTTI (bare-metal friendly)

The Cortex-M3 UMULL instruction (32×32→64) runs in 3-5 cycles,
comparable to ESP32's Xtensa MULL.

## Platform Macro

Defined via CMake: `SECP256K1_PLATFORM_STM32=1`

This activates:
- 32-bit Comba mul/sqr (shared with ESP32)
- GLV+Shamir scalar multiplication
- Optimized dbl_inplace (5S+2M)
- No-exception error handling
- Embedded selftest paths
