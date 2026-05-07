# ESP32 Integration Guide — UltrafastSecp256k1

**Status:** Tested on ESP-IDF v5.4, GCC 14.2.0, May 2026.  
**Supported targets:** ESP32 (PICO-D4 / LX6), ESP32-S3 (LX7), ESP32-P4 (RISC-V HP), ESP32-C6 (RISC-V)

---

## Table of Contents

1. [Supported Targets & Benchmark Results](#1-supported-targets--benchmark-results)
2. [Required sdkconfig Settings Per Target](#2-required-sdkconfig-settings-per-target)
3. [Critical: Stack Size Requirements](#3-critical-stack-size-requirements)
4. [Critical: IDF v5.4 SMP Startup Bug](#4-critical-idf-v54-smp-startup-bug)
5. [Global Constructor Rules (No-Crash Checklist)](#5-global-constructor-rules-no-crash-checklist)
6. [Source Files to Include](#6-source-files-to-include)
7. [CMakeLists Template](#7-cmakelists-template)
8. [Build & Flash Commands](#8-build--flash-commands)
9. [API Usage on ESP32](#9-api-usage-on-esp32)
10. [Known Limitations](#10-known-limitations)

---

## 1. Supported Targets & Benchmark Results

All measurements: single-threaded, `esp_timer` (1 µs resolution), median of 3 runs.

| Target | Arch | Freq | CT ECDSA sign | CT Schnorr sign | ECDSA verify | Schnorr verify |
|--------|------|------|--------------|-----------------|-------------|----------------|
| ESP32-P4 | RISC-V HP | 360 MHz | 2,609 µs | 2,107 µs | 6,514 µs | 7,081 µs |
| ESP32-S3 | Xtensa LX7 | 240 MHz | 12,705 µs | 6,568 µs | 20,462 µs | 22,344 µs |
| ESP32-C6 | RISC-V | 160 MHz | ~5,400 µs | ~3,000 µs | ~17,250 µs | ~18,900 µs |
| ESP32 (PICO-D4) | Xtensa LX6 | 240 MHz | ~8,000 µs | 6,564 µs | ~23,500 µs | 26,336 µs |

**vs libsecp256k1 on ESP32-S3:** Schnorr sign 1.41×, Schnorr verify 1.47×, ECDSA verify 1.57×.

---

## 2. Required sdkconfig Settings Per Target

### ESP32-P4 (RISC-V, dual-core HP @ 360 MHz)

```ini
CONFIG_IDF_TARGET="esp32p4"
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_360=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=65536
CONFIG_FREERTOS_IDLE_TASK_STACKSIZE=8192
CONFIG_ESP_IPC_TASK_STACK_SIZE=8192
CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0=n
CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n
# NOTE: Do NOT set CONFIG_FREERTOS_UNICORE=y — P4 startup requires SMP mode
```

### ESP32-S3 (Xtensa LX7, dual-core @ 240 MHz)

```ini
CONFIG_IDF_TARGET="esp32s3"
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_FREERTOS_UNICORE=y            # Required — avoids IDF v5.4 SMP race
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=131072  # 128 KB — v4.0.0 has larger footprint
CONFIG_FREERTOS_IDLE_TASK_STACKSIZE=8192
CONFIG_ESP_IPC_TASK_STACK_SIZE=8192
CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0=n
CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n
CONFIG_ESPTOOLPY_FLASHSIZE_8MB=y    # Adjust for your board
```

### ESP32-C6 (RISC-V, single-core @ 160 MHz)

```ini
CONFIG_IDF_TARGET="esp32c6"
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_160=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=65536
CONFIG_FREERTOS_IDLE_TASK_STACKSIZE=8192
CONFIG_ESP_IPC_TASK_STACK_SIZE=8192
CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0=n
# Workaround: coexist ROM fires timer before FreeRTOS on rev v0.2 + IDF v5.4
CONFIG_ESP_COEX_ENABLED=n
CONFIG_ESP_WIFI_ENABLED=n
CONFIG_BT_ENABLED=n
CONFIG_IEEE802154_ENABLED=n
CONFIG_ESP_CONSOLE_SECONDARY_NONE=y
CONFIG_NVS_FLASH=n                  # Removes NVS global constructor
```

### ESP32 / ESP32-PICO-D4 (Xtensa LX6, dual-core @ 240 MHz)

```ini
CONFIG_IDF_TARGET="esp32"
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_FREERTOS_UNICORE=y            # Required for dual-core ESP32
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=65536
CONFIG_FREERTOS_IDLE_TASK_STACKSIZE=8192
CONFIG_ESP_IPC_TASK_STACK_SIZE=8192
CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0=n
CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y
```

---

## 3. Critical: Stack Size Requirements

UltrafastSecp256k1 v4.x uses significantly more stack than v3.x due to new modules (FROST, MuSig2, BIP-352, etc.). **The most common cause of crashes on ESP32 is insufficient stack.**

| Stack | Minimum | Recommended | Why |
|-------|---------|-------------|-----|
| `ESP_MAIN_TASK_STACK_SIZE` | 32 KB | **64–128 KB** | Crypto ops use large stack frames (point tables, wNAF arrays) |
| `FREERTOS_IDLE_TASK_STACKSIZE` | 1.5 KB (default) | **8 KB** | IDF idle hooks + C++ global initialization |
| `ESP_IPC_TASK_STACK_SIZE` | 1.3 KB (default) | **8 KB** | IPC task handles inter-core calls |

**Symptoms of stack overflow:**
- `***ERROR*** A stack overflow in task IDLE detected`
- Assert failure at `esp_startup_start_app.c:XX (rdsp == pdTRUE)` — main task creation failed
- Corrupt backtrace containing `0xa5a5a5a5` (FreeRTOS stack sentinel)

**How to measure actual usage after the bench runs:**

```cpp
// Add to app_main() at the end
UBaseType_t stack_remaining = uxTaskGetStackHighWaterMark(NULL);
printf("Main task: %u bytes remaining\n", stack_remaining * sizeof(StackType_t));
```

---

## 4. Critical: IDF v5.4 SMP Startup Bug

**Affected targets:** All dual-core ESP32 targets (S3, P4) running IDF v5.4 with a large binary (v4.x).

**Root cause:** The coexist ROM fires a hardware timer interrupt before `vTaskStartScheduler()` finishes initializing `pxCurrentTCBs`. On larger binaries the timing window is wider, making the race more likely to hit.

**Crash signature:**
- ESP32-S3 (Xtensa): `Guru Meditation Error: Core 0 panic'ed (LoadProhibited)` with `EXCVADDR=0x44`
- ESP32-P4 (RISC-V): `Guru Meditation Error: Core 0 panic'ed (Load access fault)`
- Backtrace containing `prvSelectHighestPriorityTaskSMP` at `tasks.c:362x`

**Fix per target:**

| Target | Fix |
|--------|-----|
| ESP32-S3 | `CONFIG_FREERTOS_UNICORE=y` — bench is single-threaded anyway |
| ESP32-P4 | Cannot use UNICORE (startup assert fails). Use 64 KB main stack instead of 128 KB. The smaller binary fits without triggering the race. |
| ESP32-C6 | Disable coexist/WiFi/BT in sdkconfig to prevent ROM timer from registering. |
| ESP32 / PICO | `CONFIG_FREERTOS_UNICORE=y` |

**IDF version note:** This bug is fixed in IDF v5.5+. If you can upgrade, do so — all targets should then work without the workarounds above.

---

## 5. Global Constructor Rules (No-Crash Checklist)

UltrafastSecp256k1 has several C++ objects that require runtime initialization (global constructors). These run **before `app_main()`**, in the context of the IDF startup stack. If any constructor triggers an IDF mutex or FreeRTOS call, the device will crash.

**Built-in guards (already in the library):**

| Object | Platform | Guard mechanism |
|--------|----------|-----------------|
| `SHA256 g_*_midstate` (×12) | All ESP32 | `EspLazySHA256` — constexpr-constructed, lazy SHA-256 on first use |
| `FieldElement kGeneratorX/Y` | All ESP32 | Lazy inline functions — computed on first call to `Point::generator()` |
| `std::once_flag g_comb_table_once` | All ESP32 | Replaced with `volatile bool g_comb_init` (zero-init in BSS) |
| `FieldElement B7` | All ESP32 | `get_b7()` inline helper — no global object |

**Rules for adding new code that targets ESP32:**

1. **No file-scope `static const` of complex types.** They generate `_GLOBAL__sub_I_` constructors that run before FreeRTOS.
   ```cpp
   // BAD — generates global constructor
   static const FieldElement kMyConst = FieldElement::from_bytes({...});
   
   // GOOD — lazy init, only runs on first use
   static inline const FieldElement& get_my_const() {
       static FieldElement v = FieldElement::from_bytes({...});
       return v;
   }
   ```

2. **No `std::once_flag` or `std::mutex` at file scope.** Their constructors set non-zero state and go into `.data` (triggers dynamic init).
   ```cpp
   // BAD
   static std::once_flag g_init_once;
   
   // GOOD for ESP32
   #if defined(ESP_PLATFORM)
   static volatile bool g_init_done = false;
   #else
   static std::once_flag g_init_once;
   #endif
   ```

3. **No default member initializers on large struct arrays.**
   ```cpp
   // BAD — generates loop constructor for all entries
   struct Table { bool initialized = false; };
   
   // GOOD — BSS zero-init is sufficient
   struct Table { bool initialized; };
   ```

4. **`FieldElement` is now trivially constructible** (`limbs_{}` removed). Local variables `FieldElement fe;` have indeterminate values — use `FieldElement::zero()` or assign before use.

5. **Verify init_array is clean** before releasing for ESP32:
   ```bash
   xtensa-esp32s3-elf-nm build/app.elf | grep '_GLOBAL__sub_I_.*secp256k1'
   # Should return ZERO secp256k1 entries
   ```

---

## 6. Source Files to Include

The ESP32 bench uses a reduced set of source files (no GPU, no bindings, no coin layer). Minimum required for full sign/verify:

```cmake
# Core cryptography (required)
src/cpu/src/field.cpp
src/cpu/src/scalar.cpp
src/cpu/src/point.cpp
src/cpu/src/glv.cpp

# Constant-time layer (required for CT signing)
src/cpu/src/ct_field.cpp
src/cpu/src/ct_scalar.cpp
src/cpu/src/ct_point.cpp
src/cpu/src/ct_sign.cpp

# Signing algorithms
src/cpu/src/ecdsa.cpp
src/cpu/src/schnorr.cpp

# Hardware acceleration (SHA-256)
src/cpu/src/hash_accel.cpp

# Optional: batch verification
src/cpu/src/multiscalar.cpp
src/cpu/src/batch_verify.cpp
src/cpu/src/pippenger.cpp

# Optional: field variants (for optimal selection)
src/cpu/src/field_26.cpp   # 10x26 for 32-bit platforms
src/cpu/src/field_52.cpp   # 5x52 (skipped if SECP256K1_NO_INT128)

# Self-test (recommended)
src/cpu/src/selftest.cpp
```

**Do NOT include:**
- `src/cpu/src/precompute.cpp` — requires large precomputed tables, PSRAM recommended
- `src/cpu/src/bip32.cpp` etc. — add only if you need HD wallet derivation
- Any GPU sources

---

## 7. CMakeLists Template

```cmake
idf_component_register(
    SRCS "main.cpp"
         "${SECP_ROOT}/src/field.cpp"
         "${SECP_ROOT}/src/field_26.cpp"
         "${SECP_ROOT}/src/field_52.cpp"
         "${SECP_ROOT}/src/scalar.cpp"
         "${SECP_ROOT}/src/point.cpp"
         "${SECP_ROOT}/src/glv.cpp"
         "${SECP_ROOT}/src/selftest.cpp"
         "${SECP_ROOT}/src/ct_field.cpp"
         "${SECP_ROOT}/src/ct_scalar.cpp"
         "${SECP_ROOT}/src/ct_point.cpp"
         "${SECP_ROOT}/src/ct_sign.cpp"
         "${SECP_ROOT}/src/ecdsa.cpp"
         "${SECP_ROOT}/src/schnorr.cpp"
         "${SECP_ROOT}/src/hash_accel.cpp"
    INCLUDE_DIRS "."
                 "${SECP_ROOT}/include"     # public headers (secp256k1/)
                 "${SECP_ROOT}/cpu/include" # internal (field.hpp etc.)
    REQUIRES esp_timer
)

set(SECP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/path/to/UltrafastSecp256k1/src/cpu")

# Required ESP32 defines
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    SECP256K1_PLATFORM_ESP32=1
    SECP256K1_NO_INT128=1   # All ESP32 are 32-bit (no __int128)
    SECP256K1_NO_ASM=1
    NDEBUG=1
)

# Optimization flags
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -fno-exceptions
    -fno-rtti
    -fomit-frame-pointer
    -fno-stack-protector
    -fno-threadsafe-statics    # Avoids mutex on local statics
    -fno-use-cxa-atexit        # Avoids __cxa_atexit destructor registration
    -Wno-error=return-type
    -Wno-unused-variable
    -Wno-unused-function
)
```

---

## 8. Build & Flash Commands

```bash
# Source IDF environment
. /path/to/esp-idf/export.sh

# Navigate to project
cd examples/esp32s3_bench_hornet   # or your project

# Delete old sdkconfig to pick up sdkconfig.defaults changes
rm sdkconfig

# Build
idf.py build

# Flash (replace PORT with actual device port)
idf.py -p /dev/ttyACM0 flash

# Flash + monitor
idf.py -p /dev/ttyACM0 flash monitor

# Read serial output via Python (works when idf.py monitor requires TTY)
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyACM0', 115200, timeout=5)
s.setDTR(False); s.setRTS(False)
start = time.time()
while time.time()-start < 600:
    data = s.read(2000)
    if data: print(data.decode('utf-8', errors='replace'), end='', flush=True)
s.close()
"
```

**Port identification:**
- ESP32-S3, ESP32-P4 (USB-JTAG): `/dev/ttyACM*` with description "USB JTAG/serial debug unit"
- ESP32-C6: `/dev/ttyACM*` or `/dev/ttyUSB*` — check `lsusb` VID:PID=303A:1001
- ESP32, ESP32-PICO: usually `/dev/ttyUSB0` via CH340/CP2102

---

## 9. API Usage on ESP32

### Signing (CT — constant-time, recommended for production)

```cpp
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/scalar.hpp"

// Generate key
auto sk = secp256k1::fast::Scalar::from_bytes(privkey_bytes);

// ECDSA sign (constant-time)
std::array<uint8_t, 32> msg_hash = compute_sha256(message);
auto sig = secp256k1::ct::ecdsa_sign(msg_hash, sk);
auto sig_bytes = sig.to_der();  // DER-encoded

// Schnorr sign (constant-time, BIP-340)
auto keypair = secp256k1::schnorr_keypair_create(sk);
std::array<uint8_t, 32> aux_rand{};  // fill with random bytes
auto schnorr_sig = secp256k1::ct::schnorr_sign(keypair, msg_hash, aux_rand);
auto sig64 = schnorr_sig.to_bytes();  // 64 bytes
```

### Verification (variable-time, correct for public data)

```cpp
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"

// ECDSA verify
auto pubkey = secp256k1::fast::Point::from_compressed(pubkey_bytes);
bool ok = secp256k1::ecdsa_verify(msg_hash, pubkey, sig);

// Schnorr verify (BIP-340, x-only pubkey)
std::array<uint8_t, 32> pubkey_x32 = ...;
bool ok = secp256k1::schnorr_verify(pubkey_x32, msg_hash, schnorr_sig);
```

### Memory allocation note

The CT generator multiplication table (`g_comb_table`, ~25 KB) is a static global in SRAM. On devices with limited SRAM (ESP32 original: 520 KB, ESP32-C6: 512 KB), this is fine. On very constrained devices, consider using `Point::generator().scalar_mul(k)` instead of `ct::generator_mul(k)` — it uses less memory but is not constant-time.

---

## 10. Known Limitations

### All ESP32 targets

| Limitation | Detail |
|-----------|--------|
| No `__int128` | All ESP32 are 32-bit (RV32 or Xtensa 32-bit). The 5x52 FE52 path is disabled. 4x64 path uses software 64-bit arithmetic. |
| No SIMD | No AVX2/SSE/NEON equivalent. Pure C++ arithmetic only. |
| `FieldElement` local vars | After v4.0.0, `FieldElement fe;` has indeterminate limbs. Always initialize: `FieldElement fe = FieldElement::zero();` |
| `precompute.hpp` excluded | The large precomputed generator table (`precompute.hpp`) is excluded for ESP32 (compile guard). Use `Point::generator().scalar_mul(k)` for key generation instead of the precomputed path. |

### ESP32-C6 specific

| Limitation | Detail |
|-----------|--------|
| WiFi/BT disabled | The workaround for the IDF v5.4 coexist ROM crash requires disabling WiFi/BT in sdkconfig. If your application needs WiFi/BT, upgrade to IDF v5.5+. |
| Hardware rev v0.2 | The coexist ROM startup race is specifically pronounced on rev v0.2 chips with IDF v5.4. Newer revisions or newer IDF may not need the workaround. |

### ESP32-S3 specific

| Limitation | Detail |
|-----------|--------|
| Single-core mode | `CONFIG_FREERTOS_UNICORE=y` forces single-core operation. All tasks run on CPU0. If you need CPU1 for other work, upgrade to IDF v5.5+ which fixes the SMP race. |
| 128 KB main task | The 128 KB main task stack is larger than typical ESP-IDF apps. This is necessary because v4.0.0's global initialization and crypto stack frames are larger. |

### ESP32-P4 specific

| Limitation | Detail |
|-----------|--------|
| SMP mode required | P4's startup code requires SMP FreeRTOS. `CONFIG_FREERTOS_UNICORE=y` causes an assertion failure. Keep SMP mode and use 64 KB stack. |
| IDF v5.4 SMP race | With 64 KB stack (smaller binary), the race condition window is narrow enough to not crash consistently. With 128 KB stack, it crashes reliably. Upgrade to IDF v5.5+ for a proper fix. |

---

## Appendix: IDF Patch for SMP Race (Advanced)

If you cannot upgrade IDF and need a more robust fix than the sdkconfig workarounds, the root cause is in two IDF files:

**`components/freertos/FreeRTOS-Kernel/portable/xtensa/portasm.S` (for S3)**

Line ~222 — change:
```asm
beqz    a2,  1f    # null TCB → goes to dispatcher (crashes)
```
to:
```asm
beqz    a2,  .Lnoswitch   # null TCB → skip context switch (safe)
```

**`components/freertos/FreeRTOS-Kernel/tasks.c` (for all SMP targets)**

Inside `prvSelectHighestPriorityTaskSMP`, before `taskIS_AFFINITY_COMPATIBLE`:
```c
if (pxTCBCur == NULL) { goto get_next_task; }
```

These patches prevent the crash but may cause Core 1 to stall briefly during startup. The `UNICORE`/stack-size workarounds above are safer for production use.
