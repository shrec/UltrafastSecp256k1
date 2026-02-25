# UltrafastSecp256k1 Examples

Usage examples for different platforms and use cases.

## 📁 Available Examples

### ESP32 (Embedded)

**Location:** [`esp32_test/`](esp32_test/)

Complete ESP32-S3 example demonstrating:
- Library self-test execution
- Field arithmetic benchmarks
- Point multiplication performance
- ESP-IDF integration

**Status:** [OK] Working (28/28 tests pass)

**Quick Start:**
```bash
cd esp32_test
idf.py build
idf.py flash monitor
```

See [esp32_test/README.md](esp32_test/README.md) for detailed instructions.

---

## 🚧 Planned Examples

The following examples will be added:

- **basic_operations.cpp** - Simple field/point arithmetic
- **signature_verification.cpp** - ECDSA verification
- **batch_processing.cpp** - CUDA batch operations
- **custom_curves.cpp** - Using library with other curves
- **hardware_wallet.cpp** - Secure signing example
- **address_generation.cpp** - Bitcoin/Ethereum address derivation
