# CLion + ESP-IDF Full Development Environment

## Prerequisites

1. **ESP-IDF 5.5.1** installed (`C:\Espressif\`)
2. **CLion 2024.x** or newer

---

## Step 1: Install CLion ESP-IDF Plugin

1. **File → Settings → Plugins**
2. Search in Marketplace: **"ESP-IDF"**
3. Install and restart CLion

---

## Step 2: Configure ESP-IDF in CLion

1. **File → Settings → Languages & Frameworks → ESP-IDF**
2. Fill in:
   - **ESP-IDF Path:** `C:\Espressif\frameworks\esp-idf-v5.5.1`
   - **Python:** `C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe`
   - **Tools Path:** `C:\Espressif`

3. **Apply** and **OK**

---

## Step 3: Open Project

1. **File → Open**
2. Select: `D:\Dev\Secp256K1\libs\UltrafastSecp256k1\examples\esp32_test`
3. CLion will find CMakeLists.txt and start configuration

---

## Step 4: Target Device Configuration

1. **Run → Edit Configurations**
2. Click **+** → **ESP-IDF**
3. Fill in:
   - **Name:** `ESP32-S3 Flash & Monitor`
   - **Target:** `esp32s3`
   - **Serial Port:** `COM3`
   - **Flash:** ✓
   - **Monitor:** ✓
   - **Baud rate:** `115200`

---

## Usage

| Action | How |
|-----------|-------|
| **Build** | `Ctrl+F9` or 🔨 button |
| **Flash** | Select configuration → `Shift+F10` |
| **Monitor** | Opens automatically after flash |
| **Debug** | `Shift+F9` (JTAG required) |

---

## Serial Monitor in CLion

1. **View → Tool Windows → Serial Monitor**
2. Port: `COM3`
3. Baud: `115200`
4. **Connect**

---

## Troubleshooting

### "IDF_PATH not found"
- Settings → Languages & Frameworks → ESP-IDF → Check paths

### "Cannot open COM port"
- Close other programs (Arduino IDE, PuTTY)
- Check COM port in Device Manager

### Build errors
- Run in terminal: `idf.py fullclean`
- Rebuild

---

## Alternative: ESP-IDF CMD + CLion

If the plugin doesn't work:

1. Open **ESP-IDF 5.5.1 PowerShell** (from Start menu)
2. Run:
   ```cmd
   cd D:\Dev\Secp256K1\libs\UltrafastSecp256k1\examples\esp32_test
   clion .
   ```

This will open CLion with the correct ESP-IDF environment.
