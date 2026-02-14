"""
STM32F103ZET6 Flash + Run + Monitor — one-command workflow.

Usage:
    python flash_and_run.py               # flash + run + monitor
    python flash_and_run.py --run-only    # skip flash, just GO + monitor
    python flash_and_run.py --monitor     # just open serial monitor

Before FIRST flash: set BOOT0=HIGH, press RESET.
After flash: set BOOT0=LOW (normal). GO command starts execution.
Subsequent runs: just press RESET button or use --run-only.
"""
import serial
import time
import sys
import subprocess
import argparse

PORT = "COM4"
BAUD = 115200
TIMEOUT = 180
BIN_FILE = "build/stm32_secp256k1_test.bin"


def flash(port, bin_file):
    """Flash via UART bootloader (BOOT0 must be HIGH)."""
    print(f"[FLASH] Flashing {bin_file} to {port}...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "stm32loader",
         "-p", port, "-e", "-w", "-v", "-f", "F1", bin_file],
        capture_output=True, text=True, timeout=120
    )
    if "Verification OK" in result.stdout or "Verification OK" in result.stderr:
        print("[FLASH] Verification OK!", flush=True)
        return True
    else:
        print(f"[FLASH] FAILED!", flush=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        return False


def go_command(port):
    """Send GO command to start execution from flash (BOOT0 can be LOW)."""
    print(f"[GO] Sending GO 0x08000000...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stm32loader",
             "-p", port, "-g", "0x08000000"],
            capture_output=True, text=True, timeout=15
        )
        print(f"[GO] Done.", flush=True)
        return True
    except Exception as e:
        print(f"[GO] Error: {e}", flush=True)
        return False


def monitor(port, baud, timeout_sec):
    """Open serial monitor and capture output until completion or timeout."""
    time.sleep(3)  # Wait for port release

    retries = 5
    ser = None
    for i in range(retries):
        try:
            ser = serial.Serial(port, baud, timeout=2)
            break
        except Exception as e:
            print(f"[MONITOR] Retry {i+1}/{retries}: {e}", flush=True)
            time.sleep(2)

    if not ser:
        print("[MONITOR] Cannot open serial port!", flush=True)
        return []

    ser.reset_input_buffer()
    print(f"[MONITOR] Listening on {port} @ {baud}...", flush=True)
    print("=" * 60, flush=True)

    start = time.time()
    lines = []
    empty_count = 0

    while time.time() - start < timeout_sec:
        data = ser.readline()
        if data:
            text = data.decode('utf-8', errors='replace').strip()
            if text:
                print(text, flush=True)
                lines.append(text)
                empty_count = 0
                if 'COMPLETE' in text.upper() or 'ALL DONE' in text.upper():
                    time.sleep(2)
                    while ser.in_waiting:
                        extra = ser.readline().decode('utf-8', errors='replace').strip()
                        if extra:
                            print(extra, flush=True)
                            lines.append(extra)
                    break
        else:
            empty_count += 1
            if lines and empty_count > 15:
                break
            if not lines and empty_count > 30:
                print("[MONITOR] No data received.", flush=True)
                break

    ser.close()
    print("=" * 60, flush=True)
    print(f"[MONITOR] {len(lines)} lines in {time.time()-start:.1f}s", flush=True)

    if lines:
        with open("stm32_output.txt", "w") as f:
            f.write("\n".join(lines))
        print("[MONITOR] Saved to stm32_output.txt", flush=True)

    return lines


def main():
    parser = argparse.ArgumentParser(description="STM32 Flash + Run + Monitor")
    parser.add_argument("--run-only", action="store_true",
                        help="Skip flash, just send GO + monitor")
    parser.add_argument("--monitor", action="store_true",
                        help="Just open serial monitor (press RESET on board)")
    parser.add_argument("-p", "--port", default=PORT, help=f"Serial port (default: {PORT})")
    parser.add_argument("-b", "--baud", type=int, default=BAUD)
    parser.add_argument("--bin", default=BIN_FILE, help="Binary file to flash")
    args = parser.parse_args()

    if args.monitor:
        # Just monitor — user presses RESET manually
        print("Press RESET on the board, then wait for output...", flush=True)
        monitor(args.port, args.baud, TIMEOUT)
        return

    if args.run_only:
        # Send GO command and monitor
        go_command(args.port)
        monitor(args.port, args.baud, TIMEOUT)
        return

    # Full workflow: flash + go + monitor
    print("=" * 60, flush=True)
    print("  STM32F103ZET6 Flash + Run + Monitor", flush=True)
    print("=" * 60, flush=True)
    print("  BOOT0 must be HIGH for flashing!", flush=True)
    print("=" * 60, flush=True)

    if not flash(args.port, args.bin):
        print("\nFlash failed. Check BOOT0 jumper and connection.", flush=True)
        sys.exit(1)

    print("\n>>> Set BOOT0 back to LOW and press RESET <<<", flush=True)
    print(">>> Or just wait — GO command will try to start... <<<\n", flush=True)
    time.sleep(2)

    go_command(args.port)
    monitor(args.port, args.baud, TIMEOUT)


if __name__ == "__main__":
    main()
