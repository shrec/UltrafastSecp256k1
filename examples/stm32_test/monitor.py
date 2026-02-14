"""Flash-go + serial monitor for STM32F103ZET6"""
import serial
import time
import sys
import subprocess

PORT = "COM4"
BAUD = 115200
TIMEOUT = 180  # seconds max wait

print(f"=== STM32 Monitor ({PORT} @ {BAUD}) ===", flush=True)
print("Sending GO command to start firmware...", flush=True)

# Send GO command via stm32loader to start execution
try:
    result = subprocess.run(
        [sys.executable, "-m", "stm32loader", "-p", PORT, "-g", "0x08000000"],
        capture_output=True, text=True, timeout=15
    )
    print(f"GO command sent. stdout: {result.stdout.strip()}", flush=True)
    if result.returncode != 0:
        print(f"stderr: {result.stderr.strip()}", flush=True)
except Exception as e:
    print(f"GO command error: {e}", flush=True)

# Wait for port to be available after GO command
time.sleep(3)

# Open serial port for monitoring
retries = 5
ser = None
for i in range(retries):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
        print(f"Serial port opened (attempt {i+1})", flush=True)
        break
    except Exception as e:
        print(f"Retry {i+1}/{retries}: {e}", flush=True)
        time.sleep(2)

if not ser:
    print("FATAL: Cannot open serial port", flush=True)
    sys.exit(1)

# Flush any stale data
ser.reset_input_buffer()

# Read output
start = time.time()
lines = []
empty_count = 0

while time.time() - start < TIMEOUT:
    data = ser.readline()
    if data:
        text = data.decode('utf-8', errors='replace').strip()
        if text:
            print(text, flush=True)
            lines.append(text)
            empty_count = 0
            # Check for completion
            if 'ALL DONE' in text.upper() or 'COMPLETE' in text.upper():
                time.sleep(2)
                while ser.in_waiting:
                    extra = ser.readline().decode('utf-8', errors='replace').strip()
                    if extra:
                        print(extra, flush=True)
                        lines.append(extra)
                break
    else:
        empty_count += 1
        # If no data for 30 seconds after we got some output, stop
        if lines and empty_count > 15:
            print("(no more data, stopping)", flush=True)
            break
        # If no data at all for 60 seconds, timeout
        if not lines and empty_count > 30:
            print("(no data received, firmware may not be sending on USART1)", flush=True)
            break

ser.close()
print(f"\n=== Captured {len(lines)} lines in {time.time()-start:.1f}s ===", flush=True)

# Save output
if lines:
    with open("stm32_output.txt", "w") as f:
        f.write("\n".join(lines))
    print("Output saved to stm32_output.txt", flush=True)
