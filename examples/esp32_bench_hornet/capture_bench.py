import serial
import time
import sys

PORT = 'COM3'
BAUD = 115200
TIMEOUT_TOTAL = 900  # 15 minutes max
OUTPUT_FILE = 'esp32_bench_hornet_output.txt'
DONE_MARKER = 'UltrafastSecp256k1 v'  # appears in the final line

print(f"Opening {PORT} at {BAUD} baud...", flush=True)
s = serial.Serial(PORT, BAUD, timeout=1)

# Reset ESP32 via RTS toggle
s.setDTR(False)
s.setRTS(False)
time.sleep(0.3)
s.setRTS(True)
time.sleep(0.1)
s.setRTS(False)
time.sleep(1.0)
s.flushInput()

print("Monitoring... waiting for benchmark output (this may take 10+ minutes)", flush=True)

buf = ''
start = time.time()
done_time = None
all_output = []
saw_header = False
# Wait for the final summary line that has "UltrafastSecp256k1 v"
# after we've already seen the benchmark sections

while True:
    data = s.read(4096)
    if data:
        txt = data.decode('utf-8', 'replace')
        buf += txt
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.rstrip('\r')
            print(line, flush=True)
            all_output.append(line)
            if 'Bitcoin Consensus CPU Benchmark' in line:
                saw_header = True
            if saw_header and 'BENCH_HORNET_COMPLETE' in line:
                done_time = time.time()

    # Wait 3s after seeing completion marker
    if done_time and (time.time() - done_time > 3):
        break

    # Total timeout
    if time.time() - start > TIMEOUT_TOTAL:
        print("!!! TIMEOUT -- saving partial output", flush=True)
        break

s.close()

# Save output
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for line in all_output:
        f.write(line + '\n')

print(f"\n=== Output saved to {OUTPUT_FILE} ({len(all_output)} lines) ===", flush=True)
