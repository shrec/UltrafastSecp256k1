"""Send STM32 bootloader GO command and monitor output on the same serial connection."""
import serial
import time
import sys

PORT = "COM4"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(0.1)
ser.reset_input_buffer()

# 1) Activate bootloader: send 0x7F
ser.write(b'\x7f')
time.sleep(0.1)
ack = ser.read(1)
print("Init ACK:", ack.hex() if ack else "none")

# 2) GO command: 0x21 + complement 0xDE
ser.write(b'\x21\xde')
time.sleep(0.1)
ack = ser.read(1)
print("GO CMD ACK:", ack.hex() if ack else "none")

# 3) Address 0x08000000 + checksum
addr = bytes([0x08, 0x00, 0x00, 0x00])
chk = 0x08 ^ 0x00 ^ 0x00 ^ 0x00
ser.write(addr + bytes([chk]))
time.sleep(0.1)
ack = ser.read(1)
print("GO ADDR ACK:", ack.hex() if ack else "none")
print("--- Firmware output ---")

# Read firmware output (port stays open!)
lines = []
deadline = time.time() + 25
while time.time() < deadline:
    line = ser.readline().decode("utf-8", errors="replace").strip()
    if line:
        print(line)
        lines.append(line)
        if "Test Complete" in line:
            break

ser.close()
print("=== Captured %d lines ===" % len(lines))

# Save to file
with open("stm32_output.txt", "w") as f:
    f.write("\n".join(lines))
