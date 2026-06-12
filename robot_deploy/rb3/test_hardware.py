#!/usr/bin/env python3
"""
Verify the Pico USB connection and stream live odometry.
Run on the RB3 with the Pico plugged in via USB.

Usage:
    python3 test_hardware.py               # auto-detect Pico
    python3 test_hardware.py --port /dev/ttyACM0
"""
import argparse, sys, time
import config as rb3_cfg
from hardware import HardwareBridge, find_pico_port

p = argparse.ArgumentParser()
p.add_argument("--port", default=None, help="Serial port (default: auto-detect)")
args = p.parse_args()

# Show what ports are available
from hardware import serial
print("Available serial ports:")
import serial.tools.list_ports
for port in serial.tools.list_ports.comports():
    vid = f"{port.vid:04X}" if port.vid else "????"
    pid = f"{port.pid:04X}" if port.pid else "????"
    print(f"  {port.device}  VID:PID={vid}:{pid}  {port.description}")

hw = HardwareBridge(
    port         = args.port,
    wheel_radius = rb3_cfg.WHEEL_RADIUS_M,
    wheel_base   = rb3_cfg.WHEEL_BASE_M,
)

if not hw.connect():
    sys.exit(1)

print(f"\nStreaming odometry — Ctrl-C to stop\n")
print(f"{'v_l (m/s)':>12}  {'v_r (m/s)':>12}  {'dt (ms)':>10}")
print("-" * 42)

try:
    while True:
        od = hw.get_odometry()
        print(f"{od['v_l']:>12.4f}  {od['v_r']:>12.4f}  {od['dt']*1000:>10.1f}")
        hw.send_cmd(0.0, 0.0)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")
    hw.disconnect()
