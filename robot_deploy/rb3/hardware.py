"""
rb3/hardware.py — Serial client that talks to the Pi Pico over USB.

The Pico appears as /dev/ttyACM0 (or ttyACM1) on the RB3.
Protocol: newline-delimited JSON at 115200 baud.

  Pico → RB3:  {"tick_l": int, "tick_r": int, "tpr": int, "ts": float}
  RB3 → Pico:  {"v_lin": float, "v_ang": float}
"""

import json
import math
import threading
import time

import serial
import serial.tools.list_ports


def find_pico_port() -> str | None:
    """Auto-detect the Pico's USB serial port."""
    for p in serial.tools.list_ports.comports():
        # Pico shows up as VID:PID 2E8A:0005 (MicroPython USB CDC)
        if p.vid == 0x2E8A:
            return p.device
    return None


class HardwareBridge:
    """
    Reads encoder ticks from the Pico and sends velocity commands.
    Drop-in for SyntheticSensors — feeds EKF2D.predict(v_l, v_r, dt).
    """

    BAUD = 115200

    def __init__(self, port: str | None, wheel_radius: float, wheel_base: float):
        self._port        = port           # e.g. "/dev/ttyACM0", or None → auto
        self.wheel_radius = wheel_radius
        self.wheel_base   = wheel_base

        self._ser      = None
        self._running  = False
        self._lock     = threading.Lock()

        # Latest odometry
        self._v_l     = 0.0
        self._v_r     = 0.0
        self._dt      = 0.02
        self._last_ts = time.time()
        self._tpr     = 360

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, retries: int = 10) -> bool:
        port = self._port

        for attempt in range(1, retries + 1):
            # Auto-detect if no port specified
            if port is None:
                port = find_pico_port()
                if port:
                    print(f"[HW] Pico detected on {port}")
                else:
                    print(f"[HW] Attempt {attempt}/{retries} — Pico not found")
                    time.sleep(1.5)
                    continue

            try:
                ser = serial.Serial(port, self.BAUD, timeout=0.05)
                # Flush any boot messages from MicroPython
                time.sleep(0.5)
                ser.reset_input_buffer()

                self._ser     = ser
                self._running = True
                threading.Thread(target=self._reader, daemon=True).start()
                print(f"[HW] Connected to Pico on {port}")
                return True

            except serial.SerialException as e:
                print(f"[HW] Attempt {attempt}/{retries} — {e}")
                port = None   # retry auto-detect
                time.sleep(1.5)

        print("[HW] ERROR: could not open Pico serial port")
        return False

    def disconnect(self):
        self._running = False
        if self._ser:
            try:
                self.send_cmd(0.0, 0.0)   # stop motors
                self._ser.close()
            except serial.SerialException:
                pass
            self._ser = None

    @property
    def connected(self) -> bool:
        return self._running and self._ser is not None

    # ── background reader ─────────────────────────────────────────────────────

    def _reader(self):
        buf = b""
        while self._running and self._ser:
            try:
                buf += self._ser.read(256)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._parse(line.decode(errors="ignore").strip())
            except serial.SerialException:
                break
        self._running = False
        print("[HW] Serial reader stopped")

    def _parse(self, line: str):
        if not line:
            return
        try:
            pkt = json.loads(line)
        except json.JSONDecodeError:
            return

        tick_l = int(pkt.get("tick_l", 0))
        tick_r = int(pkt.get("tick_r", 0))
        tpr    = int(pkt.get("tpr",    self._tpr))

        now = time.time()
        dt  = max(1e-3, now - self._last_ts)
        self._last_ts = now
        self._tpr     = tpr

        circ = 2 * math.pi * self.wheel_radius
        v_l  = (tick_l / tpr) * circ / dt
        v_r  = (tick_r / tpr) * circ / dt

        with self._lock:
            self._v_l = v_l
            self._v_r = v_r
            self._dt  = dt

    # ── public API ────────────────────────────────────────────────────────────

    def get_odometry(self) -> dict:
        """Latest wheel speeds — feed directly into EKF2D.predict()."""
        with self._lock:
            return {"v_l": self._v_l, "v_r": self._v_r, "dt": self._dt}

    def send_cmd(self, v_lin: float, v_ang: float):
        """Send velocity command to Pico."""
        if not self.connected:
            return
        pkt = json.dumps({"v_lin": round(float(v_lin), 4),
                           "v_ang": round(float(v_ang), 4)}) + "\n"
        try:
            self._ser.write(pkt.encode())
        except serial.SerialException:
            self._running = False
