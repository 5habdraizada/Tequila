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
import subprocess
import re

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

    def __init__(self, port: str | None, wheel_radius: float, wheel_base: float,
                 new_data_callback=None,
                 accel_fwd_idx: int = 0, accel_fwd_sign: float = 1.0):
        self._port        = port           # e.g. "/dev/ttyACM0", or None → auto
        self.wheel_radius = wheel_radius
        self.wheel_base   = wheel_base

        self.new_data_callback = new_data_callback
        self._accel_fwd_idx  = accel_fwd_idx
        self._accel_fwd_sign = accel_fwd_sign

        self._ser      = None
        self._running  = False
        self._lock     = threading.Lock()

        # Latest odometry
        self._v_l     = 0.0
        self._v_r     = 0.0
        self._dt      = 0.02
        self.distance_left = 0
        self.ticks_left = 0

        self.pos_theta = 0
        self.pos_x = 0
        self.pos_y = 0

        # IMU readings (updated by gyro_read / accel_read threads)
        self._gyro_z   = 0.0   # yaw rate  (rad/s, RB3 frame)
        self._accel_fwd = 0.0  # forward acceleration (m/s², robot frame)

        self._last_ts = time.time()
        self._tpr = 990

        self._gyro_proc  = None
        self._accel_proc = None

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
                threading.Thread(target=self._reader,   daemon=True).start()
                threading.Thread(target=self.gyro_read,  daemon=True).start()
                threading.Thread(target=self.accel_read, daemon=True).start()
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

        if self._gyro_proc:
            self._gyro_proc.terminate()
        if self._accel_proc:
            self._accel_proc.terminate()

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

##########################################################################
    def gyro_read(self):
        cmd = [
            "sudo",
            "see_workhorse",
            "-sensor=gyro",
            "-sample_rate=500",
            "-display_events=1",
            "-duration=86400"
        ]

        try:
            self._gyro_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )

            print("[GYRO] SEE gyro stream started")

        except FileNotFoundError:
            print("[GYRO] see_workhorse not found")
            return

        buf = []
        for line in self._gyro_proc.stdout:
            # print(line)

            if not self._running:
                break

            line = line.strip()

            if not line:
                continue

            buf.append(line)

            # End of one sensor event block
            if '"Time Elapsed"' in line:
                block = "\n".join(buf)
                # print(block)

                try:
                    # Extract:
                    # "data" : [ x, y, z ]

                    match = re.search(
                        r'"data"\s*:\s*\[\s*'
                        r'([-0-9.eE]+)\s*,\s*'
                        r'([-0-9.eE]+)\s*,\s*'
                        r'([-0-9.eE]+)',
                        block,
                        re.MULTILINE
                    )

                    if match:

                        gx = float(match.group(1))
                        gy = float(match.group(2))
                        gz = float(match.group(3))

                        with self._lock:
                            self._gyro_z = gz

                except (ValueError, IndexError):
                    pass

                buf.clear()

        if self._gyro_proc:
            self._gyro_proc.terminate()

##########################################################################
    def accel_read(self):
        """Read forward linear acceleration from the RB3 IMU at 200 Hz.

        Uses the same see_workhorse mechanism as gyro_read().  The forward
        component (configured by accel_fwd_idx / accel_fwd_sign) is stored
        in _accel_fwd and passed to the EKF predict step so the state's
        velocity dimension integrates real acceleration rather than relying
        solely on wheel encoders.
        """
        cmd = [
            "sudo",
            "see_workhorse",
            "-sensor=accel",
            "-sample_rate=200",
            "-display_events=1",
            "-duration=86400",
        ]

        try:
            self._accel_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            print("[ACCEL] SEE accel stream started")
        except FileNotFoundError:
            print("[ACCEL] see_workhorse not found — accel disabled")
            return

        buf = []
        for line in self._accel_proc.stdout:
            if not self._running:
                break

            line = line.strip()
            if not line:
                continue

            buf.append(line)

            if '"Time Elapsed"' in line:
                block = "\n".join(buf)
                try:
                    match = re.search(
                        r'"data"\s*:\s*\[\s*'
                        r'([-0-9.eE]+)\s*,\s*'
                        r'([-0-9.eE]+)\s*,\s*'
                        r'([-0-9.eE]+)',
                        block,
                        re.MULTILINE,
                    )
                    if match:
                        vals = [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                        fwd = vals[self._accel_fwd_idx] * self._accel_fwd_sign
                        with self._lock:
                            self._accel_fwd = fwd
                except (ValueError, IndexError):
                    pass

                buf.clear()

        if self._accel_proc:
            self._accel_proc.terminate()

############################################################
    # ── background reader ────────────────────────────────────────────────────
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
            gyro_z   = self._gyro_z    # snapshot latest IMU readings
            accel_fwd = self._accel_fwd

        self.new_data_callback(v_l, v_r, dt, gyro_z, accel_fwd)


        
    def wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    
    def update_odometry(self):
        v_delta_l = self._v_l * self._dt
        v_delta_r = self._v_r * self._dt
        
        
        v = (v_delta_l + v_delta_r) / 2
        d_theta = (v_delta_r - v_delta_l) / self.wheel_base
        
        if abs(d_theta) < 1e-4:
            self.pos_x += v * math.cos(self.pos_theta)
            self.pos_y += v * math.sin(self.pos_theta)
        else:
            r = v / d_theta;
            self.pos_x += r * (math.sin(self.pos_theta + d_theta) - math.sin(self.pos_theta));
            self.pos_y += r * (math.cos(self.pos_theta) - math.cos(self.pos_theta + d_theta));
        
        self.pos_theta = HardwareBridge.wrap_angle(self.pos_theta + d_theta)
        
        

    # ── public API ────────────────────────────────────────────────────────────

    def get_ticks(self):
        return self._tpr

    def get_odometry(self) -> dict:
        """Latest wheel speeds and IMU readings."""
        with self._lock:
            return {
                "v_l":       self._v_l,
                "v_r":       self._v_r,
                "dt":        self._dt,
                "gyro_z":    self._gyro_z,
                "accel_fwd": self._accel_fwd,
                "x":         self.pos_x,
                "y":         self.pos_y,
                "theta":     self.pos_theta,
            }

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