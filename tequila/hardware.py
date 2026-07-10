"""TCP client run on the RB3 that talks to pi_bridge/bridge.py on the Pi:
receives encoder ticks, sends velocity commands. Drop-in replacement for
SyntheticSensors in the real robot pipeline."""

import json
import math
import socket
import threading
import time


class HardwareBridge:
    """TCP client for the Pi bridge.

    hw = HardwareBridge(pi_ip="192.168.1.42")
    hw.connect()
    data = hw.get_odometry()   # {v_l, v_r, dt}
    hw.send_cmd(v_lin, v_ang)
    """

    PORT     = 9100
    TIMEOUT  = 2.0    # seconds to wait for connection

    def __init__(self, pi_ip: str, wheel_radius: float = 0.035,
                 wheel_base: float = 0.20):
        self.pi_ip        = pi_ip
        self.wheel_radius = wheel_radius
        self.wheel_base   = wheel_base

        self._sock        = None
        self._lock        = threading.Lock()
        self._running     = False

        self._v_l  = 0.0   # left  wheel speed  (m/s)
        self._v_r  = 0.0   # right wheel speed  (m/s)
        self._dt   = 0.02  # time since last packet (s)
        self._last_ts     = time.time()
        self._ticks_per_rev = 360  # updated from first packet

    def connect(self, retries: int = 10) -> bool:
        for attempt in range(1, retries + 1):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.TIMEOUT)
                s.connect((self.pi_ip, self.PORT))
                self._sock    = s
                self._running = True
                threading.Thread(target=self._reader, daemon=True).start()
                print(f"[HW] Connected to Pi at {self.pi_ip}:{self.PORT}")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                print(f"[HW] Attempt {attempt}/{retries} failed: {e}")
                time.sleep(1.5)
        print("[HW] Could not connect to Pi bridge.")
        return False

    def disconnect(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _reader(self):
        buf = ""
        while self._running and self._sock:
            try:
                chunk = self._sock.recv(512).decode(errors="ignore")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._handle_packet(line.strip())
            except (socket.timeout, OSError):
                break
        print("[HW] Reader thread exited")
        self._running = False

    def _handle_packet(self, line: str):
        try:
            pkt = json.loads(line)
        except json.JSONDecodeError:
            return

        tick_l = int(pkt.get("tick_l", 0))
        tick_r = int(pkt.get("tick_r", 0))
        tpr    = int(pkt.get("tpr",    self._ticks_per_rev))
        ts     = float(pkt.get("ts",   time.time()))

        now = time.time()
        dt  = max(1e-3, now - self._last_ts)
        self._last_ts = now
        self._ticks_per_rev = tpr

        # Ticks → wheel speed (m/s)
        circ  = 2 * math.pi * self.wheel_radius
        v_l   = (tick_l / tpr) * circ / dt
        v_r   = (tick_r / tpr) * circ / dt

        with self._lock:
            self._v_l = v_l
            self._v_r = v_r
            self._dt  = dt

    def get_odometry(self) -> dict:
        """Latest wheel speeds, compatible with EKF2D.predict(v_l, v_r, dt)."""
        with self._lock:
            return {
                "v_l": self._v_l,
                "v_r": self._v_r,
                "dt":  self._dt,
            }

    def send_cmd(self, v_lin: float, v_ang: float):
        """Send a velocity command to the Pi (and on to the motors)."""
        if not self._sock or not self._running:
            return
        pkt = json.dumps({"v_lin": round(v_lin, 4),
                           "v_ang": round(v_ang, 4)}) + "\n"
        try:
            self._sock.sendall(pkt.encode())
        except OSError:
            pass

    @property
    def connected(self) -> bool:
        return self._running and self._sock is not None
