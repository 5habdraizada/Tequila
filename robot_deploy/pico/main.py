"""
pico/main.py — MicroPython firmware for the Pi Pico.

Flash alongside config.py.  Pico connects to RB3 via USB (/dev/ttyACM0).

"""

import sys
import uselect as select
import time
import ujson
from machine import Pin, PWM
import math

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder  (both-edge count on A, direction from a==b comparison)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder:
    def __init__(self, pin_a: int, pin_b: int):
        self._ticks = 0
        self._a = Pin(pin_a, Pin.IN, Pin.PULL_UP)
        self._b = Pin(pin_b, Pin.IN, Pin.PULL_UP)
        self._last_a = self._a.value()
        self._a.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=self._isr)
        self._last_a = self._a.value()
        self._a.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=self._isr, hard=True)

    def _isr(self, _pin):
        a = self._a.value()
        b = self._b.value()
        if a != self._last_a:
            self._ticks += 1 if (a == b) else -1
        self._last_a = a
        a = self._a.value()
        b = self._b.value()
        if a != self._last_a:
            self._ticks += 1 if (a == b) else -1
        self._last_a = a

    def pop(self) -> int:
        """Return ticks since last call and reset counter."""
        t = self._ticks
        self._ticks = 0
        return t
    

class Odometry:
    def __init__(self):
        self.theta = 0
        self.x = 0
        self.y = 0
        
    def wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    
    def update_odometry(self, enc_l_ticks, enc_r_ticks):
        v_delta_l = (enc_l_ticks  / cfg.TICKS_PER_REV) * cfg.WHEEL_CIRC
        v_delta_r = (enc_r_ticks  / cfg.TICKS_PER_REV) * cfg.WHEEL_CIRC
        
        v = (v_delta_l + v_delta_r) / 2
        d_theta = (v_delta_r - v_delta_l) / cfg.WHEEL_BASE
        
        if abs(d_theta) < 1e-4:
            self.x += v * math.cos(self.theta)
            self.x += v * math.sin(self.theta)
        else:
            r = v / d_theta;
            self.x += r * (math.sin(self.theta + d_theta) - math.sin(self.theta));
            self.y += r * (math.cos(self.theta) - math.cos(self.theta + d_theta));
        
        self.theta = Odometry.wrap_angle(self.theta + d_theta);



# ─────────────────────────────────────────────────────────────────────────────
#  Motor  (dual-PWM driver: P1=fwd PWM, P2=rev PWM)
# ─────────────────────────────────────────────────────────────────────────────

class Motor:
    def __init__(self, p1_pin: int, p2_pin: int):
        self._p1 = PWM(Pin(p1_pin))
        self._p2 = PWM(Pin(p2_pin))
        self._p1.freq(cfg.MOTOR_PWM_FREQ)
        self._p2.freq(cfg.MOTOR_PWM_FREQ)
        self.stop()

    def set_duty(self, duty: float):
        """duty in [-100.0, 100.0].  Positive = forward."""
        duty = max(-100.0, min(100.0, duty))
        u16  = int(abs(duty) / 100.0 * 65535)
        if duty >= 0:
            self._p1.duty_u16(u16)
            self._p2.duty_u16(0)
        else:
            self._p1.duty_u16(0)
            self._p2.duty_u16(u16)

    def stop(self):
        self._p1.duty_u16(0)
        self._p2.duty_u16(0)

    def brake(self):
        """Active brake — both pins full."""
        self._p1.duty_u16(65535)
        self._p2.duty_u16(65535)


# ─────────────────────────────────────────────────────────────────────────────
#  Drive train
# ─────────────────────────────────────────────────────────────────────────────

class DriveTrain:
    def __init__(self):
        self._r = Motor(cfg.M_R1, cfg.M_R2)
        self._l = Motor(cfg.M_L1, cfg.M_L2)
        self._max_v = (cfg.MAX_RPM / 60.0) * cfg.WHEEL_CIRC  # m/s at full duty

    def set_velocity(self, v_lin: float, v_ang: float):
        """Convert body velocity → individual wheel duties."""
        v_l = v_lin - v_ang * cfg.WHEEL_BASE / 2.0
        v_r = v_lin + v_ang * cfg.WHEEL_BASE / 2.0
        # Normalise so neither wheel exceeds 100 %
        peak = max(abs(v_l), abs(v_r), self._max_v)
        d_l  = v_l / peak * 100.0
        d_r  = v_r / peak * 100.0
        # Apply invert flags (fix circles caused by reversed motor wiring)
        if cfg.INVERT_LEFT:  d_l = -d_l
        if cfg.INVERT_RIGHT: d_r = -d_r
        self._l.set_duty(d_l)
        self._r.set_duty(d_r)

    def set_individual(self, d_l: float, d_r: float):
        """Set duties directly for motor testing."""
        if cfg.INVERT_LEFT:  d_l = -d_l
        if cfg.INVERT_RIGHT: d_r = -d_r
        self._l.set_duty(d_l)
        self._r.set_duty(d_r)

    def stop(self):
        self._l.stop()
        self._r.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  USB serial helpers
# ─────────────────────────────────────────────────────────────────────────────

_rx_buf = ""

def _try_readline():
    """Non-blocking read from USB serial.  Returns stripped line or None."""
    global _rx_buf
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r"):
            line    = _rx_buf.strip()
            _rx_buf = ""
            return line if line else None
        _rx_buf += ch
    return None


def _send(data: dict):
    sys.stdout.write(ujson.dumps(data) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run():
    enc_l = Encoder(cfg.ENC_L_A, cfg.ENC_L_B)
    enc_r = Encoder(cfg.ENC_R_A, cfg.ENC_R_B)
    drive = DriveTrain()
    odometry = Odometry()

    last_broadcast = time.ticks_ms()
    last_cmd       = time.ticks_ms()

    while True:
        now = time.ticks_ms()

        # Receive velocity command from RB3
        line = _try_readline()
        if line:
            try:
                pkt = ujson.loads(line)
                if "d_l" in pkt or "d_r" in pkt:
                    # Individual motor test command
                    drive.set_individual(float(pkt.get("d_l", 0.0)),
                                         float(pkt.get("d_r", 0.0)))
                else:
                    drive.set_velocity(float(pkt.get("v_lin", 0.0)),
                                       float(pkt.get("v_ang", 0.0)))
                last_cmd = now
            except (ValueError, KeyError):
                pass

        # Watchdog — stop if RB3 goes silent
        if time.ticks_diff(now, last_cmd) > cfg.CMD_TIMEOUT_MS:
            drive.stop()

        # Broadcast encoder snapshot at 50 Hz
        if time.ticks_diff(now, last_broadcast) >= cfg.BROADCAST_MS:
            enc_l_ticks = enc_l.pop()
            enc_r_ticks = enc_r.pop()   
            odometry.update_odometry(enc_l_ticks, enc_r_ticks)            
            
            _send({
                "pos_x": odometry.x,
                "pos_y": odometry.y,
                "pos_theta": (odometry.theta/math.pi)*180,
                "tpr":    cfg.TICKS_PER_REV,    # 990
                "ts":     now / 1000.0,
            })
            last_broadcast = now

        time.sleep_us(200)


run()


