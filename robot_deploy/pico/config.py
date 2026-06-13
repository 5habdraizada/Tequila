# pico/config.py — Real robot constants

import math

# ── Encoder ───────────────────────────────────────────────────────────────────
CPR   = 11    # encoder cycles per shaft revolution
RATIO = 45    # gearbox ratio

COUNTS_PER_WHEEL_REV = CPR * RATIO * 2      # 990 (both edges counted)
TICKS_PER_REV        = COUNTS_PER_WHEEL_REV # alias used by bridge protocol

# ── Robot geometry ────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0335   # metres
WHEEL_BASE   = 0.190    # metres (wheel contact point to wheel contact point)

WHEEL_CIRC   = 2 * math.pi * WHEEL_RADIUS   # ~0.2105 m
COUNTS_PER_M = COUNTS_PER_WHEEL_REV / WHEEL_CIRC  # ~4703

# ── Encoder pins (GPIO) ───────────────────────────────────────────────────────
ENC_R_A = 11
ENC_R_B = 10
ENC_L_A = 12
ENC_L_B = 13

# ── Motor pins — dual PWM driver (one pin forward, one pin backward) ──────────
# Forward:  P1 = duty,  P2 = 0
# Backward: P1 = 0,     P2 = duty
# Brake:    P1 = full,  P2 = full
M_R1 = 7
M_R2 = 6
M_L1 = 9
M_L2 = 8
MOTOR_PWM_FREQ = 20000   # Hz (matches your existing setup)

# ── Max speed ─────────────────────────────────────────────────────────────────
MAX_RPM = 150   # motor output shaft RPM at full duty — measure or check datasheet

# ── Motor direction — set True if that motor runs backwards ───────────────────
# Symptom: robot spins in circles when it should go straight → flip one of these
INVERT_LEFT  = True
INVERT_RIGHT = True

# ── Timing ────────────────────────────────────────────────────────────────────
BROADCAST_MS   = 20    # send encoder packet every 20 ms = 50 Hz
CMD_TIMEOUT_MS = 500   # stop motors if silent for 500 ms

