"""
rb3/config.py — RB3-specific overrides for the TEQUILA pipeline.
All values here are merged on top of tequila/config.py at startup.
"""

# ── Pico USB serial connection ────────────────────────────────────────────────
# Set to None to auto-detect the Pico (recommended).
# Or set explicitly e.g. "/dev/ttyACM0" if auto-detect fails.
PICO_PORT = None   # None = auto-detect by USB vendor ID (2E8A = Raspberry Pi)

# ── Robot geometry (must match pi/config.py) ──────────────────────────────────
WHEEL_RADIUS_M = 0.0325   # metres  0.0325
WHEEL_BASE_M   = 0.190    # metres

# ── Odometry-driven map stitching (level-1 fusion) ────────────────────────────
# When True, each camera frame is placed in the map using the robot's measured
# wheel-odometry pose (pos_x/pos_y/pos_theta) instead of visual odometry.
# This is the quick experiment: if wheel odometry alone gives a clean map, VO is
# not needed.  If the map shears on turns/slip, move to level 3 (odom as VO prior).
ODOM_FUSION = True

# Camera mount offset relative to the wheel/encoder centre, expressed in the
# robot body frame (metres): how far the camera lens sits forward / to the left /
# above the centre of the wheelbase.  Measure this once for your build — it
# matters because the camera swings on a lever arm when the robot turns.
CAM_MOUNT_FWD  = 0.03   # +forward
CAM_MOUNT_LEFT = 0.04   # +left
CAM_MOUNT_UP   = 0.195   # +up

# ── RB3 Gen 2 camera ─────────────────────────────────────────────────────────
# The RB3 Gen 2 exposes cameras as V4L2 devices.
# Run `ls /dev/video*` on the RB3 to confirm the index.
CAMERA_INDEX = 0            # /dev/video0  (change if needed)
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# ── Navigation ────────────────────────────────────────────────────────────────
# Maximum linear and angular speed sent to the Pi
MAX_V_LIN  = 0.1           # m/s  — conservative for first runs
MAX_V_ANG  = 1.0           # rad/s

# Look-ahead distance for pure-pursuit path following
LOOKAHEAD_M = 0.50          # metres ahead on the path to aim for

# Stop if next waypoint is closer than this (waypoint reached)
WP_REACHED_M = 0.20         # metres

# ── Depth model (override tequila/config.py defaults) ────────────────────────
# Use the small indoor metric model — fast enough on RB3 CPU
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
INFER_WIDTH    = 640         # lower res for speed on embedded CPU
MAP_MAX_DEPTH_M = 3.0

# ── Viser viewer ─────────────────────────────────────────────────────────────
PORT = 8080                  # open http://<rb3-ip>:8080 on any browser on the LAN


