"""
rb3/config.py — RB3-specific overrides for the TEQUILA pipeline.
All values here are merged on top of tequila/config.py at startup.
"""

# Pico USB serial connection
# Set to None to auto-detect the Pico (recommended).
# Or set explicitly e.g. "/dev/ttyACM0" if auto-detect fails.
PICO_PORT = None   # None = auto-detect by USB vendor ID (2E8A = Raspberry Pi)

# Robot geometry (must match pi/config.py)
WHEEL_RADIUS_M = 0.033   # metres  0.0325
WHEEL_BASE_M   = 0.172    # metres

# Odometry-driven map stitching (level-1 fusion)
# When True, each camera frame is placed in the map using the robot's measured
# wheel-odometry pose (pos_x/pos_y/pos_theta) instead of visual odometry.
# This is the quick experiment: if wheel odometry alone gives a clean map, VO is
# not needed.  If the map shears on turns/slip, move to level 3 (odom as VO prior).
ODOM_FUSION = True

# Visual-odometry drift correction of the EKF.  When True, VO measures the
# actual camera motion each frame and nudges the EKF (odometry + vision fused).
# When False, the map stitches from PURE wheel+gyro odometry with no visual
# correction — useful for isolating how good the raw odometry is on its own.
# Also live-toggleable from the "VO drift correction" checkbox in the viewer.
USE_VO_CORRECTION = True

# Camera mount offset relative to the wheel/encoder centre, expressed in the
# robot body frame (metres): how far the camera lens sits forward / to the left /
# above the centre of the wheelbase.  Measure this once for your build — it
# matters because the camera swings on a lever arm when the robot turns.
CAM_MOUNT_FWD  = 0.03   # +forward
CAM_MOUNT_LEFT = 0.04   # +left
CAM_MOUNT_UP   = 0.195   # +up

# RB3 onboard IMU (Qualcomm SDF / see_workhorse)
# Gyroscope yaw rate — used in EKF predict instead of the encoder differential.
# The IMU is far more accurate for rotation than wheel-speed difference, which
# suffers from wheel-radius mismatch and slip on turns.
#
# GYRO_SCALE:    unit conversion factor.  see_workhorse reports in rad/s → 1.0.
#                Set to math.pi/180 if it reports in deg/s.
# GYRO_YAW_SIGN: 1 if a left turn gives a positive gyro_z reading, -1 otherwise.
#                Quick test: command the robot to turn left and check the sign
#                printed by the "[EKF/VO]" lines.
GYRO_SCALE    = 1.0   # rad/s (Qualcomm SDF default)
GYRO_YAW_SIGN = 1     # flip to -1 if yaw direction appears inverted
# Safety clamp applied to the gyro yaw rate before it enters the EKF.
# Catches unit mismatches (e.g. deg/s read as rad/s) and sensor glitches.
# Set to roughly 2× the robot's real maximum angular speed.
MAX_GYRO_RATE = 3.0   # rad/s  (~172 °/s; RB3 max is 1 rad/s so 3× headroom)

# Accelerometer forward axis — data is collected and available via
# get_odometry()["accel_fwd"] for diagnostics and future use.
# NOTE: raw accelerometer output includes gravity (≈9.81 m/s²) whenever the
# forward axis is not perfectly level.  This constant bias accumulates into
# an unbounded velocity if integrated without subtracting the static offset
# measured while the robot is stationary.  Accel integration is therefore
# disabled in the EKF until a calibration step is added; only the gyro is
# fused right now.
ACCEL_FWD_IDX  = 0   # X assumed forward on the RB3 Gen 2 mounting
ACCEL_FWD_SIGN = 1   # positive = forward; flip to -1 if wrong

# RB3 Gen 2 camera
# The RB3 Gen 2 exposes cameras as V4L2 devices.
# Run `ls /dev/video*` on the RB3 to confirm the index.
CAMERA_INDEX = 0            # /dev/video0  (change if needed)
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# Horizontal field of view of the capture, in degrees.
# Thundercomm kit lens (1.42 mm focal) on the 1/2.3" IMX577 (1.55 µm pixels):
#   sensor width 4056·1.55µm = 6.287 mm  →  HFOV = 2·atan(6.287/(2·1.42)) ≈ 128°
# This assumes the 1280×720 capture uses the FULL sensor width (vertical crop to
# 16:9).  If the live FOV slider's sweet-spot is far from 128, the ISP is
# cropping horizontally instead — set this to whatever the slider lands on.
# NOTE: this is a *fisheye* lens.  128° is only a pinhole approximation that is
# correct at the image centre; the edges follow r=f·θ, not r=f·tanθ, so they
# still distort.  With FISHEYE=True below this value is unused (the frame is
# undistorted to a true rectilinear image first).
FOV_H_DEG = 128.0

# Fisheye undistortion (Thundercomm kit lens on the IMX577)
# Undistort each frame from fisheye to rectilinear before depth, so the pinhole
# back-projection is valid (kills the radial fan-splay in the map).
# Equidistant model from the lens optics:
#   f_full = 1.42 mm / 1.55 µm ≈ 916 px at the 4056-px full sensor width.
FISHEYE              = True
FISHEYE_FOCAL_MM     = 1.42     # lens focal length (mm)
SENSOR_PIXEL_UM      = 1.55     # sensor pixel pitch (µm)
SENSOR_FULL_WIDTH_PX = 4056     # full sensor width (px); 1280 capture = full-width
UNDISTORT_FOV_DEG    = 90.0     # output rectilinear HFOV (tune live; keep < ~128°)
# Measured fisheye calibration from tools/camera_calibration.py.  Once the .npz
# exists (repo root or cwd) it overrides the equidistant approximation above.
FISHEYE_CALIB_NPZ    = "tools/fishEye_RB3_CameraCalibration_fisheye.npz"
FISHEYE_CALIB_WH     = (1280, 720)   # capture_calibration.py grabs at 1280×720

# Navigation
# Maximum linear and angular speed sent to the Pi
MAX_V_LIN  = 0.1           # m/s  — conservative for first runs
MAX_V_ANG  = 0.8           # rad/s

# Look-ahead distance for pure-pursuit path following
LOOKAHEAD_M = 0.50          # metres ahead on the path to aim for

# Stop if next waypoint is closer than this (waypoint reached)
WP_REACHED_M = 0.20         # metres

# Depth model (override tequila/config.py defaults)
# Use the small indoor metric model — fast enough on RB3 CPU
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
INFER_WIDTH    = 640         # lower res for speed on embedded CPU
MAP_MAX_DEPTH_M = 2.0        # display cloud: keep only nearby, reliable geometry
                             # (far points smear into fan arms)
NAV_MAX_DEPTH_M = 4.0        # navmesh cloud: a bit further for look-ahead, but
                             # still capped to drop far obstacle/floor noise

# TSDF volumetric fusion
# Average overlapping depth observations into a voxel volume (noise cancels)
# instead of accumulating raw points.  Needs Open3D on the RB3
# (`pip install open3d`); if it's missing the pipeline falls back automatically.
USE_TSDF     = True
TSDF_VOXEL_M = 0.03          # voxel size (m) — smaller = finer + slower/more RAM
TSDF_TRUNC_M = 0.12          # signed-distance truncation (~4 voxels)

# Timing / capture
MIN_FRAME_BRIGHTNESS = 10.0  # skip near-black warm-up frames (mean < this)
NAV_INTERVAL_S       = 2.5   # recompute the navmesh less often so its (blocking)
                             # pass doesn't stall the map display as often

# Viser viewer
PORT = 8080                  # open http://<rb3-ip>:8080 on any browser on the LAN


