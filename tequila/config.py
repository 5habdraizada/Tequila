"""
config.py — All tunable constants for the TEQUILA pipeline.

All values here can be overridden from the CLI via main.py.
Edit this file to change defaults permanently.
"""

# ── Depth model ───────────────────────────────────────────────────────────────
# Hugging Face metric-depth model — outputs real distances in metres.
# Available models (indoor):
#   Small  → depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf   (24.8M params, fast)
#   Large  → depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf   (335M params, accurate)
# Outdoor/mixed:
#   Large  → depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

INFER_WIDTH    = 1280    # inference image width (try 640 on CPU for speed)
MAX_DEPTH_M    = 10.0    # clip anything beyond this distance (metres)
FOV_H_DEG      = 70.0   # horizontal field of view of the camera (degrees)
                        # (pinhole back-projection FOV; ignored when FISHEYE=True)

# ── Fisheye undistortion ──────────────────────────────────────────────────────
# When True, each frame is undistorted from a fisheye to a rectilinear image
# before depth inference, so the pinhole back-projection (r = f·tanθ) becomes
# valid.  The lens is modelled as an equidistant fisheye (r = f·θ); near the
# centre all fisheye models agree, and the paraxial focal is fixed by the optics
# (focal_mm / pixel_pitch), so the central undistorted region is well-determined.
# After undistortion the back-projection focal is the OUTPUT rectilinear focal
# derived from UNDISTORT_FOV_DEG.
FISHEYE              = False
FISHEYE_FOCAL_MM     = 1.42     # lens focal length (mm)
SENSOR_PIXEL_UM      = 1.55     # sensor pixel pitch (µm)
SENSOR_FULL_WIDTH_PX = 4056     # full sensor width (px); capture assumed full-width
UNDISTORT_FOV_DEG    = 90.0     # output rectilinear HFOV — keep below the lens's
                                # true FOV (~128°) so there are no black borders
# Measured calibration (preferred over the equidistant approximation above).
# Point this at the .npz from tools/camera_calibration.py (keys camMatrix,
# distCoeff).  Empty string → fall back to the equidistant model.
FISHEYE_CALIB_NPZ    = ""
FISHEYE_CALIB_WH     = (1280, 720)   # resolution the calibration was captured at

# ── Capture / timing ──────────────────────────────────────────────────────────
CAPTURE_INTERVAL_S = 3.0   # seconds between webcam captures
FRAME_SKIP         = 30    # process every Nth video frame (video-file mode)
NAV_INTERVAL_S     = 1.0   # seconds between navmesh recomputes
PORT               = 8080  # viser web-viewer port

# ── Segmentation (unused in default pipeline, kept for reference) ─────────────
SAT_THRESH    = 22
VAL_THRESH    = 45
BG_DILATE_PX  = 10
BG_FILL_COLOR = (114, 114, 114)

# ── Point-cloud / Gaussian splat display ─────────────────────────────────────
VOXEL_SIZE       = 0.02   # voxel grid cell size (metres) for display cloud
SOR_NB_NEIGHBORS = 20     # Statistical Outlier Removal: neighbours to consider
SOR_STD_RATIO    = 2.0    # SOR: keep points within mean + ratio×std

# Gaussian splat rendering — replaces the raw point cloud when USE_SPLATS=True.
# Each accumulated map point is rendered as a small isotropic 3-D Gaussian blob
# using viser's built-in splat renderer (alpha-sorted on the GPU).
USE_SPLATS    = False   # True = splat render; False = raw point cloud
                        # Raw points render crisp and solid; splats can look
                        # blobby / washed-out depending on the viser version.
SPLAT_RADIUS  = 0.012   # splat half-width in metres (only used if USE_SPLATS)
SPLAT_OPACITY = 0.9     # per-splat opacity in [0, 1]
POINT_SIZE    = 0.012   # raw point cloud dot size (metres) when USE_SPLATS=False

# ── Flying-pixel / depth-edge removal ────────────────────────────────────────
# Three-stage pipeline:
#
#  1. Erosion mask  — pixels whose depth exceeds the local neighbourhood
#                    minimum by more than EDGE_THRESHOLD are flying pixels.
#
#  2. Gradient mask — pixels where the local depth gradient is large relative
#                    to the depth value itself are at a depth discontinuity.
#                    GRAD_THRESHOLD is relative: grad_mag / depth > threshold.
#                    Typical value 0.3–0.6 (lower = remove more edge pixels).
#
#  3. Dilation     — the combined bad-pixel mask is dilated by EDGE_DILATE_PX
#                    pixels to widen the removal band around each edge, catching
#                    partially-blended transition pixels.
EDGE_WINDOW_PX  = 11    # erosion kernel size (pixels)
EDGE_THRESHOLD  = 0.20  # relative depth jump to flag as flying pixel
GRAD_THRESHOLD  = 0.40  # relative gradient limit: grad/depth > this → edge pixel
EDGE_DILATE_PX  = 2     # pixels to dilate the bad-pixel mask (0 = no dilation)

# ── Occlusion / wall helpers (available but not used in default pipeline) ─────
OCCLUSION_TOLERANCE     = 0.20
OCCLUSION_WIN_PX        = 20
REMOVE_WALLS            = True
WALL_DEPTH_PERCENTILE   = 78
WALL_FLATNESS_THRESHOLD = 0.018
WALL_LOCAL_RADIUS       = 20

# ── Floor detection — GPP (Ground Principal Plane) ────────────────────────────
# Divides the point cloud into angular sectors, fits a PCA plane per sector,
# finds the largest consensus group, then refines with least-squares.
UP_AXIS            = "y"
MAX_FLOOR_TILT_DEG = 30.0   # max angle between floor normal and up-axis (degrees)
FLOOR_RANSAC_DIST  = 0.20   # inlier band width (metres) for plane membership
                             # wider = more floor points pulled into floor_idx
                             # → fewer floor points leaking into obstacle cloud
MIN_FLOOR_POINTS   = 200    # minimum inliers to accept a floor plane
GPP_N_SECTORS      = 8      # angular sectors around the camera (8 = 45° each)
GPP_MIN_SECTOR_PTS = 10     # ignore sectors with fewer points
GPP_CONSENSUS_DEG  = 15.0   # max normal-angle difference (°) for two sectors to agree
GPP_MIN_SECTORS    = 1      # min consensus sectors required to accept a plane
                             # (1 = lenient, works for product/single-room videos)

# ── Navmesh grid ──────────────────────────────────────────────────────────────
NAV_PATH_ONLY     = True   # True = show only obstacles + planned path (clean);
                           # False = also draw the full free/blocked node grid,
                           # edge web, and trajectory trail (debug view).
NODE_SPACING      = 0.15   # metres between navmesh grid nodes
NODE_ABOVE_FLOOR  = 0.04   # metres above floor for each node (avoids z-fighting)
OBS_CLEARANCE_R   = 0.40   # obstacle clearance radius (metres) per node
OBS_HEIGHT_MIN    = 0.18   # ignore obstacles below this height above floor
                             # must be > floor noise + VO drift (~10-15 cm)
OBS_HEIGHT_MAX    = 0.80   # ignore obstacles above this height above floor
OBS_VOXEL_SIZE    = 0.05   # voxel size for obstacle deduplication
OBS_DENOISE_NB    = 10     # SOR neighbours for obstacle denoising
OBS_DENOISE_STD   = 1.5    # SOR std-ratio for obstacle denoising
FLOOR_PROX_FACTOR = 1      # remove obstacle pts within this × NODE_SPACING of floor
EDGE_MAX_DIST     = NODE_SPACING * 2.0  # max edge length between free nodes
EDGE_CHECK_STEPS  = 20     # line-of-sight samples per edge (more → fewer clip-throughs)

# ── Point accumulation ────────────────────────────────────────────────────────
# Planar-motion lock — this is a ground robot driving on a flat floor, so the
# true camera motion is only 3-DOF (x, z, yaw).  After each frame the cumulative
# pose is projected back onto that plane: pitch/roll are removed and the camera
# height is pinned to the floor-0 level.  This stops the per-frame VO error in
# pitch/roll/vertical-translation from accumulating, which is what otherwise
# stacks the floor at different heights across the map.
# Assumes the camera is mounted ~level (forward-facing); if it is tilted down,
# the lock axis would need to come from the detected floor normal instead.
PLANAR_LOCK       = True
NAV_ACCUM_VOXEL   = VOXEL_SIZE * 3  # dedup voxel for accumulated cloud
NAV_ACCUM_MAX_PTS = 500_000         # hard cap to prevent unbounded memory growth
ACCUM_ENABLED     = True            # False = single-frame mode (product/turntable)
MAP_MAX_DEPTH_M   = 3.0             # only accumulate points within this distance
                                     # (limits fan-arm length when alignment drifts)
NAV_MAX_DEPTH_M   = 5.0             # cap depth used for the navmesh cloud — distant
                                     # depth error is multiplied by range, so far
                                     # points become spurious obstacles / floor noise

# ── ICP frame alignment (fallback when ORB+PnP fails) ────────────────────────
ICP_MAX_DIST    = 1.0    # max correspondence distance (metres)
ICP_MAX_ITER    = 50     # iterations per frame
ICP_FITNESS_MIN = 0.35   # min inlier fraction to accept alignment  (was 0.25)
ICP_SUBSAMPLE   = 3000   # subsample size per cloud (for speed)
ICP_MAX_SHIFT_M = 2.0    # max translation per frame — rejects wild drifts
ICP_MAX_ROT_DEG = 15.0   # max rotation per frame  — rejects wild spins  (was 45.0)

# ── Visual Odometry (SIFT + PnP) — primary alignment ─────────────────────────
VO_MAX_FEATURES = 2000   # SIFT features per frame
VO_RATIO_TEST   = 0.75   # Lowe's ratio test threshold for match filtering
VO_MIN_INLIERS  = 20     # minimum PnP inliers to accept alignment  (was 12)
VO_RANSAC_ITER  = 200    # solvePnPRansac iterations
VO_REPROJ_ERR   = 4.0    # reprojection error threshold (pixels)
VO_MAX_SHIFT_M  = 2.0    # max translation per frame (metres)
VO_MAX_ROT_DEG  = 15.0   # max rotation per frame (degrees)  (was 45.0)
VO_MIN_SHIFT_M  = 0.03   # min translation to accumulate; below this = duplicate view
