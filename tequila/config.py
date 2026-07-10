"""All tunable constants for the TEQUILA pipeline. Overridable from the CLI via main.py."""

# Depth model — Hugging Face metric-depth model, outputs real distances in metres.
# Indoor: Small (24.8M params, fast) / Large (335M params, accurate); also an
# Outdoor-Large variant. See depth-anything/Depth-Anything-V2-Metric-* on the Hub.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

INFER_WIDTH    = 1280    # inference image width (try 640 on CPU for speed)
MAX_DEPTH_M    = 10.0    # clip anything beyond this distance (metres)
FOV_H_DEG      = 70.0    # horizontal FOV in degrees; ignored when FISHEYE=True

# Fisheye undistortion — rectify to rectilinear before depth inference so the
# pinhole back-projection (r = f·tanθ) is valid. Modelled as an equidistant
# fisheye (r = f·θ); the back-projection focal is derived from UNDISTORT_FOV_DEG.
FISHEYE              = False
FISHEYE_FOCAL_MM     = 1.42     # lens focal length (mm)
SENSOR_PIXEL_UM      = 1.55     # sensor pixel pitch (µm)
SENSOR_FULL_WIDTH_PX = 4056     # full sensor width (px); capture assumed full-width
UNDISTORT_FOV_DEG    = 90.0     # output rectilinear HFOV, kept below the lens's ~128°
# Measured calibration (preferred over the equidistant approximation above).
# Point at the .npz from tools/camera_calibration.py (keys camMatrix, distCoeff).
FISHEYE_CALIB_NPZ    = ""
FISHEYE_CALIB_WH     = (1280, 720)   # resolution the calibration was captured at

# Capture / timing
CAPTURE_INTERVAL_S = 3.0   # seconds between webcam captures
FRAME_SKIP         = 30    # process every Nth video frame (video-file mode)
NAV_INTERVAL_S     = 1.0   # seconds between navmesh recomputes
MAP_INTERVAL_S     = 0.5   # seconds between map refreshes; decoupled from
                           # NAV_INTERVAL_S so the cloud updates without waiting
                           # on the (slower) navmesh recompute
MIN_FRAME_BRIGHTNESS = 0.0 # skip webcam frames below this mean brightness
                           # (camera warm-up / black frames); 0 = disabled
PORT               = 8080  # viser web-viewer port

# Segmentation (unused in default pipeline, kept for reference)
SAT_THRESH    = 22
VAL_THRESH    = 45
BG_DILATE_PX  = 10
BG_FILL_COLOR = (114, 114, 114)

# Point-cloud / Gaussian splat display
VOXEL_SIZE       = 0.02   # voxel grid cell size (metres) for display cloud
SOR_NB_NEIGHBORS = 20     # Statistical Outlier Removal: neighbours to consider
SOR_STD_RATIO    = 2.0    # SOR: keep points within mean + ratio×std

# Gaussian splat rendering — replaces the raw point cloud when USE_SPLATS=True,
# rendering each map point as a small isotropic 3-D Gaussian via viser's splat renderer.
USE_SPLATS    = False   # raw points render crisper; splats can look washed-out
SPLAT_RADIUS  = 0.012   # splat half-width in metres (only used if USE_SPLATS)
SPLAT_OPACITY = 0.9     # per-splat opacity in [0, 1]
POINT_SIZE    = 0.012   # raw point cloud dot size (metres) when USE_SPLATS=False

# Flying-pixel / depth-edge removal — see depth_edge_mask() for the erosion +
# gradient + dilation pipeline these feed into.
EDGE_WINDOW_PX  = 11    # erosion kernel size (pixels)
EDGE_THRESHOLD  = 0.20  # relative depth jump to flag as flying pixel
GRAD_THRESHOLD  = 0.40  # relative gradient limit: grad/depth > this → edge pixel
EDGE_DILATE_PX  = 2     # pixels to dilate the bad-pixel mask (0 = no dilation)

# Occlusion / wall helpers (available but not used in default pipeline)
OCCLUSION_TOLERANCE     = 0.20
OCCLUSION_WIN_PX        = 20
REMOVE_WALLS            = True
WALL_DEPTH_PERCENTILE   = 78
WALL_FLATNESS_THRESHOLD = 0.018
WALL_LOCAL_RADIUS       = 20

# Floor detection — GPP (Ground Principal Plane): divides the cloud into angular
# sectors, fits a PCA plane per sector, finds the largest consensus group, then
# refines with least-squares.
UP_AXIS            = "y"
MAX_FLOOR_TILT_DEG = 30.0   # max angle between floor normal and up-axis (degrees)
FLOOR_RANSAC_DIST  = 0.20   # inlier band width (metres); wider pulls more floor
                             # points in, keeping them out of the obstacle cloud
MIN_FLOOR_POINTS   = 200    # minimum inliers to accept a floor plane
GPP_N_SECTORS      = 8      # angular sectors around the camera (8 = 45° each)
GPP_MIN_SECTOR_PTS = 10     # ignore sectors with fewer points
GPP_CONSENSUS_DEG  = 15.0   # max normal-angle difference (°) for two sectors to agree
GPP_MIN_SECTORS    = 1      # min consensus sectors to accept a plane (1 = lenient)

# Navmesh grid
NAV_PATH_ONLY     = True   # True = obstacles + path only; False = also draw the
                           # full free/blocked grid, edge web, trail (debug view)
NAV_GOAL_REACHED_M = 0.5   # exploration goal commitment: keep heading to the
                           # current goal until within this distance (or it becomes
                           # unreachable) instead of re-picking every recompute
NODE_SPACING      = 0.15   # metres between navmesh grid nodes
NODE_ABOVE_FLOOR  = 0.04   # metres above floor for each node (avoids z-fighting)
OBS_CLEARANCE_R   = 0.40   # obstacle clearance radius (metres) per node
OBS_HEIGHT_MIN    = 0.18   # ignore obstacles below this height above floor
                             # (must exceed floor noise + VO drift, ~10-15 cm)
OBS_HEIGHT_MAX    = 0.80   # ignore obstacles above this height above floor
OBS_VOXEL_SIZE    = 0.05   # voxel size for obstacle deduplication
OBS_DENOISE_NB    = 10     # SOR neighbours for obstacle denoising
OBS_DENOISE_STD   = 1.5    # SOR std-ratio for obstacle denoising
FLOOR_PROX_FACTOR = 1      # remove obstacle pts within this × NODE_SPACING of floor
EDGE_MAX_DIST     = NODE_SPACING * 2.0  # max edge length between free nodes
EDGE_CHECK_STEPS  = 20     # line-of-sight samples per edge (more → fewer clip-throughs)

# Point accumulation. Planar-motion lock: the robot drives on a flat floor, so
# true camera motion is only 3-DOF (x, z, yaw). Each frame's cumulative pose is
# projected back onto that plane — pitch/roll removed, height pinned to floor-0 —
# stopping per-frame VO error from stacking the floor at different heights.
# Assumes a level, forward-facing camera mount.
PLANAR_LOCK       = True

# Volumetric TSDF fusion — integrate depth into a voxel volume that averages
# overlapping observations instead of piling up raw points. Requires Open3D;
# falls back to point accumulation if missing.
USE_TSDF          = False
TSDF_VOXEL_M      = 0.03   # TSDF voxel size (metres) — smaller = finer + slower
TSDF_TRUNC_M      = 0.12   # signed-distance truncation (≈4 voxels)

NAV_ACCUM_VOXEL   = VOXEL_SIZE * 3  # dedup voxel for accumulated cloud
NAV_ACCUM_MAX_PTS = 500_000         # hard cap to prevent unbounded memory growth
ACCUM_ENABLED     = True            # False = single-frame mode (product/turntable)
MAP_MAX_DEPTH_M   = 3.0             # accumulate only within this distance
                                     # (limits fan-arm length when alignment drifts)
NAV_MAX_DEPTH_M   = 5.0             # cap depth for the navmesh cloud — distant
                                     # depth error scales with range into spurious
                                     # obstacles / floor noise

# ICP frame alignment (fallback when ORB+PnP fails)
ICP_MAX_DIST    = 1.0    # max correspondence distance (metres)
ICP_MAX_ITER    = 50     # iterations per frame
ICP_FITNESS_MIN = 0.35   # min inlier fraction to accept alignment
ICP_SUBSAMPLE   = 3000   # subsample size per cloud (for speed)
ICP_MAX_SHIFT_M = 2.0    # max translation per frame — rejects wild drifts
ICP_MAX_ROT_DEG = 15.0   # max rotation per frame — rejects wild spins

# Visual Odometry (SIFT + PnP) — primary alignment
VO_MAX_FEATURES = 2000   # SIFT features per frame
VO_RATIO_TEST   = 0.75   # Lowe's ratio test threshold for match filtering
VO_MIN_INLIERS  = 20     # minimum PnP inliers to accept alignment
VO_RANSAC_ITER  = 200    # solvePnPRansac iterations
VO_REPROJ_ERR   = 4.0    # reprojection error threshold (pixels)
VO_MAX_SHIFT_M  = 2.0    # max translation per frame (metres)
VO_MAX_ROT_DEG  = 15.0   # max rotation per frame (degrees)
VO_MIN_SHIFT_M  = 0.03   # min translation to accumulate; below this = duplicate view
