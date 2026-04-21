"""
config.py — All tunable constants for the TEQUILA pipeline.

All values here can be overridden from the CLI via main.py.
Edit this file to change defaults permanently.
"""

# ── Depth model ───────────────────────────────────────────────────────────────
# DEPTH_METRIC = True  → use a Hugging Face metric-depth model.
#   The model outputs real distances in metres with no per-frame scale drift.
#   anchor_depth_scale() becomes a no-op.
#
# DEPTH_METRIC = False → fall back to the local relative-depth ViT-S model.
#   Faster (no download), but requires the per-frame scale anchor to avoid
#   fan/stretch artefacts in the accumulated map.
#
# Available metric models (indoor):
#   Small  → depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf   (24.8M params, fast)
#   Large  → depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf   (335M params, accurate)
# Outdoor/mixed:
#   Large  → depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
DEPTH_METRIC   = True
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
CHECKPOINT     = r"checkpoints\depth_anything_v2_vits.pth"  # relative-depth fallback

INFER_WIDTH    = 1280    # inference image width (try 640 on CPU for speed)
MAX_DEPTH_M    = 10.0    # clip anything beyond this distance (metres)
FOV_H_DEG      = 70.0   # horizontal field of view of the camera (degrees)

# ── Capture / timing ──────────────────────────────────────────────────────────
CAPTURE_INTERVAL_S = 3.0   # seconds between webcam captures
FRAME_SKIP         = 30    # process every Nth video frame (video-file mode)
NAV_INTERVAL_S     = 6.0   # seconds between navmesh recomputes
PORT               = 8080  # viser web-viewer port

# ── Segmentation (unused in default pipeline, kept for reference) ─────────────
SAT_THRESH    = 22
VAL_THRESH    = 45
BG_DILATE_PX  = 10
BG_FILL_COLOR = (114, 114, 114)

# ── Point-cloud display ───────────────────────────────────────────────────────
VOXEL_SIZE       = 0.02   # voxel grid cell size (metres) for display cloud
SOR_NB_NEIGHBORS = 20     # Statistical Outlier Removal: neighbours to consider
SOR_STD_RATIO    = 2.0    # SOR: keep points within mean + ratio×std

# ── Flying-pixel / depth-edge removal ────────────────────────────────────────
# Depth models bleed intermediate values onto boundary pixels between a near
# object and a far background.  Back-projected, these become rays or spikes
# extending behind every object edge.
#
# Fix: any pixel whose depth exceeds the local neighbourhood minimum by more
# than EDGE_THRESHOLD (relative) is on the far side of an edge — mask it out.
EDGE_WINDOW_PX = 11    # erosion kernel size (pixels); larger → catches wider edges
EDGE_THRESHOLD = 0.20  # relative depth jump limit:
                        #   pixel removed if depth > local_min × (1 + threshold)

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

# ── Depth scale normalisation (relative-depth mode only) ─────────────────────
# Monocular relative depth is normalised per-frame, causing scale drift across
# frames and fan/stretch artefacts in the accumulated map.
#
# Auto-fix: on the first frame the floor is detected and the camera-to-floor
# distance is recorded as a baseline.  Every subsequent frame is rescaled so
# its floor appears at the same height — fixing scale drift without --cam-height.
#
# Metric fix: DEPTH_METRIC = True eliminates the problem entirely.
CAM_HEIGHT_M      = 0.0    # 0 = auto-baseline; >0 = user-supplied metric height
SCALE_RANSAC_ITER = 300    # RANSAC iterations for quick floor detection
SCALE_INLIER_DIST = 0.12   # inlier band (metres) for scale RANSAC
SCALE_MIN_INLIERS = 20     # abort if fewer inliers than this
SCALE_CLAMP       = (0.65, 1.45)  # reject corrections outside this band
                                   # (avoids latching onto walls / tables)

# ── Point accumulation ────────────────────────────────────────────────────────
NAV_ACCUM_VOXEL   = VOXEL_SIZE * 3  # dedup voxel for accumulated cloud
NAV_ACCUM_MAX_PTS = 500_000         # hard cap to prevent unbounded memory growth
ACCUM_ENABLED     = True            # False = single-frame mode (product/turntable)
MAP_MAX_DEPTH_M   = 4.0             # only accumulate points within this distance
                                     # (limits fan-arm length when alignment drifts)

# ── ICP frame alignment (fallback when ORB+PnP fails) ────────────────────────
ICP_MAX_DIST    = 1.0    # max correspondence distance (metres)
ICP_MAX_ITER    = 50     # iterations per frame
ICP_FITNESS_MIN = 0.25   # min inlier fraction to accept alignment
ICP_SUBSAMPLE   = 3000   # subsample size per cloud (for speed)
ICP_MAX_SHIFT_M = 2.0    # max translation per frame — rejects wild drifts
ICP_MAX_ROT_DEG = 45.0   # max rotation per frame  — rejects wild spins

# ── Visual Odometry (ORB + PnP) — primary alignment ──────────────────────────
VO_MAX_FEATURES = 2000   # ORB features per frame
VO_RATIO_TEST   = 0.75   # Lowe's ratio test threshold for match filtering
VO_MIN_INLIERS  = 12     # minimum PnP inliers to accept alignment
VO_RANSAC_ITER  = 200    # solvePnPRansac iterations
VO_REPROJ_ERR   = 4.0    # reprojection error threshold (pixels)
VO_MAX_SHIFT_M  = 2.0    # max translation per frame (metres)
VO_MAX_ROT_DEG  = 45.0   # max rotation per frame (degrees)
VO_MIN_SHIFT_M  = 0.03   # min translation to accumulate; below this = duplicate view
