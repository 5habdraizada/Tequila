"""
video_navmesh.py — TEQUILA Live Video + Navigation Mesh

Combines the live Gaussian splat pipeline with real-time navmesh generation.
Each frame updates the 3D scene; the navmesh recomputes on a slower interval.

Sources
-------
  python video_navmesh.py                         # default webcam
  python video_navmesh.py --source 1              # second camera
  python video_navmesh.py --source clip.mp4       # video file

Options
-------
  --source        INT or PATH   webcam index or video file  (default: 0)
  --checkpoint    PATH          model weights
  --width         INT           inference width (try 640 on CPU)
  --interval      FLOAT         seconds between webcam captures (default: 3.0)
  --frame-skip    INT           process every Nth video frame  (default: 30)
  --nav-interval  FLOAT         seconds between navmesh recomputes (default: 6.0)
  --up-axis       x|y|z|auto    floor normal axis               (default: y)
  --max-tilt      FLOAT         max floor tilt in degrees       (default: 15.0)
  --port          INT           viser port                      (default: 8080)

Architecture — 4 threads
------------------------
  [Thread 1] CaptureThread    — reads frames from webcam or video file
  [Thread 2] InferenceThread  — depth + segmentation → Gaussian splats + raw pts
  [Thread 3] NavmeshThread    — floor RANSAC, node grid, edges, A* path
  [Thread 4] Main thread      — viser server, updates both layers independently
"""

import argparse
import heapq
import queue
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image as PilImage
from scipy.spatial import KDTree

# ─────────────────────────────────────────────
#  CONFIGURATION  (all overridable via CLI)
# ─────────────────────────────────────────────

# ── Depth model ──────────────────────────────
# Set DEPTH_METRIC = True  to use a Hugging Face metric-depth model.
# The model outputs real distances in metres — no per-frame scale drift,
# no stretching artefacts, and anchor_depth_scale() becomes a no-op.
#
# Recommended models:
#   Outdoor / mixed:  depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
#   Indoor robot:     depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
#
# Set DEPTH_METRIC = False to fall back to the local relative-depth ViT-S
# model (faster, but requires the per-frame scale anchor to avoid stretching).
DEPTH_METRIC       = True
DEPTH_MODEL_ID     = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf'
CHECKPOINT         = r'checkpoints\depth_anything_v2_vits.pth'   # used only when DEPTH_METRIC=False
INFER_WIDTH        = 1280
MAX_DEPTH_M        = 10.0   # indoor rooms; clip anything beyond 10 m
FOV_H_DEG          = 70.0
CAPTURE_INTERVAL_S = 3.0
FRAME_SKIP         = 30
NAV_INTERVAL_S     = 6.0    # navmesh recompute rate (independent of frame rate)
PORT               = 8080

# ── Segmentation ─────────────────────────────
SAT_THRESH    = 22
VAL_THRESH    = 45
BG_DILATE_PX  = 10
BG_FILL_COLOR = (114, 114, 114)

# ── Gaussian splats ───────────────────────────
VOXEL_SIZE       = 0.02
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO    = 2.0

# ── Flying-pixel / depth-edge removal ─────────
# At depth discontinuities (e.g. chair leg vs background) the depth model
# bleeds intermediate values onto boundary pixels.  Back-projected, these
# appear as rays/spikes extending behind every object edge.
# We detect them by finding pixels whose depth is >> the local minimum depth
# in a small neighbourhood (i.e. they sit on the far side of an edge).
EDGE_WINDOW_PX   = 9     # neighbourhood size (pixels); larger = catches wider edges
EDGE_THRESHOLD   = 0.25  # max relative depth jump: pixel removed if
                          # depth > local_min × (1 + threshold)

# ── Occlusion / wall removal ──────────────────
OCCLUSION_TOLERANCE     = 0.20
OCCLUSION_WIN_PX        = 20
REMOVE_WALLS            = True
WALL_DEPTH_PERCENTILE   = 78
WALL_FLATNESS_THRESHOLD = 0.018
WALL_LOCAL_RADIUS       = 20

# ── Floor detection — GPP (Ground Principal Plane) ────────────────────────
# Based on GCSLAM-B: partitioned sector fitting + PCA consensus
UP_AXIS              = 'y'
MAX_FLOOR_TILT_DEG   = 30.0   # max angle between floor normal and up-axis
FLOOR_RANSAC_DIST    = 0.12   # inlier band (metres) for plane membership
MIN_FLOOR_POINTS     = 200    # minimum inliers to accept a floor plane
GPP_N_SECTORS        = 8      # angular sectors around camera (8 = 45° each)
GPP_MIN_SECTOR_PTS   = 10     # ignore sectors with fewer points than this
GPP_CONSENSUS_DEG    = 15.0   # max normal-angle diff (°) for two sectors to agree
GPP_MIN_SECTORS      = 1      # min consensus sectors needed to accept a plane
                               # (1 = works on product videos; 2 = stricter for rooms)

# ── Navmesh ───────────────────────────────────
NODE_SPACING      = 0.15
NODE_ABOVE_FLOOR  = 0.04
OBS_CLEARANCE_R   = 0.40
OBS_HEIGHT_MIN    = 0.05
OBS_HEIGHT_MAX    = 0.80   # only detect obstacles up to 80 cm above the floor
OBS_VOXEL_SIZE    = 0.05
OBS_DENOISE_NB    = 10
OBS_DENOISE_STD   = 1.5
FLOOR_PROX_FACTOR = 1
EDGE_MAX_DIST     = NODE_SPACING * 2.0
EDGE_CHECK_STEPS  = 20   # more samples per edge → fewer wall clip-throughs

# ── Depth scale normalisation ─────────────────
# Monocular depth is relative (per-frame min/max normalised to MAX_DEPTH_M).
# As the robot moves, the depth scale shifts between frames, causing the
# accumulated map to fan/stretch.
#
# Fix (automatic): on the first frame we detect the floor and record the
# camera-to-floor distance as the baseline.  Every subsequent frame is
# rescaled so its floor appears at that same baseline height.  No flag needed.
#
# Fix (metric): pass --cam-height H to specify the actual camera mounting
# height in metres.  This makes all depths truly metric.
CAM_HEIGHT_M       = 0.0    # 0 = auto-baseline from first frame
SCALE_RANSAC_ITER  = 300    # iterations for quick floor-RANSAC in scale estimation
SCALE_INLIER_DIST  = 0.12   # inlier band (metres) for scale RANSAC
SCALE_MIN_INLIERS  = 20     # abort scale estimation if fewer inliers than this
SCALE_CLAMP        = (0.65, 1.45) # reject corrections outside this band — anything
                                   # beyond ±40% of baseline usually means the RANSAC
                                   # latched onto a wall/table instead of the floor

# ── Point accumulation ────────────────────────
NAV_ACCUM_VOXEL    = VOXEL_SIZE * 3   # dedup voxel for accumulated cloud
NAV_ACCUM_MAX_PTS  = 500_000          # cap to avoid unbounded memory growth
ACCUM_ENABLED      = True             # set False (--no-accum) for product/turntable video
MAP_MAX_DEPTH_M    = 3.0              # only accumulate points within this distance (metres)
                                      # limits fan-arm length when alignment drifts

# ── ICP frame alignment (fallback) ───────────
ICP_MAX_DIST    = 1.0    # max correspondence distance (metres)
ICP_MAX_ITER    = 50     # iterations per frame
ICP_FITNESS_MIN = 0.25   # min inlier fraction to accept alignment
ICP_SUBSAMPLE   = 3000   # pts to subsample per cloud for speed
ICP_MAX_SHIFT_M = 2.0    # max translation per frame (metres) — rejects wild drifts
ICP_MAX_ROT_DEG = 45.0   # max rotation per frame (degrees)  — rejects wild spins

# ── Visual Odometry (ORB + PnP) ───────────────
VO_MAX_FEATURES = 2000   # ORB features per frame
VO_RATIO_TEST   = 0.75   # Lowe's ratio for match filtering
VO_MIN_INLIERS  = 12     # minimum PnP inliers to accept alignment
VO_RANSAC_ITER  = 200    # solvePnPRansac iterations
VO_REPROJ_ERR   = 4.0    # reprojection error threshold (pixels)
VO_MAX_SHIFT_M  = 2.0    # max translation per frame (metres)
VO_MAX_ROT_DEG  = 45.0   # max rotation per frame (degrees)
VO_MIN_SHIFT_M  = 0.03   # min translation to bother accumulating (metres)
                          # frames with less motion are treated as duplicate views
# ─────────────────────────────────────────────

# Shared queues — all maxsize=1 so we always work on the latest data
frame_queue   = queue.Queue(maxsize=1)   # Capture  → Inference
splat_queue   = queue.Queue(maxsize=1)   # Inference → Viewer  (Gaussian data)
pts_queue     = queue.Queue(maxsize=1)   # Inference → Navmesh (raw 3-D points + colors)
navmesh_queue = queue.Queue(maxsize=1)   # Navmesh  → Viewer  (navmesh overlay)
map_queue     = queue.Queue(maxsize=1)   # Navmesh  → Viewer  (accumulated colored map)
stop_event    = threading.Event()

# Auto-scale baseline: set from the first frame that has a detectable floor.
# After that every frame is rescaled to match this height, giving consistent
# metric scale across the whole accumulated map without needing --cam-height.
_scale_baseline_m = None   # metres; None = not yet established


# ─────────────────────────────────────────────
#  DEPTH / SEGMENTATION HELPERS
# ─────────────────────────────────────────────

def segment_product(bgr):
    h, w  = bgr.shape[:2]
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v  = hsv[:, :, 1], hsv[:, :, 2]
    candidate  = ((s < SAT_THRESH) & (v > VAL_THRESH)).astype(np.uint8)
    filled     = candidate.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    border = ([(0, x) for x in range(w)] + [(h-1, x) for x in range(w)] +
              [(y, 0) for y in range(h)] + [(y, w-1) for y in range(h)])
    for y, x in border:
        if filled[y, x] == 1:
            cv2.floodFill(filled, flood_mask, (x, y), 2)
    bg_mask = (filled == 2).astype(np.uint8)
    if BG_DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (BG_DILATE_PX * 2 + 1,) * 2)
        bg_mask = cv2.dilate(bg_mask, k)
    return ((1 - bg_mask) * 255).astype(np.uint8)


def raycast_occlusion_mask(depth_m, product_mask):
    sentinel   = depth_m.max() + 1.0
    prod_depth = np.where(product_mask > 0, depth_m, sentinel)
    kernel     = np.ones((OCCLUSION_WIN_PX,) * 2, np.float32)
    local_min  = cv2.erode(prod_depth.astype(np.float32), kernel)
    return depth_m <= (local_min + OCCLUSION_TOLERANCE)


def wall_removal_mask(depth_m):
    d_norm = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)
    r  = WALL_LOCAL_RADIUS * 2 + 1
    k  = np.ones((r, r), np.float32) / (r * r)
    mu  = cv2.filter2D(d_norm, -1, k)
    mu2 = cv2.filter2D(d_norm ** 2, -1, k)
    local_std = np.sqrt(np.clip(mu2 - mu ** 2, 0, None))
    depth_thr = np.percentile(d_norm, WALL_DEPTH_PERCENTILE)
    is_wall   = (d_norm > depth_thr) & (local_std < WALL_FLATNESS_THRESHOLD)
    ksize  = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    is_wall = cv2.morphologyEx(is_wall.astype(np.uint8),
                               cv2.MORPH_CLOSE, kernel).astype(bool)
    return ~is_wall


def voxel_downsample_colored(pts, colors, voxel_size):
    vox  = np.floor(pts / voxel_size).astype(np.int64)
    vox -= vox.min(axis=0)
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    keys     = (x << 42) | (y << 21) | z
    _, first = np.unique(keys, return_index=True)
    return pts[first], colors[first]


def voxel_downsample_pts(pts, voxel_size):
    """Points-only voxel downsample used by the navmesh pipeline."""
    vox  = np.floor(pts / voxel_size).astype(np.int64)
    keys = (vox[:, 0] * 1_000_000_000 + vox[:, 1] * 1_000_000 + vox[:, 2])
    _, first = np.unique(keys, return_index=True)
    return pts[first]


def sor_colored(pts, colors, nb=20, std_ratio=2.0):
    """Statistical outlier removal that keeps pts and colors in sync."""
    if len(pts) < nb + 1:
        return pts, colors
    tree       = KDTree(pts)
    dists, _   = tree.query(pts, k=nb + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + std_ratio * mean_dists.std()
    keep       = mean_dists <= threshold
    return pts[keep], colors[keep]


def sor_pts(pts, nb=10, std_ratio=1.5):
    """Points-only outlier removal for the navmesh obstacle cloud."""
    if len(pts) < nb + 1:
        return pts
    tree       = KDTree(pts)
    dists, _   = tree.query(pts, k=nb + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= threshold]


# ─────────────────────────────────────────────
#  FLYING-PIXEL REMOVAL
# ─────────────────────────────────────────────

def depth_edge_mask(depth_m):
    """
    Returns a boolean mask (H×W) that is True for valid pixels and False for
    flying pixels at depth discontinuities.

    Method: morphological erosion finds the local depth minimum in an
    EDGE_WINDOW_PX neighbourhood.  Any pixel whose depth exceeds that minimum
    by more than EDGE_THRESHOLD (relative) is on the far side of an edge and
    is treated as a flying pixel — it would back-project to a wrong 3-D
    position, appearing as a ray or spike behind the nearer object.

    Example with metric depth:
      chair leg at 1.5 m, background at 4.0 m, threshold=0.25
        → local_min = 1.5 m,  limit = 1.5 × 1.25 = 1.875 m
        → 4.0 m > 1.875 m  ✓  background pixel removed
      wall corner at 2.0 m, adjacent wall at 2.4 m
        → local_min = 2.0 m,  limit = 2.5 m
        → 2.4 m < 2.5 m  ✓  wall pixel kept
    """
    k         = np.ones((EDGE_WINDOW_PX, EDGE_WINDOW_PX), np.float32)
    local_min = cv2.erode(depth_m.astype(np.float32), k,
                          borderType=cv2.BORDER_REPLICATE)
    return depth_m <= local_min * (1.0 + EDGE_THRESHOLD)


# ─────────────────────────────────────────────
#  FRAME → GAUSSIANS + RAW PTS
# ─────────────────────────────────────────────

def frame_to_result(img, model, device):
    """
    Full per-frame pipeline.

    Two separate processing paths share ONE depth inference:
      • Splat path  — segmented + occlusion/wall filtered → display Gaussians
      • Nav path    — full raw depth map (no masking)     → floor RANSAC + navmesh

    Returns a dict with:
      splat    — Gaussian data ready for viser
      nav_pts  — raw (N, 3) float32 point cloud for navmesh (full frame, no mask)
    Returns None if no foreground survived the splat masks.
    """
    oh, ow = img.shape[:2]
    ih  = int(oh * INFER_WIDTH / ow)
    img = cv2.resize(img, (INFER_WIDTH, ih), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    focal = w / (2.0 * np.tan(np.radians(FOV_H_DEG / 2.0)))

    # ── Depth inference ───────────────────────────────────────────────────────
    if DEPTH_METRIC:
        # Transformers metric-depth model → output is already in metres
        depth_processor, depth_model_hf = model   # unpacked tuple
        img_pil  = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs   = depth_processor(images=img_pil, return_tensors="pt")
        inputs   = {k: v.to(depth_model_hf.device) for k, v in inputs.items()}
        with torch.no_grad():
            raw = depth_model_hf(**inputs).predicted_depth   # (1, H', W')
        depth_m_full = torch.nn.functional.interpolate(
            raw.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)
        depth_m_full = np.clip(depth_m_full, 0.1, MAX_DEPTH_M)
        # No per-frame normalisation needed — values are metric metres.
        # anchor_depth_scale() is a no-op when DEPTH_METRIC=True.
    else:
        # Legacy relative-depth ViT-S model
        with torch.no_grad():
            depth_raw_full = model.infer_image(img)
        depth_raw_full = cv2.bilateralFilter(
            depth_raw_full.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
        dmin, dmax   = depth_raw_full.min(), depth_raw_full.max()
        depth_m_full = (1.0 - (depth_raw_full - dmin) / (dmax - dmin)) * MAX_DEPTH_M
        # Anchor depth scale so that VO's prev_depth is metric-consistent
        depth_m_full = anchor_depth_scale(depth_m_full, focal)

    # ── Remove flying pixels at depth discontinuities ────────────────────────
    # Zero out pixels on the far side of depth edges so they are excluded from
    # all subsequent back-projections (nav pts, map pts, and splat display).
    edge_valid   = depth_edge_mask(depth_m_full)          # True = keep
    depth_m_full = np.where(edge_valid, depth_m_full, 0.0)

    # ── Nav path — full depth, no masking ────────────────────────────────────
    nav_pts, map_pts, map_colors = frame_to_nav_pts(img, depth_m_full, focal)

    # ── Splat path — full scene, no segmentation ─────────────────────────────
    cx, cy = w / 2.0, h / 2.0
    px, py = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x3 = (px - cx) * depth_m_full / focal
    y3 = (py - cy) * depth_m_full / focal
    z3 = depth_m_full

    pts    = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=-1)
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    # Flip Y and Z to standard Y-up, camera-at-origin convention
    pts[:, 1] *= -1
    pts[:, 2] *= -1

    # Exclude masked pixels (depth=0 after edge removal → z=0 after flip)
    valid  = pts[:, 2] < 0
    pts    = pts[valid]
    colors = colors[valid]

    pts, colors = voxel_downsample_colored(pts, colors, VOXEL_SIZE)
    pts, colors = sor_colored(pts, colors, SOR_NB_NEIGHBORS, SOR_STD_RATIO)

    if len(pts) == 0:
        return None

    # ── Gaussian data ─────────────────────────────────────────────────────────
    scales = np.maximum(np.abs(pts[:, 2]) / focal, VOXEL_SIZE * 0.5).astype(np.float32)
    N  = len(pts)
    s2 = scales ** 2
    covs          = np.zeros((N, 3, 3), dtype=np.float32)
    covs[:, 0, 0] = s2
    covs[:, 1, 1] = s2
    covs[:, 2, 2] = s2

    splat = dict(
        centers     = pts.astype(np.float32),
        covariances = covs,
        rgbs        = np.clip(colors, 0, 1).astype(np.float32),
        opacities   = np.full((N, 1), 0.95, dtype=np.float32),
    )

    return dict(splat=splat, nav_pts=nav_pts, map_pts=map_pts, map_colors=map_colors,
                img=img, depth_m=depth_m_full, focal=focal, cx=cx, cy=cy)


def anchor_depth_scale(depth_m, focal):
    """
    Rescale a relative monocular depth map so that every frame shares the
    same camera-to-floor distance (the "scale baseline").

    On the first call the baseline is auto-detected from the current frame's
    floor plane.  If --cam-height was passed that value is used instead,
    making depths truly metric.  Subsequent frames are rescaled to match
    the baseline, eliminating per-frame scale drift and the resulting
    fan/stretch artefacts.

    Returns a rescaled depth map (same shape), or the original if the floor
    cannot be reliably detected.
    """
    global _scale_baseline_m

    if DEPTH_METRIC:
        return depth_m   # already metric — no rescaling needed

    h, w   = depth_m.shape
    cx, cy = w / 2.0, h / 2.0

    def _sample_pts(y_start, step):
        py, px = np.meshgrid(np.arange(y_start, h, step, dtype=np.float32),
                             np.arange(0, w, step, dtype=np.float32), indexing='ij')
        d_sub  = depth_m[y_start::step, ::step].astype(np.float32)
        x3 =  (px - cx) * d_sub / focal
        y3 = -((py - cy) * d_sub / focal)
        z3 = -d_sub
        pts   = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=-1)
        valid = (d_sub.ravel() > 0.05) & (d_sub.ravel() < MAX_DEPTH_M * 0.98)
        return pts[valid]

    # Try bottom 40 % first (floor most likely here); fall back to full image
    pts = _sample_pts(int(h * 0.60), 2)
    if len(pts) < SCALE_MIN_INLIERS:
        pts = _sample_pts(0, 4)   # full image, sparser sampling
    if len(pts) < SCALE_MIN_INLIERS:
        return depth_m   # genuinely not enough data

    idx = np.random.choice(len(pts), min(800, len(pts)), replace=False)
    pts = pts[idx]

    up = np.array([0.0, 1.0, 0.0])
    cos_thresh   = np.cos(np.radians(MAX_FLOOR_TILT_DEG))
    best_inliers = 0
    best_floor_y = None

    for _ in range(SCALE_RANSAC_ITER):
        i, j, k = np.random.choice(len(pts), 3, replace=False)
        n = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(np.dot(n, up)) < cos_thresh:
            continue
        if np.dot(n, up) < 0:
            n = -n
        d_plane = -n @ pts[i]
        n_inl   = int((np.abs(pts @ n + d_plane) < SCALE_INLIER_DIST).sum())
        if n_inl > best_inliers:
            best_inliers = n_inl
            best_floor_y = float(pts[np.abs(pts @ n + d_plane) < SCALE_INLIER_DIST, 1].mean())

    if best_floor_y is None or best_floor_y >= -0.05 or best_inliers < SCALE_MIN_INLIERS:
        # Floor not reliably detected this frame — return unscaled but still
        # apply baseline if one exists (keeps already-locked scale consistent)
        return depth_m

    detected_h = abs(best_floor_y)   # camera-to-floor distance in this frame's units

    # --- Establish or look up the target height ---
    if CAM_HEIGHT_M > 0:
        target_h = CAM_HEIGHT_M          # user-supplied metric height
    elif _scale_baseline_m is None:
        # First successful detection: record as baseline
        _scale_baseline_m = detected_h
        print(f"[Scale]  Baseline locked at {detected_h:.3f} m  "
              f"(inliers={best_inliers}) — all frames will match this scale")
        return depth_m                   # first frame needs no rescaling
    else:
        target_h = _scale_baseline_m     # auto-baseline

    raw_scale = target_h / detected_h

    # Sanity check: if the raw scale is outside SCALE_CLAMP the RANSAC almost
    # certainly hit a wall / table / ceiling instead of the floor.  Skip the
    # correction for this frame rather than applying a wild distortion.
    lo, hi = SCALE_CLAMP
    if not (lo <= raw_scale <= hi):
        print(f"[Scale]  floor={best_floor_y:.3f}m  scale={raw_scale:.3f}× — "
              f"out of [{lo},{hi}], skipping (wrong surface?)")
        return depth_m

    scale = float(raw_scale)
    if abs(scale - 1.0) > 0.02:
        print(f"[Scale]  floor={best_floor_y:.3f}m  scale={scale:.3f}×")
    return depth_m * scale


def frame_to_nav_pts(img, depth_m, focal):
    """
    Back-project the FULL depth map.

    Returns (nav_pts, map_pts, map_colors):
      nav_pts    — coarse (VOXEL_SIZE×3) position-only cloud for RANSAC / navmesh
      map_pts    — fine  (VOXEL_SIZE)    positions for accumulated display map
      map_colors — colors matching map_pts
    """
    # depth_m is already scale-anchored and edge-masked by frame_to_result
    h, w   = depth_m.shape
    cx, cy = w / 2.0, h / 2.0
    px, py = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x3 = (px - cx) * depth_m / focal
    y3 = (py - cy) * depth_m / focal
    z3 = depth_m

    all_pts    = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=-1)
    all_colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    all_pts[:, 1] *= -1
    all_pts[:, 2] *= -1

    # Exclude pixels that were zeroed by edge masking (z = 0 after flip)
    valid      = all_pts[:, 2] < 0
    all_pts    = all_pts[valid]
    all_colors = all_colors[valid]

    # Depth filter for accumulation map — only keep close points.
    # In nav convention z = -depth, so |z| = depth.
    # Far points amplify any alignment error into huge fan arms.
    close_mask = np.abs(all_pts[:, 2]) <= MAP_MAX_DEPTH_M
    map_all_pts    = all_pts[close_mask]
    map_all_colors = all_colors[close_mask]

    # Fine resolution — for display (same voxel as Gaussian splats)
    if len(map_all_pts) > 0:
        map_pts, map_colors = voxel_downsample_colored(map_all_pts, map_all_colors, VOXEL_SIZE)
    else:
        map_pts    = np.zeros((0, 3), dtype=np.float32)
        map_colors = np.zeros((0, 3), dtype=np.float32)

    # Coarse resolution — for RANSAC / navmesh (uses full depth, not capped)
    nav_pts = voxel_downsample_pts(all_pts, VOXEL_SIZE * 3)

    return nav_pts.astype(np.float32), map_pts.astype(np.float32), map_colors.astype(np.float32)


# ─────────────────────────────────────────────
#  ICP FRAME ALIGNMENT
# ─────────────────────────────────────────────

def icp_align(source, target):
    """
    Point-to-point ICP. Aligns source onto target using numpy + KDTree only.

    Returns (R, t, fitness):
      R       — (3, 3) rotation matrix
      t       — (3,)  translation vector
      fitness — fraction of source pts within ICP_MAX_DIST of their match

    Apply as:  aligned = (R @ source.T).T + t
    """
    src      = source.astype(np.float64)
    R_acc    = np.eye(3)
    t_acc    = np.zeros(3)
    tree     = KDTree(target.astype(np.float64))
    prev_err = np.inf

    for _ in range(ICP_MAX_ITER):
        dists, idx = tree.query(src, k=1)
        inliers    = dists < ICP_MAX_DIST
        if inliers.sum() < 10:
            break

        s = src[inliers]
        t_pts = target[idx[inliers]].astype(np.float64)

        # Optimal rigid transform via SVD
        sc, tc = s.mean(0), t_pts.mean(0)
        H = (s - sc).T @ (t_pts - tc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:   # fix reflection
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t_step = tc - R @ sc

        src   = (R @ src.T).T + t_step
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t_step

        err = dists[inliers].mean()
        if abs(prev_err - err) < 1e-4:
            break
        prev_err = err

    fitness = inliers.sum() / len(source)
    return R_acc, t_acc, float(fitness)


# ─────────────────────────────────────────────
#  ORB + PnP VISUAL ODOMETRY
# ─────────────────────────────────────────────

def vo_align(img_prev, depth_prev, img_curr, focal, cx, cy):
    """
    Visual odometry via ORB feature matching + PnP pose estimation.

    Detects ORB keypoints in both frames, applies Lowe's ratio test, lifts
    the matched previous-frame keypoints to 3D via depth_prev (standard
    pinhole: Y-down, Z-into-scene), runs solvePnPRansac, then converts the
    result to the navmesh Y-up / Z-toward-viewer convention.

    Returns (R_rel, t_rel, n_inliers):
      R_rel, t_rel — transform that maps current-frame nav pts → prev-frame
                     nav pts  (same convention as icp_align, apply as
                     (R_rel @ pts.T).T + t_rel).
      n_inliers    — PnP inlier count; None R_rel/t_rel means failure.
    """
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

    orb       = cv2.ORB_create(nfeatures=VO_MAX_FEATURES)
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, 0

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < VO_RATIO_TEST * n.distance]

    if len(good) < VO_MIN_INLIERS:
        return None, None, len(good)

    # Back-project matched prev-frame keypoints to 3D.
    # Use STANDARD camera coords (Y-down, Z-into-scene) so K applies correctly.
    hd, wd = depth_prev.shape
    obj_pts, img_pts = [], []
    for m in good:
        u1, v1 = kp1[m.queryIdx].pt   # prev-frame pixel
        u2, v2 = kp2[m.trainIdx].pt   # curr-frame pixel

        ui, vi = int(round(u1)), int(round(v1))
        if not (0 <= ui < wd and 0 <= vi < hd):
            continue
        d = float(depth_prev[vi, ui])
        if d < 0.05 or d >= MAX_DEPTH_M * 0.99:
            continue   # skip missing / saturated depth

        # Standard pinhole back-projection (no Y/Z flip)
        x3 = (u1 - cx) * d / focal
        y3 = (v1 - cy) * d / focal
        z3 = d
        obj_pts.append([x3, y3, z3])
        img_pts.append([u2, v2])

    if len(obj_pts) < VO_MIN_INLIERS:
        return None, None, len(obj_pts)

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    K = np.array([[focal, 0.0, cx],
                  [0.0, focal, cy],
                  [0.0,   0.0, 1.0]], dtype=np.float64)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None,
        iterationsCount = VO_RANSAC_ITER,
        reprojectionError = VO_REPROJ_ERR,
        confidence = 0.99,
        flags = cv2.SOLVEPNP_EPNP,
    )

    if not ret or inliers is None or len(inliers) < VO_MIN_INLIERS:
        return None, None, (0 if inliers is None else len(inliers))

    # R_std maps prev-cam-std coords → curr-cam-std coords:
    #   p_curr_std = R_std @ p_prev_std + t_std
    R_std, _ = cv2.Rodrigues(rvec)
    t_std     = tvec.ravel()

    # Convert to nav convention (Y-up, Z-toward-viewer):
    #   F = diag(1, -1, -1)  →  R_nav = F @ R_std @ F,  t_nav = F @ t_std
    F = np.diag([1.0, -1.0, -1.0])
    R_curr_from_prev = F @ R_std @ F
    t_curr_from_prev = F @ t_std

    # Invert to get T_prev_from_curr (matches icp_align convention)
    R_rel = R_curr_from_prev.T
    t_rel = -(R_curr_from_prev.T @ t_curr_from_prev)

    return R_rel, t_rel, len(inliers)


# ─────────────────────────────────────────────
#  NAVMESH HELPERS  (ported from generate_nodes.py)
# ─────────────────────────────────────────────

def up_vector(axis_idx):
    v = np.zeros(3); v[axis_idx] = 1.0; return v


def _ransac_fallback(pts, up_idx):
    """
    Classic RANSAC plane fit — used when GPP sectors are too sparse.
    Returns (plane, inlier_indices).  Raises RuntimeError on failure.
    """
    up         = up_vector(up_idx)
    cos_thresh = np.cos(np.radians(MAX_FLOOR_TILT_DEG))
    n          = len(pts)
    best_inliers = []; best_plane = None; best_h = np.inf
    for _ in range(400):
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm   = np.linalg.norm(normal)
        if norm < 1e-10: continue
        normal /= norm
        if abs(np.dot(normal, up)) < cos_thresh: continue
        d = -normal @ p0
        inl = np.where(np.abs(pts @ normal + d) < FLOOR_RANSAC_DIST)[0]
        h   = pts[inl, up_idx].mean() if len(inl) else np.inf
        if len(inl) > len(best_inliers) or (len(inl) == len(best_inliers) and h < best_h):
            best_inliers = inl; best_plane = np.array([*normal, d]); best_h = h
    if best_plane is None or len(best_inliers) < MIN_FLOOR_POINTS:
        raise RuntimeError(f"RANSAC fallback failed ({len(best_inliers)} inliers).")
    ip = pts[best_inliers]; c = ip.mean(0)
    _, _, Vt = np.linalg.svd(ip - c, full_matrices=False)
    normal = Vt[-1]; d = -normal @ c
    if np.dot(normal, up) < 0: normal = -normal; d = -d
    inliers = np.where(np.abs(pts @ normal + d) < FLOOR_RANSAC_DIST)[0]
    return np.array([*normal, d]), inliers


def extract_gpp(pts, up_idx):
    """
    Ground Principal Plane extraction via partitioned sector fitting.
    Based on GCSLAM-B (2024): divides the point cloud into GPP_N_SECTORS
    angular sectors, fits a PCA plane per sector, finds the largest group
    of sectors whose normals agree within GPP_CONSENSUS_DEG, then refines
    the consensus plane with least-squares on all inliers.

    Returns (plane, inlier_indices, lambda_min):
      plane       — [nx, ny, nz, d]
      inliers     — indices into pts
      lambda_min  — PCA minimum eigenvalue / N  (flatness score; lower = flatter)

    Raises RuntimeError if floor cannot be found.
    """
    up         = up_vector(up_idx)
    cos_thresh = np.cos(np.radians(MAX_FLOOR_TILT_DEG))

    # Horizontal axes for sector angle calculation (the two non-up axes)
    h_axes = [i for i in range(3) if i != up_idx]
    a0, a1 = h_axes
    angles = np.arctan2(pts[:, a1], pts[:, a0])   # (N,) in [-π, π]

    sector_normals = []
    sector_ds      = []
    sector_width   = 2.0 * np.pi / GPP_N_SECTORS

    # ── Step 1: fit one plane per sector via PCA ──────────────────────────
    for s in range(GPP_N_SECTORS):
        lo   = -np.pi + s * sector_width
        hi   = lo + sector_width
        mask = (angles >= lo) & (angles < hi)
        if mask.sum() < GPP_MIN_SECTOR_PTS:
            continue

        sp       = pts[mask]
        centroid = sp.mean(0)
        _, _, Vt = np.linalg.svd(sp - centroid, full_matrices=False)
        normal   = Vt[-1]   # eigenvector of the smallest singular value

        # Reject sectors whose plane is not roughly horizontal
        if abs(np.dot(normal, up)) < cos_thresh:
            continue

        if np.dot(normal, up) < 0:
            normal = -normal
        sector_normals.append(normal)
        sector_ds.append(-normal @ centroid)

    if len(sector_normals) < GPP_MIN_SECTORS:
        raise RuntimeError(
            f"GPP: only {len(sector_normals)} valid sectors "
            f"(need {GPP_MIN_SECTORS}). Try --max-tilt 45.")

    normals = np.array(sector_normals)   # (S, 3)

    # ── Step 2: consensus — largest group with agreeing normals ───────────
    cos_consensus = np.cos(np.radians(GPP_CONSENSUS_DEG))
    best_group    = []
    for i in range(len(normals)):
        # dot product with each other normal (both positive — already flipped up)
        dots  = normals @ normals[i]
        group = np.where(dots >= cos_consensus)[0].tolist()
        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < GPP_MIN_SECTORS:
        raise RuntimeError("GPP: no consensus group found.")

    # ── Step 3: average consensus normal + initial inlier pass ───────────
    avg_normal = normals[best_group].mean(0)
    avg_normal /= np.linalg.norm(avg_normal)
    avg_d      = float(np.mean([sector_ds[i] for i in best_group]))

    rough_inliers = np.where(np.abs(pts @ avg_normal + avg_d) < FLOOR_RANSAC_DIST)[0]
    if len(rough_inliers) < MIN_FLOOR_POINTS:
        raise RuntimeError(
            f"GPP: {len(rough_inliers)} rough inliers (need {MIN_FLOOR_POINTS}). "
            "Try --max-tilt 45.")

    # ── Step 4: least-squares refinement on all inliers ──────────────────
    ip       = pts[rough_inliers]
    centroid = ip.mean(0)
    _, sv, Vt = np.linalg.svd(ip - centroid, full_matrices=False)
    normal   = Vt[-1]
    d        = -normal @ centroid
    if np.dot(normal, up) < 0:
        normal = -normal; d = -d

    # λ_min normalised by point count — flatness quality score
    lambda_min = float(sv[-1]) / len(ip)

    # Final inlier set with refined plane
    inliers = np.where(np.abs(pts @ normal + d) < FLOOR_RANSAC_DIST)[0]
    if len(inliers) < MIN_FLOOR_POINTS:
        raise RuntimeError(
            f"GPP refined: {len(inliers)} inliers (need {MIN_FLOOR_POINTS}).")

    return np.array([*normal, d]), inliers, lambda_min


def point_to_plane_signed(points, plane):
    a, b, c, d = plane
    n = np.array([a, b, c])
    return (points @ n + d) / np.linalg.norm(n)


def build_floor_axes(normal):
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(normal, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, up); u /= np.linalg.norm(u)
    v = np.cross(normal, u);  v /= np.linalg.norm(v)
    return u, v


def generate_nodes(floor_pts, plane, normal):
    u, v = build_floor_axes(normal)
    uc, vc = floor_pts @ u, floor_pts @ v
    ug = np.arange(uc.min(), uc.max(), NODE_SPACING)
    vg = np.arange(vc.min(), vc.max(), NODE_SPACING)
    uu, vv = np.meshgrid(ug, vg); uu, vv = uu.ravel(), vv.ravel()
    floor_tree = KDTree(np.stack([uc, vc], axis=1))
    dists, idx = floor_tree.query(np.stack([uu, vv], axis=1), k=1)
    valid  = dists < NODE_SPACING * 1.5
    return floor_pts[idx[valid]] + normal * NODE_ABOVE_FLOOR


def denoise_obstacles(obs_pts, plane, floor_pts):
    """
    Returns (display_obs, collision_obs):
      display_obs   — denoised sparse cloud shown in the viewer (orange dots)
      collision_obs — denser cloud used for node blocking and edge checking
    """
    if len(obs_pts) == 0:
        return obs_pts, obs_pts

    heights = point_to_plane_signed(obs_pts, plane)
    obs_pts = obs_pts[(heights > OBS_HEIGHT_MIN) & (heights < OBS_HEIGHT_MAX)]
    if len(obs_pts) == 0:
        return obs_pts, obs_pts

    if len(floor_pts) > 0:
        dists, _ = KDTree(floor_pts).query(obs_pts, k=1)
        obs_pts  = obs_pts[dists > NODE_SPACING * FLOOR_PROX_FACTOR]
    if len(obs_pts) < 10:
        return obs_pts, obs_pts

    # Voxel downsample first — removes most noise cheaply
    collision_obs = voxel_downsample_pts(obs_pts, OBS_VOXEL_SIZE)
    display_obs   = collision_obs   # start from same base

    # SOR is O(n log n) — only run it on small clouds where it adds value.
    # Large accumulated obstacle clouds are already clean after voxel downsampling.
    if len(collision_obs) <= 3000:
        # Build KDTree once, derive both clouds from the same inlier mask
        tree       = KDTree(collision_obs)
        dists, _   = tree.query(collision_obs, k=OBS_DENOISE_NB + 1)
        mean_dists = dists[:, 1:].mean(axis=1)
        mu, sigma  = mean_dists.mean(), mean_dists.std()
        keep_tight = mean_dists <= (mu + OBS_DENOISE_STD        * sigma)
        keep_loose = mean_dists <= (mu + OBS_DENOISE_STD * 1.5  * sigma)
        display_obs   = collision_obs[keep_tight]
        collision_obs = collision_obs[keep_loose]

    return display_obs, collision_obs


def filter_nodes(nodes, collision_obs):
    if len(collision_obs) == 0:
        return nodes, np.zeros((0, 3))
    dists, _ = KDTree(collision_obs).query(nodes, k=1)
    free_mask = dists >= OBS_CLEARANCE_R
    return nodes[free_mask], nodes[~free_mask]


def build_edges(free_nodes, collision_obs):
    if len(free_nodes) < 2:
        return []
    pairs = KDTree(free_nodes).query_pairs(r=EDGE_MAX_DIST)
    if len(collision_obs) == 0:
        return list(pairs)
    obs_tree = KDTree(collision_obs)
    edges = []
    for i, j in pairs:
        sample = np.linspace(free_nodes[i], free_nodes[j], EDGE_CHECK_STEPS)
        if obs_tree.query(sample, k=1)[0].min() >= OBS_CLEARANCE_R:
            edges.append((i, j))
    return edges


def astar_graph(free_nodes, edges, start_idx, goal_idx):
    graph = {i: [] for i in range(len(free_nodes))}
    for i, j in edges:
        d = np.linalg.norm(free_nodes[i] - free_nodes[j])
        graph[i].append((j, d)); graph[j].append((i, d))
    h    = lambda i: np.linalg.norm(free_nodes[i] - free_nodes[goal_idx])
    heap = [(h(start_idx), 0.0, start_idx)]
    came = {start_idx: None}; g = {start_idx: 0.0}
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == goal_idx:
            path = []
            while cur is not None:
                path.append(cur); cur = came[cur]
            return path[::-1]
        for nb, d in graph.get(cur, []):
            ng = cost + d
            if ng < g.get(nb, float('inf')):
                g[nb] = ng; came[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return []


def compute_navmesh(pts, up_idx, camera_origin=None, camera_forward=None):
    """
    Full navmesh pipeline on a raw point cloud.
    camera_origin  — world-space position of the camera (default: origin)
    camera_forward — world-space forward direction    (default: -Z)
    Returns a dict of viser-ready arrays, or None on failure.
    """
    if len(pts) < MIN_FLOOR_POINTS * 2:
        return None

    if camera_origin  is None: camera_origin  = np.zeros(3)
    if camera_forward is None: camera_forward = np.array([0.0, 0.0, -1.0])
    camera_origin  = np.asarray(camera_origin,  dtype=np.float64)
    camera_forward = np.asarray(camera_forward, dtype=np.float64)
    camera_forward = camera_forward / (np.linalg.norm(camera_forward) + 1e-8)

    try:
        plane, floor_idx, lambda_min = extract_gpp(pts, up_idx)
        print(f"[GPP]   λ_min={lambda_min:.6f}  "
              f"({'flat ✓' if lambda_min < 0.001 else 'rough ~' if lambda_min < 0.005 else 'poor ✗'})")
    except RuntimeError as gpp_err:
        print(f"[GPP] {gpp_err}  — falling back to RANSAC")
        try:
            plane, floor_idx = _ransac_fallback(pts, up_idx)
            lambda_min = 0.005   # unknown quality; treated as rough
            print("[GPP] RANSAC fallback succeeded")
        except RuntimeError as e:
            print(f"[Navmesh] {e}")
            return None

    a, b, c, d = plane
    normal      = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    obstacle_idx = np.setdiff1d(np.arange(len(pts)), floor_idx)
    floor_pts    = pts[floor_idx]
    obs_pts      = pts[obstacle_idx]

    nodes                     = generate_nodes(floor_pts, plane, normal)
    display_obs, collision_obs = denoise_obstacles(obs_pts, plane, floor_pts)
    free_nodes, blocked_nodes  = filter_nodes(nodes, collision_obs)
    edges                      = build_edges(free_nodes, collision_obs)

    # Path: camera position → farthest reachable free node
    # Strategy: start at the free node nearest the camera, then try goals
    # from farthest to nearest until A* finds a connected path.
    path_pts = np.zeros((0, 3))
    if len(free_nodes) >= 2 and edges:
        node_tree = KDTree(free_nodes)
        start_idx = int(node_tree.query(camera_origin)[1])

        # Sort candidate goals by distance from camera (farthest first)
        dists_from_cam = np.linalg.norm(free_nodes - camera_origin, axis=1)
        goal_candidates = np.argsort(-dists_from_cam)   # descending

        path_idx = []
        for goal_idx in goal_candidates:
            if goal_idx == start_idx:
                continue
            path_idx = astar_graph(free_nodes, edges, start_idx, int(goal_idx))
            if path_idx:
                break   # found a valid path to the farthest reachable node

        if path_idx:
            path_pts = free_nodes[path_idx]

    print(f"[Navmesh] floor={len(floor_idx):,}  "
          f"obs(display/collision)={len(display_obs):,}/{len(collision_obs):,}  "
          f"free={len(free_nodes):,}  edges={len(edges):,}  "
          f"path={len(path_pts)} nodes")

    return dict(
        clean_obs     = display_obs.astype(np.float32),
        free_nodes    = free_nodes.astype(np.float32),
        blocked_nodes = blocked_nodes.astype(np.float32),
        edges         = edges,
        path_pts      = path_pts.astype(np.float32),
        lambda_min    = lambda_min,   # floor flatness — used by Phase 2 dynamic noise
    )


# ─────────────────────────────────────────────
#  THREAD 1 — Frame capture
# ─────────────────────────────────────────────
class CaptureThread(threading.Thread):
    def __init__(self, source, interval, frame_skip):
        super().__init__(daemon=True, name="CaptureThread")
        self.interval   = interval
        self.frame_skip = frame_skip
        try:
            cam_idx      = int(source)
            self.is_file = False
            self.cap     = cv2.VideoCapture(cam_idx)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cam_idx}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[Capture] Webcam {cam_idx} opened")
        except ValueError:
            self.is_file = True
            self.cap     = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {source}")
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = self.cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"[Capture] {source}  ({total} frames @ {fps:.1f} fps, "
                  f"every {frame_skip}th frame)")

    def run(self):
        frame_idx = 0
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    print("[Capture] Video ended — viewer open, Ctrl-C to quit.")
                    break
                time.sleep(0.1); continue

            if self.is_file and frame_idx % self.frame_skip != 0:
                frame_idx += 1; continue
            frame_idx += 1

            if self.is_file:
                # Video file: block until InferenceThread consumes the frame so
                # we don't race past the whole video before inference can run.
                frame_queue.put(frame)
            else:
                # Webcam: drop stale frames so we always have the freshest one.
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except queue.Empty: pass
                frame_queue.put(frame)
                time.sleep(self.interval)

        self.cap.release()
        print("[Capture] Stopped")


# ─────────────────────────────────────────────
#  THREAD 2 — Depth inference + Gaussian build
# ─────────────────────────────────────────────
class InferenceThread(threading.Thread):
    def __init__(self, model, device):
        super().__init__(daemon=True, name="InferenceThread")
        self.model  = model
        self.device = device

    def _push(self, q, item):
        if q.full():
            try: q.get_nowait()
            except queue.Empty: pass
        q.put(item)

    def run(self):
        count = 0
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0     = time.time()
            result = frame_to_result(frame, self.model, self.device)
            dt     = time.time() - t0
            count += 1

            if result is not None:
                n   = len(result['splat']['centers'])
                nav = len(result['nav_pts'])
                print(f"[Inference] Frame {count:3d} | {n:>7,} splats | {nav:>7,} nav pts | {dt:.2f}s")
                self._push(pts_queue, (result['nav_pts'],
                                       result['map_pts'],
                                       result['map_colors'],
                                       result['img'],
                                       result['depth_m'],
                                       result['focal'],
                                       result['cx'],
                                       result['cy']))
            else:
                print(f"[Inference] Frame {count:3d} | zero pts  | {dt:.2f}s")

        print("[Inference] Stopped")


# ─────────────────────────────────────────────
#  THREAD 3 — Navmesh computation
# ─────────────────────────────────────────────
class NavmeshThread(threading.Thread):
    def __init__(self, up_idx):
        super().__init__(daemon=True, name="NavmeshThread")
        self.up_idx = up_idx

    def run(self):
        last_run       = 0.0
        accum_pts      = None   # coarse world-space positions (for navmesh RANSAC)
        accum_pts_fine = None   # fine world-space positions   (for display)
        accum_colors   = None   # colors matching accum_pts_fine
        T_cum          = np.eye(4)
        prev_cam       = None   # coarse nav pts from previous frame (ICP fallback)
        prev_img       = None   # BGR image from previous frame       (VO)
        prev_depth     = None   # depth map from previous frame       (VO)
        prev_focal     = None
        prev_cx        = None
        prev_cy        = None

        while not stop_event.is_set():
            got_new = False
            while True:
                try:
                    (new_cam, new_map_pts, new_map_colors,
                     new_img, new_depth, focal, cx, cy) = pts_queue.get_nowait()
                except queue.Empty:
                    break

                # ── Frame alignment: ORB+PnP first, ICP fallback ──────────
                if prev_cam is None:
                    # First frame — define world = camera frame 0
                    T_cum = np.eye(4)
                else:
                    aligned    = False
                    R_rel      = None
                    t_rel      = None

                    # ── Primary: ORB + PnP ─────────────────────────────────
                    R_rel, t_rel, n_inliers = vo_align(
                        prev_img, prev_depth, new_img,
                        prev_focal, prev_cx, prev_cy)

                    if R_rel is not None and n_inliers >= VO_MIN_INLIERS:
                        shift = np.linalg.norm(t_rel)
                        angle = np.degrees(np.arccos(
                            np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
                        if shift < VO_MIN_SHIFT_M:
                            # Camera barely moved — duplicate view (e.g. turntable video).
                            # Update prev state but don't fan-accumulate this frame.
                            print(f"[VO]  inliers={n_inliers:3d}  "
                                  f"shift={shift:.3f}m  rot={angle:.1f}°  — duplicate view, skip")
                            prev_cam   = new_cam
                            prev_img   = new_img
                            prev_depth = new_depth
                            prev_focal = focal
                            prev_cx    = cx
                            prev_cy    = cy
                            continue
                        elif shift < VO_MAX_SHIFT_M and angle < VO_MAX_ROT_DEG:
                            T_rel         = np.eye(4)
                            T_rel[:3, :3] = R_rel
                            T_rel[:3,  3] = t_rel
                            T_cum         = T_cum @ T_rel
                            print(f"[VO]  inliers={n_inliers:3d}  "
                                  f"shift={shift:.3f}m  rot={angle:.1f}°")
                            aligned = True
                        else:
                            print(f"[VO]  inliers={n_inliers} but "
                                  f"shift={shift:.3f}m rot={angle:.1f}° — sanity fail")

                    # ── Fallback: ICP ─────────────────────────────────────
                    if not aligned:
                        n_s = min(ICP_SUBSAMPLE, len(new_cam))
                        n_t = min(ICP_SUBSAMPLE, len(prev_cam))
                        si  = np.random.choice(len(new_cam),  n_s, replace=False)
                        ti  = np.random.choice(len(prev_cam), n_t, replace=False)
                        R_icp, t_icp, fitness = icp_align(new_cam[si], prev_cam[ti])
                        shift = np.linalg.norm(t_icp)
                        angle = np.degrees(np.arccos(
                            np.clip((np.trace(R_icp) - 1) / 2, -1, 1)))

                        if (fitness >= ICP_FITNESS_MIN
                                and shift < ICP_MAX_SHIFT_M
                                and angle < ICP_MAX_ROT_DEG):
                            T_rel         = np.eye(4)
                            T_rel[:3, :3] = R_icp
                            T_rel[:3,  3] = t_icp
                            T_cum         = T_cum @ T_rel
                            print(f"[ICP fallback]  fitness={fitness:.2f}  "
                                  f"shift={shift:.3f}m  rot={angle:.1f}°")
                            aligned = True
                        else:
                            print(f"[Align] Both VO+ICP failed — skipping frame "
                                  f"(VO inliers={n_inliers}, "
                                  f"ICP fitness={fitness:.2f})")
                            prev_cam   = new_cam
                            prev_img   = new_img
                            prev_depth = new_depth
                            prev_focal = focal
                            prev_cx    = cx
                            prev_cy    = cy
                            continue

                R_w = T_cum[:3, :3]
                t_w = T_cum[:3,  3]

                # ── Coarse nav accumulation (position-only, for RANSAC) ────
                world_nav = (R_w @ new_cam.T).T + t_w
                if not ACCUM_ENABLED:
                    # No-accum mode: navmesh always uses only the current frame
                    accum_pts = world_nav
                elif accum_pts is None:
                    accum_pts = world_nav
                else:
                    accum_pts = np.concatenate([accum_pts, world_nav], axis=0)
                    accum_pts = voxel_downsample_pts(accum_pts, NAV_ACCUM_VOXEL)
                    if len(accum_pts) > 5000:
                        accum_pts = sor_pts(accum_pts, nb=10, std_ratio=2.0)
                    if len(accum_pts) > NAV_ACCUM_MAX_PTS:
                        accum_pts = accum_pts[-NAV_ACCUM_MAX_PTS // 2:]

                # ── Fine map accumulation (colored, for display) ───────────
                world_map = (R_w @ new_map_pts.T).T + t_w
                if len(world_map) == 0:
                    pass  # nothing to add (all points beyond MAP_MAX_DEPTH_M)
                elif not ACCUM_ENABLED:
                    # No-accum mode: replace cloud each frame (clean product view)
                    accum_pts_fine = world_map
                    accum_colors   = new_map_colors
                elif accum_colors is None:
                    accum_colors   = new_map_colors
                    accum_pts_fine = world_map
                else:
                    accum_pts_fine = np.concatenate([accum_pts_fine, world_map],      axis=0)
                    accum_colors   = np.concatenate([accum_colors,   new_map_colors], axis=0)
                    accum_pts_fine, accum_colors = voxel_downsample_colored(
                        accum_pts_fine, accum_colors, VOXEL_SIZE)
                    if len(accum_pts_fine) > NAV_ACCUM_MAX_PTS:
                        accum_pts_fine = accum_pts_fine[-NAV_ACCUM_MAX_PTS // 2:]
                        accum_colors   = accum_colors  [-NAV_ACCUM_MAX_PTS // 2:]

                prev_cam   = new_cam
                prev_img   = new_img
                prev_depth = new_depth
                prev_focal = focal
                prev_cx    = cx
                prev_cy    = cy

                # ── Push fine colored map to viewer ────────────────────────
                map_item = dict(pts=accum_pts_fine.astype(np.float32),
                                colors=accum_colors.astype(np.float32))
                if map_queue.full():
                    try: map_queue.get_nowait()
                    except queue.Empty: pass
                map_queue.put(map_item)

                got_new = True

            if accum_pts is None:
                time.sleep(0.5)
                continue
            if not got_new:
                time.sleep(0.2)
                continue

            now = time.time()
            if now - last_run < NAV_INTERVAL_S:
                time.sleep(0.2)
                continue

            # Camera is at origin in camera space; transform to world space
            cam_world_pos = T_cum[:3, 3]
            cam_world_fwd = T_cum[:3, :3] @ np.array([0.0, 0.0, -1.0])

            # Light outlier removal before navmesh — scale nb down on large
            # clouds so KDTree query doesn't dominate runtime
            nb_sor = max(4, min(10, 60_000 // max(len(accum_pts), 1)))
            clean_accum = sor_pts(accum_pts, nb=nb_sor, std_ratio=2.0)
            if len(clean_accum) < MIN_FLOOR_POINTS * 2:
                clean_accum = accum_pts

            print(f"[Navmesh] Computing on {len(clean_accum):,} pts "
                  f"(raw={len(accum_pts):,}, cam @ {cam_world_pos.round(2)})...")
            t0  = time.time()
            nav = compute_navmesh(clean_accum, self.up_idx,
                                  camera_origin=cam_world_pos,
                                  camera_forward=cam_world_fwd)
            dt  = time.time() - t0
            print(f"[Navmesh] Done in {dt:.2f}s")
            last_run = time.time()

            if nav is not None:
                # Include a downsampled copy of the accumulated cloud so the
                # viewer can show it — explains why path goes to "empty" areas
                nav['accum_pts'] = voxel_downsample_pts(
                    clean_accum, NAV_ACCUM_VOXEL * 2).astype(np.float32)
                if navmesh_queue.full():
                    try: navmesh_queue.get_nowait()
                    except queue.Empty: pass
                navmesh_queue.put(nav)

        print("[Navmesh] Stopped")


# ─────────────────────────────────────────────
#  VISER SCENE UPDATES
# ─────────────────────────────────────────────

def update_splats(server, splat):
    server.scene.add_gaussian_splats(
        "/scene/splats",
        centers     = splat['centers'],
        covariances = splat['covariances'],
        rgbs        = splat['rgbs'],
        opacities   = splat['opacities'],
    )


def update_navmesh(server, nav):
    clean_obs     = nav['clean_obs']
    free_nodes    = nav['free_nodes']
    blocked_nodes = nav['blocked_nodes']
    edges         = nav['edges']
    path_pts      = nav['path_pts']
    accum_pts     = nav.get('accum_pts', None)

    # Show the full accumulated world-space geometry the navmesh was built from.
    # This explains why the path visits areas with no current-frame Gaussians.
    if accum_pts is not None and len(accum_pts) > 0:
        server.scene.add_point_cloud(
            "/nav/accum_map",
            points     = accum_pts,
            colors     = np.tile([[0.55, 0.55, 0.55]], (len(accum_pts), 1)).astype(np.float32),
            point_size = 0.015,
        )

    if len(clean_obs) > 0:
        server.scene.add_point_cloud(
            "/nav/obstacles",
            points     = clean_obs,
            colors     = np.tile([[1.0, 0.3, 0.1]], (len(clean_obs), 1)).astype(np.float32),
            point_size = 0.04,
        )

    if edges:
        ea        = np.array(edges)
        seg_pts   = np.stack([free_nodes[ea[:, 0]],
                               free_nodes[ea[:, 1]]], axis=1)
        seg_color = np.tile([[0.13, 0.6, 1.0]], (len(edges), 2, 1)).astype(np.float32)
        server.scene.add_line_segments(
            "/nav/edges", points=seg_pts, colors=seg_color, line_width=2.0)

    if len(blocked_nodes) > 0:
        server.scene.add_point_cloud(
            "/nav/blocked",
            points     = blocked_nodes,
            colors     = np.tile([[0.8, 0.1, 0.1]], (len(blocked_nodes), 1)).astype(np.float32),
            point_size = 0.08,
        )

    if len(free_nodes) > 0:
        server.scene.add_point_cloud(
            "/nav/free",
            points     = free_nodes,
            colors     = np.tile([[1.0, 0.8, 0.0]], (len(free_nodes), 1)).astype(np.float32),
            point_size = 0.08,
        )

    if len(path_pts) >= 2:
        path_segs  = np.stack([path_pts[:-1], path_pts[1:]], axis=1)
        path_color = np.tile([[0.0, 0.95, 0.88]], (len(path_segs), 2, 1)).astype(np.float32)
        server.scene.add_line_segments(
            "/nav/path", points=path_segs, colors=path_color, line_width=4.0)
        server.scene.add_point_cloud(
            "/nav/path_nodes",
            points     = path_pts,
            colors     = np.tile([[0.0, 0.95, 0.88]], (len(path_pts), 1)).astype(np.float32),
            point_size = 0.12,
        )


# ─────────────────────────────────────────────
#  MAIN THREAD — viser viewer
# ─────────────────────────────────────────────
def run_viewer(model, device, source, interval, frame_skip, up_idx, port):
    import viser

    threads = [
        CaptureThread(source, interval, frame_skip),
        InferenceThread(model, device),
        NavmeshThread(up_idx),
    ]
    for t in threads:
        t.start()

    server = viser.ViserServer(port=port)
    print(f"\n[Viewer] Open  http://localhost:{port}  in your browser")
    print("[Viewer] Controls: left-drag=orbit  right-drag=pan  scroll=zoom")
    print("[Viewer] Ctrl-C to quit\n")

    camera_set = False

    def _set_camera(client, centroid):
        client.camera.position = (0.0, 0.0, 0.0)
        client.camera.look_at  = tuple(centroid.tolist())
        client.camera.up       = (0.0, 1.0, 0.0)

    while not stop_event.is_set():
        # ── Accumulated colored map (updates every inference frame) ──────────
        try:
            m = map_queue.get_nowait()
            server.scene.add_point_cloud(
                "/scene/map",
                points     = m['pts'],
                colors     = m['colors'],
                point_size = 0.008,
            )
            if not camera_set:
                centroid = m['pts'].mean(axis=0)
                for client in server.get_clients().values():
                    _set_camera(client, centroid)

                @server.on_client_connect
                def _init_cam(client) -> None:
                    _set_camera(client, centroid)

                camera_set = True
        except queue.Empty:
            pass

        # ── Navmesh overlay (every nav recompute) ────────────────────────────
        try:
            nav = navmesh_queue.get_nowait()
            update_navmesh(server, nav)
        except queue.Empty:
            pass

        time.sleep(0.05)

    for t in threads:
        t.join(timeout=5)
    print("\nAll threads stopped.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    global INFER_WIDTH, NAV_INTERVAL_S, MAX_FLOOR_TILT_DEG, OBS_HEIGHT_MAX, \
           ACCUM_ENABLED, MAP_MAX_DEPTH_M, CAM_HEIGHT_M

    parser = argparse.ArgumentParser(
        description="TEQUILA — live Gaussian splat + navigation mesh")
    parser.add_argument('--source',       default='0')
    parser.add_argument('--checkpoint',   default=CHECKPOINT)
    parser.add_argument('--width',        type=int,   default=INFER_WIDTH)
    parser.add_argument('--interval',     type=float, default=CAPTURE_INTERVAL_S)
    parser.add_argument('--frame-skip',   type=int,   default=FRAME_SKIP)
    parser.add_argument('--nav-interval', type=float, default=NAV_INTERVAL_S)
    parser.add_argument('--up-axis',      default=UP_AXIS,
                        choices=['x', 'y', 'z', 'auto'])
    parser.add_argument('--max-tilt',     type=float, default=MAX_FLOOR_TILT_DEG)
    parser.add_argument('--obs-max-height', type=float, default=OBS_HEIGHT_MAX,
                        help='max obstacle height above floor in metres (default: 0.80)')
    parser.add_argument('--cam-height',   type=float, default=CAM_HEIGHT_M,
                        help='camera height above floor in metres (e.g. 0.5). '
                             'Enables per-frame depth rescaling so accumulated frames '
                             'share a consistent metric scale — fixes map stretching '
                             'on robot footage.  0 = disabled (default).')
    parser.add_argument('--no-accum',     action='store_true',
                        help='disable frame accumulation — shows only the current frame '
                             '(use for product/turntable video to avoid stretching artefacts)')
    parser.add_argument('--map-depth',    type=float, default=MAP_MAX_DEPTH_M,
                        help='max depth (metres) of points added to the accumulated map (default: 3.0)')
    parser.add_argument('--port',         type=int,   default=PORT)
    args = parser.parse_args()

    INFER_WIDTH        = args.width
    NAV_INTERVAL_S     = args.nav_interval
    MAX_FLOOR_TILT_DEG = args.max_tilt
    OBS_HEIGHT_MAX     = args.obs_max_height
    CAM_HEIGHT_M       = args.cam_height
    ACCUM_ENABLED      = not args.no_accum
    MAP_MAX_DEPTH_M    = args.map_depth
    if CAM_HEIGHT_M > 0:
        print(f"Depth scale anchor enabled — camera height {CAM_HEIGHT_M:.2f} m above floor.")
    if not ACCUM_ENABLED:
        print("Accumulation disabled — single-frame display mode (product/turntable video).")

    up_idx = ({'x': 0, 'y': 1, 'z': 2}.get(args.up_axis) if args.up_axis != 'auto'
              else None)   # None = detect from first frame (handled in NavmeshThread)

    # If auto, default to Y until we can detect from data
    if up_idx is None:
        up_idx = 1
        print("Up-axis: auto → defaulting to Y (index 1). "
              "Adjust with --up-axis if floor is not found.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("TIP: pass --width 640 for faster inference on CPU.\n")

    if DEPTH_METRIC:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_id = DEPTH_MODEL_ID
        print(f"Loading metric depth model: {model_id}")
        print("(First run will download ~1.3 GB — cached afterwards)")
        depth_processor = AutoImageProcessor.from_pretrained(model_id)
        depth_model_hf  = AutoModelForDepthEstimation.from_pretrained(model_id)
        depth_model_hf.to(device).eval()
        model = (depth_processor, depth_model_hf)   # passed as a tuple
        print("Metric depth model ready.\n")
    else:
        from depth_anything_v2.dpt import DepthAnythingV2
        print("Loading DepthAnythingV2 (vits) — relative depth mode...")
        model = DepthAnythingV2(encoder='vits', features=64,
                                out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        model.to(device).eval()
        print("Model ready.\n")

    try:
        run_viewer(model, device,
                   source     = args.source,
                   interval   = args.interval,
                   frame_skip = args.frame_skip,
                   up_idx     = up_idx,
                   port       = args.port)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nInterrupted.")


if __name__ == '__main__':
    main()
