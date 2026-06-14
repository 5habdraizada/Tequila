"""
depth.py — Depth model inference and map pre-processing.

Provides:
  • load_model(device)
      Load the Hugging Face metric-depth model (Depth Anything V2).
  • depth_edge_mask(depth_m)
      Detect flying pixels at depth discontinuities (returns boolean keep-mask).
  • run_inference(img, model)
      Run depth inference on one BGR frame; return a metric depth map in metres.
  • frame_to_nav_pts(img, depth_m, focal)
      Back-project a depth map to coarse nav points and fine coloured map points.
  • frame_to_result(img, model)
      Full per-frame pipeline: inference → edge removal → back-projection.
      Returns a dict ready for the inference thread to pass downstream.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image as PilImage

import tequila.config as cfg
from tequila.pointcloud import voxel_downsample_colored, voxel_downsample_pts


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: str):
    """Load and return the Hugging Face metric-depth model.

    Downloads and caches Depth Anything V2 (~1.3 GB on first run).

    Returns:
        Tuple (processor, model) accepted by run_inference().
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print(f"Loading metric depth model: {cfg.DEPTH_MODEL_ID}")
    print("(First run will download ~1.3 GB — cached afterwards)")
    processor = AutoImageProcessor.from_pretrained(cfg.DEPTH_MODEL_ID)
    model_hf  = AutoModelForDepthEstimation.from_pretrained(cfg.DEPTH_MODEL_ID)
    model_hf.to(device).eval()
    print("Metric depth model ready.\n")
    return (processor, model_hf)


# ─────────────────────────────────────────────────────────────────────────────
#  Flying-pixel removal
# ─────────────────────────────────────────────────────────────────────────────

def depth_edge_mask(depth_m: np.ndarray) -> np.ndarray:
    """Return a boolean mask (H×W) — True = valid pixel, False = flying pixel.

    Three-stage pipeline:

    1. Erosion mask — the local neighbourhood minimum flags pixels that sit
       on the far side of a depth discontinuity (classic flying-pixel test).

    2. Gradient mask — Sobel gradient magnitude relative to the pixel depth
       flags transition pixels at depth edges that pass the erosion test.

    3. Dilation — the combined bad-pixel mask is dilated by EDGE_DILATE_PX
       to widen the removal band and catch partially-blended neighbours.
    """
    d = depth_m.astype(np.float32)

    # Stage 1: erosion-based flying-pixel detection
    k         = np.ones((cfg.EDGE_WINDOW_PX, cfg.EDGE_WINDOW_PX), np.float32)
    local_min = cv2.erode(d, k, borderType=cv2.BORDER_REPLICATE)
    bad_erosion = d > local_min * (1.0 + cfg.EDGE_THRESHOLD)

    # Stage 2: gradient-based edge detection (relative gradient)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    safe_depth = np.maximum(d, 0.01)
    bad_gradient = (grad_mag / safe_depth) > cfg.GRAD_THRESHOLD

    # Combine: bad if either test flags the pixel
    bad = (bad_erosion | bad_gradient).astype(np.uint8)

    # Stage 3: dilate bad-pixel mask to widen the removal band
    if cfg.EDGE_DILATE_PX > 0:
        dk  = np.ones((cfg.EDGE_DILATE_PX * 2 + 1,
                       cfg.EDGE_DILATE_PX * 2 + 1), np.uint8)
        bad = cv2.dilate(bad, dk)

    return bad == 0   # True = keep, False = remove


# ─────────────────────────────────────────────────────────────────────────────
#  Fisheye → rectilinear undistortion
# ─────────────────────────────────────────────────────────────────────────────

_undistort_cache: dict = {}
_calib_cache: dict = {}   # cached (K, D, (W_cal, H_cal)) loaded from the .npz


def _resolve_calib_path(path: str) -> str | None:
    """Locate the calibration .npz: try as given, then relative to the repo root."""
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):                       # relative to cwd
        return os.path.abspath(path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(repo_root, path)
    return cand if os.path.exists(cand) else None


def _load_calibration():
    """Load measured fisheye intrinsics from cfg.FISHEYE_CALIB_NPZ, or None.

    Expects the keys written by tools/camera_calibration.py:
      camMatrix (3×3 K) and distCoeff (4×1 D), captured at cfg.FISHEYE_CALIB_WH.
    """
    path = getattr(cfg, "FISHEYE_CALIB_NPZ", "") or ""
    resolved = _resolve_calib_path(path)
    if resolved is None:
        return None
    if resolved in _calib_cache:
        return _calib_cache[resolved]
    try:
        data = np.load(resolved)
        K = np.asarray(data["camMatrix"], dtype=np.float64)
        D = np.asarray(data["distCoeff"], dtype=np.float64).reshape(4, 1)
        wh = tuple(getattr(cfg, "FISHEYE_CALIB_WH", (1280, 720)))
        result = (K, D, wh)
        print(f"[Depth] Loaded fisheye calibration from {resolved} "
              f"(calibrated at {wh[0]}×{wh[1]})")
    except (OSError, KeyError, ValueError) as e:
        print(f"[Depth] Could not load fisheye calibration ({e}); "
              "using equidistant approximation")
        result = None
    _calib_cache[resolved] = result
    return result


def _get_undistort_maps(w: int, h: int):
    """Build (and cache) the fisheye→rectilinear remap for an image of size w×h.

    Uses the measured calibration (cfg.FISHEYE_CALIB_NPZ) when available —
    scaling the camera matrix from the calibration resolution to w×h and using
    the real distortion coefficients.  Falls back to an equidistant model
    (cv2.fisheye, distortion=0) derived from the lens optics:
      f_full = FISHEYE_FOCAL_MM / SENSOR_PIXEL_UM   (px at full sensor)
      f_fish = f_full × (w / SENSOR_FULL_WIDTH_PX)

    The output focal is chosen so the rectilinear result spans UNDISTORT_FOV_DEG.
    Returns (map1, map2, f_out) for cv2.remap; f_out is the focal of the
    undistorted (rectilinear) image, used for back-projection.
    """
    key = (w, h, round(cfg.UNDISTORT_FOV_DEG, 3))
    cached = _undistort_cache.get(key)
    if cached is not None:
        return cached

    calib = _load_calibration()
    if calib is not None:
        # Measured intrinsics — scale K from the calibration resolution to w×h
        # (intrinsics scale linearly with uniform image resize; D is unchanged).
        K_cal, D, (w_cal, h_cal) = calib
        sx, sy = w / float(w_cal), h / float(h_cal)
        K = K_cal.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
    else:
        # Equidistant approximation from the lens optics (distortion = 0).
        f_full = cfg.FISHEYE_FOCAL_MM / (cfg.SENSOR_PIXEL_UM * 1e-3)
        f_fish = f_full * (w / cfg.SENSOR_FULL_WIDTH_PX)
        K = np.array([[f_fish, 0.0, w / 2.0],
                      [0.0, f_fish, h / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)   # pure equidistant (r = f·θ)

    f_out = (w / 2.0) / np.tan(np.radians(cfg.UNDISTORT_FOV_DEG / 2.0))
    P = np.array([[f_out, 0.0, w / 2.0],
                  [0.0, f_out, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), P, (w, h), cv2.CV_16SC2)
    result = (map1, map2, float(f_out))
    _undistort_cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Depth inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(img: np.ndarray, model) -> tuple[np.ndarray, float, float, float]:
    """Run depth inference on one BGR frame.

    Resizes the image to cfg.INFER_WIDTH, runs the metric depth model, and
    returns a depth map clipped to [0.1, MAX_DEPTH_M] metres.

    Args:
        img:   BGR uint8 input frame (any resolution).
        model: (processor, model_hf) tuple returned by load_model().

    Returns:
        Tuple (img, depth_m, focal, cx, cy):
          img     — resized BGR frame at inference resolution.
          depth_m — (H, W) float32 depth map in metres (flying pixels = 0.0).
          focal   — pinhole focal length in pixels at inference resolution.
          cx, cy  — principal point (image centre) in pixels.
    """
    oh, ow = img.shape[:2]
    ih  = int(oh * cfg.INFER_WIDTH / ow)
    img = cv2.resize(img, (cfg.INFER_WIDTH, ih), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    if getattr(cfg, "FISHEYE", False):
        # Undistort fisheye → rectilinear so the pinhole back-projection is valid.
        # Depth then runs on the rectilinear image and back-projection uses the
        # output focal.  (Output FOV < lens FOV ⇒ no black borders to mask.)
        map1, map2, focal = _get_undistort_maps(w, h)
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    else:
        focal = w / (2.0 * np.tan(np.radians(cfg.FOV_H_DEG / 2.0)))
    cx, cy = w / 2.0, h / 2.0

    depth_processor, depth_model_hf = model
    img_pil  = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs   = depth_processor(images=img_pil, return_tensors="pt")
    inputs   = {k: v.to(depth_model_hf.device) for k, v in inputs.items()}
    with torch.no_grad():
        raw = depth_model_hf(**inputs).predicted_depth   # (1, H', W')
    depth_m = torch.nn.functional.interpolate(
        raw.unsqueeze(1), size=(h, w),
        mode="bicubic", align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)
    depth_m = np.clip(depth_m, 0.1, cfg.MAX_DEPTH_M)

    # Remove flying pixels at depth discontinuities (set to 0.0 so they are
    # filtered out by the z < 0 check in back-projection).
    edge_valid = depth_edge_mask(depth_m)
    depth_m    = np.where(edge_valid, depth_m, 0.0)

    return img, depth_m, focal, cx, cy


# ─────────────────────────────────────────────────────────────────────────────
#  Back-projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def frame_to_nav_pts(img: np.ndarray,
                     depth_m: np.ndarray,
                     focal: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project the full depth map to 3-D nav points and a coloured map.

    Two output clouds are produced from the same depth map:
      nav_pts  — coarse (3× voxel) position-only cloud for floor RANSAC / navmesh.
      map_pts  — fine  (1× voxel) coloured cloud for display accumulation.

    Args:
        img:     BGR uint8 frame at inference resolution.
        depth_m: (H, W) float32 depth map; pixels = 0.0 are masked out.
        focal:   pinhole focal length in pixels.

    Returns:
        Tuple (nav_pts, map_pts, map_colors) all float32.
    """
    h, w   = depth_m.shape
    cx, cy = w / 2.0, h / 2.0
    px, py = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    # Standard pinhole back-projection, then flip to Y-up / Z-toward-viewer.
    x3 = (px - cx) * depth_m / focal
    y3 = (py - cy) * depth_m / focal
    z3 = depth_m

    all_pts    = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=-1)
    all_colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    all_pts[:, 1] *= -1   # flip Y to nav convention (Y-up)
    all_pts[:, 2] *= -1   # flip Z (Z-toward-viewer)

    # Pixels zeroed by edge masking have z = 0 after the flip; exclude them.
    valid      = all_pts[:, 2] < 0
    all_pts    = all_pts[valid]
    all_colors = all_colors[valid]

    # Fine coloured cloud — only accumulate nearby points.
    # Distant points amplify any alignment error into long fan arms.
    close_mask = np.abs(all_pts[:, 2]) <= cfg.MAP_MAX_DEPTH_M
    map_all_pts    = all_pts[close_mask]
    map_all_colors = all_colors[close_mask]

    if len(map_all_pts) > 0:
        map_pts, map_colors = voxel_downsample_colored(
            map_all_pts, map_all_colors, cfg.VOXEL_SIZE)
    else:
        map_pts    = np.zeros((0, 3), dtype=np.float32)
        map_colors = np.zeros((0, 3), dtype=np.float32)

    # Coarse position-only cloud — full depth range for navmesh RANSAC.
    nav_pts = voxel_downsample_pts(all_pts, cfg.VOXEL_SIZE * 3)

    return (nav_pts.astype(np.float32),
            map_pts.astype(np.float32),
            map_colors.astype(np.float32))


def frame_to_result(img: np.ndarray, model) -> dict | None:
    """Full per-frame pipeline: inference → edge removal → back-projection.

    This is the single function called by InferenceThread for every frame.
    It runs depth inference, removes flying pixels, and produces both the
    coarse nav cloud (for the navmesh) and the fine coloured cloud (for the
    display map).

    Args:
        img:   BGR uint8 input frame (any resolution).
        model: depth model returned by load_model().

    Returns:
        Dict with keys:
          nav_pts    — (N, 3) float32  coarse nav points (camera coords, Y-up)
          map_pts    — (M, 3) float32  fine coloured cloud positions
          map_colors — (M, 3) float32  RGB in [0, 1] for map_pts
          img        — resized BGR frame at inference resolution
          depth_m    — (H, W) float32  metric depth map (flying pixels = 0.0)
          focal      — pinhole focal length (pixels)
          cx, cy     — principal point (pixels)
        Returns None if back-projection produces no points (blank frame).
    """
    img, depth_m, focal, cx, cy = run_inference(img, model)
    nav_pts, map_pts, map_colors = frame_to_nav_pts(img, depth_m, focal)

    if len(nav_pts) == 0:
        return None

    return dict(
        nav_pts    = nav_pts,
        map_pts    = map_pts,
        map_colors = map_colors,
        img        = img,
        depth_m    = depth_m,
        focal      = focal,
        cx         = cx,
        cy         = cy,
    )
