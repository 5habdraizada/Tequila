"""
depth.py — Depth model inference and map pre-processing.

Provides:
  • load_model(device)
      Load whichever depth model is configured (metric HF model or local ViT-S).
  • depth_edge_mask(depth_m)
      Detect flying pixels at depth discontinuities (returns boolean keep-mask).
  • anchor_depth_scale(depth_m, focal)
      Rescale relative-depth maps so all frames share the same metric scale.
      No-op when DEPTH_METRIC = True.
  • run_inference(img, model)
      Run depth inference on one BGR frame; return a metric depth map in metres.
  • frame_to_nav_pts(img, depth_m, focal)
      Back-project a depth map to coarse nav points and fine coloured map points.
  • frame_to_result(img, model)
      Full per-frame pipeline: inference → edge removal → back-projection.
      Returns a dict ready for the inference thread to pass downstream.
"""

import cv2
import numpy as np
import torch
from PIL import Image as PilImage

import tequila.config as cfg
from tequila.pointcloud import voxel_downsample_colored, voxel_downsample_pts

# Auto-scale baseline: set from the first frame that has a detectable floor.
# Subsequent frames are rescaled to match this height, giving consistent metric
# scale across the whole accumulated map without needing --cam-height.
_scale_baseline_m: float | None = None   # metres; None = not yet established


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: str):
    """Load and return the configured depth model.

    When DEPTH_METRIC = True (default) this downloads / caches the Hugging Face
    metric-depth transformer (~1.3 GB on first run).

    When DEPTH_METRIC = False this loads the local relative-depth ViT-S weights
    from cfg.CHECKPOINT — much faster to start, but requires scale anchoring.

    Returns:
        A model object (or tuple) accepted by run_inference().
    """
    if cfg.DEPTH_METRIC:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f"Loading metric depth model: {cfg.DEPTH_MODEL_ID}")
        print("(First run will download ~1.3 GB — cached afterwards)")
        processor = AutoImageProcessor.from_pretrained(cfg.DEPTH_MODEL_ID)
        model_hf  = AutoModelForDepthEstimation.from_pretrained(cfg.DEPTH_MODEL_ID)
        model_hf.to(device).eval()
        print("Metric depth model ready.\n")
        return (processor, model_hf)   # passed as a tuple to run_inference
    else:
        from depth_anything_v2.dpt import DepthAnythingV2
        print("Loading DepthAnythingV2 (vits) — relative depth mode...")
        model = DepthAnythingV2(encoder="vits", features=64,
                                out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(cfg.CHECKPOINT, map_location="cpu"))
        model.to(device).eval()
        print("Relative depth model ready.\n")
        return model


# ─────────────────────────────────────────────────────────────────────────────
#  Flying-pixel removal
# ─────────────────────────────────────────────────────────────────────────────

def depth_edge_mask(depth_m: np.ndarray) -> np.ndarray:
    """Return a boolean mask (H×W) — True = valid pixel, False = flying pixel.

    Depth models blend intermediate values onto boundary pixels between a near
    object and a far background.  Back-projected, these become rays or spikes
    behind every object edge.

    Detection: morphological erosion gives the local depth minimum in an
    EDGE_WINDOW_PX neighbourhood.  A pixel is a flying pixel if its depth
    exceeds that minimum by more than EDGE_THRESHOLD (relative).

    Example (metric depth, threshold=0.25):
      chair leg at 1.5 m, background at 4.0 m
        local_min = 1.5 m  →  limit = 1.5 × 1.25 = 1.875 m
        4.0 m > 1.875 m  ✓  background pixel removed (it's behind the chair)
      wall corner at 2.0 m, adjacent wall at 2.4 m
        local_min = 2.0 m  →  limit = 2.5 m
        2.4 m < 2.5 m  ✓  wall pixel kept (small depth jump, same surface)
    """
    k         = np.ones((cfg.EDGE_WINDOW_PX, cfg.EDGE_WINDOW_PX), np.float32)
    local_min = cv2.erode(depth_m.astype(np.float32), k,
                          borderType=cv2.BORDER_REPLICATE)
    return depth_m <= local_min * (1.0 + cfg.EDGE_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
#  Depth scale anchoring (relative-depth mode only)
# ─────────────────────────────────────────────────────────────────────────────

def anchor_depth_scale(depth_m: np.ndarray, focal: float) -> np.ndarray:
    """Rescale a relative-depth map so every frame shares the same metric scale.

    Relative-depth models normalise per-frame (min→0, max→MAX_DEPTH_M), so the
    same physical distance can map to different pixel values across frames.
    When accumulated, this causes the map to fan and stretch.

    Fix: detect the floor plane via quick RANSAC on the bottom 40 % of the
    image.  Record the first successful camera-to-floor distance as a baseline.
    Rescale all subsequent frames so their floor sits at the same baseline
    height.  Corrections outside SCALE_CLAMP are skipped (wrong surface).

    This function is a no-op when cfg.DEPTH_METRIC = True (the model already
    outputs metric metres with no per-frame drift).

    Args:
        depth_m: (H, W) float32 depth map in the model's units.
        focal:   pinhole focal length (pixels) for this frame.

    Returns:
        Rescaled depth map (same shape), or the original if detection fails.
    """
    global _scale_baseline_m

    if cfg.DEPTH_METRIC:
        return depth_m   # already metric — nothing to do

    h, w   = depth_m.shape
    cx, cy = w / 2.0, h / 2.0

    def _sample_pts(y_start: int, step: int) -> np.ndarray:
        """Back-project a sparse grid of pixels to 3-D nav coords (Y-up)."""
        py, px = np.meshgrid(np.arange(y_start, h, step, dtype=np.float32),
                             np.arange(0, w, step, dtype=np.float32), indexing="ij")
        d_sub  = depth_m[y_start::step, ::step].astype(np.float32)
        x3 =  (px - cx) * d_sub / focal
        y3 = -((py - cy) * d_sub / focal)   # Y-up convention
        z3 = -d_sub
        pts   = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=-1)
        valid = (d_sub.ravel() > 0.05) & (d_sub.ravel() < cfg.MAX_DEPTH_M * 0.98)
        return pts[valid]

    # Try the bottom 40 % of the image (floor most likely here).
    # Fall back to the full image at sparser sampling if too few points.
    pts = _sample_pts(int(h * 0.60), 2)
    if len(pts) < cfg.SCALE_MIN_INLIERS:
        pts = _sample_pts(0, 4)
    if len(pts) < cfg.SCALE_MIN_INLIERS:
        return depth_m   # not enough data to detect the floor

    # Quick RANSAC plane fit
    idx = np.random.choice(len(pts), min(800, len(pts)), replace=False)
    pts = pts[idx]
    up           = np.array([0.0, 1.0, 0.0])
    cos_thresh   = np.cos(np.radians(cfg.MAX_FLOOR_TILT_DEG))
    best_inliers = 0
    best_floor_y = None

    for _ in range(cfg.SCALE_RANSAC_ITER):
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
        inlier_mask = np.abs(pts @ n + d_plane) < cfg.SCALE_INLIER_DIST
        n_inl = int(inlier_mask.sum())
        if n_inl > best_inliers:
            best_inliers = n_inl
            best_floor_y = float(pts[inlier_mask, 1].mean())

    if best_floor_y is None or best_floor_y >= -0.05 or best_inliers < cfg.SCALE_MIN_INLIERS:
        return depth_m   # floor not reliably detected this frame

    detected_h = abs(best_floor_y)   # camera-to-floor in this frame's units

    # Establish or look up the target height
    if cfg.CAM_HEIGHT_M > 0:
        target_h = cfg.CAM_HEIGHT_M          # user-supplied metric height
    elif _scale_baseline_m is None:
        _scale_baseline_m = detected_h
        print(f"[Scale]  Baseline locked at {detected_h:.3f} m "
              f"(inliers={best_inliers}) — all frames will match this scale")
        return depth_m                       # first frame needs no rescaling
    else:
        target_h = _scale_baseline_m

    raw_scale = target_h / detected_h

    # Reject corrections outside SCALE_CLAMP — they almost certainly mean the
    # RANSAC latched onto a wall, table, or ceiling instead of the floor.
    lo, hi = cfg.SCALE_CLAMP
    if not (lo <= raw_scale <= hi):
        print(f"[Scale]  floor={best_floor_y:.3f}m  scale={raw_scale:.3f}× "
              f"— out of [{lo},{hi}], skipping (wrong surface?)")
        return depth_m

    scale = float(raw_scale)
    if abs(scale - 1.0) > 0.02:
        print(f"[Scale]  floor={best_floor_y:.3f}m  scale={scale:.3f}×")
    return depth_m * scale


# ─────────────────────────────────────────────────────────────────────────────
#  Depth inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(img: np.ndarray, model) -> tuple[np.ndarray, float, float, float]:
    """Run depth inference on one BGR frame.

    Resizes the image to cfg.INFER_WIDTH, runs the depth model, and returns
    a metric depth map clipped to [0.1, MAX_DEPTH_M] metres.

    Args:
        img:   BGR uint8 input frame (any resolution).
        model: depth model object returned by load_model().

    Returns:
        Tuple (depth_m, focal, cx, cy):
          depth_m — (H, W) float32 depth map in metres (flying pixels = 0.0).
          focal   — pinhole focal length in pixels at inference resolution.
          cx, cy  — principal point (image centre) in pixels.
    """
    oh, ow = img.shape[:2]
    ih  = int(oh * cfg.INFER_WIDTH / ow)
    img = cv2.resize(img, (cfg.INFER_WIDTH, ih), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    focal = w / (2.0 * np.tan(np.radians(cfg.FOV_H_DEG / 2.0)))
    cx, cy = w / 2.0, h / 2.0

    if cfg.DEPTH_METRIC:
        # Transformers metric-depth model — output is already in metres.
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
    else:
        # Local relative-depth ViT-S model + per-frame scale anchor.
        with torch.no_grad():
            depth_raw = model.infer_image(img)
        depth_raw = cv2.bilateralFilter(
            depth_raw.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
        dmin, dmax = depth_raw.min(), depth_raw.max()
        depth_m    = (1.0 - (depth_raw - dmin) / (dmax - dmin)) * cfg.MAX_DEPTH_M
        depth_m    = anchor_depth_scale(depth_m, focal)

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
