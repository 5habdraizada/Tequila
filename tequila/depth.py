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

    focal  = w / (2.0 * np.tan(np.radians(cfg.FOV_H_DEG / 2.0)))
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
