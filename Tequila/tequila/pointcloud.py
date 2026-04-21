"""
pointcloud.py — Point-cloud utilities.

Provides:
  • voxel_downsample_colored / voxel_downsample_pts
      Fast voxel-grid deduplication using integer hash keys.
  • sor_colored / sor_pts
      Statistical Outlier Removal (SOR) based on mean neighbour distance.
  • segment_product / raycast_occlusion_mask / wall_removal_mask
      Optional masking helpers (not used in the default pipeline but available
      for product-photography or indoor-scene filtering).
"""

import cv2
import numpy as np
from scipy.spatial import KDTree

import tequila.config as cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Voxel downsampling
# ─────────────────────────────────────────────────────────────────────────────

def voxel_downsample_colored(pts: np.ndarray, colors: np.ndarray,
                              voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a coloured point cloud to one point per voxel cell.

    Uses the first point that falls into each cell (no averaging).

    Args:
        pts:        (N, 3) float32 positions.
        colors:     (N, 3) float32 RGB in [0, 1].
        voxel_size: cell edge length in metres.

    Returns:
        Tuple of (downsampled_pts, downsampled_colors).
    """
    vox  = np.floor(pts / voxel_size).astype(np.int64)
    vox -= vox.min(axis=0)
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    keys     = (x << 42) | (y << 21) | z
    _, first = np.unique(keys, return_index=True)
    return pts[first], colors[first]


def voxel_downsample_pts(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a position-only point cloud to one point per voxel cell.

    Cheaper than the coloured variant — used by the navmesh pipeline where
    colours are not needed.
    """
    vox  = np.floor(pts / voxel_size).astype(np.int64)
    keys = vox[:, 0] * 1_000_000_000 + vox[:, 1] * 1_000_000 + vox[:, 2]
    _, first = np.unique(keys, return_index=True)
    return pts[first]


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical Outlier Removal (SOR)
# ─────────────────────────────────────────────────────────────────────────────

def sor_colored(pts: np.ndarray, colors: np.ndarray,
                nb: int = 20, std_ratio: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """SOR that keeps pts and colors synchronised.

    Removes points whose mean distance to their ``nb`` nearest neighbours
    exceeds  mean + std_ratio × std  across all points.

    Returns:
        Filtered (pts, colors).
    """
    if len(pts) < nb + 1:
        return pts, colors
    tree       = KDTree(pts)
    dists, _   = tree.query(pts, k=nb + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + std_ratio * mean_dists.std()
    keep       = mean_dists <= threshold
    return pts[keep], colors[keep]


def sor_pts(pts: np.ndarray, nb: int = 10, std_ratio: float = 1.5) -> np.ndarray:
    """Position-only SOR.  Used by the navmesh obstacle cloud.

    Returns:
        Filtered pts.
    """
    if len(pts) < nb + 1:
        return pts
    tree       = KDTree(pts)
    dists, _   = tree.query(pts, k=nb + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= threshold]


# ─────────────────────────────────────────────────────────────────────────────
#  Masking helpers (available but not active in the default pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def segment_product(bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 foreground mask for product-photography images.

    Flood-fills from the image border to identify connected background
    regions (low saturation, high value), then dilates the result.  The
    return value is 255 for foreground and 0 for background.
    """
    h, w  = bgr.shape[:2]
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v  = hsv[:, :, 1], hsv[:, :, 2]
    candidate  = ((s < cfg.SAT_THRESH) & (v > cfg.VAL_THRESH)).astype(np.uint8)
    filled     = candidate.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    border = ([(0, x) for x in range(w)] + [(h - 1, x) for x in range(w)] +
              [(y, 0) for y in range(h)] + [(y, w - 1) for y in range(h)])
    for y, x in border:
        if filled[y, x] == 1:
            cv2.floodFill(filled, flood_mask, (x, y), 2)
    bg_mask = (filled == 2).astype(np.uint8)
    if cfg.BG_DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (cfg.BG_DILATE_PX * 2 + 1,) * 2)
        bg_mask = cv2.dilate(bg_mask, k)
    return ((1 - bg_mask) * 255).astype(np.uint8)


def raycast_occlusion_mask(depth_m: np.ndarray,
                            product_mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask that keeps pixels not occluded behind the product.

    Uses a sliding-window local minimum on product pixels to decide whether
    a background pixel is in the shadow of a foreground object.
    """
    sentinel   = depth_m.max() + 1.0
    prod_depth = np.where(product_mask > 0, depth_m, sentinel)
    kernel     = np.ones((cfg.OCCLUSION_WIN_PX,) * 2, np.float32)
    local_min  = cv2.erode(prod_depth.astype(np.float32), kernel)
    return depth_m <= (local_min + cfg.OCCLUSION_TOLERANCE)


def wall_removal_mask(depth_m: np.ndarray) -> np.ndarray:
    """Return a boolean mask that removes flat far-depth regions (walls).

    Pixels that are both deep (top WALL_DEPTH_PERCENTILE %) and locally flat
    (low local std) are classified as walls and excluded.
    """
    d_norm = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)
    r  = cfg.WALL_LOCAL_RADIUS * 2 + 1
    k  = np.ones((r, r), np.float32) / (r * r)
    mu  = cv2.filter2D(d_norm, -1, k)
    mu2 = cv2.filter2D(d_norm ** 2, -1, k)
    local_std = np.sqrt(np.clip(mu2 - mu ** 2, 0, None))
    depth_thr = np.percentile(d_norm, cfg.WALL_DEPTH_PERCENTILE)
    is_wall   = (d_norm > depth_thr) & (local_std < cfg.WALL_FLATNESS_THRESHOLD)
    ksize  = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    is_wall = cv2.morphologyEx(is_wall.astype(np.uint8),
                               cv2.MORPH_CLOSE, kernel).astype(bool)
    return ~is_wall
