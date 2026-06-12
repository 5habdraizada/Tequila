"""
odometry.py — Frame-to-frame camera alignment.

Two methods are provided and tried in order each frame:

  1. ORB + PnP (primary)
     Detects ORB keypoints in both frames, applies Lowe's ratio test, lifts
     the matched previous-frame keypoints to 3-D using the stored depth map,
     then calls solvePnPRansac to recover the relative pose.  Fast and accurate
     when there is enough texture.

  2. ICP (fallback)
     Point-to-point Iterative Closest Point on subsampled nav clouds.  Less
     accurate but works when there are too few texture features (blank walls,
     low light).

Both functions return (R_rel, t_rel) in the *navmesh convention*:
  • Y-up, Z-toward-viewer  (opposite of the standard OpenCV camera frame)
  • The transform maps current-frame points → previous-frame points, so the
    accumulated world cloud grows correctly as:
        world_pts = (R_w @ cam_pts.T).T + t_w
"""

import cv2
import numpy as np
from scipy.spatial import KDTree

import tequila.config as cfg


# ─────────────────────────────────────────────────────────────────────────────
#  ORB + PnP Visual Odometry
# ─────────────────────────────────────────────────────────────────────────────

def vo_align(img_prev: np.ndarray,
             depth_prev: np.ndarray,
             img_curr: np.ndarray,
             focal: float,
             cx: float,
             cy: float) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Estimate relative camera pose between two consecutive frames.

    Steps:
      1. Detect ORB keypoints in both frames.
      2. Match descriptors with BFMatcher + Lowe's ratio test.
      3. Back-project matched previous-frame keypoints to 3-D using depth_prev
         (standard camera coords: Y-down, Z-into-scene).
      4. solvePnPRansac to recover R_std, t_std.
      5. Convert to nav convention (Y-up, Z-toward-viewer) via F = diag(1,-1,-1).
      6. Invert so the result maps current → previous (matches ICP convention).

    Args:
        img_prev:   BGR frame at time t−1.
        depth_prev: (H, W) float32 depth map at time t−1 (metres, 0.0 = invalid).
        img_curr:   BGR frame at time t.
        focal:      pinhole focal length in pixels (same for both frames).
        cx, cy:     principal point in pixels.

    Returns:
        (R_rel, t_rel, n_inliers):
          R_rel, t_rel — relative transform in nav coords; None on failure.
          n_inliers    — number of PnP inliers (0 on failure).
    """
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

    orb       = cv2.ORB_create(nfeatures=cfg.VO_MAX_FEATURES)
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, 0

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < cfg.VO_RATIO_TEST * n.distance]

    if len(good) < cfg.VO_MIN_INLIERS:
        return None, None, len(good)

    # Back-project matched previous-frame keypoints to 3-D.
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
        if d < 0.05 or d >= cfg.MAX_DEPTH_M * 0.99:
            continue   # skip missing or saturated depth

        # Standard pinhole back-projection (no Y/Z flip)
        x3 = (u1 - cx) * d / focal
        y3 = (v1 - cy) * d / focal
        z3 = d
        obj_pts.append([x3, y3, z3])
        img_pts.append([u2, v2])

    if len(obj_pts) < cfg.VO_MIN_INLIERS:
        return None, None, len(obj_pts)

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    K = np.array([[focal, 0.0, cx],
                  [0.0, focal, cy],
                  [0.0,   0.0, 1.0]], dtype=np.float64)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None,
        iterationsCount   = cfg.VO_RANSAC_ITER,
        reprojectionError = cfg.VO_REPROJ_ERR,
        confidence        = 0.99,
        flags             = cv2.SOLVEPNP_EPNP,
    )

    if not ret or inliers is None or len(inliers) < cfg.VO_MIN_INLIERS:
        return None, None, (0 if inliers is None else len(inliers))

    # R_std: prev-cam-std coords → curr-cam-std coords
    #   p_curr_std = R_std @ p_prev_std + t_std
    R_std, _ = cv2.Rodrigues(rvec)
    t_std     = tvec.ravel()

    # Convert to nav convention (Y-up, Z-toward-viewer):
    #   F = diag(1, -1, -1)
    #   R_nav = F @ R_std @ F
    #   t_nav = F @ t_std
    F = np.diag([1.0, -1.0, -1.0])
    R_curr_from_prev = F @ R_std @ F
    t_curr_from_prev = F @ t_std

    # Invert so the result maps current → previous (ICP convention):
    #   (R_rel @ pts_curr.T).T + t_rel  gives pts in prev-frame coords
    R_rel = R_curr_from_prev.T
    t_rel = -(R_curr_from_prev.T @ t_curr_from_prev)

    return R_rel, t_rel, len(inliers)


# ─────────────────────────────────────────────────────────────────────────────
#  ICP frame alignment (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def icp_align(source: np.ndarray,
              target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP.  Aligns source onto target using numpy + KDTree.

    This is the fallback when ORB+PnP fails (too few texture features).

    Args:
        source: (N, 3) float  point cloud to align (current frame, nav coords).
        target: (M, 3) float  reference cloud     (previous frame, nav coords).

    Returns:
        (R, t, fitness):
          R       — (3, 3) rotation matrix.
          t       — (3,)   translation vector.
          fitness — fraction of source points within ICP_MAX_DIST of a match.

    Apply the result as:  aligned = (R @ source.T).T + t
    """
    src      = source.astype(np.float64)
    R_acc    = np.eye(3)
    t_acc    = np.zeros(3)
    tree     = KDTree(target.astype(np.float64))
    prev_err = np.inf
    inliers  = np.zeros(len(src), dtype=bool)   # initialised for the fitness return

    for _ in range(cfg.ICP_MAX_ITER):
        dists, idx = tree.query(src, k=1)
        inliers    = dists < cfg.ICP_MAX_DIST
        if inliers.sum() < 10:
            break

        s     = src[inliers]
        t_pts = target[idx[inliers]].astype(np.float64)

        # Optimal rigid transform via SVD (Umeyama method)
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