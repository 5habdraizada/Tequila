"""Frame-to-frame camera alignment.

Tries SIFT+PnP first (matches keypoints, lifts the previous frame to 3-D via
its depth map, solvePnPRansac for the relative pose), falling back to
point-to-point ICP when there's too little texture (blank walls, low light).

Both return (R_rel, t_rel) in the navmesh convention (Y-up, Z-toward-viewer,
opposite of OpenCV's camera frame), mapping current-frame points to
previous-frame points so the world cloud accumulates as:
    world_pts = (R_w @ cam_pts.T).T + t_w
"""

import cv2
import numpy as np
from scipy.spatial import KDTree

import tequila.config as cfg


def vo_align(img_prev: np.ndarray,
             depth_prev: np.ndarray,
             img_curr: np.ndarray,
             focal: float,
             cx: float,
             cy: float) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Estimate relative camera pose (R_rel, t_rel, n_inliers) between two frames.

    Matches SIFT descriptors (BFMatcher L2 + Lowe's ratio test), back-projects
    the matched previous-frame keypoints to 3-D via depth_prev, then recovers
    the pose with solvePnPRansac. Returns (None, None, 0) on failure.
    """
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

    sift      = cv2.SIFT_create(nfeatures=cfg.VO_MAX_FEATURES)
    kp1, des1 = sift.detectAndCompute(gray_prev, None)
    kp2, des2 = sift.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, 0

    bf  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < cfg.VO_RATIO_TEST * n.distance]

    if len(good) < cfg.VO_MIN_INLIERS:
        return None, None, len(good)

    # Back-project to 3-D in standard camera coords (Y-down, Z-into-scene)
    # so the pinhole intrinsics K apply directly.
    hd, wd = depth_prev.shape
    obj_pts, img_pts = [], []
    for m in good:
        u1, v1 = kp1[m.queryIdx].pt   # prev-frame pixel
        u2, v2 = kp2[m.trainIdx].pt   # curr-frame pixel

        ui, vi = int(round(u1)), int(round(v1))
        if not (0 <= ui < wd and 0 <= vi < hd):
            continue
        d = float(depth_prev[vi, ui])
        if d < 0.05 or d > cfg.MAP_MAX_DEPTH_M or d >= cfg.MAX_DEPTH_M * 0.99:
            continue   # skip missing, saturated, or far depth (noisy for PnP)

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

    # solvePnPRansac gives p_curr_std = R_std @ p_prev_std + t_std in standard
    # camera coords. Flip to nav convention (Y-up, Z-toward-viewer) via
    # F = diag(1,-1,-1): R_nav = F @ R_std @ F, t_nav = F @ t_std.
    R_std, _ = cv2.Rodrigues(rvec)
    t_std     = tvec.ravel()

    F = np.diag([1.0, -1.0, -1.0])
    R_curr_from_prev = F @ R_std @ F
    t_curr_from_prev = F @ t_std

    # Invert so the result maps current → previous, matching ICP's convention:
    # (R_rel @ pts_curr.T).T + t_rel gives points in the previous frame.
    R_rel = R_curr_from_prev.T
    t_rel = -(R_curr_from_prev.T @ t_curr_from_prev)

    return R_rel, t_rel, len(inliers)


def icp_align(source: np.ndarray,
              target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP aligning source onto target (both (N,3) nav-coord clouds).

    Fallback when SIFT+PnP fails. Returns (R, t, fitness); apply as
    aligned = (R @ source.T).T + t. fitness is the inlier fraction within ICP_MAX_DIST.
    """
    src      = source.astype(np.float64)
    R_acc    = np.eye(3)
    t_acc    = np.zeros(3)
    tree     = KDTree(target.astype(np.float64))
    prev_err = np.inf
    inliers  = np.zeros(len(src), dtype=bool)   # in case the loop never runs

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