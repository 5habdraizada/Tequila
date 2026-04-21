"""
navmesh.py — Navigation mesh generation from a 3-D point cloud.

Pipeline (called once per navmesh interval):
  1. extract_gpp()      — detect the floor plane via partitioned sector PCA
                          (falls back to classic RANSAC if sectors are sparse)
  2. generate_nodes()   — lay a uniform grid of waypoint nodes on the floor
  3. denoise_obstacles()— height-filter and SOR-clean the non-floor points
  4. filter_nodes()     — remove nodes that are too close to obstacles
  5. build_edges()      — connect nearby free nodes with line-of-sight check
  6. astar_graph()      — A* shortest path from camera to farthest free node
  7. compute_navmesh()  — orchestrates all of the above; called from NavmeshThread

Coordinate convention: Y-up, Z-toward-viewer (the "nav convention" used
throughout the rest of the codebase).
"""

import heapq

import numpy as np
from scipy.spatial import KDTree

import tequila.config as cfg
from tequila.pointcloud import voxel_downsample_pts, sor_pts


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def up_vector(axis_idx: int) -> np.ndarray:
    """Return the unit vector along the given axis index (0=X, 1=Y, 2=Z)."""
    v = np.zeros(3)
    v[axis_idx] = 1.0
    return v


def point_to_plane_signed(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Signed distance of each point to a plane [nx, ny, nz, d].

    Positive = on the normal side of the plane.
    """
    a, b, c, d = plane
    n = np.array([a, b, c])
    return (points @ n + d) / np.linalg.norm(n)


def build_floor_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors that span the floor plane.

    These are used to project floor points to 2-D for grid generation.
    """
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(normal, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])   # avoid degenerate cross product
    u = np.cross(normal, up); u /= np.linalg.norm(u)
    v = np.cross(normal, u);  v /= np.linalg.norm(v)
    return u, v


# ─────────────────────────────────────────────────────────────────────────────
#  Floor detection
# ─────────────────────────────────────────────────────────────────────────────

def _ransac_fallback(pts: np.ndarray,
                     up_idx: int,
                     max_centroid_up: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """Classic RANSAC plane fit — used when GPP sectors are too sparse.

    Randomly samples triplets of points, tests whether the resulting plane is
    roughly horizontal (within MAX_FLOOR_TILT_DEG of up_idx), and keeps the
    plane with the most inliers.  Refines the winning plane with least-squares.

    Args:
        pts:             (N, 3) float  point cloud in nav coords.
        up_idx:          index of the up axis (0=X, 1=Y, 2=Z).
        max_centroid_up: upper bound on the inlier centroid along the up axis.
                         Passing camera_up ensures the floor is always below
                         the camera, preventing walls or furniture tops from
                         being mistaken for the floor.

    Returns:
        (plane, inlier_indices):
          plane   — [nx, ny, nz, d]  unit-normal plane equation.
          inliers — indices into pts.

    Raises:
        RuntimeError if no plane with enough inliers is found.
    """
    up         = up_vector(up_idx)
    cos_thresh = np.cos(np.radians(cfg.MAX_FLOOR_TILT_DEG))
    n          = len(pts)
    best_inliers: list = []
    best_plane   = None
    best_h       = np.inf

    for _ in range(400):
        idx    = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm   = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if abs(np.dot(normal, up)) < cos_thresh:
            continue
        d   = -normal @ p0
        inl = np.where(np.abs(pts @ normal + d) < cfg.FLOOR_RANSAC_DIST)[0]
        h   = pts[inl, up_idx].mean() if len(inl) else np.inf
        if h > max_centroid_up:
            continue   # plane is above the camera — not a floor candidate
        if len(inl) > len(best_inliers) or (
                len(inl) == len(best_inliers) and h < best_h):
            best_inliers = inl
            best_plane   = np.array([*normal, d])
            best_h       = h

    if best_plane is None or len(best_inliers) < cfg.MIN_FLOOR_POINTS:
        raise RuntimeError(
            f"RANSAC fallback failed ({len(best_inliers)} inliers).")

    # Least-squares refinement on all inliers
    ip = pts[best_inliers]
    c  = ip.mean(0)
    _, _, Vt = np.linalg.svd(ip - c, full_matrices=False)
    normal = Vt[-1]
    d      = -normal @ c
    if np.dot(normal, up) < 0:
        normal = -normal; d = -d

    inliers = np.where(np.abs(pts @ normal + d) < cfg.FLOOR_RANSAC_DIST)[0]
    return np.array([*normal, d]), inliers


def extract_gpp(pts: np.ndarray,
                up_idx: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Ground Principal Plane (GPP) extraction via partitioned sector fitting.

    Based on GCSLAM-B (2024):
      Step 1 — Divide the cloud into GPP_N_SECTORS angular sectors around the
               vertical axis.  Fit a PCA plane to each sector.
      Step 2 — Find the largest group of sectors whose normals agree within
               GPP_CONSENSUS_DEG (majority-vote consensus).
      Step 3 — Average the consensus normals and collect rough inliers.
      Step 4 — Refine with least-squares on all inliers.

    Args:
        pts:    (N, 3) float  point cloud in nav coords.
        up_idx: index of the up axis (0=X, 1=Y, 2=Z).

    Returns:
        (plane, inlier_indices, lambda_min):
          plane      — [nx, ny, nz, d]  unit-normal plane equation.
          inliers    — indices into pts.
          lambda_min — min PCA eigenvalue / N (flatness score; lower = flatter).

    Raises:
        RuntimeError if the floor cannot be found.
    """
    up         = up_vector(up_idx)
    cos_thresh = np.cos(np.radians(cfg.MAX_FLOOR_TILT_DEG))

    # Horizontal axes for sector angle calculation (the two non-up axes)
    h_axes = [i for i in range(3) if i != up_idx]
    a0, a1 = h_axes
    angles  = np.arctan2(pts[:, a1], pts[:, a0])   # (N,) in [−π, π]

    sector_normals = []
    sector_ds      = []
    sector_width   = 2.0 * np.pi / cfg.GPP_N_SECTORS

    # Step 1: fit one plane per sector via PCA ─────────────────────────────
    for s in range(cfg.GPP_N_SECTORS):
        lo   = -np.pi + s * sector_width
        hi   = lo + sector_width
        mask = (angles >= lo) & (angles < hi)
        if mask.sum() < cfg.GPP_MIN_SECTOR_PTS:
            continue

        sp       = pts[mask]
        centroid = sp.mean(0)
        _, _, Vt = np.linalg.svd(sp - centroid, full_matrices=False)
        normal   = Vt[-1]   # eigenvector of the smallest singular value

        if abs(np.dot(normal, up)) < cos_thresh:
            continue   # plane is too tilted for a floor
        if np.dot(normal, up) < 0:
            normal = -normal
        sector_normals.append(normal)
        sector_ds.append(-normal @ centroid)

    if len(sector_normals) < cfg.GPP_MIN_SECTORS:
        raise RuntimeError(
            f"GPP: only {len(sector_normals)} valid sectors "
            f"(need {cfg.GPP_MIN_SECTORS}). Try --max-tilt 45.")

    normals = np.array(sector_normals)   # (S, 3)

    # Step 2: consensus — largest group with agreeing normals ─────────────
    cos_consensus = np.cos(np.radians(cfg.GPP_CONSENSUS_DEG))
    best_group: list = []
    for i in range(len(normals)):
        dots  = normals @ normals[i]
        group = np.where(dots >= cos_consensus)[0].tolist()
        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < cfg.GPP_MIN_SECTORS:
        raise RuntimeError("GPP: no consensus group found.")

    # Step 3: average consensus normal + rough inliers ────────────────────
    avg_normal = normals[best_group].mean(0)
    avg_normal /= np.linalg.norm(avg_normal)
    avg_d      = float(np.mean([sector_ds[i] for i in best_group]))

    rough_inliers = np.where(
        np.abs(pts @ avg_normal + avg_d) < cfg.FLOOR_RANSAC_DIST)[0]
    if len(rough_inliers) < cfg.MIN_FLOOR_POINTS:
        raise RuntimeError(
            f"GPP: {len(rough_inliers)} rough inliers (need {cfg.MIN_FLOOR_POINTS}). "
            "Try --max-tilt 45.")

    # Step 4: least-squares refinement on all inliers ──────────────────────
    ip       = pts[rough_inliers]
    centroid = ip.mean(0)
    _, sv, Vt = np.linalg.svd(ip - centroid, full_matrices=False)
    normal    = Vt[-1]
    d         = -normal @ centroid
    if np.dot(normal, up) < 0:
        normal = -normal; d = -d

    lambda_min = float(sv[-1]) / len(ip)   # flatness quality score

    inliers = np.where(np.abs(pts @ normal + d) < cfg.FLOOR_RANSAC_DIST)[0]
    if len(inliers) < cfg.MIN_FLOOR_POINTS:
        raise RuntimeError(
            f"GPP refined: {len(inliers)} inliers (need {cfg.MIN_FLOOR_POINTS}).")

    return np.array([*normal, d]), inliers, lambda_min


# ─────────────────────────────────────────────────────────────────────────────
#  Node grid generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_nodes(floor_pts: np.ndarray,
                   plane: np.ndarray,
                   normal: np.ndarray) -> np.ndarray:
    """Place a regular grid of waypoint nodes on the detected floor plane.

    Projects the floor points to 2-D floor coordinates, spans a grid at
    NODE_SPACING intervals, then lifts each grid point back to 3-D and raises
    it NODE_ABOVE_FLOOR to avoid z-fighting with the floor mesh.

    Returns:
        (K, 3) float32 array of candidate node positions (includes blocked nodes).
    """
    u, v = build_floor_axes(normal)
    uc, vc = floor_pts @ u, floor_pts @ v
    ug = np.arange(uc.min(), uc.max(), cfg.NODE_SPACING)
    vg = np.arange(vc.min(), vc.max(), cfg.NODE_SPACING)
    uu, vv = np.meshgrid(ug, vg)
    uu, vv = uu.ravel(), vv.ravel()

    floor_tree = KDTree(np.stack([uc, vc], axis=1))
    dists, _   = floor_tree.query(np.stack([uu, vv], axis=1), k=1)
    valid      = dists < cfg.NODE_SPACING * 1.5

    # Project grid nodes directly onto the plane rather than snapping to the
    # nearest floor point.  Since (u, v, normal) form an orthonormal basis and
    # normal·P = -d for any point P on the plane:
    #   P = uu*u + vv*v + (-d)*normal
    # This keeps all nodes at exactly the same height regardless of how much
    # the accumulated floor points are spread vertically by VO drift.
    d = float(plane[3])
    nodes_on_plane = (uu[valid, None] * u +
                      vv[valid, None] * v +
                      (-d) * normal)
    return (nodes_on_plane + normal * cfg.NODE_ABOVE_FLOOR).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Obstacle processing
# ─────────────────────────────────────────────────────────────────────────────

def denoise_obstacles(obs_pts: np.ndarray,
                      plane: np.ndarray,
                      floor_pts: np.ndarray,
                      up_idx: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Height-filter and denoise the obstacle point cloud.

    Uses the actual world up-axis coordinate (not the tilted plane distance) so
    that ceiling points are always excluded regardless of VO rotational drift.
    Only keeps points between OBS_HEIGHT_MIN and OBS_HEIGHT_MAX above the mean
    floor height along the up axis.

    Returns:
        (display_obs, collision_obs):
          display_obs   — tight SOR, shown in viewer as orange dots.
          collision_obs — loose SOR, used for node blocking and edge checking.
    """
    if len(obs_pts) == 0:
        return obs_pts, obs_pts

    # Use the world up-axis coordinate for height — robust against plane tilt
    # and VO rotational drift that would otherwise push ceiling points into the
    # valid obstacle height band when measured via signed plane distance.
    floor_axis_mean = float(floor_pts[:, up_idx].mean()) if len(floor_pts) > 0 else 0.0
    axis_heights    = obs_pts[:, up_idx] - floor_axis_mean
    obs_pts = obs_pts[(axis_heights > cfg.OBS_HEIGHT_MIN) & (axis_heights < cfg.OBS_HEIGHT_MAX)]
    if len(obs_pts) == 0:
        return obs_pts, obs_pts

    if len(floor_pts) > 0:
        dists, _ = KDTree(floor_pts).query(obs_pts, k=1)
        obs_pts  = obs_pts[dists > cfg.NODE_SPACING * cfg.FLOOR_PROX_FACTOR]
    if len(obs_pts) < 10:
        return obs_pts, obs_pts

    # Voxel downsample first — cheap and removes most noise.
    collision_obs = voxel_downsample_pts(obs_pts, cfg.OBS_VOXEL_SIZE)
    display_obs   = collision_obs

    # SOR is O(n log n) — only run it on small clouds where it adds value.
    # Large accumulated obstacle clouds are already clean after voxel downsampling.
    if len(collision_obs) <= 3000:
        # Build the KDTree once and derive both clouds from the same query.
        tree       = KDTree(collision_obs)
        dists, _   = tree.query(collision_obs, k=cfg.OBS_DENOISE_NB + 1)
        mean_dists = dists[:, 1:].mean(axis=1)
        mu, sigma  = mean_dists.mean(), mean_dists.std()
        keep_tight = mean_dists <= (mu + cfg.OBS_DENOISE_STD       * sigma)
        keep_loose = mean_dists <= (mu + cfg.OBS_DENOISE_STD * 1.5 * sigma)
        display_obs   = collision_obs[keep_tight]
        collision_obs = collision_obs[keep_loose]

    return display_obs, collision_obs


def filter_nodes(nodes: np.ndarray,
                 collision_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split nodes into free and blocked based on obstacle clearance.

    A node is blocked if any obstacle point falls within OBS_CLEARANCE_R metres.

    Returns:
        (free_nodes, blocked_nodes).
    """
    if len(collision_obs) == 0:
        return nodes, np.zeros((0, 3))
    dists, _ = KDTree(collision_obs).query(nodes, k=1)
    free_mask = dists >= cfg.OBS_CLEARANCE_R
    return nodes[free_mask], nodes[~free_mask]


# ─────────────────────────────────────────────────────────────────────────────
#  Graph construction and pathfinding
# ─────────────────────────────────────────────────────────────────────────────

def build_edges(free_nodes: np.ndarray,
                collision_obs: np.ndarray) -> list[tuple[int, int]]:
    """Connect adjacent free nodes that have a clear line of sight.

    For each pair of free nodes within EDGE_MAX_DIST, samples EDGE_CHECK_STEPS
    points along the connecting segment and checks that none are closer than
    OBS_CLEARANCE_R to an obstacle.

    Returns:
        List of (i, j) index pairs representing passable edges.
    """
    if len(free_nodes) < 2:
        return []
    pairs = KDTree(free_nodes).query_pairs(r=cfg.EDGE_MAX_DIST)
    if len(collision_obs) == 0:
        return list(pairs)
    obs_tree = KDTree(collision_obs)
    edges = []
    for i, j in pairs:
        sample = np.linspace(free_nodes[i], free_nodes[j], cfg.EDGE_CHECK_STEPS)
        if obs_tree.query(sample, k=1)[0].min() >= cfg.OBS_CLEARANCE_R:
            edges.append((i, j))
    return edges


def astar_graph(free_nodes: np.ndarray,
                edges: list[tuple[int, int]],
                start_idx: int,
                goal_idx: int) -> list[int]:
    """A* shortest path on the free-node graph.

    Args:
        free_nodes: (N, 3) array of waypoint positions.
        edges:      list of (i, j) connectivity pairs from build_edges().
        start_idx:  index of the start node.
        goal_idx:   index of the goal node.

    Returns:
        List of node indices from start to goal, or [] if no path exists.
    """
    graph: dict[int, list[tuple[int, float]]] = {
        i: [] for i in range(len(free_nodes))}
    for i, j in edges:
        d = float(np.linalg.norm(free_nodes[i] - free_nodes[j]))
        graph[i].append((j, d))
        graph[j].append((i, d))

    def heuristic(i: int) -> float:
        return float(np.linalg.norm(free_nodes[i] - free_nodes[goal_idx]))

    heap = [(heuristic(start_idx), 0.0, start_idx)]
    came: dict[int, int | None] = {start_idx: None}
    g:    dict[int, float]      = {start_idx: 0.0}

    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == goal_idx:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        for nb, d in graph.get(cur, []):
            ng = cost + d
            if ng < g.get(nb, float("inf")):
                g[nb]    = ng
                came[nb] = cur
                heapq.heappush(heap, (ng + heuristic(nb), ng, nb))
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level navmesh pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_navmesh(pts: np.ndarray,
                    up_idx: int,
                    camera_origin: np.ndarray | None = None,
                    camera_forward: np.ndarray | None = None) -> dict | None:
    """Run the full navmesh pipeline on an accumulated point cloud.

    Orchestrates floor detection → node generation → obstacle denoising →
    node filtering → edge building → A* path to the farthest reachable node.

    The A* goal is chosen by trying free nodes from farthest to nearest until
    a connected path is found.  This implements a simple frontier-exploration
    strategy: always head for the most distant reachable point.

    Args:
        pts:            (N, 3) float  accumulated world-space point cloud.
        up_idx:         index of the up axis (0=X, 1=Y, 2=Z).
        camera_origin:  (3,) world-space camera position  (default: origin).
        camera_forward: (3,) world-space forward direction (default: −Z).

    Returns:
        Dict with keys:
          clean_obs, free_nodes, blocked_nodes, edges, path_pts, lambda_min
        or None if the floor cannot be detected.
    """
    if len(pts) < cfg.MIN_FLOOR_POINTS * 2:
        return None

    if camera_origin  is None: camera_origin  = np.zeros(3)
    if camera_forward is None: camera_forward = np.array([0.0, 0.0, -1.0])
    camera_origin  = np.asarray(camera_origin,  dtype=np.float64)
    camera_forward = np.asarray(camera_forward, dtype=np.float64)
    camera_forward /= np.linalg.norm(camera_forward) + 1e-8

    # The floor must be below the camera.  Any flat surface above this threshold
    # (wall, table top, ceiling) is rejected immediately and RANSAC is retried
    # with the camera height as a hard upper bound.
    camera_up    = float(camera_origin[up_idx])
    max_floor_up = camera_up - 0.05   # floor centroid must be at least 5 cm below camera

    def _floor_is_valid(f_idx: np.ndarray) -> bool:
        return float(pts[f_idx, up_idx].mean()) < max_floor_up

    # Floor detection: try GPP first, fall back to RANSAC.
    try:
        plane, floor_idx, lambda_min = extract_gpp(pts, up_idx)
        quality = ("flat ✓"  if lambda_min < 0.001 else
                   "rough ~" if lambda_min < 0.005 else "poor ✗")
        print(f"[GPP]   λ_min={lambda_min:.6f}  ({quality})")
        if not _floor_is_valid(floor_idx):
            centroid_up = float(pts[floor_idx, up_idx].mean())
            print(f"[GPP]   floor centroid ({centroid_up:.2f}) is above camera "
                  f"({camera_up:.2f}) — wrong surface, retrying with RANSAC")
            raise RuntimeError("GPP found surface above camera")
    except RuntimeError as gpp_err:
        print(f"[GPP] {gpp_err}  — falling back to RANSAC")
        try:
            plane, floor_idx = _ransac_fallback(pts, up_idx,
                                                 max_centroid_up=max_floor_up)
            lambda_min = 0.005   # unknown quality; treated as rough
            print("[GPP] RANSAC fallback succeeded")
        except RuntimeError as e:
            print(f"[Navmesh] {e}")
            return None

    a, b, c, d = plane
    normal       = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    obstacle_idx = np.setdiff1d(np.arange(len(pts)), floor_idx)
    floor_pts    = pts[floor_idx]
    obs_pts      = pts[obstacle_idx]

    nodes                      = generate_nodes(floor_pts, plane, normal)
    display_obs, collision_obs = denoise_obstacles(obs_pts, plane, floor_pts, up_idx)
    free_nodes,  blocked_nodes = filter_nodes(nodes, collision_obs)
    edges                      = build_edges(free_nodes, collision_obs)

    # Frontier-exploration path: start at the free node nearest the camera,
    # then try goals from farthest to nearest until A* finds a connected path.
    path_pts = np.zeros((0, 3))
    if len(free_nodes) >= 2 and edges:
        node_tree  = KDTree(free_nodes)
        start_idx  = int(node_tree.query(camera_origin)[1])

        dists_from_cam  = np.linalg.norm(free_nodes - camera_origin, axis=1)
        goal_candidates = np.argsort(-dists_from_cam)   # farthest first

        path_idx: list[int] = []
        for goal_idx in goal_candidates:
            if goal_idx == start_idx:
                continue
            path_idx = astar_graph(free_nodes, edges, start_idx, int(goal_idx))
            if path_idx:
                break   # found the farthest reachable node

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
        lambda_min    = lambda_min,
    )
