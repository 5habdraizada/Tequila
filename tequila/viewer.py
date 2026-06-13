"""
viewer.py — Viser 3-D viewer: scene update helpers and the main viewer loop.

The viewer loop polls two queues every 50 ms:
  map_queue     — accumulated coloured point cloud (updates every inference frame)
  navmesh_queue — navmesh overlay: obstacles, free nodes, edges, A* path

Visual legend
-------------
  Orange points (/nav/obstacles)  — detected obstacles (height-filtered + denoised).
  Red   points (/nav/blocked)     — navmesh nodes blocked by obstacles.
  Yellow points (/nav/free)       — free (passable) navmesh nodes.
  Blue  lines  (/nav/edges)       — passable edges between free nodes.
  Teal  line   (/nav/path)        — A* path to the farthest reachable node.
  Teal  points (/nav/path_nodes)  — waypoints along the path.
"""

import queue
import time

import numpy as np

import tequila.config as cfg
from tequila.threads import (
    CaptureThread, InferenceThread, NavmeshThread,
    map_queue, navmesh_queue, stop_event,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Scene update helpers
# ─────────────────────────────────────────────────────────────────────────────

def update_navmesh(server, nav: dict) -> None:
    """Push all navmesh layers to the viser scene.

    Each layer is updated atomically — stale items from the previous navmesh
    recompute are automatically replaced when the new ones are added.

    Args:
        server: viser.ViserServer instance.
        nav:    dict returned by compute_navmesh(), plus nav["accum_pts"].
    """
    clean_obs     = nav["clean_obs"]
    free_nodes    = nav["free_nodes"]
    blocked_nodes = nav["blocked_nodes"]
    edges         = nav["edges"]
    path_pts      = nav["path_pts"]

    # Note: the full coloured map (/scene/map) already shows the geometry the
    # navmesh was built from, so we no longer draw a second grey copy here —
    # the duplicate cloud only muddied the overlay.

    # Obstacle cloud (orange)
    if len(clean_obs) > 0:
        server.scene.add_point_cloud(
            "/nav/obstacles",
            points     = clean_obs,
            colors     = np.tile([[1.0, 0.3, 0.1]], (len(clean_obs), 1)).astype(np.float32),
            point_size = 0.015,
        )

    # Graph edges (blue lines)
    if edges:
        ea        = np.array(edges)
        seg_pts   = np.stack([free_nodes[ea[:, 0]],
                               free_nodes[ea[:, 1]]], axis=1)
        seg_color = np.tile([[0.13, 0.6, 1.0]], (len(edges), 2, 1)).astype(np.float32)
        server.scene.add_line_segments(
            "/nav/edges", points=seg_pts, colors=seg_color, line_width=2.0)

    # Blocked nodes (red)
    if len(blocked_nodes) > 0:
        server.scene.add_point_cloud(
            "/nav/blocked",
            points     = blocked_nodes,
            colors     = np.tile([[0.8, 0.1, 0.1]], (len(blocked_nodes), 1)).astype(np.float32),
            point_size = 0.025,
        )

    # Free nodes (yellow)
    if len(free_nodes) > 0:
        server.scene.add_point_cloud(
            "/nav/free",
            points     = free_nodes,
            colors     = np.tile([[1.0, 0.8, 0.0]], (len(free_nodes), 1)).astype(np.float32),
            point_size = 0.025,
        )

    # Robot trajectory — every camera position since frame 1 (green line)
    trajectory = nav.get("trajectory", None)
    if trajectory is not None and len(trajectory) >= 2:
        traj_segs  = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
        traj_color = np.tile([[0.1, 0.9, 0.1]], (len(traj_segs), 2, 1)).astype(np.float32)
        server.scene.add_line_segments(
            "/nav/trajectory", points=traj_segs, colors=traj_color, line_width=3.0)
        # Mark each recorded position as a small dot
        server.scene.add_point_cloud(
            "/nav/trajectory_pts",
            points     = trajectory,
            colors     = np.tile([[0.1, 0.9, 0.1]], (len(trajectory), 1)).astype(np.float32),
            point_size = 0.02,
        )

    # A* path (teal line + waypoints)
    if len(path_pts) >= 2:
        path_segs  = np.stack([path_pts[:-1], path_pts[1:]], axis=1)
        path_color = np.tile([[0.0, 0.95, 0.88]], (len(path_segs), 2, 1)).astype(np.float32)
        server.scene.add_line_segments(
            "/nav/path", points=path_segs, colors=path_color, line_width=4.0)
        server.scene.add_point_cloud(
            "/nav/path_nodes",
            points     = path_pts,
            colors     = np.tile([[0.0, 0.95, 0.88]], (len(path_pts), 1)).astype(np.float32),
            point_size = 0.04,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Main viewer loop
# ─────────────────────────────────────────────────────────────────────────────

def run_viewer(model,
               device: str,
               source: str,
               interval: float,
               frame_skip: int,
               up_idx: int,
               port: int) -> None:
    """Start the three worker threads, open the viser server, and run the loop.

    The loop polls map_queue and navmesh_queue at ~20 Hz and updates the
    scene.  Blocks until stop_event is set (Ctrl-C from main).

    Args:
        model:      depth model returned by load_model().
        device:     "cuda" or "cpu".
        source:     webcam index (str of int) or path to a video file.
        interval:   seconds between webcam captures.
        frame_skip: process every Nth frame (video-file mode).
        up_idx:     index of the up axis (0=X, 1=Y, 2=Z).
        port:       viser web-viewer port.
    """
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

    def _set_camera(client, centroid: np.ndarray) -> None:
        """Point the viewer camera at the centroid of the first map cloud."""
        client.camera.position = (0.0, 0.0, 0.0)
        client.camera.look_at  = tuple(centroid.tolist())
        client.camera.up       = (0.0, 1.0, 0.0)

    while not stop_event.is_set():

        # Accumulated coloured map — updates every inference frame
        try:
            m = map_queue.get_nowait()
            pts    = m["pts"]
            colors = m["colors"]
            if cfg.USE_SPLATS and len(pts) > 0:
                r    = cfg.SPLAT_RADIUS
                cov  = np.eye(3, dtype=np.float32) * (r * r)
                covs = np.ascontiguousarray(np.tile(cov, (len(pts), 1, 1)))
                opacities = np.full((len(pts), 1), cfg.SPLAT_OPACITY, dtype=np.float32)
                server.scene.add_gaussian_splats(
                    "/scene/map",
                    centers     = np.ascontiguousarray(pts),
                    covariances = covs,
                    rgbs        = np.ascontiguousarray(colors),
                    opacities   = opacities,
                )
            else:
                server.scene.add_point_cloud(
                    "/scene/map",
                    points     = pts,
                    colors     = colors,
                    point_size = 0.008,
                )
            if not camera_set:
                centroid = m["pts"].mean(axis=0)
                for client in server.get_clients().values():
                    _set_camera(client, centroid)

                @server.on_client_connect
                def _init_cam(client) -> None:
                    _set_camera(client, centroid)

                camera_set = True
        except queue.Empty:
            pass

        # Navmesh overlay — updates every navmesh recompute
        try:
            nav = navmesh_queue.get_nowait()
            update_navmesh(server, nav)
        except queue.Empty:
            pass

        time.sleep(0.05)   # ~20 Hz polling

    for t in threads:
        t.join(timeout=5)
    print("\nAll threads stopped.")
