"""Viser 3-D viewer: scene update helpers and the main viewer loop.

The loop polls map_queue (accumulated coloured cloud, every inference frame)
and navmesh_queue (obstacles/nodes/edges/A* path, every navmesh recompute)
every 50 ms. Scene layers: orange obstacles, red blocked nodes, yellow free
nodes, blue edges, teal A* path.
"""

import queue
import time

import numpy as np

import tequila.config as cfg
from tequila.threads import (
    CaptureThread, InferenceThread, NavmeshThread,
    map_queue, navmesh_queue, stop_event,
)


def update_navmesh(server, nav: dict) -> None:
    """Push all navmesh layers to the viser scene (dict from compute_navmesh()).

    Each layer is replaced atomically, so stale items from the previous
    recompute disappear as soon as the new ones are added.
    """
    clean_obs     = nav["clean_obs"]
    free_nodes    = nav["free_nodes"]
    blocked_nodes = nav["blocked_nodes"]
    edges         = nav["edges"]
    path_pts      = nav["path_pts"]

    if len(clean_obs) > 0:
        server.scene.add_point_cloud(
            "/nav/obstacles",
            points     = clean_obs,
            colors     = np.tile([[1.0, 0.3, 0.1]], (len(clean_obs), 1)).astype(np.float32),
            point_size = 0.015,
        )

    # Debug layers (full node grid, edge web, trajectory) are hidden by
    # default since they clutter the overlay; NAV_PATH_ONLY=False re-enables.
    if not cfg.NAV_PATH_ONLY:
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
                point_size = 0.025,
            )

        if len(free_nodes) > 0:
            server.scene.add_point_cloud(
                "/nav/free",
                points     = free_nodes,
                colors     = np.tile([[1.0, 0.8, 0.0]], (len(free_nodes), 1)).astype(np.float32),
                point_size = 0.025,
            )

        trajectory = nav.get("trajectory", None)
        if trajectory is not None and len(trajectory) >= 2:
            traj_segs  = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
            traj_color = np.tile([[0.1, 0.9, 0.1]], (len(traj_segs), 2, 1)).astype(np.float32)
            server.scene.add_line_segments(
                "/nav/trajectory", points=traj_segs, colors=traj_color, line_width=3.0)
            server.scene.add_point_cloud(
                "/nav/trajectory_pts",
                points     = trajectory,
                colors     = np.tile([[0.1, 0.9, 0.1]], (len(trajectory), 1)).astype(np.float32),
                point_size = 0.02,
            )

    # A* path is always shown, regardless of NAV_PATH_ONLY.
    if len(path_pts) >= 2:
        path_segs  = np.stack([path_pts[:-1], path_pts[1:]], axis=1)
        path_color = np.tile([[0.0, 0.95, 0.88]], (len(path_segs), 2, 1)).astype(np.float32)
        server.scene.add_line_segments(
            "/nav/path", points=path_segs, colors=path_color, line_width=4.0)
        server.scene.add_point_cloud(
            "/nav/path_nodes",
            points     = path_pts,
            colors     = np.tile([[0.0, 0.95, 0.88]], (len(path_pts), 1)).astype(np.float32),
            point_size = 0.05,
        )


def run_viewer(model,
               device: str,
               source: str,
               interval: float,
               frame_skip: int,
               up_idx: int,
               port: int) -> None:
    """Start the worker threads and viser server, then poll the queues at ~20 Hz
    updating the scene until stop_event is set (Ctrl-C from main)."""
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
                    point_size = cfg.POINT_SIZE,
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

        try:
            nav = navmesh_queue.get_nowait()
            update_navmesh(server, nav)
        except queue.Empty:
            pass

        time.sleep(0.05)   # ~20 Hz polling

    for t in threads:
        t.join(timeout=5)
    print("\nAll threads stopped.")
