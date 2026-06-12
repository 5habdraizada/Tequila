#!/usr/bin/env python3
"""
digital_twin.py — Interactive indoor mapping simulator for TEQUILA.

Drop a virtual robot into a configurable 3-D room.  The robot explores
automatically using the live A* navmesh path.  A camera POV feed and a
top-down minimap are shown as billboards in the 3-D scene.

Usage:   python digital_twin.py [--scene Office] [--port 8081]
Viewer:  http://localhost:8081

Scene layers
------------
  /scene/*         — room walls + obstacles (static)
  /robot/true      — ground-truth pose      (green axes)
  /robot/odom      — dead-reckoning         (red axes)
  /robot/ekf       — EKF estimate           (blue axes)
  /trails/*        — trajectory history
  /map/cloud       — accumulated splat map
  /nav/*           — navmesh overlay
  /hud/pov         — robot camera POV billboard
  /hud/minimap     — top-down minimap billboard
"""

import time
import threading
import numpy as np
import cv2
import viser

import tequila.config as cfg
from tequila.pointcloud import voxel_downsample_colored, voxel_downsample_pts
from tequila.navmesh import compute_navmesh
from tequila.viewer import update_navmesh


# ─────────────────────────────────────────────────────────────────────────────
#  Scene library
#  Obstacle format: (center_x, center_z, half_x, half_z, height, (R, G, B))
# ─────────────────────────────────────────────────────────────────────────────

SCENES: dict = {
    "Empty Room": {
        "room": (8.0, 6.0), "ceiling": 2.5, "obstacles": [],
    },
    "Office": {
        "room": (8.0, 6.0), "ceiling": 2.5,
        "obstacles": [
            ( 1.5, -1.0,  0.60, 0.30, 0.75, (0.55, 0.35, 0.20)),
            ( 1.5, -0.2,  0.20, 0.20, 0.50, (0.20, 0.20, 0.70)),
            (-2.5,  0.0,  0.20, 0.50, 1.80, (0.50, 0.50, 0.50)),
            ( 0.0,  1.8,  0.15, 0.15, 2.50, (0.80, 0.80, 0.80)),
            (-1.0, -1.8,  0.60, 0.30, 0.75, (0.55, 0.35, 0.20)),
        ],
    },
    "Warehouse": {
        "room": (12.0, 10.0), "ceiling": 4.0,
        "obstacles": [
            (-3.5,  1.0,  0.20, 1.50, 2.20, (0.65, 0.45, 0.25)),
            ( 3.5,  1.0,  0.20, 1.50, 2.20, (0.65, 0.45, 0.25)),
            (-1.0,  2.5,  0.40, 0.40, 0.60, (0.50, 0.30, 0.10)),
            ( 1.0,  2.5,  0.40, 0.40, 0.60, (0.50, 0.30, 0.10)),
            ( 0.0, -3.5,  0.50, 0.50, 0.60, (0.50, 0.30, 0.10)),
        ],
    },
    "Maze": {
        "room": (10.0, 10.0), "ceiling": 2.5,
        "obstacles": [
            ( 0.0, -1.5,  2.50, 0.10, 2.0, (0.70, 0.70, 0.70)),
            ( 0.0,  1.5,  2.50, 0.10, 2.0, (0.70, 0.70, 0.70)),
            (-2.0,  0.0,  0.10, 2.00, 2.0, (0.70, 0.70, 0.70)),
            ( 2.0,  0.0,  0.10, 2.00, 2.0, (0.70, 0.70, 0.70)),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Raycaster
# ─────────────────────────────────────────────────────────────────────────────

class Raycaster:
    W      = 320
    H      = 240
    CAM_H  = 0.55
    TILT   = 15.0
    MAX_D  = 10.0

    def __init__(self, scene_name: str):
        self.load_scene(scene_name)
        self._build_ray_grid()

    def load_scene(self, name: str):
        s = SCENES[name]
        self.room_w, self.room_d = s["room"]
        self.ceiling = s["ceiling"]
        self.obs = s["obstacles"]

    def _build_ray_grid(self):
        focal = self.W / (2.0 * np.tan(np.radians(cfg.FOV_H_DEG / 2.0)))
        cx, cy = self.W / 2.0, self.H / 2.0
        u = np.arange(self.W, dtype=np.float32)
        v = np.arange(self.H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        rx = (uu - cx) / focal
        ry = -((vv - cy) / focal)
        rz = -np.ones_like(rx)
        self._rays_cam  = np.stack([rx, ry, rz], axis=-1)
        self._cos_theta = 1.0 / np.sqrt(rx**2 + ry**2 + 1.0)
        self.focal = focal
        self.cx, self.cy = cx, cy

    def render(self, rx: float, rz: float, yaw: float):
        cam_pos = np.array([rx, self.CAM_H, rz], dtype=np.float64)
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[ cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        t  = np.radians(-self.TILT)
        ct, st = np.cos(t), np.sin(t)
        Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        R  = (Ry @ Rx).astype(np.float64)
        rays_w = (R @ self._rays_cam.reshape(-1, 3).T).T
        N      = len(rays_w)
        t_min  = np.full(N, self.MAX_D, np.float64)
        colors = np.zeros((N, 3), np.float32)

        hw, hd = self.room_w / 2, self.room_d / 2
        self._plane(cam_pos, rays_w, t_min, colors, 1, 0.0,
                    (-hw, hw), (-0.1, 0.1), (-hd, hd), (0.75, 0.75, 0.72))
        self._plane(cam_pos, rays_w, t_min, colors, 1, self.ceiling,
                    (-hw, hw), (self.ceiling-0.1, self.ceiling+0.1), (-hd, hd),
                    (0.95, 0.95, 0.95))
        for ax, val, col in [
            (0, -hw, (0.80, 0.80, 0.85)), (0, hw, (0.80, 0.80, 0.85)),
            (2, -hd, (0.75, 0.78, 0.85)), (2, hd, (0.75, 0.78, 0.85)),
        ]:
            self._plane(cam_pos, rays_w, t_min, colors, ax, val,
                        (-hw-0.1, hw+0.1), (0.0, self.ceiling), (-hd-0.1, hd+0.1), col)
        for (cx_o, cz_o, hx, hz, ht, ocol) in self.obs:
            self._aabb(cam_pos, rays_w, t_min, colors,
                       cx_o-hx, cx_o+hx, 0.0, ht, cz_o-hz, cz_o+hz, ocol)

        valid   = t_min < self.MAX_D * 0.99
        depth_m = np.where(valid, t_min * self._cos_theta.ravel(), 0.0)
        depth_m[valid] *= np.random.normal(1.0, 0.004, int(valid.sum()))
        depth_map = depth_m.reshape(self.H, self.W).astype(np.float32)

        shade = np.clip(1.0 - t_min / (self.MAX_D * 0.6), 0.25, 1.0)[:, None]
        rgb   = np.clip(colors * shade, 0, 1)
        bgr   = (rgb[:, ::-1] * 255).astype(np.uint8).reshape(self.H, self.W, 3)

        return depth_map, bgr, cam_pos.astype(np.float32)

    def _plane(self, o, rays, t_min, colors, axis, val, bx, by, bz, col):
        d  = rays[:, axis]
        ok = np.abs(d) > 1e-9
        t  = np.where(ok, (val - o[axis]) / np.where(ok, d, 1.0), np.inf)
        t  = np.where(np.isfinite(t), t, np.inf)
        hp = o + np.where(np.isfinite(t), t, 0.0)[:, None] * rays
        hit = ((t > 0.02) &
               (hp[:, 0] >= bx[0]) & (hp[:, 0] <= bx[1]) &
               (hp[:, 1] >= by[0]) & (hp[:, 1] <= by[1]) &
               (hp[:, 2] >= bz[0]) & (hp[:, 2] <= bz[1]))
        c = hit & (t < t_min)
        t_min[c]  = t[c]
        colors[c] = col

    def _aabb(self, o, rays, t_min, colors, x0, x1, y0, y1, z0, z1, col):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(rays) > 1e-12, 1.0 / rays,
                           np.sign(rays) * 1e12)
        tx0, tx1 = (x0-o[0])*inv[:,0], (x1-o[0])*inv[:,0]
        ty0, ty1 = (y0-o[1])*inv[:,1], (y1-o[1])*inv[:,1]
        tz0, tz1 = (z0-o[2])*inv[:,2], (z1-o[2])*inv[:,2]
        te = np.maximum(np.maximum(np.minimum(tx0,tx1), np.minimum(ty0,ty1)),
                        np.minimum(tz0,tz1))
        tx = np.minimum(np.minimum(np.maximum(tx0,tx1), np.maximum(ty0,ty1)),
                        np.maximum(tz0,tz1))
        hit   = (te < tx) & (tx > 0.02)
        t_hit = np.where(hit, np.maximum(te, 0.02), np.inf)
        c = hit & (t_hit < t_min)
        t_min[c]  = t_hit[c]
        colors[c] = col


# ─────────────────────────────────────────────────────────────────────────────
#  Robot / sensors / estimators
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    WHEEL_BASE = 0.35
    def __init__(self, x=0.0, z=0.0, yaw=0.0): self.reset(x, z, yaw)
    def reset(self, x=0.0, z=0.0, yaw=0.0):
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
    def step(self, v_lin, v_ang, dt):
        self.yaw += v_ang * dt
        self.x   += v_lin * np.cos(self.yaw) * dt
        self.z   -= v_lin * np.sin(self.yaw) * dt
    @property
    def pose(self): return self.x, self.z, self.yaw


class SyntheticSensors:
    def __init__(self):
        self.sigma_wheel = 0.02
        self.sigma_gyro  = 0.03
    def wheels(self, v_lin, v_ang):
        wr = v_lin + v_ang * Robot.WHEEL_BASE / 2
        wl = v_lin - v_ang * Robot.WHEEL_BASE / 2
        return (wl + np.random.normal(0, self.sigma_wheel),
                wr + np.random.normal(0, self.sigma_wheel))
    def gyro_z(self, v_ang):
        return v_ang + np.random.normal(0, self.sigma_gyro)


class EKF2D:
    def __init__(self, x=0.0, z=0.0, yaw=0.0):
        self.mu  = np.array([x, z, yaw], np.float64)
        self.P   = np.diag([1e-4, 1e-4, 1e-4])
        self.Q   = np.diag([4e-5, 4e-5, 2e-6])
        self.R_g = np.array([[9e-4]])
    def reset(self, x=0.0, z=0.0, yaw=0.0):
        self.mu = np.array([x, z, yaw], np.float64)
        self.P  = np.diag([1e-4, 1e-4, 1e-4])
    def predict(self, v_l, v_r, dt):
        vl = (v_r + v_l) / 2.0; va = (v_r - v_l) / Robot.WHEEL_BASE
        x, z, y = self.mu; ny = y + va * dt
        self.mu = np.array([x + vl*np.cos(ny)*dt, z - vl*np.sin(ny)*dt, ny])
        F = np.array([[1,0,-vl*np.sin(ny)*dt],[0,1,-vl*np.cos(ny)*dt],[0,0,1]])
        self.P = F @ self.P @ F.T + self.Q
    def update_gyro(self, gz, dt):
        H = np.array([[0.0, 0.0, 1.0]])
        z = np.array([self.mu[2] + gz * dt])
        S = H @ self.P @ H.T + self.R_g
        K = self.P @ H.T / S[0, 0]
        inn = (z[0] - self.mu[2] + np.pi) % (2*np.pi) - np.pi
        self.mu += K.ravel() * inn
        self.P   = (np.eye(3) - K.reshape(3,1) @ H) @ self.P
    @property
    def pose(self): return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])


class DeadReckoning:
    def __init__(self, x=0.0, z=0.0, yaw=0.0): self.reset(x, z, yaw)
    def reset(self, x=0.0, z=0.0, yaw=0.0):
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
    def step(self, v_l, v_r, dt):
        vl = (v_r+v_l)/2; va = (v_r-v_l)/Robot.WHEEL_BASE
        self.yaw += va*dt; self.x += vl*np.cos(self.yaw)*dt
        self.z   -= vl*np.sin(self.yaw)*dt
    @property
    def pose(self): return self.x, self.z, self.yaw


def yaw_to_wxyz(yaw):
    return (float(np.cos(yaw/2)), 0.0, float(np.sin(yaw/2)), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  DigitalTwin
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwin:
    SIM_HZ      = 50
    MAP_EVERY_S = 2.0
    MAX_NAV_PTS = 300_000

    def __init__(self, scene_name: str = "Office", port: int = 8081):
        self.scene_name = scene_name
        self.port       = port
        self.robot   = Robot()
        self.sensors = SyntheticSensors()
        self.ekf     = EKF2D()
        self.dr      = DeadReckoning()
        self.rc      = Raycaster(scene_name)

        self._lock    = threading.Lock()
        self._running = True

        # Manual velocity (used only when auto_explore is off)
        self.v_lin = 0.0
        self.v_ang = 0.0

        # Auto-exploration state
        self.auto_explore     = True
        self._path_wps: list  = []   # list of (x, z) waypoints
        self._wp_idx          = 0
        self._stuck_t         = 0.0
        self._prev_pos        = (0.0, 0.0)
        self._spin_t          = 0.0
        self._spin_dir        = 1.0

        # Visited-cell grid: 0.4 m cells, tracks which areas are explored
        self._visit_res       = 0.4  # metres per cell
        self._visited: set    = set()   # set of (ix, iz) integer cell coords
        self._current_goal: tuple | None = None  # (x, z) current target node

        # Map
        self.nav_pts    = np.zeros((0, 3), np.float32)
        self.map_pts    = np.zeros((0, 3), np.float32)
        self.map_colors = np.zeros((0, 3), np.float32)

        # Trails
        self.trail_true: list = []
        self.trail_dr:   list = []
        self.trail_ekf:  list = []

        self._nav_dict: dict | None = None
        self._last_bgr: np.ndarray  = np.zeros((Raycaster.H, Raycaster.W, 3), np.uint8)

        self.sigma_wheel = 0.02
        self.sigma_gyro  = 0.03

    # ── scene management ──────────────────────────────────────────────────────

    def load_scene(self, name: str):
        self.scene_name = name
        self.rc.load_scene(name)
        self._clear_map()

    def _clear_map(self):
        self.nav_pts    = np.zeros((0, 3), np.float32)
        self.map_pts    = np.zeros((0, 3), np.float32)
        self.map_colors = np.zeros((0, 3), np.float32)
        self.trail_true.clear(); self.trail_dr.clear(); self.trail_ekf.clear()
        self._nav_dict = None
        self._path_wps = []; self._wp_idx = 0
        self._visited.clear(); self._current_goal = None

    def reset(self):
        self.robot.reset(); self.ekf.reset(); self.dr.reset()
        self._stuck_t = 0.0; self._spin_t = 0.0
        self._prev_pos = (0.0, 0.0)
        self._clear_map()

    # ── visited-grid helpers ──────────────────────────────────────────────────

    def _cell(self, x: float, z: float) -> tuple[int, int]:
        return (int(np.floor(x / self._visit_res)),
                int(np.floor(z / self._visit_res)))

    def _mark_visited(self, x: float, z: float):
        self._visited.add(self._cell(x, z))

    def _is_visited(self, x: float, z: float) -> bool:
        return self._cell(x, z) in self._visited

    # ── random frontier sampling ──────────────────────────────────────────────

    def _sample_goal(self) -> tuple[float, float] | None:
        """
        Random frontier sampling: draw candidates across the whole room floor,
        score by (unvisited bonus + distance from robot), pick the best.
        Avoids the navmesh A* local-minimum problem entirely.
        """
        hw = self.rc.room_w / 2 - 0.5   # stay 0.5 m from walls
        hd = self.rc.room_d / 2 - 0.5
        rx, rz, _ = self.robot.pose

        # Sample 80 random positions on the floor
        xs = np.random.uniform(-hw, hw, 80)
        zs = np.random.uniform(-hd, hd, 80)

        best_score = -np.inf
        best       = None
        for x, z in zip(xs, zs):
            if not self._pos_valid(x, z, clearance=0.40):
                continue

            dist    = float(np.sqrt((x-rx)**2 + (z-rz)**2))
            if dist < 0.5:
                continue          # too close
            unvisited = 0.0 if self._is_visited(x, z) else 4.0
            # Also reward distance from current goal to spread exploration
            goal_spread = 0.0
            if self._current_goal is not None:
                gx, gz = self._current_goal
                goal_spread = min(float(np.sqrt((x-gx)**2+(z-gz)**2)), 3.0)

            score = dist + unvisited + goal_spread * 0.5
            if score > best_score:
                best_score = score
                best       = (float(x), float(z))

        return best

    # ── auto-navigation ───────────────────────────────────────────────────────

    def _auto_velocity(self, dt: float) -> tuple[float, float]:
        rx, rz, ryaw = self.robot.pose
        self._mark_visited(rx, rz)

        # Stuck detection (position barely moved)
        ddx = rx - self._prev_pos[0]; ddz = rz - self._prev_pos[1]
        if np.sqrt(ddx*ddx + ddz*ddz) > 0.06:
            self._stuck_t  = 0.0
            self._prev_pos = (rx, rz)
        else:
            self._stuck_t += dt

        # Stuck escape: reverse + spin
        if self._stuck_t > 1.5 or self._spin_t > 0:
            self._stuck_t      = 0.0
            self._current_goal = None
            self._path_wps     = []
            if self._spin_t <= 0:
                self._spin_t   = float(np.random.uniform(2.0, 3.5))
                self._spin_dir = float(np.random.choice([-1, 1]))
            self._spin_t -= dt
            return -0.15, self._spin_dir * 1.4

        # Reactive look-ahead: if path ahead is blocked, turn in place
        if not self._forward_clear(rx, rz, ryaw, look_dist=0.55):
            self._current_goal = None
            self._path_wps     = []
            # Pick turn direction away from the nearer wall
            turn_dir = 1.0 if not self._forward_clear(rx, rz, ryaw - 0.7, 0.45) else -1.0
            return 0.0, turn_dir * 1.3

        # No goal → pick one
        if self._current_goal is None:
            self._current_goal = self._sample_goal()
            if self._current_goal is None:
                return 0.0, 0.5
            gx, gz = self._current_goal
            # Use navmesh A* if available, else fall back to straight-line
            nav_wps = self._navmesh_path_to(gx, gz)
            if nav_wps:
                self._path_wps = nav_wps
            else:
                dist_g = float(np.sqrt((gx-rx)**2 + (gz-rz)**2))
                steps  = max(3, int(dist_g / 0.22))
                self._path_wps = [
                    (float(rx + (gx-rx)*i/steps), float(rz + (gz-rz)*i/steps))
                    for i in range(1, steps+1)
                ]
            self._wp_idx = 0

        # Skip already-reached waypoints
        while self._wp_idx < len(self._path_wps):
            tx, tz = self._path_wps[self._wp_idx]
            if float(np.sqrt((tx-rx)**2 + (tz-rz)**2)) < 0.22:
                self._mark_visited(tx, tz)
                self._wp_idx += 1
            else:
                break
        else:
            self._current_goal = None
            return 0.0, 0.0

        tx, tz   = self._path_wps[self._wp_idx]
        dx2, dz2 = tx - rx, tz - rz
        dist     = float(np.sqrt(dx2*dx2 + dz2*dz2))

        target_yaw = float(np.arctan2(-dz2, dx2))
        yaw_err    = float(((target_yaw - ryaw) + np.pi) % (2*np.pi) - np.pi)
        v_ang = float(np.clip(3.0 * yaw_err, -1.5, 1.5))
        v_lin = float(np.clip(dist * 0.9, 0.0, 0.6)) if abs(yaw_err) < 0.45 else 0.0
        return v_lin, v_ang

    # ── collision helpers ─────────────────────────────────────────────────────

    def _pos_valid(self, x: float, z: float, clearance: float = 0.32) -> bool:
        """Return True if (x, z) is inside the room and not inside any obstacle."""
        hw = self.rc.room_w / 2 - clearance
        hd = self.rc.room_d / 2 - clearance
        if abs(x) > hw or abs(z) > hd:
            return False
        for (cx_o, cz_o, hx, hz, ht, _) in self.rc.obs:
            if abs(x - cx_o) < hx + clearance and abs(z - cz_o) < hz + clearance:
                return False
        return True

    def _clamp_robot(self):
        """Hard constraint: push robot back inside valid space after each step."""
        hw = self.rc.room_w / 2 - 0.28
        hd = self.rc.room_d / 2 - 0.28
        self.robot.x = float(np.clip(self.robot.x, -hw, hw))
        self.robot.z = float(np.clip(self.robot.z, -hd, hd))
        for (cx_o, cz_o, hx, hz, ht, _) in self.rc.obs:
            dx = self.robot.x - cx_o
            dz = self.robot.z - cz_o
            ox, oz = hx + 0.32, hz + 0.32
            if abs(dx) < ox and abs(dz) < oz:
                pen_x = ox - abs(dx)
                pen_z = oz - abs(dz)
                if pen_x < pen_z:
                    self.robot.x += float(np.sign(dx)) * pen_x
                else:
                    self.robot.z += float(np.sign(dz)) * pen_z

    def _forward_clear(self, rx: float, rz: float, ryaw: float,
                        look_dist: float = 0.55) -> bool:
        """Return True if the corridor ahead is free of walls and obstacles."""
        steps = 7
        for i in range(1, steps + 1):
            d  = look_dist * i / steps
            fx = rx + d * np.cos(ryaw)
            fz = rz - d * np.sin(ryaw)
            if not self._pos_valid(fx, fz, clearance=0.28):
                return False
        return True

    # ── navmesh A* routing ────────────────────────────────────────────────────

    def _navmesh_path_to(self, gx: float, gz: float) -> list[tuple[float, float]]:
        """A* through navmesh free nodes.  Returns list of (x, z) waypoints or []."""
        import heapq
        nav = self._nav_dict
        if nav is None:
            return []
        free  = nav.get("free_nodes", None)
        edges = nav.get("edges",      None)
        if free is None or len(free) == 0 or not edges:
            return []

        rx, rz, _ = self.robot.pose
        si = int(np.argmin((free[:,0]-rx)**2 + (free[:,2]-rz)**2))
        ei = int(np.argmin((free[:,0]-gx)**2 + (free[:,2]-gz)**2))
        if si == ei:
            return [(gx, gz)]

        adj: dict[int, list] = {i: [] for i in range(len(free))}
        for e in edges:
            a, b = int(e[0]), int(e[1])
            d = float(np.sqrt((free[a,0]-free[b,0])**2 +
                               (free[a,2]-free[b,2])**2))
            adj[a].append((b, d)); adj[b].append((a, d))

        def h(i):
            return float(np.sqrt((free[i,0]-free[ei,0])**2 +
                                  (free[i,2]-free[ei,2])**2))

        heap = [(h(si), 0.0, si, [si])]
        seen: set[int] = set()
        while heap:
            _, g, cur, path = heapq.heappop(heap)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == ei:
                return [(float(free[i,0]), float(free[i,2])) for i in path[1:]]
            for nxt, d in adj[cur]:
                if nxt not in seen:
                    heapq.heappush(heap, (g+d+h(nxt), g+d, nxt, path+[nxt]))
        return []

    def _update_path(self):
        """Called after each navmesh recompute — rebuild waypoints through navmesh."""
        if self._current_goal is not None:
            gx, gz = self._current_goal
            nav_wps = self._navmesh_path_to(gx, gz)
            if nav_wps:
                self._path_wps = nav_wps
                self._wp_idx   = 0

    # ── back-projection ───────────────────────────────────────────────────────

    def _backproject(self, depth_m, bgr, robot_x, robot_z, yaw):
        h, w   = depth_m.shape
        cx, cy = w / 2.0, h / 2.0
        focal  = self.rc.focal
        px, py = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        x3 =  (px - cx) * depth_m / focal
        y3 = -((py - cy) * depth_m / focal)
        z3 = -depth_m
        pts    = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=-1)
        colors = (bgr[..., ::-1].reshape(-1, 3) / 255.0).astype(np.float32)
        valid  = z3.ravel() < -0.01
        pts    = pts[valid].astype(np.float32)
        colors = colors[valid]

        # Full camera rotation (yaw + tilt) + camera world position
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[ cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], np.float32)
        t  = np.radians(-self.rc.TILT)
        ct, st = np.cos(t), np.sin(t)
        Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], np.float32)
        R  = (Ry @ Rx).astype(np.float32)
        cam_pos   = np.array([robot_x, self.rc.CAM_H, robot_z], np.float32)
        world_pts = (R @ pts.T).T + cam_pos

        close = np.sqrt((world_pts[:,0]-robot_x)**2 +
                        (world_pts[:,2]-robot_z)**2) < cfg.MAP_MAX_DEPTH_M
        nav_pts = voxel_downsample_pts(world_pts, cfg.VOXEL_SIZE * 3)
        if close.sum() > 0:
            mp, mc = voxel_downsample_colored(world_pts[close], colors[close],
                                              cfg.VOXEL_SIZE)
        else:
            mp, mc = np.zeros((0,3),np.float32), np.zeros((0,3),np.float32)
        return nav_pts, mp, mc

    # ── minimap ───────────────────────────────────────────────────────────────

    def _make_minimap(self, size: int = 300) -> np.ndarray:
        """Return a top-down (size×size) RGB uint8 minimap."""
        hw, hd = self.rc.room_w / 2, self.rc.room_d / 2
        img    = np.full((size, size, 3), 25, np.uint8)

        def w2p(x, z):
            px = int((x + hw) / self.rc.room_w * (size - 20) + 10)
            pz = int((z + hd) / self.rc.room_d * (size - 20) + 10)
            return (np.clip(px, 0, size-1), np.clip(pz, 0, size-1))

        # Room border
        cv2.rectangle(img, w2p(-hw, -hd), w2p(hw, hd), (80, 80, 80), 2)

        # Obstacles
        for (cx_o, cz_o, hx, hz, ht, col) in self.rc.obs:
            p0, p1 = w2p(cx_o-hx, cz_o-hz), w2p(cx_o+hx, cz_o+hz)
            c = tuple(int(v * 200) for v in col)
            cv2.rectangle(img, p0, p1, c, -1)

        # Accumulated map points (grey dots)
        if len(self.nav_pts) > 0:
            sub = self.nav_pts[::8]
            pxs = ((sub[:,0]+hw)/(self.rc.room_w)*(size-20)+10).astype(int)
            pzs = ((sub[:,2]+hd)/(self.rc.room_d)*(size-20)+10).astype(int)
            valid = (pxs>=0)&(pxs<size)&(pzs>=0)&(pzs<size)
            img[pzs[valid], pxs[valid]] = (110, 110, 110)

        # Navmesh free nodes + path
        if self._nav_dict is not None:
            free = self._nav_dict.get("free_nodes", np.array([]))
            if free is not None and len(free):
                for pt in free[::3]:
                    cv2.circle(img, w2p(pt[0], pt[2]), 2, (0, 180, 180), -1)
            path = self._nav_dict.get("path_pts", np.array([]))
            if path is not None and len(path) >= 2:
                for i in range(len(path)-1):
                    cv2.line(img, w2p(path[i][0], path[i][2]),
                             w2p(path[i+1][0], path[i+1][2]), (0, 220, 180), 2)

        # Trajectories
        for trail, col in [(self.trail_dr, (60,60,220)),
                            (self.trail_ekf, (60,160,255))]:
            pts = trail[-200:]
            for i in range(len(pts)-1):
                cv2.line(img, w2p(pts[i][0],pts[i][2]),
                         w2p(pts[i+1][0],pts[i+1][2]), col, 1)

        # Robot (green arrow)
        rx, rz, ryaw = self.robot.pose
        rp = w2p(rx, rz)
        scale = (size - 20) / max(self.rc.room_w, self.rc.room_d)
        arrow_len = max(8, int(0.4 * scale))
        ap = (int(rp[0] + arrow_len * np.cos(ryaw)),
              int(rp[1] - arrow_len * np.sin(ryaw)))
        cv2.circle(img, rp, 6, (50, 220, 50), -1)
        cv2.arrowedLine(img, rp, ap, (50, 255, 50), 2, tipLength=0.4)

        # Label
        cv2.putText(img, "MINIMAP", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        return img  # RGB uint8

    # ── simulation thread ─────────────────────────────────────────────────────

    def _sim_loop(self):
        dt       = 1.0 / self.SIM_HZ
        last_map = 0.0

        while self._running:
            t0 = time.time()

            # Velocity command
            if self.auto_explore:
                vl, va = self._auto_velocity(dt)
            else:
                with self._lock:
                    vl, va = self.v_lin, self.v_ang

            self.sensors.sigma_wheel = self.sigma_wheel
            self.sensors.sigma_gyro  = self.sigma_gyro

            self.robot.step(vl, va, dt)
            self._clamp_robot()          # hard wall / obstacle constraint
            rx, rz, ryaw = self.robot.pose

            v_l, v_r = self.sensors.wheels(vl, va)
            gz       = self.sensors.gyro_z(va)
            self.dr.step(v_l, v_r, dt)
            self.ekf.predict(v_l, v_r, dt)
            self.ekf.update_gyro(gz, dt)

            self.trail_true.append(np.array([rx,  0.02, rz],  np.float32))
            dx, dz, _ = self.dr.pose
            self.trail_dr.append(np.array([dx,   0.02, dz],   np.float32))
            ex, ez, _ = self.ekf.pose
            self.trail_ekf.append(np.array([ex,  0.02, ez],   np.float32))

            now = time.time()
            if now - last_map >= self.MAP_EVERY_S:
                last_map = now
                depth_m, bgr, cam_pos = self.rc.render(rx, rz, ryaw)
                self._last_bgr = bgr.copy()

                nav_new, mp_new, mc_new = self._backproject(depth_m, bgr, rx, rz, ryaw)

                if len(nav_new) > 0:
                    self.nav_pts = np.concatenate([self.nav_pts, nav_new])
                    self.nav_pts = voxel_downsample_pts(self.nav_pts, cfg.VOXEL_SIZE*3)
                    if len(self.nav_pts) > self.MAX_NAV_PTS:
                        self.nav_pts = self.nav_pts[
                            np.random.choice(len(self.nav_pts), self.MAX_NAV_PTS, False)]

                if len(mp_new) > 0:
                    self.map_pts    = np.concatenate([self.map_pts, mp_new])
                    self.map_colors = np.concatenate([self.map_colors, mc_new])
                    self.map_pts, self.map_colors = voxel_downsample_colored(
                        self.map_pts, self.map_colors, cfg.VOXEL_SIZE)
                    if len(self.map_pts) > self.MAX_NAV_PTS:
                        idx = np.random.choice(len(self.map_pts), self.MAX_NAV_PTS, False)
                        self.map_pts    = self.map_pts[idx]
                        self.map_colors = self.map_colors[idx]

                if len(self.nav_pts) >= cfg.MIN_FLOOR_POINTS:
                    try:
                        nav = compute_navmesh(self.nav_pts, 1, camera_origin=cam_pos)
                        nav["accum_pts"]  = self.nav_pts
                        nav["trajectory"] = None
                        self._nav_dict = nav
                        self._update_path()
                    except Exception as e:
                        print(f"[Navmesh] {e}")

            time.sleep(max(0.0, dt - (time.time() - t0)))

    # ── scene & display helpers ───────────────────────────────────────────────

    def _build_room(self, server):
        hw, hd = self.rc.room_w / 2, self.rc.room_d / 2
        ch     = self.rc.ceiling
        gray   = (180, 180, 185)
        server.scene.add_box("/scene/floor",
            dimensions=(self.rc.room_w, 0.04, self.rc.room_d),
            position=(0, -0.02, 0), color=(160, 160, 155))
        for name, dim, pos in [
            ("wall_L", (0.08, ch, self.rc.room_d), (-hw, ch/2, 0)),
            ("wall_R", (0.08, ch, self.rc.room_d), ( hw, ch/2, 0)),
            ("wall_B", (self.rc.room_w, ch, 0.08), (0, ch/2, -hd)),
            ("wall_F", (self.rc.room_w, ch, 0.08), (0, ch/2,  hd)),
        ]:
            server.scene.add_box(f"/scene/{name}", dimensions=dim,
                                  position=pos, color=gray)
        for i, (cx_o, cz_o, hx, hz, ht, col) in enumerate(self.rc.obs):
            server.scene.add_box(f"/scene/obs_{i}",
                dimensions=(hx*2, ht, hz*2),
                position=(cx_o, ht/2, cz_o),
                color=tuple(int(c*255) for c in col))

    def _update_poses(self, server):
        rx, rz, ryaw = self.robot.pose
        dx, dz, dyaw = self.dr.pose
        ex, ez, eyaw = self.ekf.pose
        for name, x, z, y in [("/robot/true",rx,rz,ryaw),
                                ("/robot/odom",dx,dz,dyaw),
                                ("/robot/ekf", ex,ez,eyaw)]:
            server.scene.add_frame(name, wxyz=yaw_to_wxyz(y),
                                   position=(x,0.15,z),
                                   axes_length=0.4, axes_radius=0.025)
        server.scene.add_box("/robot/body",
            dimensions=(0.3, 0.2, 0.4), wxyz=yaw_to_wxyz(ryaw),
            position=(rx, 0.1, rz), color=(50, 200, 50))
        for trail, name, col in [
            (self.trail_true, "/trails/true", (0.1,0.9,0.1)),
            (self.trail_dr,   "/trails/odom", (0.9,0.1,0.1)),
            (self.trail_ekf,  "/trails/ekf",  (0.1,0.4,0.9)),
        ]:
            if len(trail) >= 2:
                pts  = np.array(trail[-500:], np.float32)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                cols = np.tile(col, (len(segs),2,1)).astype(np.float32)
                server.scene.add_line_segments(name, points=segs,
                                               colors=cols, line_width=2.5)

    # Splat radius for the digital twin — much larger than real-camera mode
    # so individual splats are visible at room scale (voxel = 0.02 m → splat ≈ 0.03 m)
    SPLAT_R = 0.03

    def _update_map(self, server):
        # Consistent snapshot (avoids race with sim thread)
        pts  = self.map_pts.copy()
        cols = self.map_colors.copy()
        n    = min(len(pts), len(cols))
        if n < 10:
            return
        pts  = pts[:n].reshape(n, 3)
        cols = cols[:n].reshape(n, 3).clip(0.0, 1.0).astype(np.float32)

        # Isotropic Gaussian covariance: r² on each axis
        r   = self.SPLAT_R
        cov = np.eye(3, dtype=np.float32) * (r * r)
        covs = np.ascontiguousarray(np.tile(cov, (n, 1, 1)))   # (N,3,3)
        ops  = np.full((n, 1), 0.92, np.float32)                # high opacity

        server.scene.add_gaussian_splats(
            "/map/cloud",
            centers     = np.ascontiguousarray(pts),
            covariances = covs,
            rgbs        = np.ascontiguousarray(cols),
            opacities   = ops,
        )

    def _make_map_image(self, size: int = 300) -> np.ndarray:
        """Top-down projection of the accumulated coloured point cloud."""
        hw, hd = self.rc.room_w / 2, self.rc.room_d / 2
        img    = np.full((size, size, 3), 20, np.uint8)

        pts  = self.map_pts.copy()
        cols = self.map_colors.copy()
        n    = min(len(pts), len(cols))
        if n > 0:
            pts  = pts[:n];  cols = cols[:n]
            px = ((pts[:,0]+hw) / self.rc.room_w * (size-20) + 10).astype(int)
            pz = ((pts[:,2]+hd) / self.rc.room_d * (size-20) + 10).astype(int)
            valid = (px >= 0) & (px < size) & (pz >= 0) & (pz < size)
            # Sort by height so higher points paint over lower ones
            order = np.argsort(pts[valid, 1])
            px_v  = px[valid][order];  pz_v = pz[valid][order]
            rgb_v = (cols[valid][order] * 255).clip(0,255).astype(np.uint8)
            img[pz_v, px_v] = rgb_v

        # Room border + obstacles
        cv2.rectangle(img, (10,10), (size-10, size-10), (80,80,80), 1)
        for (cx_o, cz_o, hx, hz, ht, col) in self.rc.obs:
            p0 = (int((cx_o-hx+hw)/(self.rc.room_w)*(size-20)+10),
                  int((cz_o-hz+hd)/(self.rc.room_d)*(size-20)+10))
            p1 = (int((cx_o+hx+hw)/(self.rc.room_w)*(size-20)+10),
                  int((cz_o+hz+hd)/(self.rc.room_d)*(size-20)+10))
            cv2.rectangle(img, p0, p1, (60,60,60), 1)

        # Robot arrow
        rx, rz, ryaw = self.robot.pose
        rp  = (int((rx+hw)/(self.rc.room_w)*(size-20)+10),
               int((rz+hd)/(self.rc.room_d)*(size-20)+10))
        asc = max(8, int(0.4 * (size-20) / max(self.rc.room_w, self.rc.room_d)))
        ap  = (int(rp[0] + asc*np.cos(ryaw)), int(rp[1] - asc*np.sin(ryaw)))
        cv2.circle(img, rp, 5, (50,255,50), -1)
        cv2.arrowedLine(img, rp, ap, (50,255,50), 2, tipLength=0.4)

        cv2.putText(img, "3D MAP (top-down)", (6,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
        return img

    def _update_hud(self, server):
        """Update minimap billboard in the 3D scene.
        The accumulated 3D point cloud is already rendered live as /map/cloud splats.
        """
        hw = self.rc.room_w / 2
        hd = self.rc.room_d / 2

        # Minimap — navmesh overlay, trails, robot position
        mm = self._make_minimap(350)
        server.scene.add_image("/hud/minimap", mm,
                                render_width=2.5, render_height=2.5,
                                position=(0.0, self.rc.ceiling + 0.6, -hd + 1.0),
                                wxyz=(1.0, 0.0, 0.0, 0.0))

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self):
        server = viser.ViserServer(port=self.port)
        print(f"\n[Digital Twin] Open  http://localhost:{self.port}  in your browser\n")

        # ── GUI ───────────────────────────────────────────────────────────────
        with server.gui.add_folder("Scene"):
            g_scene  = server.gui.add_dropdown(
                "Room", list(SCENES.keys()), initial_value=self.scene_name)
            g_reload = server.gui.add_button("Load Scene")
            g_reset  = server.gui.add_button("Reset Robot")

        with server.gui.add_folder("Exploration"):
            g_auto = server.gui.add_checkbox("Auto Explore", initial_value=True)
            g_vlin = server.gui.add_slider(
                "Linear  (m/s)", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
            g_vang = server.gui.add_slider(
                "Angular (°/s)", min=-90.0, max=90.0, step=5.0, initial_value=0.0)
            g_stop = server.gui.add_button("Stop")
            g_vlin.disabled = True
            g_vang.disabled = True

        with server.gui.add_folder("Sensor Noise"):
            g_wn = server.gui.add_slider(
                "Wheel σ", min=0.0, max=0.2, step=0.005, initial_value=0.02)
            g_gn = server.gui.add_slider(
                "Gyro σ",  min=0.0, max=0.2, step=0.005, initial_value=0.03)

        with server.gui.add_folder("Map"):
            g_interval = server.gui.add_slider(
                "Capture interval (s)", min=0.5, max=10.0, step=0.5,
                initial_value=self.MAP_EVERY_S)
            g_clear = server.gui.add_button("Clear Map")

        # ── callbacks ─────────────────────────────────────────────────────────
        @g_auto.on_update
        def _toggle_auto(_):
            self.auto_explore   = g_auto.value
            g_vlin.disabled     = g_auto.value
            g_vang.disabled     = g_auto.value

        @g_reload.on_click
        def _load(_):
            self.load_scene(g_scene.value)
            self._build_room(server)

        @g_reset.on_click
        def _reset(_): self.reset()

        @g_stop.on_click
        def _stop(_):
            g_vlin.value = 0.0; g_vang.value = 0.0

        @g_clear.on_click
        def _clear(_): self._clear_map()

        self._build_room(server)
        threading.Thread(target=self._sim_loop, daemon=True).start()

        # ── viewer loop ───────────────────────────────────────────────────────
        try:
            while True:
                self.auto_explore  = bool(g_auto.value)
                if not self.auto_explore:
                    with self._lock:
                        self.v_lin = float(g_vlin.value)
                        self.v_ang = float(np.radians(g_vang.value))
                self.sigma_wheel = float(g_wn.value)
                self.sigma_gyro  = float(g_gn.value)
                self.MAP_EVERY_S = float(g_interval.value)

                self._update_poses(server)
                self._update_map(server)
                self._update_hud(server)
                if self._nav_dict is not None:
                    update_navmesh(server, self._nav_dict)

                time.sleep(0.05)
        except KeyboardInterrupt:
            self._running = False
            print("\nStopped.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="TEQUILA digital twin simulator")
    p.add_argument("--scene", default="Office", choices=list(SCENES.keys()))
    p.add_argument("--port",  type=int, default=8081)
    args = p.parse_args()
    DigitalTwin(scene_name=args.scene, port=args.port).run()
