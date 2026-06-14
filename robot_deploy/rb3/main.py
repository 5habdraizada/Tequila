#!/usr/bin/env python3
"""
rb3/main.py — Full TEQUILA pipeline on the RB3 Gen 2 with real hardware.

  Camera → depth → point cloud → navmesh → motor commands (Pico)
  Live 3D map + control panel at http://<rb3-ip>:8080

Usage:
    python3 main.py
    python3 main.py --no-nav          # mapping only
    python3 main.py --pico-port /dev/ttyACM0
"""

import argparse
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import viser

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tequila.config as cfg
from tequila.depth   import load_model
from tequila.viewer  import update_navmesh
from tequila.threads import (
    CaptureThread, InferenceThread, NavmeshThread,
    map_queue, navmesh_queue, stop_event, reset_map_event,
)

import config as rb3_cfg
from hardware import HardwareBridge


# ─────────────────────────────────────────────────────────────────────────────
#  Odometry → camera pose (level-1 fusion source)
# ─────────────────────────────────────────────────────────────────────────────

def make_odom_source(ekf: "EKF2D"):
    """Build a callable that returns the camera's world pose (4×4, nav coords).

    The Pico streams raw encoder ticks; HardwareBridge turns them into wheel
    velocities and the EKF integrates those into the robot pose (x, z, yaw),
    already expressed in the map's ground plane (nav X-Z, Y up).  This converts
    that pose into the *camera's* world pose for map stitching, accounting for:

      • the camera mount offset (forward / left / up of the wheel centre), which
        swings on a lever arm as the robot turns, and
      • the 90° between the EKF body frame (forward = +X at yaw 0) and the camera
        convention (the camera looks along −Z).

    Passed to CaptureThread so each frame is stamped with the pose at capture.
    """
    mf = rb3_cfg.CAM_MOUNT_FWD
    ml = rb3_cfg.CAM_MOUNT_LEFT
    mu = rb3_cfg.CAM_MOUNT_UP
    HALF_PI = math.pi / 2.0

    def _source():
        ex, ez, eyaw = ekf.pose              # robot pose in nav X-Z plane (Y up)

        # Robot body axes expressed in world (nav) coords.
        cf, sf = math.cos(eyaw), math.sin(eyaw)
        fwd_w  = np.array([ cf, 0.0, -sf])   # robot forward (R_y(yaw)·+X)
        left_w = np.array([-sf, 0.0, -cf])   # robot left    (up × forward)
        up_w   = np.array([0.0, 1.0,  0.0])  # robot up

        # Camera position = robot body position + mounted lever-arm offset.
        p_cam = (np.array([ex, 0.0, ez], dtype=np.float64)
                 + mf * fwd_w + ml * left_w + mu * up_w)

        # Camera looks along robot forward.  Camera −Z must map to robot forward,
        # so the camera rotation is yaw − 90° about +Y (body +X-forward → −Z-cam).
        phi  = eyaw - HALF_PI
        c, s = math.cos(phi), math.sin(phi)
        R = np.array([[ c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = p_cam
        return T

    return _source


def make_vo_update_cb(ekf: "EKF2D"):
    """Return a callback that corrects EKF drift using a VO-derived camera pose.

    NavmeshThread calls this whenever visual odometry succeeds, passing the
    estimated camera world position and rotation.  Here we:
      1. Extract robot yaw from the camera rotation (camera looks along −Z,
         and is rotated −90° about +Y relative to robot body).
      2. Subtract the mount-offset lever arm to get robot body position.
      3. Run an EKF measurement update to pull the dead-reckoning estimate
         toward what vision actually observed.

    Args (of the returned callable):
        p_cam (np.ndarray): (3,) camera world position in nav coords.
        R_cam (np.ndarray): (3,3) camera world rotation in nav coords.
        n_inliers (int): PnP inlier count — more → tighter noise covariance.
    """
    mf = rb3_cfg.CAM_MOUNT_FWD
    ml = rb3_cfg.CAM_MOUNT_LEFT

    def _cb(p_cam: np.ndarray, R_cam: np.ndarray, n_inliers: int) -> None:
        # Camera −Z in world = robot forward direction (nav convention).
        fwd = R_cam @ np.array([0.0, 0.0, -1.0])
        fwd[1] = 0.0                     # project onto the ground plane
        n = float(np.linalg.norm(fwd))
        if n < 1e-6:
            return
        fwd /= n
        # R_y(phi) @ [0,0,-1] = [-sin(phi), 0, -cos(phi)]  →  phi = atan2(-fx, -fz)
        phi  = float(np.arctan2(-fwd[0], -fwd[2]))   # camera yaw in nav
        eyaw = phi + math.pi / 2.0                    # robot yaw

        # Robot body position = camera position − lever arm (XZ components only;
        # height is irrelevant for the 2D EKF state).
        cf, sf = math.cos(eyaw), math.sin(eyaw)
        fwd_w  = np.array([cf, 0.0, -sf])
        left_w = np.array([-sf, 0.0, -cf])
        p_robot = p_cam - mf * fwd_w - ml * left_w

        # Measurement noise: scales inversely with sqrt of inlier count so
        # frames with many matches get more weight in the update.
        scale    = math.sqrt(max(cfg.VO_MIN_INLIERS, 1) / max(n_inliers, 1))
        sig_xy   = 0.08 * scale    # metres  (8 cm at 20 inliers)
        sig_yaw  = 0.05 * scale    # radians (3° at 20 inliers)
        R_noise  = np.diag([sig_xy**2, sig_xy**2, sig_yaw**2])

        ekf.update(float(p_robot[0]), float(p_robot[2]), eyaw, R_noise)
        print(f"[EKF/VO] inliers={n_inliers}  "
              f"x={p_robot[0]:+.3f}  z={p_robot[2]:+.3f}  "
              f"yaw={math.degrees(eyaw):+.1f}°")

    return _cb


# ─────────────────────────────────────────────────────────────────────────────
#  EKF
# ─────────────────────────────────────────────────────────────────────────────

class EKF2D:
    """Extended Kalman Filter for 2D ground-robot localisation.

    State:  mu = [x, z, yaw]
              x, z  — position in nav X-Z plane (metres)
              yaw   — heading about nav +Y (radians)

    Predict inputs (from HardwareBridge at 50 Hz):
      • gyro_z   — IMU yaw rate (rad/s); replaces the noisy encoder differential.
                   The gyro measures rotation directly and is unaffected by wheel
                   slip or unequal tyre radii — the main source of dead-reckoning
                   yaw error.  Falls back to encoder differential when gyro = 0
                   (not yet started or see_workhorse unavailable).
      • accel_fwd — forward acceleration (m/s²) is received but NOT integrated
                   into the state.  Raw accelerometer output includes ≈9.81 m/s²
                   of gravity whenever the forward axis is not perfectly level;
                   integrating that bias causes position to fly away.  The data
                   is available in get_odometry()["accel_fwd"] for diagnostics.
                   Proper accel fusion requires subtracting a calibrated static
                   bias measured while the robot is stationary.

    Update inputs (from NavmeshThread on VO success):
      • (x_meas, z_meas, yaw_meas) — VO-derived robot world pose; H = I.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.mu = np.zeros(3, np.float64)
        self.P  = np.diag([1e-4, 1e-4, 1e-4])
        # Q[2,2] (yaw) is smaller than before because the gyro is far more
        # accurate than the encoder differential for rotation.
        self.Q  = np.diag([4e-5, 4e-5, 1e-6])

    def predict(self, v_l: float, v_r: float, dt: float,
                gyro_z: float = 0.0, accel_fwd: float = 0.0) -> None:
        with self._lock:
            v     = (v_r + v_l) / 2.0
            w_enc = (v_r - v_l) / rb3_cfg.WHEEL_BASE_M

            # Use gyro yaw rate when available; clamp to the robot's physical
            # maximum to catch unit errors (e.g. deg/s read as rad/s).
            if abs(gyro_z) > 1e-9:
                w_gyro = gyro_z * rb3_cfg.GYRO_SCALE * rb3_cfg.GYRO_YAW_SIGN
                w = float(np.clip(w_gyro, -rb3_cfg.MAX_GYRO_RATE,
                                           rb3_cfg.MAX_GYRO_RATE))
            else:
                w = w_enc   # gyro not yet reading; fall back to encoders

            x, z, yaw = self.mu
            ny = yaw + w * dt
            self.mu = np.array([x + v * math.cos(ny) * dt,
                                 z - v * math.sin(ny) * dt,
                                 ny])
            F = np.array([[1., 0., -v * math.sin(ny) * dt],
                          [0., 1., -v * math.cos(ny) * dt],
                          [0., 0.,  1.]])
            self.P = F @ self.P @ F.T + self.Q

    def update(self, x_meas: float, z_meas: float, yaw_meas: float,
               R_noise: np.ndarray) -> None:
        """EKF measurement update with a VO-derived pose observation (x, z, yaw).

        H = I — direct state observation.  Includes a Mahalanobis gate to
        reject VO estimates that are implausibly far from the EKF prediction.
        """
        with self._lock:
            z_obs = np.array([x_meas, z_meas, yaw_meas])
            inn   = z_obs - self.mu
            inn[2] = (inn[2] + math.pi) % (2 * math.pi) - math.pi
            S = self.P + R_noise   # H = I, so H P Hᵀ = P
            try:
                mahal_sq = float(inn @ np.linalg.solve(S, inn))
                if mahal_sq > 11.3:   # chi²(3) 99% gate — reject outliers
                    return
                K = self.P @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return
            self.mu = self.mu + K @ inn
            self.mu[2] = (self.mu[2] + math.pi) % (2 * math.pi) - math.pi
            self.P = (np.eye(3) - K) @ self.P

    def reset(self):
        with self._lock:
            self.mu = np.zeros(3, np.float64)
            self.P  = np.diag([1e-4, 1e-4, 1e-4])

    @property
    def pose(self):
        with self._lock:
            return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])


# ─────────────────────────────────────────────────────────────────────────────
#  Shared robot state (read by GUI, written by controller)
# ─────────────────────────────────────────────────────────────────────────────

class RobotState:
    def __init__(self):
        self._lock    = threading.Lock()
        self.v_l      = 0.0
        self.v_r      = 0.0
        self.v_lin    = 0.0
        self.v_ang    = 0.0
        self.ekf_x    = 0.0
        self.ekf_z    = 0.0
        self.ekf_yaw  = 0.0
        self.gyro_z   = 0.0   # raw IMU yaw rate (rad/s, before GYRO_SCALE/sign)
        self.tick_l   = 0
        self.tick_r   = 0
        self.connected = False

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Controller
# ─────────────────────────────────────────────────────────────────────────────

class Controller:
    HZ = 20

    def __init__(self, hw: HardwareBridge, ekf: EKF2D, state: RobotState):
        self.hw    = hw
        self.ekf   = ekf
        self.state = state

        # Mode
        self.manual   = False   # True = GUI sliders drive the robot
        self.v_lin_manual = 0.0
        self.v_ang_manual = 0.0

        self._path_pts: list = []
        self._lock = threading.Lock()

    def update_path(self, path_pts: np.ndarray):
        if len(path_pts) >= 2:
            with self._lock:
                self._path_pts = [(float(p[0]), float(p[2])) for p in path_pts]

    def send_motor_test(self, d_l: float, d_r: float):
        """Send individual motor duties (for diagnosis)."""
        self.hw._ser and self.hw._ser.write(
            (f'{{"d_l":{d_l:.1f},"d_r":{d_r:.1f}}}\n').encode())

    def run(self):
        dt = 1.0 / self.HZ
        print("[Controller] Running")

        while not stop_event.is_set():
            t0 = time.time()

            if not self.hw.connected:
                time.sleep(dt)
                continue

            od = self.hw.get_odometry()
            rx, rz, ryaw = self.ekf.pose

            self.state.update(
                v_l      = od["v_l"],
                v_r      = od["v_r"],
                ekf_x    = rx,
                ekf_z    = rz,
                ekf_yaw  = math.degrees(ryaw),
                gyro_z   = od["gyro_z"],
                connected = self.hw.connected,
            )

            if self.manual:
                v_lin = self.v_lin_manual
                v_ang = self.v_ang_manual
            else:
                with self._lock:
                    path = list(self._path_pts)
                
                # print(self._path_pts)
                v_lin, v_ang = self._pure_pursuit(rx, rz, ryaw, path)
                

            self.hw.send_cmd(
                float(np.clip(v_lin, -rb3_cfg.MAX_V_LIN, rb3_cfg.MAX_V_LIN)),
                float(np.clip(v_ang, -rb3_cfg.MAX_V_ANG, rb3_cfg.MAX_V_ANG)),
            )
            
            self.state.update(v_lin=v_lin, v_ang=v_ang)

            time.sleep(max(0.0, dt - (time.time() - t0)))

        self.hw.send_cmd(0.0, 0.0)
        print("[Controller] Stopped")

    def _pure_pursuit(self, rx, rz, ryaw, path):
        if not path:
            return 0.0, 0.0
        L      = rb3_cfg.LOOKAHEAD_M
        target = next((wp for wp in reversed(path)
                        if math.hypot(wp[0]-rx, wp[1]-rz) <= L), path[0])
        tx, tz = target
        dist   = math.hypot(tx-rx, tz-rz)
        if dist < rb3_cfg.WP_REACHED_M:
            return 0.0, 0.0
        yaw_err = ((math.atan2(-(tz-rz), tx-rx) - ryaw) + math.pi) % (2*math.pi) - math.pi
        v_ang   = float(np.clip(3.0*yaw_err, -rb3_cfg.MAX_V_ANG, rb3_cfg.MAX_V_ANG))
        v_lin   = float(np.clip(dist*0.8, 0.0, rb3_cfg.MAX_V_LIN)) if abs(yaw_err) < 0.5 else 0.0
        return v_lin, v_ang


# ─────────────────────────────────────────────────────────────────────────────
#  Viewer + GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_robot(model, device, source, port, controller: Controller | None,
              state: RobotState, odom_source=None, vo_update_cb=None):

    threads = [
        CaptureThread(source, cfg.CAPTURE_INTERVAL_S, cfg.FRAME_SKIP,
                      odom_source=odom_source),
        InferenceThread(model, device),
        NavmeshThread(up_idx=1, vo_update_cb=vo_update_cb),
    ]
    for t in threads:
        t.start()

    server     = viser.ViserServer(port=port)
    camera_set = False

    print(f"\n[Viewer] Open  http://<rb3-ip>:{port}  in your browser")
    print("[Viewer] Ctrl-C to quit\n")

    # ── GUI panels ────────────────────────────────────────────────────────────

    with server.gui.add_folder("🤖 Robot Control"):
        g_manual = server.gui.add_checkbox("Manual Drive", initial_value=False)
        g_vlin   = server.gui.add_slider(
            "Forward (m/s)", min=-rb3_cfg.MAX_V_LIN, max=rb3_cfg.MAX_V_LIN,
            step=0.01, initial_value=0.0)
        g_vang   = server.gui.add_slider(
            "Turn (rad/s)", min=-rb3_cfg.MAX_V_ANG, max=rb3_cfg.MAX_V_ANG,
            step=0.05, initial_value=0.0)
        g_stop   = server.gui.add_button("⛔ STOP")
        g_vlin.disabled = True
        g_vang.disabled = True

    with server.gui.add_folder("🔧 Motor Test"):
        g_test_info = server.gui.add_markdown(
            "_Test individual motors to diagnose direction issues._")
        g_ml = server.gui.add_slider(
            "Left motor %", min=-100, max=100, step=5, initial_value=0)
        g_mr = server.gui.add_slider(
            "Right motor %", min=-100, max=100, step=5, initial_value=0)
        g_test_run  = server.gui.add_button("▶ Run Test")
        g_test_stop = server.gui.add_button("■ Stop Test")

    with server.gui.add_folder("📊 Diagnostics"):
        g_diag = server.gui.add_markdown("_Waiting for data…_")

    with server.gui.add_folder("🗺 Map"):
        g_reset_pose = server.gui.add_button("Reset EKF Pose")
        g_fov        = server.gui.add_slider(
            "Undistort FOV (°)", min=50.0, max=120.0, step=1.0,
            initial_value=float(cfg.UNDISTORT_FOV_DEG))
        g_reset_map  = server.gui.add_button("Reset Map")

    # ── callbacks ─────────────────────────────────────────────────────────────

    @g_manual.on_update
    def _toggle_manual(_):
        if controller:
            controller.manual = g_manual.value
        g_vlin.disabled = not g_manual.value
        g_vang.disabled = not g_manual.value

    @g_vlin.on_update
    def _vlin(_):
        if controller:
            controller.v_lin_manual = float(g_vlin.value)

    @g_vang.on_update
    def _vang(_):
        if controller:
            controller.v_ang_manual = float(g_vang.value)

    @g_stop.on_click
    def _stop(_):
        g_vlin.value = 0.0
        g_vang.value = 0.0
        if controller:
            controller.v_lin_manual = 0.0
            controller.v_ang_manual = 0.0
            controller.hw.send_cmd(0.0, 0.0)

    @g_test_run.on_click
    def _test_run(_):
        if controller:
            controller.send_motor_test(float(g_ml.value), float(g_mr.value))

    @g_test_stop.on_click
    def _test_stop(_):
        if controller:
            controller.send_motor_test(0.0, 0.0)
        g_ml.value = 0
        g_mr.value = 0

    @g_reset_pose.on_click
    def _reset_pose(_):
        if controller:
            controller.ekf.reset()

    @g_fov.on_update
    def _set_fov(_):
        # Output rectilinear FOV of the fisheye undistortion.  The remap is
        # rebuilt automatically on the next frame; click "Reset Map" to clear
        # the cloud built with the old FOV.
        cfg.UNDISTORT_FOV_DEG = float(g_fov.value)

    @g_reset_map.on_click
    def _reset_map(_):
        reset_map_event.set()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_camera(client, centroid):
        client.camera.position = (0.0, 2.0, 4.0)
        client.camera.look_at  = tuple(centroid.tolist())
        client.camera.up       = (0.0, 1.0, 0.0)

    diag_tick = 0
    stat_yaw0: float | None = None   # EKF yaw when the robot last went still
    stat_t0   = 0.0                  # time it went still (for drift-rate calc)

    # ── main loop ─────────────────────────────────────────────────────────────

    while not stop_event.is_set():

        # Point cloud / splat map
        try:
            m   = map_queue.get_nowait()
            pts    = m["pts"]
            colors = m["colors"]
            n   = min(len(pts), len(colors))
            if n > 0:
                pts    = np.ascontiguousarray(pts[:n])
                colors = np.ascontiguousarray(
                    colors[:n].clip(0,1).astype(np.float32))
                if cfg.USE_SPLATS:
                    r    = cfg.SPLAT_RADIUS
                    cov  = np.eye(3, dtype=np.float32) * (r*r)
                    covs = np.ascontiguousarray(np.tile(cov, (n,1,1)))
                    ops  = np.full((n,1), cfg.SPLAT_OPACITY, np.float32)
                    server.scene.add_gaussian_splats(
                        "/scene/map", centers=pts, covariances=covs,
                        rgbs=colors, opacities=ops)
                else:
                    server.scene.add_point_cloud(
                        "/scene/map", points=pts, colors=colors, point_size=0.008)

                if not camera_set:
                    centroid = pts.mean(axis=0)
                    for client in server.get_clients().values():
                        _set_camera(client, centroid)

                    @server.on_client_connect
                    def _init_cam(client):
                        _set_camera(client, centroid)
                    camera_set = True
        except queue.Empty:
            pass

        # Navmesh overlay
        try:
            nav = navmesh_queue.get_nowait()
            update_navmesh(server, nav)
            if controller:
                controller.update_path(nav.get("path_pts", np.array([])))
        except queue.Empty:
            pass

        # Robot body in scene
        if controller:
            rx, rz, ryaw = controller.ekf.pose
            qw = float(np.cos(ryaw/2))
            qy = float(np.sin(ryaw/2))
            server.scene.add_box(
                "/robot/body",
                dimensions=(0.20, 0.12, 0.25),
                wxyz=(qw, 0.0, qy, 0.0),
                position=(rx, 0.06, rz),
                color=(50, 200, 50),
            )

        # Diagnostics panel — update at ~2 Hz to avoid spam
        diag_tick += 1
        if diag_tick % 10 == 0:
            s = state.snapshot()
            conn  = "🟢 Connected" if s["connected"] else "🔴 Disconnected"
            mode  = "🕹 Manual" if (controller and controller.manual) else "🤖 Auto"

            # Stationary yaw-drift tracker — exposes gyro bias.  While the robot
            # is still, EKF yaw should hold; any creep is the gyro reading a
            # non-zero rate at rest.  Verify gyro units off `gyro_z` (a ~90 °/s
            # spin should read ~1.57 rad/s) and sign (left turn → yaw increases).
            gz    = s["gyro_z"]
            still = (abs(s["v_l"]) < 0.01 and abs(s["v_r"]) < 0.01
                     and abs(s["v_lin"]) < 0.005 and abs(s["v_ang"]) < 0.02)
            now_t = time.time()
            if not still:
                stat_yaw0 = None
                drift_str = "— (moving)"
            elif stat_yaw0 is None:
                stat_yaw0 = s["ekf_yaw"]
                stat_t0   = now_t
                drift_str = "settling…"
            else:
                elapsed = max(1e-3, now_t - stat_t0)
                d_yaw   = (s["ekf_yaw"] - stat_yaw0 + 180) % 360 - 180
                drift_str = f"{d_yaw / elapsed:+.2f} °/s  ({elapsed:.0f}s still)"

            g_diag.content = (
                f"**Pico:** {conn}  \n"
                f"**Mode:** {mode}  \n\n"
                f"| | Value |\n|---|---|\n"
                f"| EKF x | `{s['ekf_x']:+.3f} m` |\n"
                f"| EKF z | `{s['ekf_z']:+.3f} m` |\n"
                f"| EKF yaw | `{s['ekf_yaw']:+.1f} °` |\n"
                f"| gyro_z (raw) | `{gz:+.4f} rad/s` · `{math.degrees(gz):+.1f} °/s` |\n"
                f"| yaw drift (still) | `{drift_str}` |\n"
                f"| v_left | `{s['v_l']:+.3f} m/s` |\n"
                f"| v_right | `{s['v_r']:+.3f} m/s` |\n"
                f"| cmd v_lin | `{s['v_lin']:+.3f} m/s` |\n"
                f"| cmd v_ang | `{s['v_ang']:+.3f} rad/s` |\n"
            )

        time.sleep(0.05)   # ~20 Hz

    for t in threads:
        t.join(timeout=5)
    print("All threads stopped.")

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TEQUILA on RB3 Gen 2")
    parser.add_argument("--pico-port", default=rb3_cfg.PICO_PORT)
    parser.add_argument("--camera",    type=int,   default=rb3_cfg.CAMERA_INDEX)
    parser.add_argument("--width",     type=int,   default=rb3_cfg.INFER_WIDTH)
    parser.add_argument("--port",      type=int,   default=rb3_cfg.PORT)
    parser.add_argument("--map-depth", type=float, default=rb3_cfg.MAP_MAX_DEPTH_M)
    parser.add_argument("--no-nav",    action="store_true")
    parser.add_argument("--no-odom-fusion", action="store_true",
                        help="place frames with visual odometry instead of "
                             "the robot's measured wheel-odometry pose")
    args = parser.parse_args()

    cfg.DEPTH_MODEL_ID  = rb3_cfg.DEPTH_MODEL_ID
    cfg.INFER_WIDTH     = args.width
    cfg.MAP_MAX_DEPTH_M = args.map_depth
    cfg.NAV_MAX_DEPTH_M = rb3_cfg.NAV_MAX_DEPTH_M
    cfg.FOV_H_DEG       = rb3_cfg.FOV_H_DEG   # RB3 wide-angle camera (see rb3 config)
    cfg.ACCUM_ENABLED   = True

    # TSDF volumetric fusion (falls back to point accumulation if Open3D missing).
    cfg.USE_TSDF     = rb3_cfg.USE_TSDF
    cfg.TSDF_VOXEL_M = rb3_cfg.TSDF_VOXEL_M
    cfg.TSDF_TRUNC_M = rb3_cfg.TSDF_TRUNC_M

    # Timing / capture
    cfg.MIN_FRAME_BRIGHTNESS = rb3_cfg.MIN_FRAME_BRIGHTNESS
    cfg.NAV_INTERVAL_S       = rb3_cfg.NAV_INTERVAL_S

    # Fisheye undistortion for the RB3's wide-angle lens.
    cfg.FISHEYE              = rb3_cfg.FISHEYE
    cfg.FISHEYE_FOCAL_MM     = rb3_cfg.FISHEYE_FOCAL_MM
    cfg.SENSOR_PIXEL_UM      = rb3_cfg.SENSOR_PIXEL_UM
    cfg.SENSOR_FULL_WIDTH_PX = rb3_cfg.SENSOR_FULL_WIDTH_PX
    cfg.UNDISTORT_FOV_DEG    = rb3_cfg.UNDISTORT_FOV_DEG
    cfg.FISHEYE_CALIB_NPZ    = rb3_cfg.FISHEYE_CALIB_NPZ
    cfg.FISHEYE_CALIB_WH     = rb3_cfg.FISHEYE_CALIB_WH
    cfg.FISHEYE_CALIB_NPZ    = rb3_cfg.FISHEYE_CALIB_NPZ
    cfg.FISHEYE_CALIB_WH     = rb3_cfg.FISHEYE_CALIB_WH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")

    ekf   = EKF2D()
    hw    = HardwareBridge(port=args.pico_port,
                           wheel_radius=rb3_cfg.WHEEL_RADIUS_M,
                           wheel_base=rb3_cfg.WHEEL_BASE_M,
                           new_data_callback=ekf.predict,
                           accel_fwd_idx=rb3_cfg.ACCEL_FWD_IDX,
                           accel_fwd_sign=rb3_cfg.ACCEL_FWD_SIGN)
    state = RobotState()
    ctrl  = None
    odom_source  = None
    vo_update_cb = None

    if not args.no_nav:
        if hw.connect():
            ctrl = Controller(hw, ekf, state)
            threading.Thread(target=ctrl.run, daemon=True).start()

            # Level-1 odometry fusion: stitch the map from the robot's measured
            # pose instead of visual odometry (unless disabled).
            use_fusion = rb3_cfg.ODOM_FUSION and not args.no_odom_fusion
            if use_fusion:
                odom_source = make_odom_source(ekf)
                print("[Main] Map stitching: WHEEL ODOMETRY via EKF (level-1 fusion)")
            else:
                print("[Main] Map stitching: VISUAL ODOMETRY")

            # EKF drift correction: VO measures actual camera motion and
            # corrects the wheel-odometry dead-reckoning via an EKF update
            # step, whether odom or VO is driving the map.
            vo_update_cb = make_vo_update_cb(ekf)
            print("[Main] EKF VO correction: ENABLED")
        else:
            print("[Main] Pico not found — mapping only")
    else:
        print("[Main] Navigation disabled")

    print("[Main] Loading depth model…")
    model = load_model(device)

    try:
        run_robot(model=model, device=device, source=str(args.camera),
                  port=args.port, controller=ctrl, state=state,
                  odom_source=odom_source, vo_update_cb=vo_update_cb)
    except KeyboardInterrupt:
        stop_event.set()
        hw.disconnect()
        print("\n[Main] Stopped.")


if __name__ == "__main__":
    main()
