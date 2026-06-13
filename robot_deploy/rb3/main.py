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
    map_queue, navmesh_queue, stop_event,
)

import config as rb3_cfg
from hardware import HardwareBridge


# ─────────────────────────────────────────────────────────────────────────────
#  EKF
# ─────────────────────────────────────────────────────────────────────────────

class EKF2D:
    def __init__(self):
        self.mu = np.zeros(3, np.float64)
        self.P  = np.diag([1e-4, 1e-4, 1e-4])
        self.Q  = np.diag([4e-5, 4e-5, 2e-6])

    def predict(self, v_l: float, v_r: float, dt: float):
        wb = rb3_cfg.WHEEL_BASE_M
        v  = (v_r + v_l) / 2.0
        w  = (v_r - v_l) / wb
        
        # x, z, y = self.mu
        # ny = y + w * dt
        # self.mu = np.array([x + v*np.cos(ny)*dt,
        #                      z - v*np.sin(ny)*dt, ny])

        
        self.mu[0] += v_l*dt
        
        self.mu[1] = rb3_cfg.WHEEL_RADIUS_M
        
        self.mu[2] = v_l
        
        
        return 
        
        
        
        F = np.array([[1,0,-v*np.sin(ny)*dt],
                      [0,1,-v*np.cos(ny)*dt],
                      [0,0,1]])
        self.P = F @ self.P @ F.T + self.Q
        # print("BRRRRRRRR")

    def reset(self):
        self.mu = np.zeros(3, np.float64)
        self.P  = np.diag([1e-4, 1e-4, 1e-4])

    @property
    def pose(self):
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
            self.ekf.predict(od["v_l"], od["v_r"], od["dt"])
            rx, rz, ryaw = self.ekf.pose

            self.state.update(
                v_l      = od["v_l"],
                v_r      = od["v_r"],
                ekf_x    = od["x"],
                ekf_z    = od["y"],
                ekf_yaw  = math.degrees(od["theta"]),
                connected = self.hw.connected,
            )

            if self.manual:
                v_lin = self.v_lin_manual
                v_ang = self.v_ang_manual
            else:
                with self._lock:
                    path = list(self._path_pts)
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
              state: RobotState):

    threads = [
        CaptureThread(source, cfg.CAPTURE_INTERVAL_S, cfg.FRAME_SKIP),
        InferenceThread(model, device),
        NavmeshThread(up_idx=1),
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

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_camera(client, centroid):
        client.camera.position = (0.0, 2.0, 4.0)
        client.camera.look_at  = tuple(centroid.tolist())
        client.camera.up       = (0.0, 1.0, 0.0)

    diag_tick = 0

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
            g_diag.content = (
                f"**Pico:** {conn}  \n"
                f"**Mode:** {mode}  \n\n"
                f"| | Value |\n|---|---|\n"
                f"| EKF x | `{s['ekf_x']:+.3f} m` |\n"
                f"| EKF z | `{s['ekf_z']:+.3f} m` |\n"
                f"| EKF yaw | `{s['ekf_yaw']:+.1f} °` |\n"
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
    args = parser.parse_args()

    cfg.DEPTH_MODEL_ID  = rb3_cfg.DEPTH_MODEL_ID
    cfg.INFER_WIDTH     = args.width
    cfg.MAP_MAX_DEPTH_M = args.map_depth
    cfg.ACCUM_ENABLED   = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")

    hw    = HardwareBridge(port=args.pico_port,
                           wheel_radius=rb3_cfg.WHEEL_RADIUS_M,
                           wheel_base=rb3_cfg.WHEEL_BASE_M)
    ekf   = EKF2D()
    state = RobotState()
    ctrl  = None

    if not args.no_nav:
        if hw.connect():
            ctrl = Controller(hw, ekf, state)
            threading.Thread(target=ctrl.run, daemon=True).start()
        else:
            print("[Main] Pico not found — mapping only")
    else:
        print("[Main] Navigation disabled")

    print("[Main] Loading depth model…")
    model = load_model(device)

    try:
        run_robot(model=model, device=device, source=str(args.camera),
                  port=args.port, controller=ctrl, state=state)
    except KeyboardInterrupt:
        stop_event.set()
        hw.disconnect()
        print("\n[Main] Stopped.")


if __name__ == "__main__":
    main()
