"""
threads.py — The three worker threads and their shared communication queues.

Thread architecture
-------------------
  CaptureThread  (Thread 1) — reads frames from webcam or video file.
  InferenceThread(Thread 2) — runs depth inference + back-projection per frame.
  NavmeshThread  (Thread 3) — accumulates the world map and recomputes the navmesh.

All queues have maxsize=1 so each consumer always sees the latest data and
stale items are dropped automatically when a newer item arrives.

Shared queues (also imported by viewer.py)
------------------------------------------
  frame_queue   — CaptureThread   → InferenceThread  (raw BGR frames)
  pts_queue     — InferenceThread → NavmeshThread    (per-frame depth data)
  navmesh_queue — NavmeshThread   → Viewer           (navmesh overlay dict)
  map_queue     — NavmeshThread   → Viewer           (accumulated coloured cloud)
  stop_event    — set by main to request a clean shutdown of all threads
"""

import queue
import threading
import time

import cv2
import numpy as np

import tequila.config as cfg
import tequila.tsdf as tsdf_mod
from tequila.depth    import frame_to_result
from tequila.navmesh  import compute_navmesh, recompute_path
from tequila.odometry import icp_align, vo_align
from tequila.pointcloud import voxel_downsample_colored, voxel_downsample_pts, sor_pts

# ─────────────────────────────────────────────────────────────────────────────
#  Shared inter-thread queues
# ─────────────────────────────────────────────────────────────────────────────

frame_queue   = queue.Queue(maxsize=1)   # raw BGR frames
pts_queue     = queue.Queue(maxsize=1)   # per-frame depth data (tuple)
navmesh_queue = queue.Queue(maxsize=1)   # navmesh overlay dict
map_queue     = queue.Queue(maxsize=1)   # accumulated coloured point cloud
stop_event    = threading.Event()        # set this to shut down all threads
reset_map_event = threading.Event()      # set to clear the accumulated map
                                         # (e.g. after changing camera FOV)


def _push(q: queue.Queue, item) -> None:
    """Drop the oldest item (if any) and put a new one — non-blocking update."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(item)


def planar_lock(T: np.ndarray) -> np.ndarray:
    """Project a 4×4 camera pose onto the ground-plane motion manifold.

    The robot drives on a flat floor with a level, forward-facing camera, so the
    only real degrees of freedom are translation in the X-Z plane and yaw about
    the world up-axis (nav-Y).  This collapses the pose to exactly that:

      • rotation  → pure yaw about +Y (pitch and roll discarded)
      • translation → X and Z kept, Y pinned to 0 (the floor-0 camera height)

    Applying this every frame prevents the small per-frame VO errors in
    pitch/roll/vertical-translation from accumulating into a tilting, drifting
    world where the floor ends up at different heights across the map.
    """
    R = T[:3, :3]
    # Camera forward (−Z in camera frame) expressed in world coords.
    f = R @ np.array([0.0, 0.0, -1.0])
    f[1] = 0.0                       # drop the vertical component → planar heading
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        theta = 0.0                  # forward points straight up/down: keep yaw=0
    else:
        f /= n
        # R_y(θ) @ [0,0,-1] = [-sinθ, 0, -cosθ]  ⇒  θ = atan2(-fx, -fz)
        theta = float(np.arctan2(-f[0], -f[2]))

    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[ c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], dtype=T.dtype)

    T_out = np.eye(4, dtype=T.dtype)
    T_out[:3, :3] = R_yaw
    T_out[:3,  3] = T[:3, 3]
    T_out[1,   3] = 0.0              # pin camera height to the floor-0 level
    return T_out


# ─────────────────────────────────────────────────────────────────────────────
#  Thread 1 — Frame capture
# ─────────────────────────────────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    """Read frames from a webcam or a video file and push them to frame_queue.

    Webcam mode:
      Captures one frame every ``interval`` seconds, dropping stale queued
      frames so the inference thread always sees the freshest image.

    Video-file mode:
      Sends every ``frame_skip``-th frame and blocks until the inference thread
      consumes it — this prevents racing past the whole video before inference
      can run.
    """

    def __init__(self, source: str, interval: float, frame_skip: int,
                 odom_source=None) -> None:
        super().__init__(daemon=True, name="CaptureThread")
        self.interval    = interval
        self.frame_skip  = frame_skip
        # Optional callable () -> (4×4 nav camera-world pose | None).  When set,
        # the robot's measured pose is sampled at capture time and travels with
        # the frame so NavmeshThread can place it without visual odometry.
        self.odom_source = odom_source

        try:
            cam_idx      = int(source)
            self.is_file = False
            # self.cap     = cv2.VideoCapture(cam_idx)

            # During CAPTURE_INTERVAL_S seconds of sleep the camera produces
            # many frames (90 at 30 fps for a 3 s interval).  Without
            # max-buffers=1 the appsink queues ALL of them.  cap.read() then
            # returns the OLDEST buffered frame — from when the sleep started —
            # while odom_source() is sampled right NOW, causing an image/pose
            # mismatch (stale image, current pose).
            #
            # appsink max-buffers=1 drop=true:
            #   GStreamer "drop=true" drops OLD buffers when the 1-slot queue
            #   is full, so only the latest camera frame is ever buffered.
            #   cap.read() will block ≤ one camera frame period (~33 ms) until
            #   the next frame arrives, then return immediately — perfectly
            #   synchronised with the odom_source() call that follows.
            # sync=false: don't stall on the presentation clock.
            gst_pipeline = (
                "qtiqmmfsrc name=camsrc camera=0 ! "
                "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
                "videoconvert ! "
                "appsink max-buffers=1 drop=true sync=false"
            )

            # Open the video stream using OpenCV and GStreamer
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cam_idx}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[Capture] Webcam {cam_idx} opened")
        except ValueError:
            self.is_file = True
            self.cap     = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {source}")
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = self.cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"[Capture] {source}  ({total} frames @ {fps:.1f} fps, "
                  f"every {frame_skip}th frame)")

    def run(self) -> None:
        frame_idx = 0
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    print("[Capture] Video ended — viewer open, Ctrl-C to quit.")
                    break
                time.sleep(0.1)
                continue

            if self.is_file and frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            # Skip near-black frames (camera warm-up / dropouts).  On boot the
            # sensor streams a few dark/garbage frames before exposure settles;
            # fusing those drops a black blob at the world origin.
            if (not self.is_file and cfg.MIN_FRAME_BRIGHTNESS > 0
                    and float(frame.mean()) < cfg.MIN_FRAME_BRIGHTNESS):
                time.sleep(0.05)
                continue

            # Sample the robot pose at the instant the frame was captured so it
            # is not stale by the time depth inference finishes downstream.
            # Guard against a faulty source: degrade to visual odometry (pose=None)
            # rather than killing the capture thread (which would stop all frames).
            pose = None
            if self.odom_source is not None:
                try:
                    pose = self.odom_source()
                except Exception as e:   # noqa: BLE001 — never let capture die
                    print(f"[Capture] odom_source error: {e} — using VO this frame")

            if self.is_file:
                # Block until InferenceThread consumes the frame so we don't
                # race past the whole video before inference can run.
                frame_queue.put((frame, pose))
            else:
                # Webcam: always keep the freshest frame.
                _push(frame_queue, (frame, pose))
                time.sleep(self.interval)

        self.cap.release()
        print("[Capture] Stopped")


# ─────────────────────────────────────────────────────────────────────────────
#  Thread 2 — Depth inference + point cloud generation
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """Run frame_to_result() on every incoming frame and forward the output.

    Pushes a tuple to pts_queue containing:
      (nav_pts, map_pts, map_colors, img, depth_m, focal, cx, cy, odom_T)
    where odom_T is the capture-time robot pose (4×4) or None.  Consumed by
    NavmeshThread.
    """

    def __init__(self, model, device: str) -> None:
        super().__init__(daemon=True, name="InferenceThread")
        self.model  = model
        self.device = device

    def run(self) -> None:
        count = 0
        while not stop_event.is_set():
            try:
                frame, pose = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0     = time.time()
            result = frame_to_result(frame, self.model)
            dt     = time.time() - t0
            count += 1

            if result is not None:
                nav = len(result["nav_pts"])
                print(f"[Inference] Frame {count:3d} | {nav:>7,} nav pts | {dt:.2f}s")
                _push(pts_queue, (
                    result["nav_pts"],
                    result["map_pts"],
                    result["map_colors"],
                    result["img"],
                    result["depth_m"],
                    result["focal"],
                    result["cx"],
                    result["cy"],
                    pose,
                ))
            else:
                print(f"[Inference] Frame {count:3d} | zero pts  | {dt:.2f}s")

        print("[Inference] Stopped")


# ─────────────────────────────────────────────────────────────────────────────
#  Thread 3 — World-map accumulation and navmesh computation
# ─────────────────────────────────────────────────────────────────────────────

class NavmeshThread(threading.Thread):
    """Accumulate a world-space point cloud and periodically recompute the navmesh.

    Frame alignment
    ~~~~~~~~~~~~~~~
    If a frame arrives stamped with a robot odometry pose (level-1 fusion), that
    pose is used directly to place the frame — anchored so the first frame is the
    world origin.  Otherwise each frame is aligned to the previous one using
    SIFT+PnP visual odometry, with ICP as a fallback (too few texture features).
    The result is a cumulative world transform T_cum.

    Map accumulation
    ~~~~~~~~~~~~~~~~
    Two clouds are maintained:
      accum_pts       — coarse position-only (for navmesh RANSAC)
      accum_pts_fine  — fine coloured (for the display map)

    Both are periodically voxel-downsampled and capped at NAV_ACCUM_MAX_PTS to
    prevent unbounded memory growth.

    Navmesh recompute
    ~~~~~~~~~~~~~~~~~
    Runs in a separate worker thread so the (1-2 s) compute_navmesh pass never
    stalls frame accumulation or the map display.  The main loop publishes the
    latest cloud + camera pose under a lock; the worker picks it up every
    NAV_INTERVAL_S, recomputes, and pushes the result to navmesh_queue.
    """

    def __init__(self, up_idx: int, vo_update_cb=None, pose_source=None) -> None:
        super().__init__(daemon=True, name="NavmeshThread")
        self.up_idx = up_idx
        # Optional callback(p_cam, R_cam, n_inliers) — called whenever VO
        # produces a reliable camera world pose so the EKF can be corrected.
        self.vo_update_cb = vo_update_cb
        # Optional callable () -> 4×4 camera world pose (nav coords).  When set,
        # the navmesh worker queries this AFTER the slow compute pass to re-run
        # just the A* path from the robot's current position, correcting for
        # movement that occurred during the ~0.5–3 s navmesh computation.
        self.pose_source = pose_source
        # Shared hand-off to the navmesh worker thread.
        self._nav_lock  = threading.Lock()
        self._nav_input = None      # (accum_pts, cam_pos, cam_fwd, trajectory)
        self._nav_reset = False     # signal the worker to drop its committed goal

    def run(self) -> None:
        last_map_run: float = 0.0  # last TSDF map extract+push (display cadence)
        placed_count      = 0      # accepted (placed) frames — for viewer display
        accum_pts         = None   # coarse world-space positions (navmesh RANSAC)
        accum_pts_fine    = None   # fine world-space positions   (display)
        accum_colors      = None   # colours matching accum_pts_fine
        T_cum             = np.eye(4)
        trajectory        = []     # list of (3,) camera positions, one per accepted frame
        prev_cam          = None   # coarse nav pts from previous frame (ICP)
        prev_img          = None   # BGR image from previous frame      (VO)
        prev_depth        = None   # depth map from previous frame      (VO)
        prev_focal        = None
        prev_cx           = None
        prev_cy           = None
        odom_T0           = None   # first odometry pose (world origin anchor)
        last_pose_T       = None   # last accumulated pose (odom duplicate skip)
        prev_odom_T       = None   # odom T_cum matching the frame in prev_img (VO)

        # Navmesh runs in its own thread so its slow recompute never blocks this
        # accumulation / map-display loop.  The committed exploration goal lives
        # in the worker.
        threading.Thread(target=self._navmesh_worker, daemon=True,
                         name="NavmeshWorker").start()

        # Optional TSDF volumetric fusion (averages overlapping views instead of
        # piling up raw points).  Falls back to point accumulation if Open3D is
        # missing.
        tsdf              = None
        frames_integrated = 0
        use_tsdf          = bool(getattr(cfg, "USE_TSDF", False))
        if use_tsdf and not tsdf_mod.available():
            print("[Navmesh] USE_TSDF set but Open3D is not installed — "
                  "using point accumulation")
            use_tsdf = False
        if use_tsdf:
            tsdf = tsdf_mod.TSDFFusion(cfg.TSDF_VOXEL_M, cfg.TSDF_TRUNC_M,
                                       cfg.MAP_MAX_DEPTH_M)
            print("[Navmesh] Map fusion: TSDF volumetric")

        while not stop_event.is_set():
            # Map reset (e.g. after changing camera FOV) — drop all accumulated
            # geometry and start fresh so old, wrongly-projected points don't
            # linger alongside newly-projected ones.
            if reset_map_event.is_set():
                reset_map_event.clear()
                accum_pts = accum_pts_fine = accum_colors = None
                T_cum       = np.eye(4)
                trajectory  = []
                prev_cam = prev_img = prev_depth = None
                prev_focal = prev_cx = prev_cy = None
                odom_T0 = last_pose_T = prev_odom_T = None
                if tsdf is not None:
                    tsdf.reset()
                    frames_integrated = 0
                with self._nav_lock:
                    self._nav_input = None
                    self._nav_reset = True
                print("[Navmesh] Map reset — accumulation cleared")

            got_new = False

            # Drain pts_queue — process all pending frames before sleeping.
            while True:
                try:
                    (new_cam, new_map_pts, new_map_colors,
                     new_img, new_depth, focal, cx, cy, odom_T) = pts_queue.get_nowait()
                except queue.Empty:
                    break

                # ── Frame alignment ───────────────────────────────────────────
                if odom_T is not None:
                    # Level-1 odometry fusion: place the frame using the robot's
                    # measured wheel-odometry pose instead of visual odometry.
                    # Anchor to the first pose so the world origin is the start.
                    if odom_T0 is None:
                        odom_T0 = odom_T.copy()
                    T_cum = np.linalg.inv(odom_T0) @ odom_T

                    # Duplicate-view skip: if the robot barely moved since the
                    # last accumulated frame, drop this one to avoid piling up
                    # redundant points at a standstill.
                    if last_pose_T is not None:
                        d_shift = float(np.linalg.norm(
                            T_cum[:3, 3] - last_pose_T[:3, 3]))
                        dR    = T_cum[:3, :3] @ last_pose_T[:3, :3].T
                        d_ang = float(np.degrees(np.arccos(
                            np.clip((np.trace(dR) - 1) / 2, -1, 1))))
                        if d_shift < cfg.VO_MIN_SHIFT_M and d_ang < 1.0:
                            prev_cam   = new_cam
                            prev_img   = new_img
                            prev_depth = new_depth
                            prev_focal = focal
                            prev_cx    = cx
                            prev_cy    = cy
                            prev_odom_T = T_cum.copy()
                            continue

                    # VO side-run for EKF correction — odom places the frame,
                    # VO measures the actual camera motion to correct EKF drift.
                    if (self.vo_update_cb is not None
                            and prev_img is not None
                            and prev_odom_T is not None):
                        R_vo, t_vo, n_vo = vo_align(
                            prev_img, prev_depth, new_img,
                            prev_focal, prev_cx, prev_cy)
                        if R_vo is not None and n_vo >= cfg.VO_MIN_INLIERS:
                            # Anchor VO delta to the odom pose of the previous
                            # accepted frame so each update is independent.
                            p_cam_vo = prev_odom_T[:3, :3] @ t_vo + prev_odom_T[:3, 3]
                            R_cam_vo = prev_odom_T[:3, :3] @ R_vo
                            self.vo_update_cb(p_cam_vo, R_cam_vo, n_vo)

                elif prev_cam is None:
                    # First frame: world coordinate system = camera frame 0.
                    T_cum = np.eye(4)
                else:
                    aligned      = False
                    n_inliers    = 0
                    vo_succeeded = False

                    # Primary: ORB + PnP ──────────────────────────────────────
                    R_rel, t_rel, n_inliers = vo_align(
                        prev_img, prev_depth, new_img,
                        prev_focal, prev_cx, prev_cy)

                    if R_rel is not None and n_inliers >= cfg.VO_MIN_INLIERS:
                        shift = float(np.linalg.norm(t_rel))
                        angle = float(np.degrees(np.arccos(
                            np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))

                        if shift < cfg.VO_MIN_SHIFT_M:
                            # Camera barely moved — duplicate view.
                            # Update state but skip this frame for accumulation.
                            print(f"[VO]  inliers={n_inliers:3d}  "
                                  f"shift={shift:.3f}m  rot={angle:.1f}°  "
                                  "— duplicate view, skip")
                            prev_cam   = new_cam
                            prev_img   = new_img
                            prev_depth = new_depth
                            prev_focal = focal
                            prev_cx    = cx
                            prev_cy    = cy
                            continue

                        elif shift < cfg.VO_MAX_SHIFT_M and angle < cfg.VO_MAX_ROT_DEG:
                            T_rel         = np.eye(4)
                            T_rel[:3, :3] = R_rel
                            T_rel[:3,  3] = t_rel
                            T_cum         = T_cum @ T_rel
                            print(f"[VO]  inliers={n_inliers:3d}  "
                                  f"shift={shift:.3f}m  rot={angle:.1f}°")
                            aligned      = True
                            vo_succeeded = True
                        else:
                            print(f"[VO]  inliers={n_inliers} but "
                                  f"shift={shift:.3f}m rot={angle:.1f}° "
                                  "— sanity check failed")

                    # Fallback: ICP ──────────────────────────────────────────
                    if not aligned:
                        n_s = min(cfg.ICP_SUBSAMPLE, len(new_cam))
                        n_t = min(cfg.ICP_SUBSAMPLE, len(prev_cam))
                        si  = np.random.choice(len(new_cam),  n_s, replace=False)
                        ti  = np.random.choice(len(prev_cam), n_t, replace=False)
                        R_icp, t_icp, fitness = icp_align(
                            new_cam[si], prev_cam[ti])
                        shift = float(np.linalg.norm(t_icp))
                        angle = float(np.degrees(np.arccos(
                            np.clip((np.trace(R_icp) - 1) / 2, -1, 1))))

                        if (fitness >= cfg.ICP_FITNESS_MIN
                                and shift < cfg.ICP_MAX_SHIFT_M
                                and angle < cfg.ICP_MAX_ROT_DEG):
                            T_rel         = np.eye(4)
                            T_rel[:3, :3] = R_icp
                            T_rel[:3,  3] = t_icp
                            T_cum         = T_cum @ T_rel
                            print(f"[ICP fallback]  fitness={fitness:.2f}  "
                                  f"shift={shift:.3f}m  rot={angle:.1f}°")
                            aligned = True
                        else:
                            print(f"[Align] Both VO+ICP failed — skipping frame "
                                  f"(VO inliers={n_inliers}, "
                                  f"ICP fitness={fitness:.2f})")
                            prev_cam   = new_cam
                            prev_img   = new_img
                            prev_depth = new_depth
                            prev_focal = focal
                            prev_cx    = cx
                            prev_cy    = cy
                            continue

                # Ground-robot planar-motion lock: re-project the cumulative
                # pose onto the floor plane each frame so pitch/roll/vertical
                # drift cannot accumulate (keeps every floor at one level).
                if cfg.PLANAR_LOCK:
                    T_cum = planar_lock(T_cum)

                last_pose_T = T_cum.copy()   # for odom duplicate-view skip

                # Diagnostic: the camera heading this frame is placed at, relative
                # to frame 0.  Rotate the robot in place and watch this: if it
                # tracks the turn the pose is fine; if it stays ~0 while the robot
                # turns, the pose feeding the map is stale (sensor/EKF lag), which
                # duplicates walls at the turn angle.
                _fwd = T_cum[:3, :3] @ np.array([0.0, 0.0, -1.0])
                _cam_yaw_deg = float(np.degrees(np.arctan2(-_fwd[0], -_fwd[2])))
                _frame_src   = "odom" if odom_T is not None else "VO"
                placed_count += 1
                print(f"[Map] placed frame #{placed_count}  cam_yaw={_cam_yaw_deg:+6.1f}°  "
                      f"pos=({T_cum[0, 3]:+.2f}, {T_cum[2, 3]:+.2f})  "
                      f"src={_frame_src}")

                # In pure-VO mode: feed the locked camera world pose into the
                # EKF so vision corrects wheel-odometry drift each frame.
                if (self.vo_update_cb is not None
                        and odom_T is None
                        and vo_succeeded):
                    self.vo_update_cb(
                        T_cum[:3, 3].copy(), T_cum[:3, :3].copy(), n_inliers)

                # Record camera position for the trajectory trail
                trajectory.append(T_cum[:3, 3].copy())

                if use_tsdf:
                    # Volumetric fusion: integrate this RGBD frame.  The fused
                    # cloud is extracted later at the navmesh cadence (extracting
                    # every frame would be too slow).
                    tsdf.integrate(new_depth, new_img, focal, cx, cy, T_cum)
                    frames_integrated += 1
                else:
                    R_w = T_cum[:3, :3]
                    t_w = T_cum[:3,  3]

                    # ── Coarse nav accumulation (navmesh RANSAC) ──────────────
                    world_nav = (R_w @ new_cam.T).T + t_w
                    if not cfg.ACCUM_ENABLED:
                        accum_pts = world_nav
                    elif accum_pts is None:
                        accum_pts = world_nav
                    else:
                        accum_pts = np.concatenate([accum_pts, world_nav], axis=0)
                        accum_pts = voxel_downsample_pts(accum_pts, cfg.NAV_ACCUM_VOXEL)
                        if len(accum_pts) > 5000:
                            accum_pts = sor_pts(accum_pts, nb=10, std_ratio=2.0)
                        if len(accum_pts) > cfg.NAV_ACCUM_MAX_PTS:
                            accum_pts = accum_pts[-cfg.NAV_ACCUM_MAX_PTS // 2:]

                    # ── Fine coloured accumulation (display map) ──────────────
                    world_map = (R_w @ new_map_pts.T).T + t_w
                    if len(world_map) == 0:
                        pass   # nothing to add (all points beyond MAP_MAX_DEPTH_M)
                    elif not cfg.ACCUM_ENABLED:
                        accum_pts_fine = world_map
                        accum_colors   = new_map_colors
                    elif accum_colors is None:
                        accum_pts_fine = world_map
                        accum_colors   = new_map_colors
                    else:
                        accum_pts_fine = np.concatenate(
                            [accum_pts_fine, world_map], axis=0)
                        accum_colors   = np.concatenate(
                            [accum_colors, new_map_colors], axis=0)
                        accum_pts_fine, accum_colors = voxel_downsample_colored(
                            accum_pts_fine, accum_colors, cfg.VOXEL_SIZE)
                        if len(accum_pts_fine) > cfg.NAV_ACCUM_MAX_PTS:
                            accum_pts_fine = accum_pts_fine[-cfg.NAV_ACCUM_MAX_PTS // 2:]
                            accum_colors   = accum_colors  [-cfg.NAV_ACCUM_MAX_PTS // 2:]

                    # Push the latest coloured map to the viewer, bundling the
                    # frame's historical pose so the viewer can display it.
                    if accum_pts_fine is not None:
                        _push(map_queue, dict(
                            pts         = accum_pts_fine.astype(np.float32),
                            colors      = accum_colors.astype(np.float32),
                            frame_count = placed_count,
                            cam_pos     = T_cum[:3, 3].copy().astype(np.float32),
                            cam_yaw_deg = _cam_yaw_deg,
                            src         = _frame_src,
                        ))

                # Save state for next frame's alignment
                if odom_T is not None:
                    prev_odom_T = T_cum.copy()
                prev_cam   = new_cam
                prev_img   = new_img
                prev_depth = new_depth
                prev_focal = focal
                prev_cx    = cx
                prev_cy    = cy

                got_new = True

            now = time.time()

            if use_tsdf:
                if frames_integrated == 0:
                    time.sleep(0.2)
                    continue
                # Map refresh (fast): extract the fused surface and push it for
                # display on its own cadence, decoupled from the navmesh worker
                # so newly-seen areas appear promptly.
                if got_new and now - last_map_run >= cfg.MAP_INTERVAL_S:
                    last_map_run = now
                    ex_pts, ex_cols = tsdf.extract()
                    if len(ex_pts) >= cfg.MIN_FLOOR_POINTS * 2:
                        _push(map_queue, dict(pts=ex_pts, colors=ex_cols))
                        accum_pts = voxel_downsample_pts(ex_pts, cfg.NAV_ACCUM_VOXEL)

            # Publish the latest cloud + camera pose for the navmesh worker.
            # This is a cheap reference hand-off — the worker does the heavy
            # SOR + compute_navmesh on its own thread.
            if got_new and accum_pts is not None:
                traj = (np.array(trajectory, dtype=np.float32)
                        if len(trajectory) >= 2 else None)
                with self._nav_lock:
                    self._nav_input = (
                        accum_pts,
                        T_cum[:3, 3].copy(),
                        T_cum[:3, :3] @ np.array([0.0, 0.0, -1.0]),
                        traj,
                    )

            time.sleep(0.1 if use_tsdf else 0.05)

        print("[Navmesh] Stopped")

    # ── Navmesh worker (separate thread) ──────────────────────────────────────
    def _navmesh_worker(self) -> None:
        """Recompute the navmesh on the latest published cloud, off the main loop.

        Reads the (accum_pts, camera pose, trajectory) snapshot published by
        run(), recomputes the navmesh every NAV_INTERVAL_S, and pushes the result
        to navmesh_queue.  Keeps its own committed exploration goal so the path
        does not churn between recomputes.
        """
        current_goal = None
        last_run     = 0.0
        while not stop_event.is_set():
            with self._nav_lock:
                if self._nav_reset:
                    self._nav_reset = False
                    current_goal = None
                inp = self._nav_input

            if inp is None:
                time.sleep(0.2)
                continue
            if time.time() - last_run < cfg.NAV_INTERVAL_S:
                time.sleep(0.1)
                continue

            accum_pts, cam_world_pos, cam_world_fwd, traj = inp
            if accum_pts is None or len(accum_pts) < cfg.MIN_FLOOR_POINTS * 2:
                time.sleep(0.2)
                continue

            # Adaptive SOR: scale nb down on large clouds so the KDTree query
            # does not dominate runtime.
            nb_sor      = max(4, min(10, 60_000 // max(len(accum_pts), 1)))
            clean_accum = sor_pts(accum_pts, nb=nb_sor, std_ratio=2.0)
            if len(clean_accum) < cfg.MIN_FLOOR_POINTS * 2:
                clean_accum = accum_pts   # not enough pts to SOR safely

            print(f"[Navmesh] Computing on {len(clean_accum):,} pts "
                  f"(raw={len(accum_pts):,}, cam @ {cam_world_pos.round(2)})...")
            t0  = time.time()
            nav = compute_navmesh(clean_accum, self.up_idx,
                                  camera_origin  = cam_world_pos,
                                  camera_forward = cam_world_fwd,
                                  prev_goal      = current_goal)
            print(f"[Navmesh] Done in {time.time() - t0:.2f}s")
            last_run = time.time()

            if nav is not None:
                current_goal = nav.get("goal")   # commit goal across recomputes

                # Path-latency correction: the robot may have moved significantly
                # during the ~0.5–3 s compute_navmesh() call.  Re-run just the
                # cheap A* step from the current measured pose so the published
                # path starts where the robot actually is now, not where it was
                # when computation began.
                if self.pose_source is not None:
                    try:
                        T_cur = self.pose_source()
                        if T_cur is not None:
                            cur_pos = np.asarray(T_cur[:3, 3], dtype=np.float64)
                            shift   = float(np.linalg.norm(cur_pos - cam_world_pos))
                            if shift > 0.02:   # moved > 2 cm during compute
                                new_path, new_goal = recompute_path(
                                    nav["free_nodes"], nav["edges"],
                                    cur_pos, current_goal)
                                nav["path_pts"] = new_path
                                if new_goal is not None:
                                    nav["goal"]  = new_goal
                                    current_goal = new_goal
                                print(f"[Navmesh] Path corrected for {shift:.3f} m "
                                      f"robot movement during compute")
                    except Exception as exc:
                        print(f"[Navmesh] pose_source error (path correction skipped): {exc}")

                nav["trajectory"]    = traj
                nav["frame_cam_pos"] = cam_world_pos.astype(np.float32)
                nav["frame_cam_yaw"] = float(np.degrees(
                    np.arctan2(-cam_world_fwd[0], -cam_world_fwd[2])))
                _push(navmesh_queue, nav)

        print("[Navmesh] Worker stopped")
