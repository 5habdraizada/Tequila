# TEQUILA

TEQUILA is a live 3D mapping and navigation system we've been building for a Qualcomm RB3 Gen 2 robot. Point a camera at a room, and it turns the video into a coloured 3D point cloud, figures out where the floor and obstacles are, and plans a path to explore the space — all live, in a browser.

It started as a webcam-only software pipeline (still works standalone, no robot needed) and has since grown a real hardware leg: a Pi Pico reading wheel encoders and a gyro, an EKF fusing that with vision, and a control panel for driving the actual robot. Two of us work on this — one on the mapping/vision side (`tequila/`), one on the odometry/EKF/firmware side (`robot_deploy/`).

It mostly works. It also still smears itself into a mess sometimes. More on that below.

---

## What it does

1. **Builds a 3D map** — every frame becomes a coloured point cloud that gets fused into a growing world-space map
2. **Finds the floor and obstacles** — fits a ground plane, lays a grid of waypoints on it, and marks which ones are blocked
3. **Plans where to go** — A* from the robot's current spot to the farthest reachable waypoint, so it keeps pushing into unexplored territory

All of it streams to a browser at `http://localhost:8080` (or `http://<rb3-ip>:8080` on the robot) via [viser](https://viser.studio) — nothing to install on the viewing device.

```
Camera / Video
      │
      ▼
 CaptureThread ──► InferenceThread ──► NavmeshThread
  (grabs frames)    (depth model,        (floor fit, obstacle
                      SIFT+PnP VO,         filter, A* — runs on
                      back-projection)     its own worker thread)
                        │                    │
                        ▼                    ▼
                   Coloured map         Navmesh overlay
                        └──────────────────┘
                                 │
                            Viewer thread
                         (http://localhost:8080)
```

On the robot, a fourth source of truth enters the picture: wheel odometry + gyro, fused through an EKF, which can drive the map instead of (or alongside) vision. More on that in [Running it on the robot](#running-it-on-the-robot).

---

## Where things stand

**Working reasonably well:**
- Metric depth (Depth Anything V2) — no per-frame scale drift, so accumulated frames actually line up
- SIFT+PnP visual odometry with ICP as a fallback when a wall has no texture to track
- Floor detection (GPP sector-plane fitting, RANSAC fallback) — handles tilted/noisy clouds and won't lock onto the ceiling
- Frontier-exploration navmesh + A* — the robot reliably picks a far, reachable spot and paths to it, and now commits to that goal instead of re-picking a new one every recompute (it used to twitch back and forth constantly)
- Navmesh recompute runs on its own thread now, so a slow (1-3 s) recompute no longer stalls the map display
- Fisheye undistortion with a real measured camera calibration (not just a lens-spec approximation)
- On the robot: wheel + gyro odometry through an EKF, with vision correcting its drift, and a live control panel (manual drive, motor test, gyro diagnostics, VO on/off toggle)

**Actively being worked on / half-working:**
- TSDF volumetric fusion as an alternative to raw point accumulation — implemented, off by default, unclear yet if it's actually better than point accumulation + SOR for our use case
- Gyro calibration on the real robot (sign, scale, bias) — still mostly manual, "spin it and check the printed sign" territory

---

## Current problems

Being honest about the state of things:

**The map still fans out and duplicates itself.** This is the big one. If yaw drifts even a couple of degrees between frames — from wheel slip, from a VO frame that barely missed its inlier threshold, from gyro bias — every frame after that is placed at a slightly wrong angle, and the same wall gets painted into the map multiple times at different angles. It looks like a wall exploding outward into a fan of ghost copies. We've attacked this from a few angles (planar-motion lock so pitch/roll can't drift, tighter VO acceptance thresholds, gyro fusion so yaw doesn't depend on wheel slip at all) and it's better than it was, but it's not solved. A held-still robot on a hard floor for a few minutes will visibly rotate in the EKF display — that's raw gyro bias creeping in, and it directly becomes map smearing over time.

**Visual odometry still fails on low-texture surfaces.** Blank walls, dim rooms, glossy floors — SIFT just doesn't find enough keypoints, and it falls back to ICP, which is worse and more prone to a bad fit if the geometry is ambiguous (a long flat corridor, for instance).

**The odom/VO fusion is a coarse toggle, not a real filter.** Right now you can either drive the map from wheel+gyro odometry, or from vision, and vision can nudge the EKF when it succeeds. It's not a tightly tuned sensor fusion — the measurement noise VO feeds into the EKF is a rough heuristic based on inlier count, not something we've actually characterized.

**The RB3 install script is out of date.** `robot_deploy/rb3/install.sh` still references a separate Pi bridge process (`pi/bridge.py`, a `PI_IP` setting) from an earlier architecture. The current code talks to the Pico directly over USB serial — the install script just hasn't caught up.

**There's dead weight in the repo.** `depth_anything_v2/` (the vendored relative-depth model code) isn't imported by anything in the current pipeline anymore — the whole thing runs on the metric Depth Anything V2 model now. `tequila/hardware.py` is in the same boat: it's a TCP client for an even older architecture (a separate Pi bridge process) that predates the RB3 talking to the Pico directly over USB serial (`robot_deploy/rb3/hardware.py` is the one actually in use). Both are just sitting there from earlier iterations.

**Recovering from a bad map is manual.** If the map does fan out, the fix today is "hit Reset Map and drive around again," not anything automatic.

If you look at the map in the viewer and it's a clean, recognizable room — that's the pipeline working as intended. If it looks like a wall got put through a blender, that's yaw drift, and it's the main thing left to fix.

---

## How it works

### Step 1 — Depth inference (`tequila/depth.py`)

Each incoming frame is resized to `--width` pixels and run through **Depth Anything V2**, a transformer depth model, in its *metric* configuration — it outputs actual distances in metres, not a 0-1 normalized map. That matters a lot: a relative-depth model rescales every frame independently, so the same real-world distance maps to a different pixel value from one frame to the next, and stitching frames together amplifies that into a stretching, fanning mess. Metric depth sidesteps that entirely.

**Flying-pixel removal:** depth models blend intermediate values at object edges (a chair leg against a far wall, say), which back-project into long spikes trailing behind every object. We remove them with a two-stage test — a pixel is flagged if it's either well past its local neighbourhood minimum (erosion test) or sitting on a sharp relative depth gradient — then dilate the bad-pixel mask a couple pixels to catch the blended edge pixels around it.

**Fisheye undistortion** (when enabled): the RB3's lens is wide enough that the pinhole back-projection math would be wrong without correcting for it first. We rectify to a narrower rectilinear FOV using either a measured calibration (`tools/camera_calibration.py`, chessboard-based) or, if that's not available, an equidistant-lens approximation from the lens spec sheet.

Each frame produces two point clouds: a coarse, position-only one (`nav_pts`) for floor detection and the navmesh, and a fine, coloured one (`map_pts`) for what you actually see in the viewer.

### Step 2 — Frame alignment (`tequila/odometry.py`)

To build a consistent map, every new frame has to be placed in the same coordinate system as everything before it. Two ways this happens:

**On the robot, if wheel+gyro fusion is on:** the EKF's pose (from `robot_deploy/rb3/main.py`) places the frame directly — no vision needed for this part. Visual odometry still runs in parallel and, when it finds a confident alignment, nudges the EKF back toward what the camera actually saw, correcting wheel-odometry drift.

**Otherwise (default, webcam/video mode):** SIFT keypoints are matched between the current and previous frame (Lowe's ratio test to kill ambiguous matches), the matched previous-frame points are lifted to 3D using the stored depth map, and `solvePnPRansac` recovers the relative camera pose. A result needs at least 20 inliers, translation under 2 m, and rotation under 15° to be accepted — anything wilder gets thrown out as noise. If that fails (not enough texture to match), it falls back to point-to-point ICP on the coarse clouds, requiring at least 35% inlier fitness under the same shift/rotation caps.

Either way, the robot only drives on a flat floor, so after alignment the pose gets projected back onto a pure X-Z-translation-plus-yaw manifold (`planar_lock`) — pitch, roll, and vertical drift get zeroed out every frame instead of being allowed to accumulate.

### Step 3 — Map accumulation (`tequila/threads.py`)

Aligned frames get transformed into world space and merged into two running clouds: a coarse one for floor/navmesh work (voxel-downsampled, periodically SOR-cleaned), and a fine coloured one for display, capped to a configurable max depth so distant, noisier points don't stretch into long fan arms. Both are hard-capped at 500,000 points. There's also an optional TSDF volumetric fusion mode (`USE_TSDF`) that integrates depth into a voxel grid instead of piling up raw points — it averages overlapping observations instead of letting noise stack, but it's newer and less battle-tested than plain accumulation.

### Step 4 — Floor detection (`tequila/navmesh.py`)

Every `--nav-interval` seconds, the accumulated cloud gets split into angular sectors around the vertical axis, a plane is PCA-fit per sector, and the largest group of sectors whose normals roughly agree becomes the floor candidate. If GPP can't find enough agreement (or the "floor" it found turns out to be above the camera — a table or the ceiling, usually from drift), it falls back to classic RANSAC with the camera height as a hard ceiling on where a floor can be.

### Step 5 — Navigation mesh + path planning (`tequila/navmesh.py`)

A grid of waypoints gets laid directly onto the fitted floor plane (not snapped to noisy floor points, so every node sits at exactly the same height). Non-floor points in a height band above the floor become obstacles; any node within the clearance radius of one is blocked. Free nodes within range of each other get connected if a line-of-sight check along the edge stays clear. A* then runs from the node nearest the camera to the farthest node it can actually reach — a simple frontier-exploration strategy — and it now *commits* to that goal across recomputes instead of picking a fresh one every time, which used to make the path visibly twitch.

---

## Running it on the robot

`robot_deploy/` is the RB3 Gen 2 deployment, split from the core pipeline:

- **`robot_deploy/pico/`** — MicroPython firmware for the Pi Pico. Reads quadrature encoders via hardware interrupts, drives the motors, and streams encoder ticks + IMU readings to the RB3 over USB serial as newline-delimited JSON.
- **`robot_deploy/rb3/`** — runs on the RB3 itself. `hardware.py` is the serial client; `main.py` wires up an `EKF2D` (wheel + gyro dead-reckoning, with an optional vision correction step), a pure-pursuit controller that follows the live navmesh path, and a viser control panel with manual drive, individual motor testing, and live diagnostics (EKF pose, raw gyro rate, stationary yaw-drift tracking to expose gyro bias).

The EKF predicts from gyro yaw rate when available (falls back to the noisy wheel-speed differential otherwise) and can take a measurement update from VO whenever vision gets a confident frame alignment — so the map can be driven by wheel+gyro odometry alone, by vision alone, or by both with vision correcting drift. That's toggleable live from the control panel.

## Testing without a robot

`digital_twin.py` is a simulated version of the whole thing — a virtual robot in a configurable room (a few preset layouts: office, warehouse, maze), with synthetic sensors and a raycaster standing in for the depth camera, running the same navmesh/exploration code the real robot does. Useful for testing navmesh and exploration logic changes without needing physical hardware or dragging a robot around a room.

---

## Project structure

```
main.py                 Entry point — CLI args, model loading, starts the viewer
digital_twin.py          Simulated robot + room for testing without hardware
tequila/
  config.py              All tunable constants
  depth.py                Depth inference, flying-pixel removal, fisheye undistortion
  pointcloud.py            Voxel downsampling, SOR, optional segmentation helpers
  odometry.py               SIFT+PnP visual odometry, ICP fallback
  navmesh.py                 GPP floor detection, node grid, obstacle filtering, A*
  threads.py                  CaptureThread, InferenceThread, NavmeshThread + queues
  viewer.py                    Viser scene updates and the main viewer loop
  tsdf.py                       Optional TSDF volumetric fusion (needs Open3D)
  hardware.py                   TCP hardware bridge — unused, predates robot_deploy/
robot_deploy/
  pico/                   MicroPython firmware — encoders, motors, IMU streaming
  rb3/                    Runs on the RB3 — EKF, hardware bridge, control panel
tools/
  camera_calibration.py  Fisheye chessboard calibration
  capture_calibration.py Grab calibration images from a live camera
depth_anything_v2/      Vendored relative-depth model code — currently unused
```

---

## Quick start

```bash
pip install -r requirements.txt
pip install viser

# Webcam (default camera index 0)
python main.py

# Video file
python main.py --source path/to/video.mp4

# Slower CPU machine — lower resolution is roughly 4x faster
python main.py --width 640
```

Open **http://localhost:8080**. Controls: `left-drag` orbits, `right-drag` pans, `scroll` zooms, `Ctrl-C` quits.

The depth model downloads automatically on first run (~100 MB for the default Small model).

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index (int) or path to a video file |
| `--width` | `1280` | Inference image width in pixels. Use `640` on CPU |
| `--interval` | `3.0` | Seconds between webcam captures |
| `--frame-skip` | `30` | Process every Nth frame (video file mode) |
| `--nav-interval` | `1.0` | Seconds between navmesh recomputes |
| `--up-axis` | `y` | Which axis points up: `x`, `y`, `z`, or `auto` |
| `--max-tilt` | `30.0` | Max floor tilt in degrees |
| `--obs-max-height` | `0.80` | Max obstacle height above floor (metres) |
| `--no-accum` | off | Single-frame mode — no map accumulation |
| `--map-depth` | `3.0` | Max depth of accumulated map points (metres) |
| `--port` | `8080` | Viser web viewer port |
| `--no-splats` | off | Use a raw point cloud instead of Gaussian splats |
| `--splat-radius` | `0.012` | Gaussian splat radius in metres |

---

## Viewer legend

| Colour | Layer | Meaning |
|--------|-------|---------|
| Coloured points | `/scene/map` | Accumulated world map |
| Orange | `/nav/obstacles` | Detected obstacles |
| Red | `/nav/blocked` | Navmesh nodes blocked by obstacles |
| Yellow | `/nav/free` | Free (passable) navmesh nodes |
| Blue lines | `/nav/edges` | Passable edges between free nodes |
| Teal | `/nav/path` | A\* path to the farthest reachable node |
| Green | `/nav/trajectory` | Robot trajectory since startup |

Yellow/red/blue debug layers are hidden by default (`NAV_PATH_ONLY = True` in config) since they clutter the view once a map has any size to it — obstacles and the planned path stay visible either way.

---

## Depth model

Default is **Depth Anything V2 Metric Indoor Small** (~100 MB, downloaded from Hugging Face on first run). Swap it in `tequila/config.py`:

```python
# Faster, smaller (default)
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# More accurate, slower (~1.3 GB)
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# Outdoor / mixed environments
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
```

| Model | AbsRel ↓ | δ₁ ↑ | RMSE ↓ | Params |
|-------|----------|------|--------|--------|
| Small | 0.073 | 96.1% | 0.261 m | 24.8 M |
| Large | 0.056 | 98.4% | 0.206 m | 335 M |

For navigation, Small is good enough — the accuracy gap rarely matters for spotting chairs and walls at the distances a robot actually cares about.

---

## Performance tips

| Hardware | Recommended settings |
|----------|---------------------|
| GPU (any CUDA) | Defaults are fine |
| Mid-range CPU | `--width 640 --frame-skip 60` |
| Low-spec CPU / RB3 | `--width 640 --nav-interval 2.5` |

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|-------------|-----|
| Map fans out into ghost copies of the same wall | Yaw drift accumulating between frames | Known issue, see [Current problems](#current-problems). Reset Map and try driving more slowly/steadily |
| Very slow | No GPU, or the Large model | `--width 640`, confirm `DEPTH_MODEL_ID` is the Small variant |
| Floor detected in mid-air | Drift rotated the cloud so GPP grabbed the wrong surface | Should self-correct — GPP validates against camera height and retries with RANSAC. If it doesn't, the drift is probably too severe already |
| False obstacles on open floor | Depth noise close to the floor plane | Raise `OBS_HEIGHT_MIN` in `config.py` |
| Rays/spikes trailing behind objects | Flying pixels not fully masked | Lower `EDGE_THRESHOLD` in `config.py` |
| `path=0 nodes` in the console | No free nodes connected to the camera's position | Camera is probably boxed in by obstacles — check `OBS_CLEARANCE_R` |
| SIFT+PnP keeps failing | Low-texture environment (plain walls, dim room) | Expected — ICP fallback should catch it. If both fail, the frame gets skipped |
| EKF pose drifts while the robot sits still | Gyro bias | Check the "yaw drift (still)" readout in the control panel; this is the main open problem on the hardware side |
