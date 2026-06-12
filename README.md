# TEQUILA — Live 3D Mapping + Navigation Mesh

Real-time indoor mapping and navigation system for robots.  
Combines monocular metric depth estimation with frontier-exploration navmesh generation, running live from a webcam or video file.

---

## What it does

TEQUILA takes a live camera feed and does three things simultaneously:

1. **Builds a 3D map** — every frame is turned into a coloured point cloud and fused into a growing world-space map
2. **Generates a navigation mesh** — detects the floor, places a grid of waypoints, and removes any waypoints blocked by obstacles
3. **Plans a path** — runs A\* from the robot's current position to the farthest reachable waypoint, implementing basic frontier exploration

Everything streams live to a browser-based 3D viewer with no extra software needed on the viewing device.

```
Camera / Video
      │
      ▼
 CaptureThread ──► InferenceThread ──► NavmeshThread
                  (depth model)        (floor detection,
                  (ORB+PnP VO)          obstacle filter,
                  (back-projection)     A* pathfinding)
                        │                    │
                        ▼                    ▼
                   Coloured map         Navmesh overlay
                        └──────────────────┘
                                 │
                            Viser viewer
                         (http://localhost:8080)
```

---

## Features

- **Metric depth** — Depth Anything V2 (Small indoor model by default) outputs real metre values with no per-frame scale drift
- **ORB + PnP visual odometry** — aligns consecutive frames using feature matching; falls back to ICP when texture is low
- **Accumulated world map** — every frame is transformed into a shared world coordinate system and merged into one growing coloured point cloud
- **GPP floor detection** — Ground Principal Plane algorithm with RANSAC fallback; validates the detected floor is always below the camera
- **Navigation mesh** — uniform grid of waypoints on the floor, obstacle clearance filtering, line-of-sight edge checking
- **A\* path planning** — always paths to the farthest reachable free node (frontier exploration strategy)
- **Robot trajectory trail** — green line showing every camera position since startup
- **Flying-pixel removal** — depth discontinuity masking to eliminate rays and spikes behind objects
- **Live browser viewer** — powered by [viser](https://viser.studio)

---

## Requirements

```
Python 3.10+
CUDA GPU recommended (runs on CPU but is slow — see Performance tips)
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install viser
```

`requirements.txt` covers: `torch`, `torchvision`, `opencv-python`, `transformers`, `Pillow`, `matplotlib`, `huggingface_hub`

---

## Quick start

```bash
# Webcam (default camera index 0)
python main.py

# Video file
python main.py --source path/to/video.mp4

# Slower CPU laptop — lower resolution is ~4x faster
python main.py --width 640

# Second camera
python main.py --source 1
```

Open **http://localhost:8080** in your browser to see the live 3D view.

Controls: `left-drag` = orbit · `right-drag` = pan · `scroll` = zoom · `Ctrl-C` = quit

The depth model downloads automatically on first run (~100 MB for the default Small model).

---

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index (int) or path to a video file |
| `--width` | `1280` | Inference image width in pixels. Use `640` on CPU |
| `--interval` | `3.0` | Seconds between webcam captures |
| `--frame-skip` | `30` | Process every Nth frame (video file mode) |
| `--nav-interval` | `6.0` | Seconds between navmesh recomputes |
| `--up-axis` | `y` | Which axis points up: `x`, `y`, `z`, or `auto` |
| `--max-tilt` | `30.0` | Max floor tilt in degrees |
| `--obs-max-height` | `0.80` | Max obstacle height above floor (metres) |
| `--cam-height` | `0` | Camera height above floor in metres. `0` = auto-detect |
| `--no-accum` | off | Single-frame mode — no map accumulation |
| `--map-depth` | `4.0` | Max depth of accumulated map points (metres) |
| `--port` | `8080` | Viser web viewer port |

---

## Viewer legend

| Colour | Layer | Meaning |
|--------|-------|---------|
| Coloured points | `/scene/map` | Accumulated world map |
| Grey points | `/nav/accum_map` | Full geometry the navmesh was built from |
| Orange | `/nav/obstacles` | Detected obstacles |
| Red | `/nav/blocked` | Navmesh nodes blocked by obstacles |
| Yellow | `/nav/free` | Free (passable) navmesh nodes |
| Blue lines | `/nav/edges` | Passable edges between free nodes |
| Teal line | `/nav/path` | A\* path to farthest reachable node |
| Green line | `/nav/trajectory` | Full robot trajectory since startup |

---

## Project structure

```
main.py                 Entry point — CLI argument parsing and model loading
tequila/
  config.py             All tunable constants (edit here to change defaults permanently)
  depth.py              Depth model inference, flying-pixel removal, scale anchoring,
                        back-projection to nav points and coloured map points
  pointcloud.py         Voxel downsampling, Statistical Outlier Removal (SOR),
                        and optional segmentation / wall-removal helpers
  odometry.py           ORB+PnP visual odometry (primary alignment),
                        ICP point-to-point (fallback alignment)
  navmesh.py            GPP floor detection, RANSAC fallback, node grid generation,
                        obstacle denoising, edge building, A* path planner
  threads.py            CaptureThread, InferenceThread, NavmeshThread,
                        and the shared inter-thread queues
  viewer.py             Viser scene update helpers and the main viewer loop
depth_anything_v2/      Local DepthAnythingV2 model code (relative-depth fallback only)
checkpoints/            Local model weights (relative-depth mode only)
```

---

## How it works — in depth

### Thread architecture

TEQUILA runs four concurrent threads so that capture, inference, navmesh computation, and display never block each other. All communication between threads happens through single-slot queues (maxsize=1), which means each consumer always sees the latest data and stale items are automatically dropped.

```
Thread 1  CaptureThread    reads frames from camera/file
Thread 2  InferenceThread  runs depth model + builds point clouds
Thread 3  NavmeshThread    accumulates world map + computes navmesh
Thread 4  Main thread      runs the viser viewer at ~20 Hz
```

---

### Step 1 — Depth inference (`tequila/depth.py`)

Each incoming BGR frame is resized to `--width` pixels and passed through **Depth Anything V2**, a transformer-based monocular depth model. The model outputs a per-pixel depth map in metres.

**Why metric depth matters:**  
Relative-depth models normalise each frame independently (min→0, max→MAX_DEPTH). This means the same physical distance maps to different pixel values across frames. When frames are accumulated into a world map, the scale differences cause the cloud to fan and stretch. The metric model outputs real distances with no per-frame normalisation, so accumulated frames align correctly without any scale correction.

**Flying-pixel removal:**  
At the boundary between a near object and a far background (e.g. a chair leg against a wall), depth models blend intermediate values onto edge pixels. When back-projected, these appear as long rays or spikes extending behind every object. TEQUILA removes them by computing the local depth minimum in an 11×11 pixel neighbourhood using morphological erosion. Any pixel whose depth exceeds that local minimum by more than 20 % is on the far side of a depth discontinuity and is masked out (set to zero).

```
local_min = erode(depth_map, 11×11 kernel)
valid = depth <= local_min × 1.20
```

**Back-projection:**  
Valid depth pixels are lifted to 3D using the standard pinhole model and then flipped to the navigation coordinate convention (Y-up, Z-toward-viewer):

```
x = (pixel_x - cx) × depth / focal
y = -(pixel_y - cy) × depth / focal   ← flip Y
z = -depth                             ← flip Z
```

Two clouds are produced per frame:
- **nav_pts** — coarse (3× voxel grid), position-only, used for floor RANSAC and navmesh
- **map_pts** — fine (1× voxel grid), coloured, used for the display accumulation map

---

### Step 2 — Frame alignment (`tequila/odometry.py`)

To build a consistent world map, each new frame must be registered into the same coordinate system as all previous frames. TEQUILA tries two methods in order:

#### Primary: ORB + PnP Visual Odometry

1. **ORB keypoints** are detected in both the previous and current frame (up to 2000 per frame)
2. Descriptors are matched with a brute-force Hamming matcher. Lowe's ratio test (threshold 0.75) keeps only unambiguous matches — a match is kept only if its distance is less than 75 % of the second-best match distance
3. Each matched previous-frame keypoint is **back-projected to 3D** using the stored depth map (in standard OpenCV camera coords — Y-down, Z-into-scene — so the camera matrix K applies correctly)
4. **solvePnPRansac** fits a camera pose that reprojects those 3D points onto the current frame's 2D positions. This gives `R_std` and `t_std` — the rotation and translation in standard camera coords
5. The result is **converted to nav convention** using the flip matrix `F = diag(1, -1, -1)`:
   ```
   R_nav = F @ R_std @ F
   t_nav = F @ t_std
   ```
6. The pose is **inverted** so it maps current-frame points into previous-frame coords (matching the ICP convention), then composed into the cumulative world transform `T_cum`

A result is accepted only if: ≥12 PnP inliers, translation 0.03–2.0 m, rotation <45°. Frames with translation <0.03 m are treated as duplicate views and skipped for accumulation (the camera hasn't moved enough to add new information).

#### Fallback: ICP (Iterative Closest Point)

If ORB+PnP fails (blank walls, low light, too few texture features), ICP is run on the coarse nav point clouds from the current and previous frames.

Each ICP iteration:
1. For every source point, find its nearest neighbour in the target cloud (KDTree)
2. Keep only pairs within 1.0 m (inliers)
3. Compute the optimal rigid transform via SVD (Umeyama method):
   ```
   H = (source_inliers - src_centroid).T @ (target_inliers - tgt_centroid)
   U, _, Vt = SVD(H)
   R = Vt.T @ U.T
   t = tgt_centroid - R @ src_centroid
   ```
4. Apply the transform and repeat up to 50 iterations or until convergence

ICP is accepted if: fitness (inlier fraction) ≥0.25, translation <2.0 m, rotation <45°. If both VO and ICP fail, the frame is skipped entirely.

---

### Step 3 — Map accumulation (`tequila/threads.py`)

Once a frame is aligned, its points are transformed into world space and merged into the accumulated cloud:

```
world_pts = (R_w @ cam_pts.T).T + t_w
```

where `R_w, t_w` are extracted from `T_cum`.

Two separate clouds are maintained:
- **Coarse nav cloud** — voxel downsampled at 3× voxel size, position only. Used for floor RANSAC and navmesh. Periodically SOR-cleaned to remove outliers
- **Fine coloured map** — voxel downsampled at 1× voxel size with colours. Used for the display. Only points within `--map-depth` metres are added to this cloud — distant points amplify any alignment error into long fan arms

Both clouds are capped at 500,000 points to prevent unbounded memory growth.

---

### Step 4 — Floor detection (`tequila/navmesh.py`)

Every `--nav-interval` seconds the navmesh pipeline runs on the accumulated nav cloud.

#### Ground Principal Plane (GPP)

GPP divides the point cloud into 8 angular sectors around the vertical axis. For each sector it fits a plane via PCA — the eigenvector of the smallest singular value is the plane normal:

```
_, _, Vt = SVD(sector_points - centroid)
normal = Vt[-1]    ← smallest singular value → flattest direction
```

Sectors whose plane is more than 30° from horizontal are discarded. The remaining sector normals are compared pairwise — the largest group of sectors whose normals agree within 15° forms the consensus. The consensus normals are averaged and a final least-squares refinement is run on all inliers.

#### Validation

The detected floor is validated against the camera position: the floor centroid must be at least 5 cm below the camera along the up-axis. If GPP returns a surface above the camera (a table top, ceiling, or wall detected due to VO drift), it is rejected and RANSAC is retried with a hard upper-bound constraint on the plane height.

#### RANSAC fallback

If GPP cannot find enough sector agreement, classic RANSAC randomly samples triplets of points, tests if they form a roughly horizontal plane, and keeps the plane with the most inliers. Candidate planes above the camera are skipped. The winning plane is refined with least-squares.

---

### Step 5 — Navigation mesh (`tequila/navmesh.py`)

Once the floor plane is detected:

**Node grid:**  
Floor points are projected to 2D floor coordinates using two orthonormal in-plane axes. A regular grid at 15 cm spacing is laid across the floor extent. Each grid cell checks whether a floor point exists within 22.5 cm — if yes, the node is placed at that grid position. Critically, nodes are projected **directly onto the mathematical plane** (not snapped to the nearest floor point's 3D position), so all nodes sit at exactly the same height regardless of depth noise or VO drift.

**Obstacle detection:**  
Non-floor points are height-filtered using the actual world Y coordinate (not the tilted plane distance, which would be corrupted by VO rotational drift). Only points between 18 cm and 80 cm above the mean floor height are kept as obstacles. These are voxel-downsampled and SOR-denoised. Any navmesh node within 40 cm of an obstacle is marked as blocked.

**Edge building:**  
Pairs of free nodes within 30 cm (2× node spacing) are candidate edges. Each candidate is sampled at 20 equally-spaced points along the connecting line. If any sample is within 40 cm of an obstacle, the edge is discarded. This ensures the robot never tries to drive through a gap it cannot physically fit through.

**Path planning:**  
A\* shortest-path is run from the free node nearest the camera to the farthest reachable free node. The goal is chosen by sorting all free nodes by distance from the camera (descending) and trying each in order until A\* finds a connected path. This implements a greedy frontier-exploration strategy — always head for the most distant area the map shows as reachable.

---

## Depth model

By default TEQUILA uses the **Depth Anything V2 Metric Indoor Small** model (~100 MB, downloaded automatically on first run from Hugging Face).

To switch models, edit `tequila/config.py`:

```python
# Faster, smaller (default)
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# More accurate, slower (~1.3 GB)
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# Outdoor / mixed environments
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

# Local relative-depth model (no download needed, but map may drift/stretch)
DEPTH_METRIC = False
```

### Small vs Large accuracy (indoor benchmark)

| Model | AbsRel ↓ | δ₁ ↑ | RMSE ↓ | Params |
|-------|----------|------|--------|--------|
| Small | 0.073 | 96.1 % | 0.261 m | 24.8 M |
| Large | 0.056 | 98.4 % | 0.206 m | 335 M |

For navigation the Small model is sufficient — the 2.3 % accuracy difference rarely matters for detecting chairs and walls at typical robot operating distances.

---

## Performance tips

| Hardware | Recommended settings |
|----------|---------------------|
| GPU (any CUDA) | Default settings work well |
| Mid-range CPU | `--width 640 --frame-skip 60` |
| Low-spec CPU | `--width 640 --frame-skip 60 --nav-interval 15` |

- **`--width 640`** — halves each image dimension, ~4× fewer pixels, ~4× faster inference
- **`--frame-skip 60`** — process half as many frames (video file mode)
- **`--nav-interval 15`** — recompute navmesh less often
- **Small model** — already set as default; ~3× faster than Large on CPU

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Very slow | No GPU / large model | `--width 640`, ensure Small model is set |
| Floor detected in mid-air | VO drift rotating the cloud | Handled automatically — RANSAC retries with camera-below constraint |
| False obstacles on open floor | Depth noise within 18 cm of floor | Increase `OBS_HEIGHT_MIN` in `config.py` |
| Ceiling / roof detected as obstacle | VO rotational drift | Fixed — obstacle height uses world Y, not tilted plane distance |
| Rays / spikes behind objects | Flying pixels at depth edges | Decrease `EDGE_THRESHOLD` in `config.py` |
| Map too sparse / shallow | Edge mask too tight or depth cutoff too short | Increase `EDGE_THRESHOLD` and `MAP_MAX_DEPTH_M` |
| Map stretching / fanning | Scale drift (relative-depth mode) | Use metric model (`DEPTH_METRIC = True`) |
| `path=0 nodes` | No free nodes connected to camera | Camera may be surrounded by obstacles; check `OBS_CLEARANCE_R` |
| `MemoryError` in SVD | Cloud too large with `full_matrices=True` | Fixed — all SVD calls use `full_matrices=False` |
| ORB+PnP always failing | Low-texture environment (plain walls) | ICP fallback handles this automatically |
