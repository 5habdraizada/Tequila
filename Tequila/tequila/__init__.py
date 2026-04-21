"""
TEQUILA — Live Video + Navigation Mesh
=======================================

A real-time 3D mapping and navigation system for indoor robots.
Combines monocular metric depth estimation with frontier-exploration
navmesh generation, running across four independent threads.

Package layout
--------------
  config.py    — all tunable constants (edit here first)
  depth.py     — depth model inference, flying-pixel removal, scale anchor
  pointcloud.py— point-cloud utilities: voxel downsample, SOR, segmentation
  odometry.py  — frame-to-frame alignment: ORB+PnP (primary), ICP (fallback)
  navmesh.py   — floor detection (GPP/RANSAC), node grid, A* path planning
  threads.py   — CaptureThread, InferenceThread, NavmeshThread + shared queues
  viewer.py    — viser scene updates and the main viewer loop

Entry point
-----------
  main.py  (project root) — CLI argument parsing and model loading
"""

__version__ = "1.0.0"
