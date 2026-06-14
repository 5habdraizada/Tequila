"""
tsdf.py — Volumetric TSDF fusion (Open3D), an alternative to point accumulation.

Point accumulation dumps every frame's points into a growing cloud, so noise and
drift pile up forever and can never be corrected.  A Truncated Signed Distance
Function volume instead integrates each frame's depth into a voxel grid where
overlapping observations are *averaged* — noise cancels instead of accumulating,
and a clean surface can be extracted.

Open3D is imported lazily; if it is unavailable the caller falls back to point
accumulation (see NavmeshThread).

Convention bridge
-----------------
The pipeline works in nav coords (Y up, Z toward viewer) with T_cum mapping the
nav camera frame → world.  Open3D's RGBD back-projection uses the standard camera
frame (Y down, Z into scene), so we bridge with F = diag(1, -1, -1):

    standard-cam → world          = T_cum @ Fh          (Fh = diag(1,-1,-1,1))
    extrinsic (world → std-cam)   = Fh @ inv(T_cum)

Points extracted from the volume are therefore already in nav world coords and
drop straight into the existing display / navmesh path.
"""

import numpy as np

try:
    import open3d as o3d
    _OPEN3D = True
except Exception:          # noqa: BLE001 — missing wheel or native-load failure
    _OPEN3D = False

# nav-camera ↔ standard-camera basis change (its own inverse)
_FH = np.diag([1.0, -1.0, -1.0, 1.0])


def available() -> bool:
    """True if Open3D is importable and TSDF fusion can be used."""
    return _OPEN3D


class TSDFFusion:
    """Thin wrapper around Open3D's ScalableTSDFVolume in nav coordinates."""

    def __init__(self, voxel_m: float, trunc_m: float, depth_trunc_m: float):
        if not _OPEN3D:
            raise RuntimeError("open3d is not installed")
        self._voxel       = float(voxel_m)
        self._trunc       = float(trunc_m)
        self.depth_trunc  = float(depth_trunc_m)
        self._make()

    def _make(self) -> None:
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length = self._voxel,
            sdf_trunc    = self._trunc,
            color_type   = o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def reset(self) -> None:
        """Discard all fused geometry and start a fresh volume."""
        self._make()

    def integrate(self, depth_m: np.ndarray, img_bgr: np.ndarray,
                  focal: float, cx: float, cy: float,
                  T_cum: np.ndarray) -> None:
        """Fuse one RGBD frame given its nav camera→world pose T_cum."""
        h, w = depth_m.shape
        rgb   = np.ascontiguousarray(img_bgr[:, :, ::-1])           # BGR → RGB
        depth = np.ascontiguousarray(depth_m.astype(np.float32))    # metres

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale          = 1.0,                # depth already in metres
            depth_trunc          = self.depth_trunc,
            convert_rgb_to_intensity = False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(w), int(h), float(focal), float(focal), float(cx), float(cy))
        extrinsic = _FH @ np.linalg.inv(T_cum)         # world → standard-cam

        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def extract(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract the fused surface as (points, colors) in nav world coords."""
        pcd  = self.volume.extract_point_cloud()
        pts  = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float32)
        return pts, cols
