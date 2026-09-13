#!/usr/bin/env python3
"""
tools/depth_probe.py — measure the metric scale of the depth pipeline.

Runs the SAME path the robot runs (undistortion, INFER_WIDTH, model, focal),
so what it reports is what the map is built from.

Test A — depth scale
    Park the robot facing a flat wall, tape-measure lens to wall, then:
        python3 tools/depth_probe.py --true-dist 1.50
    Prints the DEPTH_SCALE to put in robot_deploy/rb3/config.py.

Test B — focal / lateral scale
    Put an object of known width at a known distance, read its edge pixel
    columns off the saved PNG (it has a coordinate grid), then:
        python3 tools/depth_probe.py --true-dist 1.50 --width 0.60 --edges 210 430
    Depth right but width wrong means the focal is wrong, which points at the
    fisheye calibration rather than the model.

Viewing, on a headless robot:
        python3 tools/depth_probe.py --true-dist 1.50 --serve 8081
    then open http://<rb3-ip>:8081 from your laptop.

Offline, on a saved image:
        python3 tools/depth_probe.py --source shot.png --true-dist 1.50
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tequila.config as cfg

GST_PIPELINE = (
    "qtiqmmfsrc name=camsrc camera=0 ! "
    "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
    "videoconvert ! "
    "appsink max-buffers=1 drop=true sync=false"
)


def apply_rb3_overrides() -> bool:
    """Copy the RB3's config over tequila.config, for names tequila.config has.

    rb3/config.py says its values are "merged on top of tequila/config.py at
    startup", but main() actually merges them by hand, one assignment per line.
    Probing with the base config would undistort to a different FOV and infer
    at a different width than the robot does, so the numbers would not be the
    robot's numbers. Restricting to names that already exist in tequila.config
    keeps RB3-only constants (wheel geometry, pins) out of it.
    """
    rb3_dir = os.path.join(_ROOT, "robot_deploy", "rb3")
    if not os.path.isdir(rb3_dir):
        return False
    sys.path.insert(0, rb3_dir)
    try:
        import config as rb3_cfg   # noqa: PLC0415
    except ImportError:
        return False
    for name in dir(rb3_cfg):
        if name.isupper() and hasattr(cfg, name):
            setattr(cfg, name, getattr(rb3_cfg, name))
    return True


def grab(source: str, n_frames: int, warmup: int):
    """Return one BGR frame from the camera (or a file/still image)."""
    if source != "camera":
        img = cv2.imread(source)
        if img is not None:
            return img
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"No frame from: {source}")
        return frame

    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open the camera via GStreamer.\n"
            "Nothing else may hold it — stop main.py first.\n"
            "To probe a saved image instead: --source shot.png")
    try:
        # Early frames are dark until exposure settles; fusing those is exactly
        # what MIN_FRAME_BRIGHTNESS guards against in the live pipeline.
        for _ in range(max(warmup, 1)):
            cap.read()
            time.sleep(0.05)
        frames = []
        for _ in range(max(n_frames, 1)):
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.05)
        if not frames:
            raise RuntimeError("Camera opened but returned no frames")
        # Average to damp sensor noise — the robot is stationary for this test.
        return np.mean(frames, axis=0).astype(np.uint8)
    finally:
        cap.release()


def patch_median(depth: np.ndarray, x: int, y: int, half: int) -> tuple[float, int]:
    """Median depth over a patch, ignoring the zeros left by edge masking."""
    h, w = depth.shape
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0.0]
    if valid.size == 0:
        return float("nan"), 0
    return float(np.median(valid)), int(valid.size)


def annotate(img: np.ndarray, step: int = 80) -> np.ndarray:
    """Overlay a pixel-coordinate grid so edge columns can be read off by eye."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (60, 60, 60), 1)
        cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (60, 60, 60), 1)
        cv2.putText(out, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 255, 255), 1)
    cx, cy = w // 2, h // 2
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    return out


def colourise(depth: np.ndarray) -> np.ndarray:
    """Depth to a viewable image. Zeros (edge-masked) render black, not near."""
    valid = depth > 0.0
    out = np.zeros((*depth.shape, 3), np.uint8)
    if not valid.any():
        return out
    lo, hi = float(depth[valid].min()), float(depth[valid].max())
    norm = np.zeros_like(depth)
    if hi > lo:
        norm[valid] = (depth[valid] - lo) / (hi - lo)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid] = cm[valid]
    cv2.putText(out, f"near {lo:.2f}m", (8, out.shape[0] - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(out, f"far  {hi:.2f}m", (8, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


def serve(directory: str, port: int) -> None:
    import functools
    import http.server
    import socketserver
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"\n[serve] http://<rb3-ip>:{port}   (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] stopped")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the depth pipeline's metric scale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--source", default="camera",
                    help="'camera' (default), or a path to an image/video")
    ap.add_argument("--true-dist", type=float, default=None, metavar="M",
                    help="Test A: tape-measured lens-to-wall distance, metres")
    ap.add_argument("--width", type=float, default=None, metavar="M",
                    help="Test B: true width of the reference object, metres")
    ap.add_argument("--edges", type=int, nargs=2, default=None,
                    metavar=("X1", "X2"),
                    help="Test B: pixel columns of the object's left/right edges")
    ap.add_argument("--at", type=int, nargs=2, default=None, metavar=("X", "Y"),
                    help="sample at this pixel instead of the image centre")
    ap.add_argument("--patch", type=int, default=21,
                    help="sampling patch size in pixels (default 21)")
    ap.add_argument("--frames", type=int, default=5,
                    help="camera frames to average (default 5)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="frames to discard while exposure settles (default 10)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "tools", "probe_out"),
                    help="directory for the PNGs and report")
    ap.add_argument("--serve", type=int, default=None, metavar="PORT",
                    help="serve --out over HTTP after probing")
    ap.add_argument("--no-rb3", action="store_true",
                    help="skip the RB3 config overrides (use tequila/config.py)")
    args = ap.parse_args()

    if not args.no_rb3:
        print(f"[cfg] RB3 overrides: {'applied' if apply_rb3_overrides() else 'NOT FOUND'}")

    from tequila.depth import frame_to_nav_pts, load_model, run_inference

    print(f"[cfg] model={cfg.DEPTH_MODEL_ID}")
    print(f"[cfg] INFER_WIDTH={cfg.INFER_WIDTH}  FISHEYE={cfg.FISHEYE}  "
          f"UNDISTORT_FOV_DEG={getattr(cfg, 'UNDISTORT_FOV_DEG', 'n/a')}")
    print(f"[cfg] DEPTH_SCALE={cfg.DEPTH_SCALE}  MAX_DEPTH_M={cfg.MAX_DEPTH_M}")

    calib = getattr(cfg, "FISHEYE_CALIB_NPZ", "") or ""
    if cfg.FISHEYE and calib:
        path = calib if os.path.isabs(calib) else os.path.join(_ROOT, calib)
        if os.path.exists(path):
            size = os.path.getsize(path)
            warn = "  <-- suspiciously small, may be an unfetched LFS pointer" if size < 1000 else ""
            print(f"[cfg] calibration {calib}: {size} bytes{warn}")
        else:
            print(f"[cfg] calibration {calib}: MISSING — undistortion falls back "
                  f"to an equidistant model, so `focal` may not match the lens")

    print(f"\n[grab] source={args.source}")
    frame = grab(args.source, args.frames, args.warmup)

    print("[model] loading...")
    model = load_model("cpu")
    t0 = time.time()
    img, depth_m, focal, cx, cy = run_inference(frame, model)
    dt = time.time() - t0
    h, w = depth_m.shape
    print(f"[infer] {w}x{h} in {dt:.2f}s   focal={focal:.2f}px  cx={cx:.1f} cy={cy:.1f}")

    hfov = 2.0 * np.degrees(np.arctan((w / 2.0) / focal))
    print(f"[infer] implied horizontal FOV of the back-projection: {hfov:.1f} deg")

    sx, sy = (args.at if args.at else (int(cx), int(cy)))
    half = max(args.patch // 2, 1)
    d_centre, n_valid = patch_median(depth_m, sx, sy, half)
    coverage = float((depth_m > 0).mean())

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say()
    say("=" * 62)
    say(f"sample at ({sx},{sy}), {args.patch}x{args.patch} patch")
    say(f"  median depth      : {d_centre:.3f} m   ({n_valid} valid px in patch)")
    say(f"  valid depth pixels: {coverage*100:.1f}% of frame "
        f"(rest zeroed by edge masking)")

    # A few more samples so a tilted or non-flat target is obvious.
    say()
    say("  depth across the frame (same patch size):")
    for label, px, py in [("left  ", int(w*0.25), int(cy)),
                          ("centre", int(cx),     int(cy)),
                          ("right ", int(w*0.75), int(cy)),
                          ("upper ", int(cx),     int(h*0.25)),
                          ("lower ", int(cx),     int(h*0.75))]:
        d, n = patch_median(depth_m, px, py, half)
        say(f"    {label} ({px:4d},{py:4d}): {d:6.3f} m  ({n} px)")

    rc = 0
    scale_k = None      # Test A's correction, needed to decouple Test B
    if args.true_dist is not None:
        say()
        say("-- Test A: depth scale " + "-" * 39)
        if not np.isfinite(d_centre) or d_centre <= 0:
            say("  no valid depth at the sample point — re-aim, or raise --patch")
            rc = 1
        else:
            k = scale_k = args.true_dist / d_centre
            say(f"  measured (tape)   : {args.true_dist:.3f} m")
            say(f"  reported (model)  : {d_centre:.3f} m")
            say(f"  ratio             : {k:.3f}x")
            say()
            if abs(k - 1.0) < 0.03:
                say("  depth is metric to within 3% - nothing to correct here.")
                say("  If the cloud still looks wrong, run Test B: the error is")
                say("  then lateral (focal/calibration), not in the depth itself.")
            else:
                say(f"  reported depth is {'TOO LARGE' if k < 1 else 'TOO SMALL'} "
                    f"by {(1/k if k < 1 else k):.2f}x")
                say("  put this in robot_deploy/rb3/config.py:")
                say(f"      DEPTH_SCALE = {cfg.DEPTH_SCALE * k:.4f}")
                if abs(cfg.DEPTH_SCALE - 1.0) > 1e-9:
                    say(f"  (current DEPTH_SCALE={cfg.DEPTH_SCALE} is already applied "
                        f"above, so this folds in on top of it)")
                say("  then re-run this to confirm it lands at 1.00x.")

    if args.width is not None and args.edges is not None:
        say()
        say("-- Test B: focal / lateral scale " + "-" * 28)
        x1, x2 = sorted(args.edges)
        d1, _ = patch_median(depth_m, x1, sy, half)
        d2, _ = patch_median(depth_m, x2, sy, half)
        d_obj = np.nanmean([d1, d2])
        if not np.isfinite(d_obj) or d_obj <= 0:
            say("  no valid depth at the edge columns — re-aim, or raise --patch")
            rc = 1
        else:
            say(f"  edge columns      : x={x1} .. x={x2}  ({x2-x1} px)")
            say(f"  depth at edges    : {d1:.3f} / {d2:.3f} m  (mean {d_obj:.3f})")

            # width = (dx * depth) / focal, so a depth error inflates the width
            # just as a focal error does — measure both at once and Test B
            # simply re-reports Test A. Correcting the depth first isolates the
            # focal, which is the only thing Test B can actually speak to.
            if scale_k is None:
                say()
                say("  NO --true-dist given, so the depth here is uncorrected and")
                say("  this ratio cannot tell a focal error from a depth error.")
                say("  Re-run with --true-dist to separate them.")
                d_used, tag = d_obj, "uncorrected"
            else:
                d_used, tag = d_obj * scale_k, f"corrected x{scale_k:.3f}"
                say(f"  depth used        : {d_used:.3f} m  ({tag}, from Test A)")

            measured_w = (x2 - x1) * d_used / focal
            ratio = measured_w / args.width
            say(f"  width from cloud  : {measured_w:.3f} m")
            say(f"  true width        : {args.width:.3f} m")
            say(f"  ratio             : {ratio:.3f}x")
            say()
            if abs(ratio - 1.0) < 0.05:
                say("  focal is right. Whatever Test A found is the whole story:")
                say("  the geometry is sound and only the metric scale was off.")
            elif scale_k is None:
                say(f"  off by {ratio:.2f}x, but see the caveat above.")
            else:
                say(f"  focal is off by {ratio:.2f}x even after correcting the depth,")
                say(f"  so this is lateral: implied true focal {focal*ratio:.1f}px "
                    f"vs {focal:.1f}px in use, i.e. a real FOV of "
                    f"{2*np.degrees(np.arctan((w/2)/(focal*ratio))):.1f} deg, not "
                    f"{getattr(cfg,'UNDISTORT_FOV_DEG','?')} deg.")
                say("  That points at the fisheye calibration, not the model.")

    # Cloud extent — the thing that looked too big in viser.
    nav_pts, map_pts, _ = frame_to_nav_pts(img, depth_m, focal)
    if len(map_pts):
        lo, hi = map_pts.min(axis=0), map_pts.max(axis=0)
        say()
        say(f"cloud from this frame: {len(map_pts)} pts, extent "
            f"X {lo[0]:+.2f}..{hi[0]:+.2f}  "
            f"Y {lo[1]:+.2f}..{hi[1]:+.2f}  "
            f"Z {lo[2]:+.2f}..{hi[2]:+.2f} m")
    say("=" * 62)

    os.makedirs(args.out, exist_ok=True)
    rgb_path   = os.path.join(args.out, "probe_rgb.png")
    depth_path = os.path.join(args.out, "probe_depth.png")
    rep_path   = os.path.join(args.out, "probe_report.txt")
    cv2.imwrite(rgb_path, annotate(img))
    cv2.imwrite(depth_path, annotate(colourise(depth_m)))
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {rgb_path}")
    print(f"wrote {depth_path}")
    print(f"wrote {rep_path}")
    print("\nThe PNGs carry a pixel grid - read Test B's edge columns straight off them.")

    if args.serve is not None:
        serve(args.out, args.serve)
    return rc


if __name__ == "__main__":
    sys.exit(main())
