"""TEQUILA entry point. Run `python main.py --help` for CLI options, or edit
tequila/config.py to change defaults permanently."""

import argparse

import torch

import tequila.config as cfg
from tequila.depth  import load_model
from tequila.viewer import run_viewer
from tequila.threads import stop_event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TEQUILA — live 3-D mapping + navigation mesh for indoor robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source",         default="0",
                        help="webcam index (int) or path to a video file")
    parser.add_argument("--width",          type=int,   default=cfg.INFER_WIDTH,
                        help="inference image width in pixels (default: %(default)s)")
    parser.add_argument("--interval",       type=float, default=cfg.CAPTURE_INTERVAL_S,
                        help="seconds between webcam captures (default: %(default)s)")
    parser.add_argument("--frame-skip",     type=int,   default=cfg.FRAME_SKIP,
                        help="process every Nth video frame (default: %(default)s)")
    parser.add_argument("--nav-interval",   type=float, default=cfg.NAV_INTERVAL_S,
                        help="seconds between navmesh recomputes (default: %(default)s)")
    parser.add_argument("--up-axis",        default=cfg.UP_AXIS,
                        choices=["x", "y", "z", "auto"],
                        help="which axis points up (default: %(default)s)")
    parser.add_argument("--max-tilt",       type=float, default=cfg.MAX_FLOOR_TILT_DEG,
                        help="max floor tilt in degrees (default: %(default)s)")
    parser.add_argument("--obs-max-height", type=float, default=cfg.OBS_HEIGHT_MAX,
                        help="max obstacle height above floor in metres (default: %(default)s)")
    parser.add_argument("--no-accum",       action="store_true",
                        help="disable map accumulation — show only the current frame "
                             "(useful for product/turntable video)")
    parser.add_argument("--map-depth",      type=float, default=cfg.MAP_MAX_DEPTH_M,
                        help="max depth of accumulated map points in metres (default: %(default)s)")
    parser.add_argument("--port",           type=int,   default=cfg.PORT,
                        help="viser web-viewer port (default: %(default)s)")
    parser.add_argument("--no-splats",      action="store_true",
                        help="use raw point cloud instead of Gaussian splats")
    parser.add_argument("--splat-radius",   type=float, default=cfg.SPLAT_RADIUS,
                        help="Gaussian splat radius in metres (default: %(default)s)")
    args = parser.parse_args()

    # All modules import tequila.config, so mutating it here propagates everywhere.
    cfg.INFER_WIDTH        = args.width
    cfg.NAV_INTERVAL_S     = args.nav_interval
    cfg.MAX_FLOOR_TILT_DEG = args.max_tilt
    cfg.OBS_HEIGHT_MAX     = args.obs_max_height
    cfg.ACCUM_ENABLED      = not args.no_accum
    cfg.MAP_MAX_DEPTH_M    = args.map_depth
    cfg.USE_SPLATS         = not args.no_splats
    cfg.SPLAT_RADIUS       = args.splat_radius

    if not cfg.ACCUM_ENABLED:
        print("Accumulation disabled — single-frame display mode.")

    up_idx = {"x": 0, "y": 1, "z": 2}.get(args.up_axis)   # None if "auto"
    if up_idx is None:
        up_idx = 1
        print("Up-axis: auto → defaulting to Y (index 1). "
              "Change with --up-axis if the floor is not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("TIP: pass --width 640 for faster CPU inference.\n")

    model = load_model(device)

    try:   # run_viewer blocks until Ctrl-C
        run_viewer(
            model      = model,
            device     = device,
            source     = args.source,
            interval   = args.interval,
            frame_skip = args.frame_skip,
            up_idx     = up_idx,
            port       = args.port,
        )
    except KeyboardInterrupt:
        stop_event.set()
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
