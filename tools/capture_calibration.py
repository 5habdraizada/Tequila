#!/usr/bin/env python3
"""Capture calibration images from a live camera feed.

Controls (while the window is focused):
    SPACE or 's'  -> save the current frame
    'q' or ESC    -> quit

Saved frames go into ./calibration_images as calib_000.png, calib_001.png, ...
"""

import argparse
import os
import sys
import cv2


def open_capture(source):
    """Open a camera index (via the Qualcomm GStreamer pipeline) or a video file."""
    try:
        cam_idx = int(source)
        gst_pipeline = (
            f"qtiqmmfsrc name=camsrc camera={cam_idx} ! "
            "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
            "videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    except ValueError:
        # Not an int -> treat as a file path
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/source: {source}")
    return cap


def next_index(folder, prefix="calib_", ext=".png"):
    """Resume numbering after existing images so nothing gets overwritten."""
    nums = []
    for f in os.listdir(folder):
        if f.startswith(prefix) and f.endswith(ext):
            stem = f[len(prefix):-len(ext)]
            if stem.isdigit():
                nums.append(int(stem))
    return max(nums) + 1 if nums else 0


def main():
    parser = argparse.ArgumentParser(description="Capture camera calibration images.")
    parser.add_argument("source", nargs="?", default="0",
                        help="Camera index (default 0) or a video file path.")
    parser.add_argument("-o", "--out", default="calibration_images",
                        help="Output folder (default: calibration_images).")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    count = next_index(args.out)

    cap = open_capture(args.source)
    win = "Calibration capture  [SPACE/s = save, q/ESC = quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"Saving to '{args.out}/'. SPACE/'s' to save, 'q'/ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame; exiting.", file=sys.stderr)
                break

            # Draw the counter on a copy so the saved image stays clean.
            overlay = frame.copy()
            cv2.putText(overlay, f"saved: {count}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):          # q or ESC
                break
            if key in (ord(' '), ord('s')):    # SPACE or s
                path = os.path.join(args.out, f"calib_{count:03d}.png")
                cv2.imwrite(path, frame)       # save the clean frame, not the overlay
                print(f"saved {path}")
                count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()