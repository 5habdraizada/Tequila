#!/usr/bin/env python3
"""
Quick test to verify the RB3 camera opens and streams frames.
Run this before main.py.

Usage:
    python3 test_camera.py
    python3 test_camera.py --index 1
"""
import argparse, sys, time
import cv2

p = argparse.ArgumentParser()
p.add_argument("--index", type=int, default=0)
p.add_argument("--frames", type=int, default=10)
args = p.parse_args()

print(f"Opening /dev/video{args.index} …")
cap = cv2.VideoCapture(args.index)

if not cap.isOpened():
    print(f"ERROR: Could not open camera {args.index}")
    print("Try:  v4l2-ctl --list-devices")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened: {w}x{h}")

for i in range(args.frames):
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        print(f"Frame {i}: READ FAILED")
        continue
    ms = (time.time() - t0) * 1000
    print(f"Frame {i+1}/{args.frames}: {frame.shape}  {ms:.1f} ms")

cap.release()
print("Camera OK")
