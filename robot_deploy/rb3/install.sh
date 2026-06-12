#!/usr/bin/env bash
# rb3/install.sh — Run once on the RB3 Gen 2 to set up the TEQUILA pipeline.
set -e

echo "=== RB3 TEQUILA install ==="

# System packages
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-dev \
    libopencv-dev python3-opencv \
    v4l-utils \
    git wget curl

# Python deps
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Verify camera
echo ""
echo "=== Detected cameras ==="
v4l2-ctl --list-devices 2>/dev/null || echo "(v4l2-ctl not found)"
ls /dev/video* 2>/dev/null || echo "No /dev/video* found yet"

echo ""
echo "=== Done. ==="
echo "1. Edit rb3/config.py and set PI_IP to your Pi's IP"
echo "2. Start Pi bridge first:  python3 pi/bridge.py"
echo "3. Then on RB3:            python3 rb3/main.py"
