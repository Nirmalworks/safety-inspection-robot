#!/usr/bin/env bash
# Install Ultralytics (YOLO26 COCO models) for ml/yolo_coco.py on the ROBOT, without touching the
# ROS Python packages: --no-deps into $AME505_DATA/pylib (on the SSD, survives container rebuilds).
# torch, torchvision, OpenCV, NumPy etc. already come with the container; a plain
# `pip install ultralytics` would pull opencv-python / a newer NumPy next to ROS's.
# Laptop: just `pip install ultralytics`.
set -euo pipefail
VER="${1:-8.4.215}"
LIB="${AME505_DATA:-/home/ubuntu/share/tmp/ame505_data}/pylib"
mkdir -p "$LIB"
python3 -m pip install -q --no-deps --upgrade --target "$LIB" "ultralytics==$VER"
for m in torch torchvision cv2 matplotlib scipy psutil yaml requests; do
  python3 -c "import $m" 2>/dev/null || echo "WARNING: python module '$m' missing (pip install --user it)"
done
PYTHONPATH="$LIB" python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__, 'ready in $LIB')"
echo "Run: python3 ml/yolo_coco.py \$AME505_DATA/dataset --limit 10   (weights download once to $AME505_DATA/models/)"
