"""COCO object detection helpers shared by the live node (yolo_live) and the offline tool
(ml/yolo_coco.py). Plain NumPy, no ROS, so the ml/ scripts can import it on a laptop."""
import os
import sys

import numpy as np

DEFAULT_MODEL = 'yolo26n.pt'        # newest Ultralytics COCO model (ultralytics 8.4.x); n = nano

# COCO classes that can plausibly be (part of) a staged hazard or block a walkway / exit.
RELEVANT = {'person', 'bicycle', 'bench', 'backpack', 'umbrella', 'handbag', 'suitcase',
            'sports ball', 'bottle', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'microwave', 'oven', 'refrigerator', 'book', 'clock',
            'vase', 'teddy bear'}


def data_root():
    return os.environ.get('AME505_DATA', '/home/ubuntu/share/tmp/ame505_data')


def import_yolo():
    """ultralytics.YOLO, also from the robot-side install (scripts/setup_yolo.sh -> $AME505_DATA/pylib).
    That folder goes AFTER the system packages, so it never shadows ROS's NumPy / OpenCV."""
    lib = os.path.join(data_root(), 'pylib')
    if os.path.isdir(lib) and lib not in sys.path:
        sys.path.append(lib)
    from ultralytics import YOLO
    return YOLO


def weights_path(name=DEFAULT_MODEL):
    """Bare model names are kept (and downloaded once) in $AME505_DATA/models."""
    if os.path.sep in name or os.path.exists(name):
        return name
    d = os.path.join(data_root(), 'models')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def box_distance_m(depth_mm, box, inner=0.5):
    """Median valid depth (m) in the central `inner` fraction of box (x1, y1, x2, y2), NaN if none.
    The centre avoids background pixels at the box edges."""
    if depth_mm is None:
        return float('nan')
    h, w = depth_mm.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    xa, xb = int(max(0, cx - hw)), int(min(w, max(cx + hw, cx - hw + 1)))
    ya, yb = int(max(0, cy - hh)), int(min(h, max(cy + hh, cy - hh + 1)))
    patch = depth_mm[ya:yb, xa:xb]
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size else float('nan')


def yolo_line(cls_id, box, w, h):
    """One YOLO-format label line: class cx cy bw bh, normalized to the image size."""
    x1, y1, x2, y2 = box
    return (f'{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} '
            f'{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
