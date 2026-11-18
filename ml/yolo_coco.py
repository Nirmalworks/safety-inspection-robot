#!/usr/bin/env python3
"""Run the newest pretrained COCO YOLO (Ultralytics YOLO26) on the captured RGB-D dataset.

What it is for (no training needed, works before any CVAT labels exist):
  * a first look: which staged objects a stock COCO detector already finds (chair, backpack,
    suitcase, bench, person, ...) per scene / hazard / light / checkpoint
  * distance of every detection from the aligned depth image (median depth inside the box)
  * pre-labels for CVAT (YOLO 1.1 format) so labeling starts from boxes instead of from zero

    python3 ml/yolo_coco.py $AME505_DATA/dataset                     # all frames in index.csv
    python3 ml/yolo_coco.py $AME505_DATA/dataset --session s1 --save-images --prelabels
    python3 ml/yolo_coco.py some/folder --model yolo26s.pt            # any folder of *_rgb.png / *.png

Outputs in <dataset>/yolo_coco/ (or --out): detections.csv (one row per box), frames.csv
(one row per frame), annotated/ (with --save-images), cvat_prelabels/ (with --prelabels).

Install: laptop `pip install ultralytics`; robot `scripts/setup_yolo.sh` (installs into the SSD,
without touching the ROS Python packages). Weights download once to ml/weights/ (git-ignored).
On the Jetson Nano CPU YOLO26n takes ~0.55 s per 320x240 frame: stop the robot software first
(`scripts/free_memory.sh --robot`) or run it on a laptop.
"""
import argparse
import csv
import glob
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, 'weights')
DEFAULT_MODEL = 'yolo26n.pt'        # newest Ultralytics COCO model (8.4.x); n = nano, fits the Nano
# robot-side install location (scripts/setup_yolo.sh): keeps ultralytics out of the ROS site-packages
ROBOT_PYLIB = os.path.join(os.environ.get('AME505_DATA', '/home/ubuntu/share/tmp/ame505_data'), 'pylib')

# COCO classes that can plausibly be (part of) a staged hazard or block a walkway / exit.
# Everything else is still written to detections.csv, with relevant=0.
RELEVANT = {'person', 'bicycle', 'bench', 'backpack', 'umbrella', 'handbag', 'suitcase',
            'sports ball', 'bottle', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'microwave', 'oven', 'refrigerator', 'book', 'clock',
            'vase', 'teddy bear'}


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


def load_frames(root, session=None, checkpoint=None):
    """Rows from index.csv (capture_dataset) or, for a plain folder, one row per image."""
    index = os.path.join(root, 'index.csv')
    if os.path.exists(index):
        with open(index, newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if (not session or r['session'] == session)
                    and (not checkpoint or r['checkpoint'] == checkpoint)]
        return rows
    files = sorted(glob.glob(os.path.join(root, '**', '*_rgb.png'), recursive=True)) or \
        sorted(glob.glob(os.path.join(root, '**', '*.png'), recursive=True))
    return [{'file_rgb': os.path.relpath(p, root)} for p in files]


def resolve_model(name):
    """Bare model names download once into ml/weights/ instead of the current directory."""
    if os.path.sep in name or os.path.exists(name):
        return name
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='folder with index.csv (capture_dataset) or any image folder')
    ap.add_argument('--model', default=DEFAULT_MODEL, help='yolo26n.pt (default), yolo26s.pt, yolo11n.pt, ...')
    ap.add_argument('--imgsz', type=int, default=320, help='the Astra images are 320x240')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--session', help='only this session')
    ap.add_argument('--checkpoint', help='only this checkpoint')
    ap.add_argument('--out', help='output folder (default <dataset>/yolo_coco)')
    ap.add_argument('--save-images', action='store_true', help='write annotated copies')
    ap.add_argument('--prelabels', action='store_true',
                    help='write CVAT YOLO 1.1 pre-labels (relevant classes only)')
    ap.add_argument('--limit', type=int, default=0, help='first N frames only (quick test)')
    a = ap.parse_args()

    if os.path.isdir(ROBOT_PYLIB) and ROBOT_PYLIB not in sys.path:
        sys.path.append(ROBOT_PYLIB)                 # after the system packages: never shadows them
    try:
        from ultralytics import YOLO
    except ImportError:
        print('ultralytics is not installed. Laptop: pip install ultralytics   Robot: scripts/setup_yolo.sh')
        return 1
    import cv2

    frames = load_frames(a.dataset, a.session, a.checkpoint)
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        print(f'No frames found in {a.dataset}')
        return 1
    out = a.out or os.path.join(a.dataset, 'yolo_coco')
    os.makedirs(out, exist_ok=True)

    model = YOLO(resolve_model(a.model))
    names = model.names
    rel_ids = sorted(i for i, n in names.items() if n in RELEVANT)
    rel_map = {cid: k for k, cid in enumerate(rel_ids)}          # COCO id -> pre-label class index
    if a.prelabels:
        pre = os.path.join(out, 'cvat_prelabels')
        os.makedirs(os.path.join(pre, 'obj_train_data'), exist_ok=True)
        with open(os.path.join(pre, 'obj.names'), 'w') as f:
            f.write('\n'.join(names[i] for i in rel_ids) + '\n')
        with open(os.path.join(pre, 'obj.data'), 'w') as f:
            f.write(f'classes = {len(rel_ids)}\nnames = obj.names\ntrain = train.txt\n')
        train_list = []

    label_cols = ['session', 'checkpoint', 'scene', 'hazard', 'objects', 'light', 'occluded']
    det_f = open(os.path.join(out, 'detections.csv'), 'w', newline='')
    frm_f = open(os.path.join(out, 'frames.csv'), 'w', newline='')
    det_w = csv.DictWriter(det_f, ['file_rgb', 'cls_id', 'cls_name', 'relevant', 'conf',
                                   'x1', 'y1', 'x2', 'y2', 'dist_m'] + label_cols)
    frm_w = csv.DictWriter(frm_f, ['file_rgb', 'n_det', 'n_relevant', 'classes', 'nearest_relevant_m',
                                   'infer_s'] + label_cols)
    det_w.writeheader()
    frm_w.writeheader()

    found = defaultdict(Counter)        # (scene, hazard, light) -> class counts
    frames_per_group = Counter()
    t_all = time.time()
    for k, row in enumerate(frames):
        rgb_path = os.path.join(a.dataset, row['file_rgb'])
        img = cv2.imread(rgb_path)
        if img is None:
            print(f'  skip (unreadable): {row["file_rgb"]}')
            continue
        dpath = os.path.join(a.dataset, row['file_depth']) if row.get('file_depth') else None
        depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED) if dpath and os.path.exists(dpath) else None
        t0 = time.time()
        res = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        dt = time.time() - t0

        labels = {c: row.get(c, '') for c in label_cols}
        h, w = img.shape[:2]
        lines, cls_seen, rel_dists = [], [], []
        for b in res.boxes:
            cid, conf = int(b.cls), float(b.conf)
            box = [float(v) for v in b.xyxy[0].tolist()]
            dist = box_distance_m(depth, box)
            rel = cid in rel_map
            cls_seen.append(names[cid])
            if rel:
                lines.append(yolo_line(rel_map[cid], box, w, h))
                if not np.isnan(dist):
                    rel_dists.append(dist)
            det_w.writerow({'file_rgb': row['file_rgb'], 'cls_id': cid, 'cls_name': names[cid],
                            'relevant': int(rel), 'conf': round(conf, 3),
                            **{n: round(v, 1) for n, v in zip(('x1', 'y1', 'x2', 'y2'), box)},
                            'dist_m': '' if np.isnan(dist) else round(dist, 2), **labels})
        frm_w.writerow({'file_rgb': row['file_rgb'], 'n_det': len(res.boxes), 'n_relevant': len(lines),
                        'classes': ' '.join(sorted(set(cls_seen))),
                        'nearest_relevant_m': round(min(rel_dists), 2) if rel_dists else '',
                        'infer_s': round(dt, 3), **labels})
        group = (labels['scene'] or '-', labels['hazard'] or '-', labels['light'] or '-')
        frames_per_group[group] += 1
        found[group].update(set(c for c in cls_seen if c in RELEVANT))

        if a.save_images:
            dst = os.path.join(out, 'annotated', row['file_rgb'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, res.plot())
        if a.prelabels:
            base = os.path.splitext(os.path.basename(row['file_rgb']))[0]
            with open(os.path.join(pre, 'obj_train_data', base + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            train_list.append(f'obj_train_data/{os.path.basename(row["file_rgb"])}')
        if (k + 1) % 25 == 0 or k + 1 == len(frames):
            print(f'  {k + 1}/{len(frames)}  {dt:.2f} s/frame')
    det_f.close()
    frm_f.close()
    if a.prelabels:
        with open(os.path.join(pre, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_list) + '\n')

    print(f'\n{len(frames)} frames in {time.time() - t_all:.0f} s with {a.model} -> {out}')
    print('Frames with each relevant COCO class, per (scene, hazard, light):')
    for g in sorted(frames_per_group):
        top = ', '.join(f'{c} {n}' for c, n in found[g].most_common(6)) or 'nothing'
        print(f'  {"/".join(g):40s} {frames_per_group[g]:4d} frames: {top}')
    if a.prelabels:
        print(f'CVAT: zip the CONTENTS of {pre} (obj.names, obj.data, train.txt, obj_train_data/) and '
              'import as "YOLO 1.1" into a task whose images have the same file names.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
