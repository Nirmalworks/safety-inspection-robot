#!/usr/bin/env python3
"""Turn the unseen patches inside the room into floor (no ML). Desks and the insides of solid objects stay as they are.

    python -m ml.map_clean.fill_unknown IN.pgm OUT.pgm [--labels LABEL.png --desks 4 --desk-size L W]

The room is the largest floor region with its outline closed (9 cells) and its holes filled. Inside it, a patch of
unknown cells becomes floor unless
  - it lies inside a desk rectangle (found from the hand labels as in straighten.py, when --labels is given), or
  - at least `--solid` of the cells around it are occupied: the inside of a solid object such as a pillar or cabinet, or
  - it opens to the outside of the room over more than `--opening` cells (35 cm, the robot's width): a real opening.
A notch in the wall with a narrower opening is filled up to the room outline; no wall is drawn across it.
What is filled was never seen by the lidar (it lay in the shadow of furniture), so the map then claims floor there.
"""
import argparse
import math
import os

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, PGM, UNK, pgm_to_cls
from .straighten import desks_from_labels, main_angle


def fill_unknown(cls, lab=None, n_desks=4, desk_size=None, solid=0.6, k=9, opening=7):
    free = (cls == FREE).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), connectivity=4)
    room = (cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    room = cv2.morphologyEx(room, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnt, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    room = cv2.drawContours(np.zeros_like(room), cnt, -1, 1, cv2.FILLED) > 0
    desk = np.zeros(cls.shape, np.uint8)
    if lab is not None:
        th, _ = main_angle(cls == OCC)
        for (cx, cy), (a, b), ang in desks_from_labels(cls, lab, n_desks, th, desk_size):
            cv2.fillPoly(desk, [cv2.boxPoints(((cx, cy), (a, b), math.degrees(ang))).astype(np.int32)], 1)
    cand = ((cls == UNK) & room & (desk == 0)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    out = cls.copy(); filled = kept = 0
    for i in range(1, n):
        comp = cc == i
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
        gap = int((ring & ~room).sum())                         # how wide the patch opens to the outside of the room, in cells
        if gap > opening or (gap == 0 and (cls[ring] == OCC).mean() >= solid):
            kept += int(comp.sum()); continue                   # a real opening wider than the robot, or the inside of a solid object
        out[comp] = FREE; filled += int(comp.sum())
    return out, filled, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--labels', default='')
    ap.add_argument('--desks', type=int, default=4); ap.add_argument('--desk-size', type=float, nargs=2, default=None); ap.add_argument('--solid', type=float, default=0.6); ap.add_argument('--opening', type=int, default=7)
    a = ap.parse_args()
    cls = pgm_to_cls(cv2.imread(a.inp, 0)); lab = cv2.imread(a.labels, -1) if a.labels else None
    out, filled, kept = fill_unknown(cls, lab, a.desks, a.desk_size, a.solid, opening=a.opening)
    cv2.imwrite(a.out, PGM[out])
    y_in = os.path.splitext(a.inp)[0] + '.yaml'
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in)); y['image'] = os.path.basename(a.out)
        yaml.safe_dump(y, open(os.path.splitext(a.out)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f'{a.out}: {filled * 0.0025:.2f} m2 of unseen space inside the room turned into floor, {kept * 0.0025:.2f} m2 left unknown')


if __name__ == '__main__':
    main()
