#!/usr/bin/env python3
"""Connect the walls of the room and make everything inside known, except the desks (no ML).

    python -m ml.map_clean.connect_walls IN.pgm OUT.pgm [--labels LABEL.png --desks 4 --desk-size L W] [--max-gap 0.6]

The room is the largest floor region with its outline closed (9 cells) and its holes filled. Along the one-cell ring
just outside it, every stretch that is not a wall is a gap. A gap no longer than `--max-gap` metres (default 1.5, so a doorway is closed too) is drawn as wall;
a longer one is left open and reported. Then every unknown cell inside the room becomes floor, except
inside the desk rectangles (found from the hand labels as in straighten.py, else from the map).
The wall drawn across a gap was not seen by the lidar: the map then claims a wall there.
"""
import argparse
import math
import os

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, PGM, UNK, pgm_to_cls
from .straighten import desks_from_labels, find_desks, main_angle


def connect_walls(cls, lab=None, n_desks=4, desk_size=None, max_gap_m=1.5, res=0.05, k=9, near=2):
    free = (cls == FREE).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), connectivity=4)
    room = (cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    room = cv2.morphologyEx(room, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnt, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    room = cv2.drawContours(np.zeros_like(room), cnt, -1, 1, cv2.FILLED)
    ring = (cv2.dilate(room, np.ones((3, 3), np.uint8)) > 0) & (room == 0)
    room = room > 0
    out = cls.copy()
    # a ring cell is already walled if an occupied cell lies within `near` cells of it
    walled = ring & (cv2.dilate((cls == OCC).astype(np.uint8), np.ones((2 * near + 1, 2 * near + 1), np.uint8)) > 0)
    gaps = (ring & ~walled).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(gaps, connectivity=8)
    closed = 0; left = []
    for i in range(1, n):
        length = st[i, cv2.CC_STAT_AREA] * res
        if length <= max_gap_m:
            out[cc == i] = OCC; closed += 1
        else:
            left.append(round(float(length), 2))
    desk = np.zeros(cls.shape, np.uint8)
    if n_desks:
        th, _ = main_angle(cls == OCC)
        boxes = desks_from_labels(cls, lab, n_desks, th, desk_size) if lab is not None else find_desks(cls, n_desks, th, desk_size)
        for (cx, cy), (a, b), ang in boxes:
            cv2.fillPoly(desk, [cv2.boxPoints(((cx, cy), (a, b), math.degrees(ang))).astype(np.int32)], 1)
    # also keep any large closed pocket with an occupied rim: a desk the rectangles missed
    n, cc, st, _ = cv2.connectedComponentsWithStats((room & (out == UNK) & (desk == 0)).astype(np.uint8), connectivity=4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 600:
            comp = cc == i
            rim = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
            if (out[rim] == OCC).mean() >= 0.5:
                desk[comp] = 1
    fill = room & (out == UNK) & (desk == 0)
    out[fill] = FREE
    return out, {'gaps_closed': closed, 'openings_left_m': sorted(left, reverse=True), 'wall_cells_drawn': int((out == OCC).sum() - (cls == OCC).sum()), 'filled_m2': float(fill.sum() * res * res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--labels', default='')
    ap.add_argument('--desks', type=int, default=4); ap.add_argument('--desk-size', type=float, nargs=2, default=None); ap.add_argument('--max-gap', type=float, default=1.5, help='metres; a longer gap in the wall is left open')
    a = ap.parse_args()
    cls = pgm_to_cls(cv2.imread(a.inp, 0)); lab = cv2.imread(a.labels, -1) if a.labels else None
    out, info = connect_walls(cls, lab, a.desks, a.desk_size, a.max_gap)
    cv2.imwrite(a.out, PGM[out])
    y_in = os.path.splitext(a.inp)[0] + '.yaml'
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in)); y['image'] = os.path.basename(a.out)
        yaml.safe_dump(y, open(os.path.splitext(a.out)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f"{a.out}: {info['gaps_closed']} gaps closed with {info['wall_cells_drawn']} wall cells, openings left (m): {info['openings_left_m'] or 'none'}, {info['filled_m2']:.2f} m2 inside turned into floor")


if __name__ == '__main__':
    main()
