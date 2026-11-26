#!/usr/bin/env python3
"""Straight walls by regularising the room outline (no ML).

    python -m ml.map_clean.outline CLEANED.pgm STRAIGHT_DESKS.pgm OUT.pgm [--k 9]

The map is turned so the room's main wall direction is horizontal. There the floor region is closed and opened with a
square of `k` cells, which leaves only axis-aligned edges longer than k cells: ragged walls become straight runs and
recesses narrower than k cells are filled in or squared off. Where the new floor edge lies within `near` cells of an
original occupied cell it is drawn as wall; where the floor ends in unknown space with no wall nearby it stays open, as
in the input. Occupied things standing inside the floor (pillars, the redrawn desks of STRAIGHT_DESKS) are kept as they are.
"""
import argparse
import math
import os

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, PGM, UNK, pgm_to_cls
from .straighten import main_angle


def outline(cls, desks, k=9, near=3):
    H, W = cls.shape
    th, _ = main_angle(cls == OCC)
    c = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(c, math.degrees(th), 1.0)            # turn walls onto the axes
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W2, H2 = int(H * sin + W * cos) + 4, int(H * cos + W * sin) + 4
    M[0, 2] += W2 / 2 - c[0]; M[1, 2] += H2 / 2 - c[1]
    Mi = cv2.invertAffineTransform(M)
    r = cv2.warpAffine(cls, M, (W2, H2), flags=cv2.INTER_NEAREST, borderValue=int(UNK))
    free = (r == FREE).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), connectivity=4)
    room = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    room = cv2.morphologyEx(room, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # swallow clutter-sized notches and things on the floor
    room = cv2.morphologyEx(room, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))    # cut spikes and thin leaks
    cnt, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    room = cv2.drawContours(np.zeros_like(room), cnt, -1, 1, cv2.FILLED)   # holes (desks, unseen pockets) belong to the room: only the outer wall is regularised
    edge = (cv2.dilate(room, np.ones((3, 3), np.uint8)) > 0) & (room == 0)      # one-cell ring just outside the regular floor
    d_occ = cv2.distanceTransform((r != OCC).astype(np.uint8), cv2.DIST_L2, 3)
    wall = edge & (d_occ <= near)
    wall = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8)) & edge.astype(np.uint8) | \
           cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8)) & edge.astype(np.uint8)
    back = lambda m: cv2.warpAffine(m.astype(np.uint8), Mi, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    room_b, wall_b = back(room), back(cv2.dilate(wall, np.ones((2, 2), np.uint8)))
    out = np.full(cls.shape, UNK, np.uint8)
    band = room_b & (cls != UNK)
    out[band] = FREE                                                  # the strip between the old ragged wall and the new straight one
    inner = cv2.erode(room_b.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    keep_in = inner & (desks == OCC)                                  # things standing in the room: desks, pillars
    out[inner] = desks[inner]
    out[keep_in] = OCC
    out[wall_b & ~inner] = OCC
    # never hide structure: an occupied cell of the cleaned map inside the room that the redrawn desks do not cover is put back, so the redrawn desks can only add to what was there
    far = out != OCC
    out[inner & (cls == OCC) & far & (desks != UNK)] = OCC          # cells inside a redrawn desk are covered by it
    return out, {'angle_deg': math.degrees(th), 'wall_cells': int((wall_b & ~inner).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cleaned'); ap.add_argument('desks'); ap.add_argument('out'); ap.add_argument('--k', type=int, default=9); ap.add_argument('--near', type=int, default=3)
    a = ap.parse_args()
    cls = pgm_to_cls(cv2.imread(a.cleaned, 0)); dk = pgm_to_cls(cv2.imread(a.desks, 0))
    out, info = outline(cls, dk, a.k, a.near)
    cv2.imwrite(a.out, PGM[out])
    y_in = os.path.splitext(a.cleaned)[0] + '.yaml'
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in)); y['image'] = os.path.basename(a.out)
        yaml.safe_dump(y, open(os.path.splitext(a.out)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f"{a.out}: main direction {info['angle_deg']:.1f} deg, {info['wall_cells']} wall cells, floor {int((out == FREE).sum()) * 0.0025:.1f} m2 (input {int((cls == FREE).sum()) * 0.0025:.1f} m2)")


if __name__ == '__main__':
    main()
