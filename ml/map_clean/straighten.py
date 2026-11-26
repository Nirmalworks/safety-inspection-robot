#!/usr/bin/env python3
"""Straighten a cleaned map (no ML): walls become straight segments along the room's two main directions,
and furniture that is known to be identical (the lab's benches) is redrawn as identical rectangles.

    python -m ml.map_clean.straighten CLEANED.pgm OUT.pgm [--desks 4] [--desk-size L W]

1. Main direction: the angle (mod 90 degrees) that most wall length follows, from short Hough segments.
2. Walls: segments within `--tol` degrees of a main direction are grouped by the line they lie on (within `--snap`
   cells) and merged along it across gaps up to `--gap` cells. Each merged run of at least `--min-len` cells is drawn
   as one straight line. Occupied cells within `--snap` cells of a drawn line are replaced by it.
3. Everything else that was occupied (pillars, curved or short structure) is left exactly as it was.
4. Desks: closed pockets of unknown space with an occupied rim, of desk-like size, are fitted with rectangles. The
   `--desks` most alike are given one shared size (the median, or `--desk-size` in metres) and one shared direction,
   and redrawn: outline occupied, inside unknown.
Free and unknown space elsewhere are not touched, so the output stays on the input's grid.
"""
import argparse
import math
import os

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, PGM, UNK, pgm_to_cls


def main_angle(occ):
    """Dominant wall direction in radians, in [0, pi/2)."""
    lines = cv2.HoughLinesP(occ.astype(np.uint8) * 255, 1, np.pi / 720, threshold=10, minLineLength=10, maxLineGap=2)
    if lines is None:
        return 0.0, np.zeros((0, 4))
    L = np.asarray(lines).reshape(-1, 4).astype(float)
    ang = np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])
    w = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    th = math.atan2((w * np.sin(4 * ang)).sum(), (w * np.cos(4 * ang)).sum()) / 4          # circular mean with period 90 degrees
    return th % (math.pi / 2), L


def wall_lines(L, th, tol_deg=8.0, snap=1.5, gap=6, min_len=10):
    """-> list of (x1, y1, x2, y2) straight segments in image coordinates."""
    c, s = math.cos(th), math.sin(th)
    out = []
    for horiz in (True, False):
        segs = []
        for x1, y1, x2, y2 in L:
            a = math.atan2(y2 - y1, x2 - x1)
            d = ((a - th - (0 if horiz else math.pi / 2) + math.pi / 2) % math.pi) - math.pi / 2
            if abs(math.degrees(d)) > tol_deg:
                continue
            u1, v1, u2, v2 = x1 * c + y1 * s, -x1 * s + y1 * c, x2 * c + y2 * s, -x2 * s + y2 * c
            if not horiz:
                u1, v1, u2, v2 = v1, u1, v2, u2                                       # along = v, across = u
            segs.append((0.5 * (v1 + v2), min(u1, u2), max(u1, u2)))
        segs.sort()
        i = 0
        while i < len(segs):
            j = i
            while j + 1 < len(segs) and segs[j + 1][0] - segs[i][0] <= 2 * snap:
                j += 1
            grp = segs[i:j + 1]
            wts = np.array([g[2] - g[1] for g in grp])
            pos = float(np.average([g[0] for g in grp], weights=np.maximum(wts, 1e-6)))
            runs = []
            for a0, a1 in sorted((g[1], g[2]) for g in grp):
                if runs and a0 - runs[-1][1] <= gap:
                    runs[-1][1] = max(runs[-1][1], a1)
                else:
                    runs.append([a0, a1])
            for a0, a1 in runs:
                if a1 - a0 >= min_len:
                    if horiz:
                        p, q = (a0, pos), (a1, pos)                                   # (u, v)
                    else:
                        p, q = (pos, a0), (pos, a1)
                    out.append((p[0] * c - p[1] * s, p[0] * s + p[1] * c, q[0] * c - q[1] * s, q[0] * s + q[1] * c))
            i = j + 1
    return out


def desks_from_labels(cls, lab, n, th, size_m=None, res=0.05):
    """Desks from hand labels: closed unknown pockets rimmed by cells marked keep. One shared size and direction."""
    keep = ((cls == OCC) & (lab == 1)).astype(np.uint8)
    num, cc, st, _ = cv2.connectedComponentsWithStats((cls == UNK).astype(np.uint8), connectivity=4)
    H, W = cls.shape
    cand = []
    for i in range(1, num):
        x, y, w, h, area = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H or not (600 <= area <= 4000):
            continue
        rim = keep & (cv2.dilate((cc == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if rim.sum() < 100:
            continue
        (cx, cy), (a, b), ang = cv2.minAreaRect(np.argwhere(rim)[:, ::-1].astype(np.float32))
        ang = math.radians(ang) if a >= b else math.radians(ang) + math.pi / 2
        cand.append(((cx, cy), (max(a, b), min(a, b)), ang))
    cand = sorted(cand, key=lambda c: -c[1][0] * c[1][1])[:n]
    if not cand:
        return []
    long_ = size_m[0] / res if size_m else float(np.median([c[1][0] for c in cand]))
    short = size_m[1] / res if size_m and size_m[1] > 0 else float(np.median([c[1][1] for c in cand]))
    off = float(np.median([((c[2] - th + math.pi / 4) % (math.pi / 2)) - math.pi / 4 for c in cand]))
    return [(c[0], (long_, short), th + round((c[2] - th) / (math.pi / 2)) * (math.pi / 2) + off) for c in cand]


def find_desks(cls, n, th, size_m=None, res=0.05):
    """Closed unknown pockets of desk-like size -> n rectangles ((cx, cy), (long, short), angle_rad), all one size and direction."""
    unk = (cls == UNK).astype(np.uint8)
    num, lab, st, cen = cv2.connectedComponentsWithStats(unk, connectivity=4)
    H, W = cls.shape
    cand = []
    for i in range(1, num):
        x, y, w, h, area = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H or not (250 <= area <= 4000):
            continue
        comp = (lab == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((5, 5), np.uint8)) > 0
        if (cls[ring & (comp == 0)] == OCC).mean() < 0.35:                            # a desk is rimmed by occupied cells
            continue
        pts = np.argwhere(ring & ((cls == OCC) | (comp > 0)))[:, ::-1].astype(np.float32)
        (cx, cy), (a, b), ang = cv2.minAreaRect(pts)
        if min(a, b) < 8 or area / max(1.0, a * b) < 0.55:
            continue
        ang = math.radians(ang) if a >= b else math.radians(ang) + math.pi / 2          # direction of the long side
        cand.append(((cx, cy), (max(a, b), min(a, b)), ang, area))
    if not cand:
        return []
    if len(cand) > n:                                                                 # keep the n most alike in size
        med = np.median([c[1] for c in cand], axis=0)
        cand = sorted(cand, key=lambda c: abs(c[1][0] - med[0]) + abs(c[1][1] - med[1]))[:n]
    long_, short = (np.median([c[1][0] for c in cand]), np.median([c[1][1] for c in cand])) if size_m is None else (size_m[0] / res, size_m[1] / res)
    dirs = [((c[2] - th + math.pi / 4) % (math.pi / 2)) - math.pi / 4 for c in cand]   # offset from the nearest main direction
    base = [th + round((c[2] - th) / (math.pi / 2)) * (math.pi / 2) for c in cand]
    off = float(np.median(dirs))
    return [((c[0][0], c[0][1]), (float(long_), float(short)), b + off) for c, b in zip(cand, base)]


def straighten(cls, n_desks=0, desk_size=None, tol=8.0, snap=1.5, gap=6, min_len=10, res=0.05, lab=None):
    occ = cls == OCC
    th, L = main_angle(occ)
    out = cls.copy()
    desks = (desks_from_labels(cls, lab, n_desks, th, desk_size, res) if lab is not None else find_desks(cls, n_desks, th, desk_size, res)) if n_desks else []
    desk_band = np.zeros(cls.shape, np.uint8)
    for (cx, cy), (a, b), ang in desks:
        box = cv2.boxPoints(((cx, cy), (a + 6, b + 6), math.degrees(ang))).astype(np.int32)
        cv2.fillPoly(desk_band, [box], 1)
    fine = cv2.HoughLinesP(occ.astype(np.uint8) * 255, 1, np.pi / 360, threshold=5, minLineLength=max(4, min_len // 2), maxLineGap=2)
    if fine is not None:
        L = np.asarray(fine).reshape(-1, 4).astype(float)
    lines = wall_lines(L, th, tol, snap, gap, min_len)
    drawn = np.zeros(cls.shape, np.uint8)
    for x1, y1, x2, y2 in lines:
        mx, my = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        if 0 <= my < cls.shape[0] and 0 <= mx < cls.shape[1] and desk_band[my, mx]:
            continue                                                                  # desk edges are drawn by the desk step
        cv2.line(drawn, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1, 1)
    near = cv2.distanceTransform((drawn == 0).astype(np.uint8), cv2.DIST_L2, 3) <= snap + 0.5
    replaced = occ & near & (desk_band == 0)
    near_free = cv2.dilate((cls == FREE).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[replaced] = np.where(near_free[replaced], FREE, UNK)
    out[(drawn > 0) & (desk_band == 0)] = OCC
    # join corners: push each line end forward up to `ext` cells and keep the extension only if it reaches another wall
    ext = 6
    for x1, y1, x2, y2 in lines:
        d = np.array([x2 - x1, y2 - y1]); n = np.hypot(*d)
        if n < 1:
            continue
        d = d / n
        for (ex, ey), sgn in (((x2, y2), 1), ((x1, y1), -1)):
            path = []
            for k in range(2, ext + 2):
                px, py = int(round(ex + sgn * d[0] * k)), int(round(ey + sgn * d[1] * k))
                if not (0 <= py < cls.shape[0] and 0 <= px < cls.shape[1]) or desk_band[py, px]:
                    break
                if out[py, px] == OCC:
                    for qx, qy in path:
                        out[qy, qx] = OCC
                    break
                path.append((px, py))
    # never open a wall: a removed cell with no occupied cell within 1.5 cells of it is put back
    far = cv2.distanceTransform((out != OCC).astype(np.uint8), cv2.DIST_L2, 3) > 1.5
    back = replaced & far
    out[back] = OCC
    for (cx, cy), (a, b), ang in desks:
        outer = cv2.boxPoints(((cx, cy), (a + 6, b + 6), math.degrees(ang))).astype(np.int32)
        box = cv2.boxPoints(((cx, cy), (a, b), math.degrees(ang))).astype(np.int32)
        m = np.zeros(cls.shape, np.uint8); cv2.fillPoly(m, [outer], 1)
        old = (m > 0) & (out == OCC)
        out[old] = FREE                                                               # clear the ragged rim...
        fill = np.zeros(cls.shape, np.uint8); cv2.fillPoly(fill, [box], 1)
        out[fill > 0] = UNK                                                           # ...the inside of a desk is unknown...
        cv2.polylines(out, [box], True, int(OCC), 1)                                  # ...and its outline is one clean rectangle
    return out, {'angle_deg': math.degrees(th), 'lines': len(lines), 'desks': [(round(c[0][0], 1), round(c[0][1], 1), round(c[1][0] * res, 2), round(c[1][1] * res, 2), round(math.degrees(c[2]), 1)) for c in desks],
                 'cells_replaced': int(replaced.sum()), 'cells_drawn': int(((drawn > 0) & (desk_band == 0)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--desks', type=int, default=0, help='number of identical desks to redraw (0 = none)')
    ap.add_argument('--desk-size', type=float, nargs=2, default=None, help='desk size in metres: long side, short side (0 = take the short side from the map)')
    ap.add_argument('--labels', default='', help='hand label image; desks are then found from cells marked keep')
    ap.add_argument('--tol', type=float, default=8.0); ap.add_argument('--snap', type=float, default=1.5)
    ap.add_argument('--gap', type=int, default=6); ap.add_argument('--min-len', type=int, default=10)
    a = ap.parse_args()
    cls = pgm_to_cls(cv2.imread(a.inp, cv2.IMREAD_GRAYSCALE))
    lab = cv2.imread(a.labels, cv2.IMREAD_UNCHANGED) if a.labels else None
    out, info = straighten(cls, a.desks, a.desk_size, a.tol, a.snap, a.gap, a.min_len, lab=lab)
    cv2.imwrite(a.out, PGM[out])
    y_in = os.path.splitext(a.inp)[0] + '.yaml'
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in)); y['image'] = os.path.basename(a.out)
        yaml.safe_dump(y, open(os.path.splitext(a.out)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f"{a.inp} -> {a.out}: main direction {info['angle_deg']:.1f} deg, {info['lines']} straight wall segments, {info['cells_replaced']} ragged cells replaced by {info['cells_drawn']} line cells")
    for d in info['desks']:
        print(f'  desk at cell ({d[0]}, {d[1]}): {d[2]} x {d[3]} m, direction {d[4]} deg')


if __name__ == '__main__':
    main()
