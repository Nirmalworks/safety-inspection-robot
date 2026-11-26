#!/usr/bin/env python3
"""Planning check (no ROS): can a planner route the checkpoint tour on each candidate map, and is the route safe?

    python -m ml.map_clean.plan_check --checkpoints CP.yaml --labels LABEL.png --maps name=MAP.yaml ...

A grid version of what Nav2's global planner does: a cell is drivable if it is free and at least `--radius` metres
(the robot's half diagonal) from any occupied cell; unknown cells are not drivable. The shortest 8-connected route is
found between consecutive checkpoints (and back to the first). Per map: legs that have a route, tour length, the
smallest clearance on the route, and how many route cells come within the robot's radius of a cell hand-marked keep
(real structure the map no longer shows there: a collision the planner cannot see).
"""
import argparse
import heapq
import math

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, pgm_to_cls
import os


def route(ok, a, b):
    H, W = ok.shape
    dist = {a: 0.0}; prev = {}; pq = [(0.0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == b:
            path = [u]
            while u in prev:
                u = prev[u]; path.append(u)
            return d, path[::-1]
        if d > dist[u]:
            continue
        for dy, dx, w in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)):
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < H and 0 <= v[1] < W and ok[v] and d + w < dist.get(v, 1e18):
                dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
    return None, []


def snap(ok, p, r=6):
    if ok[p]:
        return p, 0.0
    ys, xs = np.nonzero(ok[max(0, p[0] - r):p[0] + r + 1, max(0, p[1] - r):p[1] + r + 1])
    if len(ys) == 0:
        return None, None
    ys += max(0, p[0] - r); xs += max(0, p[1] - r)
    i = int(np.argmin((ys - p[0]) ** 2 + (xs - p[1]) ** 2))
    return (int(ys[i]), int(xs[i])), float(math.hypot(ys[i] - p[0], xs[i] - p[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--maps', nargs='+', required=True); ap.add_argument('--radius', type=float, default=0.22)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    cps = yaml.safe_load(open(a.checkpoints))['checkpoints']
    lab = cv2.imread(a.labels, cv2.IMREAD_UNCHANGED)
    d_keep = cv2.distanceTransform((lab != 1).astype(np.uint8), cv2.DIST_L2, 5)
    print('| map | checkpoints in drivable space | legs with a route | tour length | smallest clearance on the route | route cells within the robot radius of real structure |\n|---|---|---|---|---|---|')
    views = []
    for spec in a.maps:
        name, path = spec.split('=', 1)
        y = yaml.safe_load(open(path)); res = y['resolution']; ox, oy = y['origin'][:2]
        cls = pgm_to_cls(cv2.imread(os.path.join(os.path.dirname(path), y['image']), 0)); H = cls.shape[0]
        d_occ = cv2.distanceTransform((cls != OCC).astype(np.uint8), cv2.DIST_L2, 5) * res
        ok = (cls == FREE) & (d_occ >= a.radius)
        pts, inside = [], 0
        for c in cps:
            p = (int(H - 1 - (c['y'] - oy) / res), int((c['x'] - ox) / res))
            q, off = snap(ok, p)
            inside += int(off == 0.0)
            pts.append(q)
        legs = total = hits = 0; clear = 1e9; cells = []
        for i in range(len(pts)):
            s, t = pts[i], pts[(i + 1) % len(pts)]
            if s is None or t is None:
                continue
            d, pth = route(ok, s, t)
            if d is None:
                continue
            legs += 1; total += d * res; cells += pth
        if cells:
            yy, xx = np.array(cells).T
            clear = float(d_occ[yy, xx].min()); hits = int((d_keep[yy, xx] * res < a.radius).sum())
        print(f"| {name} | {inside} of {len(cps)} | {legs} of {len(pts)} | {total:.1f} m | {clear * 100:.0f} cm | {hits} |", flush=True)
        if a.out:
            from .common import colour, label
            img = colour(cls, 3)
            for (r, c) in cells:
                cv2.circle(img, (c * 3 + 1, r * 3 + 1), 1, (200, 120, 0), -1)
            for i, q in enumerate(pts):
                if q is not None:
                    cv2.circle(img, (q[1] * 3 + 1, q[0] * 3 + 1), 5, (0, 0, 220), -1); cv2.putText(img, str(i + 1), (q[1] * 3 + 8, q[0] * 3 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
            nz = np.nonzero(cls != 2); s = (slice(max(0, nz[0].min() * 3 - 12), nz[0].max() * 3 + 15), slice(max(0, nz[1].min() * 3 - 12), nz[1].max() * 3 + 15))
            views.append(label(img[s], name))
    if a.out and views:
        w = max(v.shape[1] for v in views)
        cv2.imwrite(a.out, np.vstack([cv2.copyMakeBorder(v, 0, 0, 0, w - v.shape[1], cv2.BORDER_CONSTANT, value=(205, 205, 205)) for v in views]))


if __name__ == '__main__':
    main()
