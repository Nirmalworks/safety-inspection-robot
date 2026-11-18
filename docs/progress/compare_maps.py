#!/usr/bin/env python3
"""Map repeatability: align two lidar maps of the same room (made on different days) and measure
how well their walls agree.

    python3 docs/progress/compare_maps.py                        # lab_new_maps (11-13) vs lab2d_11_18_2 (11-18)
    python3 docs/progress/compare_maps.py maps/A/maps_2d/A.yaml maps/B/maps_2d/B.yaml

Method: occupied cells -> 2D points; coarse search over rotation (3 deg) and translation (0.5 m),
then trimmed point-to-point ICP (best 80 % of matches). Metric: distance from every wall cell of
map A to the nearest wall cell of map B after alignment. Furniture that moved between the days
counts as disagreement, so this is a pessimistic measure of the mapping itself.
Writes docs/progress/figures/map_repeatability.png and prints the numbers.
"""
import argparse
import math
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
DEFAULT_A = os.path.join(ROOT, 'maps', 'lab_new_maps', 'maps_2d', 'lab_new_maps.yaml')
DEFAULT_B = os.path.join(ROOT, 'maps', 'lab2d_11_18_2', 'maps_2d', 'lab2d_11_18_2.yaml')


def wall_points(path):
    m = yaml.safe_load(open(path))
    img = cv2.imread(os.path.join(os.path.dirname(path), m['image']), cv2.IMREAD_GRAYSCALE)
    r, (ox, oy) = m['resolution'], m['origin'][:2]
    ys, xs = np.nonzero(img < 255 * (1 - m.get('occupied_thresh', 0.65)))
    p = np.c_[ox + (xs + 0.5) * r, oy + (img.shape[0] - ys - 0.5) * r]
    return p, float((img > 250).sum()) * r * r


def rot(th):
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def align(A, B):
    tb = cKDTree(B)
    sub, ca, cb = A[::4], A.mean(0), B.mean(0)
    best = (np.inf, None, None)
    for deg in range(0, 360, 3):
        R = rot(math.radians(deg))
        t0 = cb - R @ ca
        for dx in np.arange(-3, 3.01, 0.5):
            for dy in np.arange(-3, 3.01, 0.5):
                t = t0 + [dx, dy]
                s = np.mean(np.minimum(tb.query(sub @ R.T + t)[0], 0.5))
                if s < best[0]:
                    best = (s, R, t)
    _, R, t = best
    for _ in range(60):                                      # trimmed ICP
        d, i = tb.query(A @ R.T + t)
        k = d < np.percentile(d, 80)
        P, M = A[k], B[i[k]]
        pm, mm = P.mean(0), M.mean(0)
        U, _, Vt = np.linalg.svd((P - pm).T @ (M - mm))
        D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mm - R @ pm
    return R, t, tb.query(A @ R.T + t)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a', nargs='?', default=DEFAULT_A)
    ap.add_argument('b', nargs='?', default=DEFAULT_B)
    ap.add_argument('--label-a', default='2026-11-13')
    ap.add_argument('--label-b', default='2026-11-18')
    a = ap.parse_args()
    A, free_a = wall_points(a.a)
    B, free_b = wall_points(a.b)
    R, t, d = align(A, B)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    within = {c: 100 * float(np.mean(d < c)) for c in (0.05, 0.10, 0.20)}
    print(f'{a.label_a}: {len(A)} wall cells, {free_a:.0f} m2 free | {a.label_b}: {len(B)} wall cells, {free_b:.0f} m2 free')
    print(f'alignment: rotation {yaw:.1f} deg, translation ({t[0]:.2f}, {t[1]:.2f}) m')
    print('walls of A within 5 / 10 / 20 cm of B: ' + ' / '.join(f'{v:.1f} %' for v in within.values()))
    print(f'median distance {np.median(d) * 100:.1f} cm')

    Aa = A @ R.T + t
    bg, fg = '#0b1020', '#e6edf7'
    fig = plt.figure(figsize=(13, 6.6), facecolor=bg)
    ax = fig.add_axes([0.04, 0.1, 0.62, 0.74], facecolor='#111a2e')
    ax.scatter(B[:, 0], B[:, 1], s=1.2, c='#5ee6ff', label=f'{a.label_b} map', lw=0)
    ok = d < 0.10
    ax.scatter(Aa[~ok, 0], Aa[~ok, 1], s=1.2, c='#ff4fd8', label=f'{a.label_a} map, > 10 cm off', lw=0)
    ax.scatter(Aa[ok, 0], Aa[ok, 1], s=1.2, c='#ffffff', label=f'{a.label_a} map, within 10 cm', lw=0)
    ax.set_aspect('equal')
    ax.tick_params(colors='#8b9bb8', labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(loc='lower right', fontsize=9, markerscale=8, facecolor=bg, edgecolor='#23304d')
    for txt in leg.get_texts():
        txt.set_color(fg)
    ax2 = fig.add_axes([0.72, 0.16, 0.25, 0.5], facecolor=bg)
    ax2.hist(np.minimum(d, 0.5) * 100, bins=50, color='#5ee6ff')
    ax2.axvline(np.median(d) * 100, color='#ffb020', lw=2)
    ax2.set_xlabel('wall-to-wall distance after alignment [cm]', color='#8b9bb8', fontsize=9)
    ax2.tick_params(colors='#8b9bb8', labelsize=8)
    for s in ax2.spines.values():
        s.set_color('#23304d')
    fig.text(0.04, 0.95, 'Map repeatability: two independent mapping runs', color=fg, fontsize=20,
             weight='bold', va='top')
    fig.text(0.04, 0.895, f'slam_toolbox · RPLidar A1 · {a.label_a} vs {a.label_b} · aligned by ICP '
             f'(rotation {yaw:.0f}°: mapping started facing another way)', color='#8b9bb8', fontsize=11, va='top')
    fig.text(0.72, 0.83, f'median {np.median(d) * 100:.0f} cm', color='#ffb020', fontsize=22, weight='bold')
    fig.text(0.72, 0.76, f'{within[0.10]:.0f} % of walls within 10 cm\n{within[0.20]:.0f} % within 20 cm',
             color=fg, fontsize=12, va='top')
    fig.text(0.04, 0.03, 'Disagreement includes furniture moved between the days (chairs, carts), '
             'so this is a pessimistic measure of mapping error.', color='#8b9bb8', fontsize=9)
    out = os.path.join(HERE, 'figures', 'map_repeatability.png')
    fig.savefig(out, dpi=180, facecolor=bg)
    print(out)


if __name__ == '__main__':
    main()
