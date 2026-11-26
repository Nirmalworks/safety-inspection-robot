#!/usr/bin/env python3
"""Real pairs from the MIT Stata Center: slam_toolbox maps of its bags, labelled from its floor plans.

    python -m ml.map_clean.stata_pairs --maps "data/map_cleaning/stata/slam_maps/*.yaml"

Each map is fitted to the plan of its floor (every rotation is tried against the plan's walls with an image
correlation at 20 cm, then trimmed ICP at full resolution). The label of an occupied cell is its distance to the
nearest plan wall: keep within `--keep-cm`, erase beyond `--erase-cm`, unsure in between, near a doorway, or
outside the building outline. The plans have no furniture, so all furniture counts as erase here.
Writes <out>/<bag>.npz (X map classes, L labels 0 / 1 keep / 2 erase / 3 unsure, floor, fit) and <bag>_view.png, FIT.md.
"""
import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .common import FREE, OCC, UNK, cells_xy, colour, label, load_map, rot, warp
from .floorplans import stata_raster
from .prelabel import refine


def plan_paths(root, floor):
    one = os.path.join(root, f'32-{floor}.xml')
    return [one] if os.path.exists(one) else [os.path.join(root, f'32-D{floor}.xml'), os.path.join(root, f'32-G{floor}.xml')]


def coarse(P, B, cell=0.2, step=2, top=4):
    """Best rotations / shifts of map points P onto plan wall points B by image correlation. -> [(score, R, t)]"""
    lo = B.min(0) - 2.0
    w, h = (np.ceil((B.max(0) + 2.0 - lo) / cell)).astype(int)
    img = np.zeros((h, w), np.float32)
    ib = ((B - lo) / cell).astype(int)
    img[ib[:, 1], ib[:, 0]] = 1
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    out = []
    for deg in range(0, 360, step):
        R = rot(math.radians(deg))
        Q = P @ R.T
        q0 = Q.min(0)
        iq = ((Q - q0) / cell).astype(int)
        tw, th = iq[:, 0].max() + 1, iq[:, 1].max() + 1
        T = np.zeros((th, tw), np.float32)
        T[iq[:, 1], iq[:, 0]] = 1
        pad = cv2.copyMakeBorder(img, th // 2, th // 2, tw // 2, tw // 2, cv2.BORDER_CONSTANT, value=0)
        if pad.shape[0] < th or pad.shape[1] < tw:
            continue
        r = cv2.matchTemplate(pad, T, cv2.TM_CCORR)
        _, mx, _, loc = cv2.minMaxLoc(r)
        t = lo + (np.array(loc, float) - [tw // 2, th // 2]) * cell - q0
        out.append((mx / T.sum(), R, t))
    out.sort(key=lambda v: -v[0])
    picked = []
    for s, R, t in out:
        if all(abs((math.degrees(math.atan2(R[1, 0], R[0, 0])) - math.degrees(math.atan2(r2[1, 0], r2[0, 0])) + 180) % 360 - 180) > 6 or np.hypot(*(t - t2)) > 3 for _, r2, t2 in picked):
            picked.append((s, R, t))
        if len(picked) == top:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='data/map_cleaning/stata/slam_maps/*.yaml')
    ap.add_argument('--plans', default='data/map_cleaning/stata/floorplans')
    ap.add_argument('--bags', default='data/map_cleaning/stata/bags.csv')
    ap.add_argument('--out', default='data/map_cleaning/stata/pairs')
    ap.add_argument('--keep-cm', type=float, default=20.0)
    ap.add_argument('--erase-cm', type=float, default=50.0)
    ap.add_argument('--tile', type=int, default=300, help='cells: the map is refitted to the plan in tiles of this size (15 m)')
    ap.add_argument('--tile-min', type=float, default=0.50, help='a tile is used only if this share of its occupied cells lies within 20 cm of a plan wall')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    floors = {}
    for row in csv.reader(open(a.bags)):
        if row and row[0].endswith('.bag'):
            floors[row[0][:-4]] = [f.strip() for f in row[6].split(',') if f.strip().isdigit()]
    md = ['# MIT Stata Center: slam_toolbox maps fitted to the floor plans\n',
          f'keep: within {a.keep_cm:.0f} cm of a plan wall; erase: more than {a.erase_cm:.0f} cm from any plan wall and inside the building outline; unsure: the rest, and doorways.\n',
          '| bag | floor | map size m | occupied cells | whole-map fit: median cm | tiles that fit | keep | erase | unsure | usable |', '|---|---|---|---|---|---|---|---|---|---|']
    plans = {}
    for y in sorted(glob.glob(a.maps)):
        m = load_map(y)
        name, cls, res = m['name'], m['cls'], m['res']
        fl = floors.get(name.split('_s')[0], [])                    # a piece of a drive is named <bag>_sNN
        if len(set(fl)) != 1:
            md.append(f"| {name} | {', '.join(fl) or '?'} | | | | | | | | no: not one floor |")
            continue
        fl = fl[0]
        if fl not in plans:
            wall, door, inside, org = stata_raster(plan_paths(a.plans, fl), res)
            plans[fl] = (wall, door, inside, org, cells_xy(wall, res, org))
        wall, door, inside, org, B = plans[fl]
        P = cells_xy(cls == OCC, res, m['origin'])
        best = None
        tb = cKDTree(B)
        for s, R, t in coarse(P, B):
            R2, t2, med = refine(P[::3], B, R, t, iters=30)
            d = tb.query(P @ R2.T + t2)[0]
            if best is None or np.mean(d < 0.2) > best[3]:
                best = (R2, t2, float(np.median(d)), float(np.mean(d < 0.2)))
        R, t, med, in20 = best
        Ri, ti = R.T, -R.T @ t                                    # plan -> map (for the picture only)
        to_map = lambda mask: warp(mask.astype(np.uint8), res, org, Ri, ti, cls.shape, m['origin']) == 1
        w_m = to_map(wall)
        # A long drive bends the SLAM map, so one rigid fit cannot hold everywhere: refit tile by tile from the global fit.
        occ = cls == OCC
        lab = np.zeros(cls.shape, np.uint8)
        lab[occ] = 3
        ddoor_img = cv2.distanceTransform((~door).astype(np.uint8), cv2.DIST_L2, 5) * res * 100
        Hp = wall.shape[0]
        ys_all, xs_all = np.nonzero(occ)
        TILE = a.tile
        n_tiles = n_good = 0
        for ty in range(0, cls.shape[0], TILE):
            for tx in range(0, cls.shape[1], TILE):
                core = (ys_all >= ty) & (ys_all < ty + TILE) & (xs_all >= tx) & (xs_all < tx + TILE)
                if core.sum() < 150:
                    continue
                near = (ys_all >= ty - 60) & (ys_all < ty + TILE + 60) & (xs_all >= tx - 60) & (xs_all < tx + TILE + 60)
                Pn = np.c_[m['origin'][0] + (xs_all[near] + 0.5) * res, m['origin'][1] + (cls.shape[0] - ys_all[near] - 0.5) * res]
                Rt, tt, _ = refine(Pn, B, R, t, iters=25)
                Pc = np.c_[m['origin'][0] + (xs_all[core] + 0.5) * res, m['origin'][1] + (cls.shape[0] - ys_all[core] - 0.5) * res] @ Rt.T + tt
                d = tb.query(Pc)[0] * 100
                n_tiles += 1
                if np.mean(d < 20) < a.tile_min:
                    continue                                      # this part of the map does not fit the plan: leave it unsure
                n_good += 1
                px = np.clip(((Pc[:, 0] - org[0]) / res).astype(int), 0, wall.shape[1] - 1)
                py = np.clip((Hp - (Pc[:, 1] - org[1]) / res).astype(int), 0, Hp - 1)
                l = np.full(len(d), 3, np.uint8)
                l[d <= a.keep_cm] = 1
                l[(d >= a.erase_cm) & inside[py, px]] = 2
                l[(ddoor_img[py, px] <= 30) & (d > a.keep_cm)] = 3
                lab[ys_all[core], xs_all[core]] = l
        tile_share = n_good / max(1, n_tiles)
        ok = n_good >= 3 and int(((lab == 1) | (lab == 2)).sum()) >= 3000
        np.savez_compressed(os.path.join(a.out, name + '.npz'), X=cls, L=lab, floor=fl, median_cm=med * 100, within20=in20, tiles=n_tiles, tiles_fit=n_good, usable=ok)
        ys, xs = np.nonzero(cls != UNK)
        s = (slice(max(0, ys.min() - 5), ys.max() + 6), slice(max(0, xs.min() - 5), xs.max() + 6))
        view = colour(np.where(occ, FREE, cls).astype(np.uint8)[s])
        view[w_m[s] & ~occ[s]] = (235, 190, 120)                  # plan walls, light blue
        for k, c in ((1, (40, 40, 40)), (3, (0, 150, 255)), (2, (40, 40, 230))):
            view[lab[s] == k] = c
        cv2.imwrite(os.path.join(a.out, name + '_view.png'), label(view, f'{name} floor {fl}: black keep, red erase, orange unsure, light blue = plan wall with no lidar'))
        md.append(f"| {name} | {fl} | {cls.shape[1] * res:.0f} x {cls.shape[0] * res:.0f} | {int(occ.sum())} | {med * 100:.1f} | {n_good} of {n_tiles} | "
                  f"{int((lab == 1).sum())} | {int((lab == 2).sum())} | {int((lab == 3).sum())} | {'yes' if ok else 'no: too little fits'} |")
        print(md[-1], flush=True)
    open(os.path.join(a.out, 'FIT.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
