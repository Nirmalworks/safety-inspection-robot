#!/usr/bin/env python3
"""Line up every pair of maps of one place and write the starting labels from agreement between them.

    python -m ml.map_clean.align_place data/map_cleaning/unseen/usc_sal

PLACE holds maps/<name>/maps_2d/<name>.yaml and labels/. Each pair is aligned directly (coarse search over rotation
and shift, then trimmed ICP: common.align), with no reference map. Writes labels/pairs.json in the format of
prelabel.py (pairs[a][b] moves map b onto map a) and labels/<name>_prelabel.png: keep where at least 70 % of the maps
that fit have something within 2 cells, erase where at most 25 % do, unsure otherwise or where fewer than 3 of them
(all of them, when fewer than 3 fit) cover the cell. A pair that fits worse than 12 cm (median) does not vote.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from .common import OCC, UNK, align, cells_xy, load_map, warp
from .prelabel import refine

K5, K21 = np.ones((5, 5), np.uint8), np.ones((21, 21), np.uint8)


def _pair(job):
    a, b, A, B = job                                   # b's wall points A onto a's wall points B
    best = None
    for search, step in ((5.0, 1), (10.0, 3)):         # drives of one building can start far apart; the wide search uses a third of the points
        R, t, _ = align(A[::step], B, search=search)
        R, t, med = refine(A, B, R, t)
        if best is None or med < best[2]:
            best = (R, t, med)
        if med <= 0.12:
            break
    return (a, b), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('place'); ap.add_argument('--max-median', type=float, default=0.12)
    ap.add_argument('--keep-frac', type=float, default=0.70); ap.add_argument('--erase-frac', type=float, default=0.25)
    a = ap.parse_args()
    lab_dir = os.path.join(a.place, 'labels'); os.makedirs(lab_dir, exist_ok=True)
    maps = {m['name']: m for m in (load_map(y) for y in sorted(glob.glob(os.path.join(a.place, 'maps', '*', 'maps_2d', '*.yaml'))))}
    names = sorted(maps); res = maps[names[0]]['res']
    pts = {n: cells_xy(m['cls'] == OCC, res, m['origin']) for n, m in maps.items()}
    jobs = [(x, y, pts[y], pts[x]) for i, x in enumerate(names) for y in names[i + 1:]]
    with Pool(min(16, len(jobs))) as pool:
        tr = dict(pool.map(_pair, jobs))
    for (x, y), (R, t, med) in list(tr.items()):       # the reverse direction: refine from the inverse
        Ri, ti, medi = refine(pts[x], pts[y], R.T, -R.T @ t)
        tr[(y, x)] = (Ri, ti, medi)
    print(f'{a.place}: {len(names)} maps\n| map | maps that fit | fit, median cm | occupied | keep | erase | unsure |\n|---|---|---|---|---|---|---|')
    out = {}
    for n in names:
        raw = maps[n]['cls']
        voters = [o for o in names if o != n and tr[(n, o)][2] <= a.max_median]
        sup, cov = np.zeros(raw.shape, np.int16), np.zeros(raw.shape, np.int16)
        for o in voters:
            R, t, _ = tr[(n, o)]
            w = warp(maps[o]['cls'], res, maps[o]['origin'], R, t, raw.shape, maps[n]['origin'])
            sup += cv2.dilate((w == OCC).astype(np.uint8), K5); cov += cv2.dilate((w != UNK).astype(np.uint8), K21)
        frac = sup / np.maximum(1, cov)
        need = max(1, min(3, len(voters)))               # with one or two other drives, every one of them must cover the cell
        lab = np.zeros(raw.shape, np.uint8); occ = raw == OCC
        lab[occ] = 3
        lab[occ & (cov >= need) & (frac >= a.keep_frac)] = 1
        lab[occ & (cov >= need) & (frac <= a.erase_frac)] = 2
        cv2.imwrite(os.path.join(lab_dir, f'{n}_prelabel.png'), lab)
        meds = [tr[(n, o)][2] * 100 for o in names if o != n]
        print(f"| {n} | {len(voters)} of {len(names) - 1} | {', '.join(f'{v:.1f}' for v in meds)} | {int(occ.sum())} | {int((lab == 1).sum())} | {int((lab == 2).sum())} | {int((lab == 3).sum())} |")
        out[n] = {o: {'R': tr[(n, o)][0].tolist(), 't': tr[(n, o)][1].tolist(), 'median_m': tr[(n, o)][2], 'votes': o in voters} for o in names if o != n}
    json.dump({'res': res, 'pairs': out}, open(os.path.join(lab_dir, 'pairs.json'), 'w'))


if __name__ == '__main__':
    main()
