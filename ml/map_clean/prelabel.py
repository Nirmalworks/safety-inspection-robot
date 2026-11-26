#!/usr/bin/env python3
"""Phase 1, step 2: starting labels for the lab maps from agreement between sessions.

    python -m ml.map_clean.prelabel --align data/map_cleaning/vote_all/alignment.json --test lab2d_11_25_7

For every usable lab map, each occupied cell gets a label from the other usable maps moved onto its grid:
keep (1) if most of them have something within 2 cells there, erase (2) if almost none do, unsure (3) otherwise.
The test session never votes on the other maps, so labels used for training are independent of it.
Writes <out>/<name>_prelabel.png (0 = not occupied, 1 keep, 2 erase, 3 unsure, same grid as the map),
<name>_prelabel_view.png, pairs.json (the map-to-map transforms, for carrying corrections over) and PRELABEL.md.
"""
import argparse
import json
import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .common import FREE, OCC, UNK, cells_xy, colour, label, load_map, rot, warp

KEEP, ERASE, UNSURE = 1, 2, 3
K5, K21 = np.ones((5, 5), np.uint8), np.ones((21, 21), np.uint8)


def refine(A, B, R, t, iters=40):
    """Trimmed ICP from a starting transform: A (other map's wall points) onto B (this map's). -> R, t, median distance."""
    tb = cKDTree(B)
    for _ in range(iters):
        d, i = tb.query(A @ R.T + t)
        k = d < np.percentile(d, 80)
        P, M = A[k], B[i[k]]
        pm, mm = P.mean(0), M.mean(0)
        U, _, Vt = np.linalg.svd((P - pm).T @ (M - mm))
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = mm - R @ pm
    d = tb.query(A @ R.T + t)[0]
    return R, t, float(np.median(d))


def _pair(job):
    n, o, A, B, R0, t0 = job
    R, t, med = refine(A, B, R0, t0)
    return (n, o), (R, t, med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', default='data/map_cleaning/vote_all/alignment.json', help='alignment of all maps to one reference (from vote.py)')
    ap.add_argument('--test', default='lab2d_11_25_7')
    ap.add_argument('--out', default='data/map_cleaning/labels')
    ap.add_argument('--keep-frac', type=float, default=0.70)
    ap.add_argument('--erase-frac', type=float, default=0.25)
    ap.add_argument('--max-median', type=float, default=0.12, help='m: a pair of maps that fits worse than this does not vote')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    al = json.load(open(a.align))
    rows = [r for r in al['maps'] if r['kept']]
    T = {r['name']: (rot(math.radians(r['yaw_deg'])), np.array(r['t'])) for r in rows}      # map -> reference frame
    maps = {n: load_map(f'maps/{n}/maps_2d/{n}.yaml') for n in T}
    res = al['res']
    pts = {n: cells_xy(m['cls'] == OCC, res, m['origin']) for n, m in maps.items()}
    jobs = []
    for n in maps:
        Rn, tn = T[n]
        for o in maps:
            if o != n:
                Ro, to = T[o]
                jobs.append((n, o, pts[o], pts[n], Rn.T @ Ro, Rn.T @ (to - tn)))                 # other -> this map, via the reference
    with Pool(a.workers) as pool:
        tr = dict(pool.map(_pair, jobs))

    md = [f'# Starting labels from agreement between sessions ({len(maps)} usable lab maps, test session `{a.test}`)\n',
          f'keep: at least {a.keep_frac * 100:.0f} % of the voting maps have something within 2 cells; erase: at most {a.erase_frac * 100:.0f} %; unsure: in between, '
          f'or fewer than 3 maps cover the cell. A map votes on another only if the pair fits within {a.max_median * 100:.0f} cm (median).\n',
          '| map | voting maps | fit to them, median cm | occupied cells | keep | erase | unsure |', '|---|---|---|---|---|---|---|']
    out_pairs = {}
    for n, m in maps.items():
        raw = m['cls']
        voters = [o for o in maps if o != n and (o != a.test or n == a.test) and tr[(n, o)][2] <= a.max_median]
        sup, cov = np.zeros(raw.shape, np.int16), np.zeros(raw.shape, np.int16)
        for o in voters:
            R, t, _ = tr[(n, o)]
            w = warp(maps[o]['cls'], res, maps[o]['origin'], R, t, raw.shape, m['origin'])
            sup += cv2.dilate((w == OCC).astype(np.uint8), K5)
            cov += cv2.dilate((w != UNK).astype(np.uint8), K21)
        frac = sup / np.maximum(1, cov)
        lab = np.zeros(raw.shape, np.uint8)
        occ = raw == OCC
        lab[occ] = UNSURE
        lab[occ & (cov >= 3) & (frac >= a.keep_frac)] = KEEP
        lab[occ & (cov >= 3) & (frac <= a.erase_frac)] = ERASE
        cv2.imwrite(os.path.join(a.out, f'{n}_prelabel.png'), lab)
        view = colour(np.where(occ, FREE, raw).astype(np.uint8), 3)
        up = lambda k: cv2.resize((lab == k).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) > 0
        view[up(KEEP)] = (40, 40, 40); view[up(UNSURE)] = (0, 150, 255); view[up(ERASE)] = (40, 40, 230)
        cv2.imwrite(os.path.join(a.out, f'{n}_prelabel_view.png'), label(view, f'{n}: black keep, orange unsure, red erase ({len(voters)} maps voted)'))
        meds = [tr[(n, o)][2] * 100 for o in voters]
        md.append(f"| {n}{' (test)' if n == a.test else ''} | {len(voters)} | {np.median(meds) if meds else float('nan'):.1f} | {int(occ.sum())} | "
                  f"{int((lab == KEEP).sum())} | {int((lab == ERASE).sum())} | {int((lab == UNSURE).sum())} |")
        out_pairs[n] = {o: {'R': tr[(n, o)][0].tolist(), 't': tr[(n, o)][1].tolist(), 'median_m': tr[(n, o)][2], 'votes': o in voters} for o in maps if o != n}
    dropped = [r['name'] + ' (' + r['why'] + ')' for r in al['maps'] if not r['kept']]
    md.append('\nNot usable, so not labelled: ' + ', '.join(dropped) + '.')
    json.dump({'test': a.test, 'res': res, 'pairs': out_pairs}, open(os.path.join(a.out, 'pairs.json'), 'w'))
    open(os.path.join(a.out, 'PRELABEL.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
