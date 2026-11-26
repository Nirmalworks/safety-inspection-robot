#!/usr/bin/env python3
"""Carry hand corrections from one labelled lab map over to the same places in the other lab maps.

    python -m ml.map_clean.propagate --src lab2d_11_24_8 --test lab2d_11_25_7

For every other usable map, the hand-labelled map is moved onto its grid (transforms from prelabel.py). An occupied
cell takes the hand label found within 2 cells of the same place (the nearer one if both keep and erase are there);
where no hand label is near, the starting label from agreement between sessions stays.
Writes <labels>/<name>_proplabel.png. The test map is never a source and never a target.
"""
import argparse
import json
import os

import cv2
import numpy as np

from .common import OCC, load_map, warp

K5 = np.ones((5, 5), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='+', default=['lab2d_11_24_8'])
    ap.add_argument('--test', default='lab2d_11_25_7')
    ap.add_argument('--labels', default='data/map_cleaning/labels')
    ap.add_argument('--max-median', type=float, default=0.12)
    a = ap.parse_args()
    pj = json.load(open(os.path.join(a.labels, 'pairs.json')))
    res, pairs = pj['res'], pj['pairs']
    src = {s: (load_map(f'maps/{s}/maps_2d/{s}.yaml'), cv2.imread(os.path.join(a.labels, s + '_label.png'), cv2.IMREAD_UNCHANGED)) for s in a.src}
    print('| map | from hand labels: keep | erase | starting labels kept: keep | erase | unsure | changed from the starting label |\n|---|---|---|---|---|---|---|')
    for n in sorted(pairs):
        if n == a.test or n in src:
            continue
        m = load_map(f'maps/{n}/maps_2d/{n}.yaml')
        pre = cv2.imread(os.path.join(a.labels, n + '_prelabel.png'), cv2.IMREAD_UNCHANGED)
        if pre is None:
            continue                                      # a map with no starting labels (it found no partners)
        occ = m['cls'] == OCC
        dk = np.full(occ.shape, 1e9, np.float32); de = np.full(occ.shape, 1e9, np.float32)
        for s, (sm, sl) in src.items():
            p = pairs[n].get(s)
            if p is None or p['median_m'] > a.max_median:
                continue
            w = warp((sl + 10).astype(np.uint8), res, sm['origin'], np.array(p['R']), np.array(p['t']), occ.shape, m['origin'])
            dk = np.minimum(dk, cv2.distanceTransform((w != 11).astype(np.uint8), cv2.DIST_L2, 3))
            de = np.minimum(de, cv2.distanceTransform((w != 12).astype(np.uint8), cv2.DIST_L2, 3))
        out = pre.copy()
        near_k, near_e = occ & (dk <= 2.5), occ & (de <= 2.5)
        out[near_k & (dk <= de)] = 1
        out[near_e & (de < dk)] = 2
        cv2.imwrite(os.path.join(a.labels, n + '_proplabel.png'), out)
        hand = near_k | near_e
        print(f"| {n} | {int((hand & (out == 1)).sum())} | {int((hand & (out == 2)).sum())} | {int((~hand & (out == 1)).sum())} | {int((~hand & (out == 2)).sum())} | {int((out == 3).sum())} | {int((out != pre).sum())} |")


if __name__ == '__main__':
    main()
