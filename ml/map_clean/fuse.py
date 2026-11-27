#!/usr/bin/env python3
"""Evidence from all sessions of one room, brought onto one map's grid.

stack(target)      every other usable session warped onto the target's grid (transforms from prelabel.py)
persistence(...)   per cell: share of the covering sessions that have a wall within one cell, and that show floor
Used by best_map.py and by the test of the fusion rule (python -m ml.map_clean.fuse).
"""
import json
import os

import cv2
import numpy as np

from .common import FREE, OCC, UNK, load_map, warp

LB = 'data/map_cleaning/labels'
K3, K21 = np.ones((3, 3), np.uint8), np.ones((21, 21), np.uint8)


def day(name):
    return name.split('_')[2] if name.startswith('lab2d_') else name


def stack(target, labels=LB, max_median=0.12, other_days_only=False, extra=None):
    """-> (target map dict, [(name, class map on the target grid)], pairs of the target). `extra`: {name: array on that session's grid} warped too."""
    pj = json.load(open(os.path.join(labels, 'pairs.json'))); pairs, res = pj['pairs'][target], pj['res']
    tm = load_map(f'maps/{target}/maps_2d/{target}.yaml')
    out, ex = [], []
    for s, p in sorted(pairs.items()):
        if p['median_m'] > max_median or not os.path.exists(f'maps/{s}/maps_2d/{s}.yaml') or (other_days_only and day(s) == day(target)):
            continue
        sm = load_map(f'maps/{s}/maps_2d/{s}.yaml'); R, t = np.array(p['R']), np.array(p['t'])
        w = warp((sm['cls'] + 10).astype(np.uint8), res, sm['origin'], R, t, tm['cls'].shape, tm['origin'])
        o = np.full(w.shape, UNK, np.uint8); m = w >= 10; o[m] = w[m] - 10
        out.append((s, o))
        if extra is not None and s in extra:
            e = warp((np.clip(extra[s], 0, 1) * 200 + 10).astype(np.uint8), res, sm['origin'], R, t, tm['cls'].shape, tm['origin']).astype(np.float32)
            ex.append((s, np.where(e >= 10, (e - 10) / 200, np.nan)))
    return tm, out, ex


def persistence(maps):
    """-> (share of covering sessions with a wall within 1 cell, share showing floor, number of covering sessions)"""
    cov = np.stack([cv2.dilate((m != UNK).astype(np.uint8), K21) for _, m in maps]).astype(bool).sum(0)
    occ = np.stack([cv2.dilate((m == OCC).astype(np.uint8), K3) for _, m in maps]).sum(0)
    fre = np.stack([(m == FREE) for _, m in maps]).sum(0)
    n = np.maximum(cov, 1)
    return occ / n, fre / n, cov


def main():
    import torch
    from .erase import apply_erase, load_model, predict
    from .score2 import score2
    D24, D25 = ['lab2d_11_24_3', 'lab2d_11_24_8'], ['lab2d_11_25_5', 'lab2d_11_25_7']
    rows = {}
    for T in D24 + D25:
        lab = cv2.imread(f'{LB}/{T}_label.png', -1)
        ps = []
        for f in (('ftfold_B1', 'ftfold_B2') if T in D24 else ('ftfold_A1', 'ftfold_A2')):       # models that never saw this day
            G, ck = load_model(f'data/map_cleaning/models2/{f}.pt', 'cuda')
            tm, others, _ = stack(T, other_days_only=True)
            with torch.no_grad():
                ps.append((predict(tm['cls'], G, 'cuda'), ck['thr']))
        raw = tm['cls']; pers, fre, cov = persistence(others)
        print(T, 'other-day sessions:', len(others), flush=True)
        for lo in (0.0, 0.2, 0.34, 0.5):
            for hi in (0.7, 0.85, 1.01):
                r = []
                for p, thr in ps:
                    erase = ((p > thr) & (pers < hi)) | ((pers < lo) & (cov >= 3))
                    s = score2(apply_erase(raw, erase.astype(np.float32), 0.5), raw, lab); r.append((s['kept'], s['removed']))
                rows.setdefault((lo, hi), []).append(np.mean(r, 0))
    print('| erase when | structure kept: mean (worst map) | clutter removed: mean (worst map) |\n|---|---|---|')
    for (lo, hi), v in rows.items():
        v = np.array(v) * 100
        name = ('model says erase' if hi > 1 else f'model says erase and fewer than {hi:.0%} of other-day sessions have a wall there') + ('' if lo == 0 else f', or fewer than {lo:.0%} have one')
        print(f'| {name} | {v[:, 0].mean():.1f} % ({v[:, 0].min():.1f}) | {v[:, 1].mean():.1f} % ({v[:, 1].min():.1f}) |')


if __name__ == '__main__':
    main()
