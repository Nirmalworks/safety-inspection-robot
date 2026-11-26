#!/usr/bin/env python3
"""Methods without ML, and the earlier redraw model, on every hand-labelled lab map.

    python -m ml.map_clean.baselines4

Per map T: speckle filter; session agreement (cautious / aggressive) from T's starting labels; the vote of all other
usable sessions moved onto T's grid (T left out; transforms from prelabel.py); the earlier model and the earlier model
with the vote's strong walls put back. Scored against T's hand labels with score2.py.
"""
import json
import os

import cv2
import numpy as np
import torch

from .clean_map import clean, load_model as load_redraw
from .common import FREE, OCC, UNK, load_map, warp
from .erase import apply_erase
from .score2 import score2 as score
from .show_real import speckle_filter
from .vote import skeletonize

LB = 'data/map_cleaning/labels'
MAPS = ['lab2d_11_24_3', 'lab2d_11_24_8', 'lab2d_11_25_5', 'lab2d_11_25_7']


def vote_on(T, tm, pairs, res):
    stack = []
    for s, p in pairs[T].items():
        if p['median_m'] > 0.12 or not os.path.exists(f'maps/{s}/maps_2d/{s}.yaml'):
            continue
        sm = load_map(f'maps/{s}/maps_2d/{s}.yaml')
        w = warp((sm['cls'] + 10).astype(np.uint8), res, sm['origin'], np.array(p['R']), np.array(p['t']), tm['cls'].shape, tm['origin'])
        o = np.full(w.shape, UNK, np.uint8); m = w >= 10; o[m] = w[m] - 10
        stack.append(o)
    stack = np.stack(stack); K = len(stack)
    k3, k21 = np.ones((3, 3), np.uint8), np.ones((21, 21), np.uint8)
    covered = np.stack([cv2.dilate((s != UNK).astype(np.uint8), k21) for s in stack]).astype(bool)
    n_cov = covered.sum(0)
    occ_sup = np.stack([cv2.dilate((s == OCC).astype(np.uint8), k3) for s in stack]).sum(0)
    n_free = (stack == FREE).sum(0)
    enough = n_cov >= min(3, K)
    thick = (occ_sup >= np.maximum(2, np.ceil(0.5 * n_cov))) & enough
    wall = skeletonize(thick)
    vote = np.full(tm['cls'].shape, UNK, np.uint8)
    vote[(n_free >= 0.4 * n_cov) & enough] = FREE
    vote[thick & ~wall & (n_free < 0.4 * n_cov)] = UNK
    vote[wall] = OCC
    strong = (occ_sup >= 0.8 * n_cov) & enough & (n_cov >= max(3, int(0.6 * K)))
    return vote, strong, K


def main():
    pj = json.load(open(f'{LB}/pairs.json')); pairs, res = pj['pairs'], pj['res']
    G = load_redraw('data/map_cleaning/models/unet_ce.pt', 'cpu')
    res_all = {}
    for T in MAPS:
        tm = load_map(f'maps/{T}/maps_2d/{T}.yaml'); raw = tm['cls']
        lab = cv2.imread(f'{LB}/{T}_label.png', -1); pre = cv2.imread(f'{LB}/{T}_prelabel.png', -1)
        kill = lambda m: apply_erase(raw, m.astype(np.float32), 0.5)
        vote, strong, K = vote_on(T, tm, pairs, res)
        with torch.no_grad():
            outs = {'Speckle filter': speckle_filter(raw), 'Session agreement, cautious': kill(pre == 2), 'Session agreement, aggressive': kill((pre == 2) | (pre == 3)),
                    f'Vote of the other sessions': vote, 'Earlier model': clean(raw, G, 'cpu'), 'Earlier model + guardrail': clean(raw, G, 'cpu', guard=strong)}
        res_all[T] = {k: score(o, raw, lab) for k, o in outs.items()}
        res_all[T]['_sessions_in_vote'] = K
    print('| method | structure kept within 10 cm: mean (range) | exact | clutter removed, none within 5 cm: mean (range) | exact | invented cells | added floor m2 |\n|---|---|---|---|---|---|---|')
    for k in [k for k in res_all[MAPS[0]] if k[0] != '_']:
        a = np.array([[res_all[T][k][f] for f in ('kept', 'kept_exact', 'removed', 'removed_exact', 'invented', 'added_floor_m2')] for T in MAPS])
        print(f'| {k} | {a[:, 0].mean() * 100:.1f} % ({a[:, 0].min() * 100:.1f} to {a[:, 0].max() * 100:.1f}) | {a[:, 1].mean() * 100:.1f} % | {a[:, 2].mean() * 100:.1f} % ({a[:, 2].min() * 100:.1f} to {a[:, 2].max() * 100:.1f}) | {a[:, 3].mean() * 100:.1f} % | {a[:, 4].mean():.0f} | {a[:, 5].mean():.1f} |')
    json.dump(res_all, open('data/map_cleaning/folds/baselines4.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
