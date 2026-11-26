#!/usr/bin/env python3
"""Two follow-up checks, both on the desktop:

    python -m ml.map_clean.eval_more thresholds     # choose each model's erase threshold with a margin, re-score on the test map
    python -m ml.map_clean.eval_more stata          # run the models on the MIT Stata maps (a building none of them has seen)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

from .common import OCC
from .erase import lab_pair, load_model, pick_threshold, predict, scores
from .show_real import speckle_filter

M = 'data/map_cleaning/models2'
LABELS = 'data/map_cleaning/labels'


def thresholds(val='lab2d_11_25_5', test='lab2d_11_25_7'):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cv_, lv, src_v = lab_pair(val, LABELS)
    ct, lt, _ = lab_pair(test, LABELS)
    print(f'threshold chosen on {val} (labels: {src_v}), scored on {test} (hand labels)\n')
    print('| model | target on validation | threshold | test: structure kept | test: clutter removed |\n|---|---|---|---|---|')
    out = {}
    for m in ('lab', 'general_ft'):
        G, _ = load_model(os.path.join(M, m + '.pt'), dev)
        pv, pt = predict(cv_, G, dev), predict(ct, G, dev)
        for target in (0.95, 0.96, 0.97, 0.98):
            thr, _, _ = pick_threshold([pv], [lv], target)
            k, e = scores(pt, lt, thr)
            out[f'{m}@{target}'] = {'thr': thr, 'kept': k, 'removed': e}
            print(f'| {m} | keep {target * 100:.0f} % | {thr:.2f} | {k * 100:.1f} % | {e * 100:.1f} % |')
    json.dump(out, open('data/map_cleaning/phase2/thresholds_margin.json', 'w'), indent=1)


def stata(pattern='data/map_cleaning/stata/pairs_seg/*.npz'):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = [f for f in sorted(glob.glob(pattern)) if ((np.load(f)['L'] == 1) | (np.load(f)['L'] == 2)).sum() >= 1500]
    data = [(os.path.basename(f)[:-4], np.load(f)['X'], np.load(f)['L'], str(np.load(f)['floor'])) for f in files]
    nk, ne = sum(int((l == 1).sum()) for _, _, l, _ in data), sum(int((l == 2).sum()) for _, _, l, _ in data)
    print(f'{len(data)} Stata pieces with at least 1500 labelled cells, floors {sorted(set(d[3] for d in data))}: {nk} keep cells, {ne} erase cells (labels from the floor plans)\n')
    res = {}
    def add(name, ps, thr=None):
        k = np.concatenate([p[l == 1] for p, (_, _, l, _) in zip(ps, data)]); e = np.concatenate([p[l == 2] for p, (_, _, l, _) in zip(ps, data)])
        row = {}
        if thr is not None:
            row['kept'], row['removed'] = float((k <= thr).mean()), float((e > thr).mean())
        best = pick_threshold(ps, [d[2] for d in data], 0.95)
        row['removed_at_95_kept'] = best[1]
        # ranking quality: chance that a random erase cell scores higher than a random keep cell
        rng = np.random.default_rng(0); a, b = rng.choice(e, 200000), rng.choice(k, 200000)
        row['auc'] = float((a > b).mean() + 0.5 * (a == b).mean())
        res[name] = row
    sp = []
    for _, x, l, _ in data:
        f = speckle_filter(x); sp.append(((x == OCC) & (f != OCC)).astype(np.float32))
    add('speckle filter', sp, 0.5)
    for m in ('lab', 'general_sim', 'general_ft'):
        G, ck = load_model(os.path.join(M, m + '.pt'), dev)
        add(m, [predict(x, G, dev) for _, x, _, _ in data], ck['thr'])
    print('| method | structure kept (own threshold) | clutter removed (own threshold) | clutter removed when keeping 95 % | ranking score (0.5 = chance, 1 = perfect) |\n|---|---|---|---|---|')
    for k, r in res.items():
        print(f"| {k} | {r['kept'] * 100:.1f} % | {r['removed'] * 100:.1f} % | {r['removed_at_95_kept'] * 100:.1f} % | {r['auc']:.3f} |")
    json.dump({'pieces': [d[0] for d in data], 'keep_cells': nk, 'erase_cells': ne, 'results': res}, open('data/map_cleaning/phase2/stata_test.json', 'w'), indent=1)


if __name__ == '__main__':
    thresholds() if sys.argv[1] == 'thresholds' else stata()
