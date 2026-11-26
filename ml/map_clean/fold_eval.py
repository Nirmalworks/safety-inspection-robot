#!/usr/bin/env python3
"""Score the leave-one-day-out rotation: each lab model on the hand-labelled maps of the day it never saw.

    python -m ml.map_clean.fold_eval [--models data/map_cleaning/models2] [--extra lab=same-day ...]

Per model and test map: structure kept / clutter removed at the threshold chosen on that model's validation map,
clutter removed at exactly 95 % structure kept on the test map (no threshold luck), and the ranking score (AUC).
"""
import argparse
import json

import numpy as np
import torch

from .erase import lab_pair, load_model, predict, scores

LB = 'data/map_cleaning/labels'
D24, D25 = ['lab2d_11_24_3', 'lab2d_11_24_8'], ['lab2d_11_25_5', 'lab2d_11_25_7']
FOLDS = {'fold_A1': D25, 'fold_A2': D25, 'fold_B1': D24, 'fold_B2': D24}


def fixed(p, lab, keep=0.95):
    pk, pe = np.sort(p[lab == 1]), p[lab == 2]
    thr = pk[min(len(pk) - 1, int(np.ceil(keep * len(pk))) - 1)]          # smallest threshold keeping `keep` of structure
    r = np.argsort(np.argsort(np.r_[pk, pe]))                              # AUC by ranks (ties are rare with float scores)
    auc = (r[len(pk):].sum() - len(pe) * (len(pe) - 1) / 2) / (len(pk) * len(pe))
    return float((pe > thr).mean()), float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='data/map_cleaning/models2')
    ap.add_argument('--extra', nargs='*', default=[], help='other models to score on all four maps, e.g. lab')
    ap.add_argument('--out', default='data/map_cleaning/folds/results.json')
    a = ap.parse_args()
    dev = 'cpu'
    maps = {n: lab_pair(n, LB) for n in D24 + D25}
    rows = []
    print('| model | test map | threshold from validation | structure kept | clutter removed | clutter removed at exactly 95 % kept | ranking score |\n|---|---|---|---|---|---|---|')
    for m, tests in list(FOLDS.items()) + [(e, D24 + D25) for e in a.extra]:
        G, ck = load_model(f'{a.models}/{m}.pt', dev)
        for n in tests:
            cls, lab, src = maps[n]
            assert src == 'label'
            with torch.no_grad():
                p = predict(cls, G, dev)
            k, e = scores(p, lab, ck['thr'])
            f, auc = fixed(p, lab)
            rows.append(dict(model=m, map=n, thr=ck['thr'], kept=k, removed=e, removed_at_95=f, auc=auc))
            print(f"| {m} | {n} | {ck['thr']:.2f} | {100 * k:.1f} % | {100 * e:.1f} % | {100 * f:.1f} % | {auc:.3f} |", flush=True)
    fr = [r for r in rows if r['model'].startswith('fold_')]
    for key, t in (('kept', 'structure kept'), ('removed', 'clutter removed'), ('removed_at_95', 'clutter removed at 95 % kept'), ('auc', 'ranking score')):
        v = np.array([r[key] for r in fr]) * (1 if key == 'auc' else 100)
        print(f'rotation, {t}: mean {v.mean():.1f} range {v.min():.1f} to {v.max():.1f}' if key != 'auc' else f'rotation, {t}: mean {v.mean():.3f} range {v.min():.3f} to {v.max():.3f}')
    json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
