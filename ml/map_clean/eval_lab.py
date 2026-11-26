#!/usr/bin/env python3
"""Score every cleaning method on the hand-labelled test session.

    python -m ml.map_clean.eval_lab --models lab general_sim --extra "earlier model=PATH.pgm"

Reference: <labels>/<test>_label.png, corrected by hand (1 keep, 2 erase, 3 unsure; unsure cells are not scored).
structure kept  = share of hand-keep cells that are still occupied in the output
clutter removed = share of hand-erase cells that are no longer occupied
For a method that redraws walls (the earlier model, the vote) a keep cell also counts as kept if the output has a wall
within 1 cell; that column is shown separately. Writes <out>/<test>__<method>.pgm + .yaml, tables.md, results.json, picture.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import yaml

from .common import FREE, OCC, PGM, UNK, colour, label, pgm_to_cls
from .erase import apply_erase, load_model, predict
from .show_real import speckle_filter

K3 = np.ones((3, 3), np.uint8)


def score(out, raw, lab):
    k, e = lab == 1, lab == 2
    occ = out == OCC
    near = cv2.dilate(occ.astype(np.uint8), K3) > 0
    return {'structure_kept': float(occ[k].mean()), 'structure_kept_1cell': float(near[k].mean()), 'clutter_removed': float((~occ[e]).mean()),
            'occupied_cells': int(occ.sum()), 'added_cells': int((occ & (raw != OCC)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='lab2d_11_25_7')
    ap.add_argument('--models', nargs='*', default=['lab'])
    ap.add_argument('--model-dir', default='data/map_cleaning/models2')
    ap.add_argument('--labels', default='data/map_cleaning/labels')
    ap.add_argument('--extra', nargs='*', default=[], help='name=path.pgm of a map cleaned by another method, on the same grid')
    ap.add_argument('--out', default='data/map_cleaning/phase1')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = a.test
    raw = pgm_to_cls(cv2.imread(f'maps/{n}/maps_2d/{n}.pgm', cv2.IMREAD_GRAYSCALE))
    lab = cv2.imread(os.path.join(a.labels, n + '_label.png'), cv2.IMREAD_UNCHANGED)
    pre = cv2.imread(os.path.join(a.labels, n + '_prelabel.png'), cv2.IMREAD_UNCHANGED)
    outs = {'raw (no cleaning)': raw, 'speckle filter': speckle_filter(raw)}
    kill = lambda m: apply_erase(raw, m.astype(np.float32), 0.5)
    outs['session agreement: erase what few sessions have'] = kill(pre == 2)
    outs['session agreement: erase all but what most sessions have'] = kill((pre == 2) | (pre == 3))
    for e in a.extra:
        name, path = e.split('=', 1)
        outs[name] = pgm_to_cls(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    curves = {}
    for m in a.models:
        G, ck = load_model(os.path.join(a.model_dir, m + '.pt'), dev)
        p = predict(raw, G, dev)
        thr = ck['thr']
        outs[f'{m} (threshold {thr:.2f} from validation)'] = apply_erase(raw, p, thr)
        curves[m] = [{'thr': float(t), **{k: v for k, v in score(apply_erase(raw, p, t), raw, lab).items() if k in ('structure_kept', 'clutter_removed')}} for t in np.linspace(0.05, 0.95, 19)]
    res = {k: score(o, raw, lab) for k, o in outs.items()}
    md = [f'# Test session `{n}` against hand labels ({int((lab == 1).sum())} keep cells, {int((lab == 2).sum())} erase cells, {int((lab == 3).sum())} left unsure and not scored)\n',
          '| method | structure kept | same, wall within 1 cell | clutter removed | occupied cells | cells added |', '|---|---|---|---|---|---|']
    for k, r in res.items():
        md.append(f"| {k} | {r['structure_kept'] * 100:.1f} % | {r['structure_kept_1cell'] * 100:.1f} % | {r['clutter_removed'] * 100:.1f} % | {r['occupied_cells']} | {r['added_cells']} |")
    for m, c in curves.items():
        md += [f'\n## {m}: the trade-off as the erase threshold moves\n', '| threshold | structure kept | clutter removed |', '|---|---|---|']
        md += [f"| {r['thr']:.2f} | {r['structure_kept'] * 100:.1f} % | {r['clutter_removed'] * 100:.1f} % |" for r in c]
    y = yaml.safe_load(open(f'maps/{n}/maps_2d/{n}.yaml'))
    tiles = []
    ys, xs = np.nonzero(raw != UNK)
    s = (slice(max(0, ys.min() - 4), ys.max() + 5), slice(max(0, xs.min() - 4), xs.max() + 5))
    ref = colour(np.where(raw == OCC, FREE, raw).astype(np.uint8)[s], 2)
    up = lambda mk: cv2.resize(mk[s].astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) > 0
    ref[up(lab == 1)] = (40, 40, 40); ref[up(lab == 3)] = (0, 150, 255); ref[up(lab == 2)] = (40, 40, 230)
    tiles.append(label(ref, 'hand labels: black keep, red erase, orange unsure'))
    for k, o in outs.items():
        stem = os.path.join(a.out, n + '__' + ''.join(ch if ch.isalnum() else '_' for ch in k).strip('_'))
        cv2.imwrite(stem + '.pgm', PGM[o])
        yy = dict(y); yy['image'] = os.path.basename(stem) + '.pgm'
        yaml.safe_dump(yy, open(stem + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
        img = colour(o[s], 2)
        img[up((lab == 1) & (cv2.dilate((o == OCC).astype(np.uint8), K3) == 0))] = (40, 40, 230)        # structure lost
        img[up((lab == 2) & (o == OCC))] = (0, 150, 255)                                                # clutter left
        r = res[k]
        tiles.append(label(img, f"{k}: kept {r['structure_kept'] * 100:.0f} %, removed {r['clutter_removed'] * 100:.0f} % (red = structure lost, orange = clutter left)", size=0.5))
    cv2.imwrite(os.path.join(a.out, f'test_{n}.png'), np.vstack(tiles))
    json.dump({'test': n, 'results': res, 'curves': curves}, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'tables.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
