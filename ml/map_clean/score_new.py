#!/usr/bin/env python3
"""Final scoring on maps that were never used for the vote map, the noise library or training.

    python -m ml.map_clean.score_new --maps lab2d_11_25_2 ... --hand PATH/lab_common_clean_edit.yaml [--old HANDBACK_DIR]

The new maps have no ground truth, so three measures are reported, each with its limit:
1. Against the hand-cleaned map of the same room, moved onto each new map by rigid alignment (approximate:
   the alignment error is printed; scores at 1 and at 3 cells).
2. Persistence across the other new maps: a raw wall cell that at least 4 of the other 5 maps also have
   (within 2 cells) is "persistent" and should be kept; one that at most 1 other map has is "transient" and
   should go. Furniture that did not move all day counts as persistent.
3. Without a reference: wall cells, floor area, number of small occupied blobs (under 12 cells).
`--old` adds the models of the first run (its own code is loaded from that folder).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import torch

from .clean_map import clean, load_model
from .common import FREE, OCC, UNK, align, cells_xy, colour, label, load_map, wall_scores, warp
from .evaluate import f1, near
from .show_real import speckle_filter

K5 = np.ones((5, 5), np.uint8)


def _align(job):
    key, A, B = job
    R, t, d = align(A, B)
    return key, (R, t, float(np.median(d)), float(np.mean(d < 0.10)))


def small_blobs(occ, max_cells=12):
    n, _, st, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
    return int((st[1:, cv2.CC_STAT_AREA] < max_cells).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--models', nargs='+', default=['unet_ce_synth', 'unet_ce', 'unet_gan'])
    ap.add_argument('--dir', default='data/map_cleaning')
    ap.add_argument('--hand', required=True)
    ap.add_argument('--old', default='', help='hand-back folder of the first run (models/, vote/, ml/map_clean/)')
    ap.add_argument('--out', default='data/map_cleaning/final')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    maps = {n: load_map(f'maps/{n}/maps_2d/{n}.yaml') for n in a.maps}
    hand = load_map(a.hand)
    z = np.load(os.path.join(a.dir, 'vote', 'stack.npz'), allow_pickle=True)
    vote_origin, strong, res = tuple(z['origin']), z['strong'].astype(np.uint8), float(z['res'])
    ref = z['stack'][0]                                   # the reference lab map, on the vote grid
    pts = {n: cells_xy(m['cls'] == OCC, res, m['origin']) for n, m in maps.items()}
    hand_pts, ref_pts = cells_xy(hand['cls'] == OCC, res, hand['origin']), cells_xy(ref == OCC, res, vote_origin)

    jobs = [((n, 'hand'), hand_pts, pts[n]) for n in maps] + [((n, 'vote'), ref_pts, pts[n]) for n in maps]
    jobs += [((n, o), pts[o], pts[n]) for n in maps for o in maps if o != n]
    with Pool(min(32, len(jobs))) as pool:
        al = dict(pool.map(_align, jobs))                 # every transform takes "other" onto the new map

    models = {m: load_model(os.path.join(a.dir, 'models', m + '.pt'), dev) for m in a.models}
    old = {}
    if a.old:
        sys.path.insert(0, os.path.join(a.old, 'ml', 'map_clean'))
        import clean_map as old_cm
        import mapio as old_io
        import model as old_model
        for m in ('unet_synth', 'unet_real', 'gan_real'):
            old[m] = old_model.load_model(os.path.join(a.old, 'models', m + '.pt'), dev)[0]

    res_all, order = {}, None
    for n, m in maps.items():
        raw = m['cls']
        to_n = lambda img, origin, key: warp(img, res, origin, al[(n, key)][0], al[(n, key)][1], raw.shape, m['origin'])
        hand_n = to_n(hand['cls'], hand['origin'], 'hand') == OCC
        guard = near(to_n(strong, vote_origin, 'vote') == 1)
        support = sum(cv2.dilate((to_n(maps[o]['cls'], maps[o]['origin'], o) == OCC).astype(np.uint8), K5) for o in maps if o != n)
        persistent, transient = (raw == OCC) & (support >= len(maps) - 2), (raw == OCC) & (support <= 1)

        outs = {'raw (no cleaning)': raw, 'speckle filter': speckle_filter(raw)}
        for k, G in models.items():
            outs['run 2: ' + k] = clean(raw, G, dev)
            outs['run 2: ' + k + ' + guardrail'] = clean(raw, G, dev, guard)
        if old:
            lab, meta = old_io.load_map(m['path'])
            for k, G in old.items():
                o = old_cm.clean_labels(lab, G, dev)
                outs['run 1: ' + k] = o
                outs['run 1: ' + k + ' + guardrail'] = old_cm.apply_guard(lab, o, meta, os.path.join(a.old, 'vote'))[0]
        order = list(outs)
        rows = {}
        for k, o in outs.items():
            occ = o == OCC
            r1, p1 = wall_scores(occ, hand_n, 1)
            r3, p3 = wall_scores(occ, hand_n, 3)
            rows[k] = {'hand_recall_1': r1, 'hand_precision_1': p1, 'hand_f1_1': f1(r1, p1), 'hand_recall_3': r3, 'hand_precision_3': p3, 'hand_f1_3': f1(r3, p3),
                       'persistent_kept': float((persistent & near(occ)).sum() / max(1, persistent.sum())),
                       'transient_removed': float((transient & ~occ).sum() / max(1, transient.sum())),
                       'wall_cells': int(occ.sum()), 'floor_m2': float((o == FREE).sum() * res * res), 'small_blobs': small_blobs(occ)}
            cv2.imwrite(os.path.join(a.out, f"{n}__{k.replace(': ', '_').replace(' + ', '_plus_').replace(' ', '_').replace('(', '').replace(')', '')}.pgm"),
                        np.array([254, 0, 205], np.uint8)[o])
        res_all[n] = {'shape': list(raw.shape), 'raw_wall_cells': int((raw == OCC).sum()), 'persistent_cells': int(persistent.sum()), 'transient_cells': int(transient.sum()),
                      'hand_align_median_cm': al[(n, 'hand')][2] * 100, 'hand_align_within_10cm': al[(n, 'hand')][3],
                      'vote_align_median_cm': al[(n, 'vote')][2] * 100,
                      'other_maps_align_median_cm': [al[(n, o)][2] * 100 for o in maps if o != n], 'rows': rows}
        ys, xs = np.nonzero(raw != UNK)
        s = (slice(max(0, ys.min() - 6), ys.max() + 7), slice(max(0, xs.min() - 6), xs.max() + 7))
        tall = raw[s].shape[0] > raw[s].shape[1]
        show = [k for k in order if 'guardrail' not in k] + ['run 2: unet_ce + guardrail']
        tiles = [label(colour(np.rot90(outs[k][s]) if tall else outs[k][s], 2), k if k != 'raw (no cleaning)' else 'raw ' + n) for k in show]
        tiles.insert(1, label(colour(np.rot90(np.where(hand_n, OCC, UNK).astype(np.uint8)[s]) if tall else np.where(hand_n, OCC, UNK).astype(np.uint8)[s], 2), 'hand-cleaned walls, moved onto this map'))
        cv2.imwrite(os.path.join(a.out, f'final_{n}.png'), np.vstack(tiles))

    keys = ['hand_precision_1', 'hand_recall_1', 'hand_f1_1', 'hand_f1_3', 'persistent_kept', 'transient_removed', 'wall_cells', 'floor_m2', 'small_blobs']
    md = [f'# Final scoring on {len(maps)} maps recorded after training (never used for the vote map, the noise library or training)\n',
          '## Alignment of the references onto each new map\n', '| map | size (cells) | raw wall cells | persistent | transient | hand-cleaned map: median distance | within 10 cm | other new maps: median distance |', '|---|---|---|---|---|---|---|---|']
    for n, r in res_all.items():
        md.append(f"| {n} | {r['shape'][1]} x {r['shape'][0]} | {r['raw_wall_cells']} | {r['persistent_cells']} | {r['transient_cells']} | {r['hand_align_median_cm']:.1f} cm | "
                  f"{r['hand_align_within_10cm'] * 100:.0f} % | {min(r['other_maps_align_median_cm']):.1f} to {max(r['other_maps_align_median_cm']):.1f} cm |")
    md += [f'\n## Mean over the {len(maps)} maps (standard deviation over maps in brackets for F1)\n',
           '| method | precision vs hand (1 cell) | recall vs hand (1 cell) | F1 (1 cell) | F1 (3 cells) | persistent walls kept | transient cells removed | wall cells | floor m2 | small blobs |', '|---|---|---|---|---|---|---|---|---|---|']
    mean = {}
    for k in order:
        v = {q: np.array([res_all[n]['rows'][k][q] for n in maps]) for q in keys}
        mean[k] = {q: float(v[q].mean()) for q in keys}
        md.append(f"| {k} | {v['hand_precision_1'].mean():.3f} | {v['hand_recall_1'].mean():.3f} | {v['hand_f1_1'].mean():.3f} ({v['hand_f1_1'].std():.3f}) | {v['hand_f1_3'].mean():.3f} ({v['hand_f1_3'].std():.3f}) | "
                  f"{v['persistent_kept'].mean() * 100:.1f} % | {v['transient_removed'].mean() * 100:.1f} % | {v['wall_cells'].mean():.0f} | {v['floor_m2'].mean():.1f} | {v['small_blobs'].mean():.0f} |")
    md += ['\n## F1 against the hand-cleaned map at 3 cells, per map\n', '| method | ' + ' | '.join(x.replace('lab2d_', '') for x in maps) + ' |', '|---|' + '---|' * len(maps)]
    for k in order:
        md.append(f'| {k} | ' + ' | '.join(f"{res_all[n]['rows'][k]['hand_f1_3']:.3f}" for n in maps) + ' |')
    json.dump({'maps': res_all, 'mean': mean}, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'tables.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
