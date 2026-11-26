#!/usr/bin/env python3
"""Steps 5 and 6: score every model and the non-learned baselines. Everything here is DEVELOPMENT data.

    python -m ml.map_clean.evaluate --models unet_ce_synth unet_ce unet_gan --hand PATH/lab_common_clean_edit.yaml

A. Synthetic validation pairs (v1_val, made with real noise): each model and the speckle filter.
B. The lab maps that built the vote map (12 maps, in the vote frame): raw, speckle filter, each model without and
   with the guardrail, scored against the vote map and the strong-wall mask.
C. The hand-cleaned map and the raw map behind it (lab2d_11_18_2, same grid): wall recall / precision against the
   hand-cleaned map, and the share of hand-erased wall cells the method also erased.
Writes <dir>/eval/results.json, tables.md, dev_<map>.png and errors_<map>.png.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch

from .clean_map import clean, load_model
from .common import FREE, OCC, UNK, colour, label, load_map, rot, wall_scores, warp
from .show_real import speckle_filter
from .train import evaluate as eval_pairs, load as load_pairs, wall_prf

K3 = np.ones((3, 3), np.uint8)


def near(mask):
    return cv2.dilate(mask.astype(np.uint8), K3) > 0


def f1(r, p):
    return 2 * r * p / max(1e-9, r + p)


def synthetic(models, a, dev):
    X, Y = load_pairs(a.val)
    rows = {}
    s = np.zeros(4)
    for i in range(0, len(X), 256):                      # the speckle filter and "do nothing" on the same pairs
        sp = torch.from_numpy(np.stack([speckle_filter(x) for x in X[i:i + 256].numpy()]))
        s += wall_prf(sp, Y[i:i + 256])
    raw = np.array(wall_prf(X, Y))
    for name, v in (('raw (no cleaning)', raw), ('speckle filter', s)):
        p, r = v[0] / max(1, v[1]), v[2] / max(1, v[3])
        rows[name] = {'wall_precision': p, 'wall_recall': r, 'wall_f1': f1(r, p)}
    for m, G in models.items():
        rows[m] = eval_pairs(G, X, Y, dev)
    return len(X), rows


def score_vs_vote(out, raw, vote, strong):
    r, p = wall_scores(out == OCC, vote == OCC)
    sw = strong & (raw == OCC)
    return {'wall_recall': r, 'wall_precision': p, 'wall_f1': f1(r, p), 'strong_kept': float((near(out == OCC) & sw).sum() / max(1, sw.sum())),
            'strong_kept_exact': float(((out == OCC) & sw).sum() / max(1, sw.sum())),
            'wall_cells': int((out == OCC).sum()), 'floor_m2': float((out == FREE).sum() * 0.0025)}


def overlay(out, ref_occ, raw, S=3):
    """Red: wall of the reference that the method lost. Blue: wall the method kept that the reference does not have."""
    img = colour(out, S)
    up = lambda m: cv2.resize(m.astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST) > 0
    img[up(ref_occ & ~near(out == OCC))] = (40, 40, 230)
    img[up((out == OCC) & ~near(ref_occ))] = (230, 140, 0)
    return img


def crop_of(*maps):
    ys, xs = np.nonzero(np.any([m != UNK for m in maps], 0))
    return slice(max(0, ys.min() - 8), ys.max() + 9), slice(max(0, xs.min() - 8), xs.max() + 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['unet_ce_synth', 'unet_ce', 'unet_gan'])
    ap.add_argument('--dir', default='data/map_cleaning')
    ap.add_argument('--val', default='data/map_cleaning/pairs/v1_val/shard_*.npz')
    ap.add_argument('--hand', default='', help='yaml of the hand-cleaned map')
    ap.add_argument('--hand-raw', default='lab2d_11_18_2')
    ap.add_argument('--show', default='lab2d_11_24_8')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(a.dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    models = {m: load_model(os.path.join(a.dir, 'models', m + '.pt'), dev) for m in a.models}
    res = {}
    md = ['# Evaluation tables (development data only)\n']

    # A. synthetic validation
    n, rows = synthetic(models, a, dev)
    res['synthetic_val'] = {'n_pairs': n, 'rows': rows}
    md += [f'## A. Synthetic validation pairs (n = {n} patches of 256 x 256, walls with 1-cell tolerance)\n',
           '| method | wall precision | wall recall | wall F1 | IoU free | IoU unknown |', '|---|---|---|---|---|---|']
    for k, r in rows.items():
        md.append(f"| {k} | {r['wall_precision']:.3f} | {r['wall_recall']:.3f} | {r['wall_f1']:.3f} | "
                  + (f"{r['iou_free']:.3f} | {r['iou_unknown']:.3f} |" if 'iou_free' in r else '- | - |'))

    # B. the lab maps of the vote
    z = np.load(os.path.join(a.dir, 'vote', 'stack.npz'), allow_pickle=True)
    names, stack, vote, strong = [str(x) for x in z['names']], z['stack'], z['vote'], z['strong']
    methods = ['raw (no cleaning)', 'speckle filter'] + [m + g for m in models for g in ('', ' + guardrail')]
    per = {k: [] for k in methods}
    outs = {}
    for nm, raw in zip(names, stack):
        o = {'raw (no cleaning)': raw, 'speckle filter': speckle_filter(raw)}
        for m, G in models.items():
            o[m] = clean(raw, G, dev)
            o[m + ' + guardrail'] = clean(raw, G, dev, strong)
        for k in methods:
            per[k].append({'map': nm, **score_vs_vote(o[k], raw, vote, strong)})
        if nm == a.show:
            outs = o
    res['lab_maps_vs_vote'] = {'n_maps': len(names), 'maps': names, 'per_map': per}
    md += [f'\n## B. {len(names)} lab maps against the vote map (mean over maps; these maps built the vote map and the noise library)\n',
           '| method | wall recall | wall precision | wall F1 | strong walls kept (1 cell) | wall cells | floor m2 |', '|---|---|---|---|---|---|---|']
    for k in methods:
        mean = lambda key: float(np.mean([r[key] for r in per[k]]))
        md.append(f"| {k} | {mean('wall_recall'):.3f} | {mean('wall_precision'):.3f} | {mean('wall_f1'):.3f} | {mean('strong_kept') * 100:.1f} % | {mean('wall_cells'):.0f} | {mean('floor_m2'):.1f} |")
    md += [f'\n### {a.show} alone\n', '| method | wall recall | wall precision | strong walls kept | wall cells |', '|---|---|---|---|---|']
    for k in methods:
        r = next(x for x in per[k] if x['map'] == a.show)
        md.append(f"| {k} | {r['wall_recall']:.3f} | {r['wall_precision']:.3f} | {r['strong_kept'] * 100:.1f} % | {r['wall_cells']} |")
    s = crop_of(outs['raw (no cleaning)'], vote)
    cv2.imwrite(os.path.join(out_dir, f'dev_{a.show}.png'),
                np.vstack([label(colour(p[s], 2), t) for t, p in [(f'raw {a.show}', outs['raw (no cleaning)']), ('speckle filter', outs['speckle filter']), ('vote map', vote)]
                           + [(k, outs[k]) for k in methods[2:]]]))
    sw = strong & (outs['raw (no cleaning)'] == OCC)
    cv2.imwrite(os.path.join(out_dir, f'errors_{a.show}.png'),
                np.vstack([label(overlay(outs[k], sw, None)[s[0].start * 3:s[0].stop * 3, s[1].start * 3:s[1].stop * 3],
                                 f'{k}: red = strong wall removed, blue = wall kept that the vote map lacks') for k in methods[2:]]))

    # C. the hand-cleaned map
    if a.hand:
        hand = load_map(a.hand)['cls']
        rm = load_map(f'maps/{a.hand_raw}/maps_2d/{a.hand_raw}.yaml')
        raw = rm['cls']
        assert raw.shape == hand.shape, 'the hand-cleaned map and its raw map must share a grid'
        al = json.load(open(os.path.join(a.dir, 'vote', 'alignment.json')))
        row = next(r for r in al['maps'] if r['name'] == a.hand_raw)
        R, t = rot(math.radians(row['yaw_deg'])), np.array(row['t'])          # raw frame -> vote frame; invert it
        back = lambda img: warp(img, al['res'], al['origin'], R.T, -R.T @ t, raw.shape, rm['origin'])
        vote_h, strong_h = back(vote), near(back(strong.astype(np.uint8)) == 1)
        hand_occ, raw_occ = hand == OCC, raw == OCC
        erased, kept = raw_occ & ~near(hand_occ), raw_occ & near(hand_occ)
        o = {'raw (no cleaning)': raw, 'speckle filter': speckle_filter(raw), 'vote map (warped to this grid)': vote_h}
        for m, G in models.items():
            o[m] = clean(raw, G, dev)
            o[m + ' + guardrail'] = clean(raw, G, dev, strong_h)
        rows = {}
        for k, out in o.items():
            r, p = wall_scores(out == OCC, hand_occ)
            r3, p3 = wall_scores(out == OCC, hand_occ, tol=3)
            rows[k] = {'wall_recall': r, 'wall_precision': p, 'wall_f1': f1(r, p), 'wall_f1_tol3': f1(r3, p3),
                       'erased_also': float((erased & (out != OCC)).sum() / max(1, erased.sum())),
                       'kept_also': float((kept & near(out == OCC)).sum() / max(1, kept.sum())), 'wall_cells': int((out == OCC).sum())}
        res['hand_cleaned'] = {'raw_map': a.hand_raw, 'raw_wall_cells': int(raw_occ.sum()), 'hand_wall_cells': int(hand_occ.sum()),
                               'hand_erased_cells': int(erased.sum()), 'hand_kept_cells': int(kept.sum()), 'rows': rows}
        md += [f'\n## C. Hand-cleaned map against the raw map behind it ({a.hand_raw}, 1 map)\n',
               f'Raw map: {int(raw_occ.sum())} wall cells. Hand-cleaned map: {int(hand_occ.sum())} wall cells. Of the raw wall cells, the hand cleaning '
               f'erased {int(erased.sum())} (no hand wall within 1 cell) and kept {int(kept.sum())}.\n',
               '| method | wall recall vs hand | wall precision vs hand | wall F1 | wall F1 at 3 cells | hand-erased cells also erased | hand-kept cells also kept (1 cell) | wall cells |', '|---|---|---|---|---|---|---|---|']
        for k, r in rows.items():
            md.append(f"| {k} | {r['wall_recall']:.3f} | {r['wall_precision']:.3f} | {r['wall_f1']:.3f} | {r['wall_f1_tol3']:.3f} | {r['erased_also'] * 100:.1f} % | {r['kept_also'] * 100:.1f} % | {r['wall_cells']} |")
        s = crop_of(raw, hand)
        cv2.imwrite(os.path.join(out_dir, f'dev_{a.hand_raw}.png'),
                    np.vstack([label(colour(p[s], 2), t) for t, p in [(f'raw {a.hand_raw}', raw), ('hand-cleaned', hand)] + [(k, v) for k, v in o.items() if k != 'raw (no cleaning)']]))
        cv2.imwrite(os.path.join(out_dir, f'errors_{a.hand_raw}.png'),
                    np.vstack([label(overlay(o[k], hand_occ, None)[s[0].start * 3:s[0].stop * 3, s[1].start * 3:s[1].stop * 3],
                                     f'{k}: red = hand-kept wall missing, blue = wall the hand cleaning does not have') for k in list(models) + [m + ' + guardrail' for m in models]]))

    json.dump(res, open(os.path.join(out_dir, 'results.json'), 'w'), indent=1)
    open(os.path.join(out_dir, 'tables.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
