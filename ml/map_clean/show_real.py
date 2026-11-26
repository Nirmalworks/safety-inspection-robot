#!/usr/bin/env python3
"""Run trained models on one real lab map and put the results side by side (a look, not a score).

    python -m ml.map_clean.show_real --map lab2d_11_24_8 --models unet_ce unet_gan

Panels: raw map, speckle filter, vote of the lab maps, then each model. The lab maps were used to build the vote map
and the noise library, so anything shown here is a DEVELOPMENT result.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from .clean_map import clean, load_model
from .common import FREE, OCC, UNK, colour, label, load_map, wall_scores


def speckle_filter(cls, min_cells=8):
    """Non-learned baseline: occupied blobs smaller than `min_cells` become floor (or unknown when no floor touches them)."""
    out = cls.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((cls == OCC).astype(np.uint8), connectivity=8)
    near_free = cv2.dilate((cls == FREE).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_cells:
            m = lab == i
            out[m] = FREE if near_free[m].any() else UNK
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='lab2d_11_24_8')
    ap.add_argument('--models', nargs='+', default=['unet_ce', 'unet_gan'])
    ap.add_argument('--dir', default='data/map_cleaning')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(os.path.join(a.dir, 'vote', 'stack.npz'), allow_pickle=True)
    names = [str(n) for n in z['names']]
    raw = z['stack'][names.index(a.map)] if a.map in names else load_map(f'maps/{a.map}/maps_2d/{a.map}.yaml')['cls']
    vote, strong = z['vote'], z['strong']
    ys, xs = np.nonzero((raw != UNK) | (vote != UNK))
    s = (slice(max(0, ys.min() - 8), ys.max() + 9), slice(max(0, xs.min() - 8), xs.max() + 9))
    panels = [('raw map ' + a.map, raw), ('speckle filter (blobs under 8 cells removed)', speckle_filter(raw))]
    if a.map in names:
        panels.append((f'vote of {len(names)} lab maps', vote))
    for m in a.models:
        G = load_model(os.path.join(a.dir, 'models', m + '.pt'), dev)
        out = clean(raw, G, dev)
        kept = float(((out == OCC) & strong).sum() / max(1, (strong & (raw == OCC)).sum())) if a.map in names else float('nan')
        rec, prec = wall_scores(out == OCC, vote == OCC, tol=1) if a.map in names else (float('nan'),) * 2
        print(f'{m}: wall cells {int((raw == OCC).sum())} -> {int((out == OCC).sum())}, floor {(raw == FREE).sum() * 0.0025:.1f} -> {(out == FREE).sum() * 0.0025:.1f} m2, '
              f'strong walls kept {kept * 100:.0f} %, vs vote map: wall recall {rec:.2f} precision {prec:.2f} (1-cell tolerance, development data)')
        panels.append((f'{m}', out))
    S = 2
    fig = np.vstack([label(colour(p[s], S), t) for t, p in panels])
    out_path = a.out or os.path.join(a.dir, 'models', f'real_{a.map}.png')
    cv2.imwrite(out_path, fig)
    print(out_path)


if __name__ == '__main__':
    main()
