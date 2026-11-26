#!/usr/bin/env python3
"""Step 1: vote baseline. Align every lab map to one reference, then take a per-cell majority vote.

    python -m ml.map_clean.vote [--ref lab2d_11_24_8] [--out data/map_cleaning/vote]

Writes: lab_vote.pgm/.yaml (consensus map, in the reference map's frame), strong_wall.png (cells
nearly all maps call a wall: the guardrail mask), stack.npz (all aligned maps, for the noise harvest),
alignment.json + ALIGNMENT.md (per-map transform and errors), vote_before_after.png.
"""
import argparse
import glob
import json
import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
try:
    from skimage.morphology import skeletonize
except ImportError:                                   # same job without scikit-image: Zhang-Suen thinning
    def skeletonize(mask):
        im = np.pad(mask.astype(np.uint8), 1)
        while True:
            changed = False
            for step in (0, 1):
                P = [np.roll(np.roll(im, dy, 0), dx, 1) for dy, dx in ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]  # N, NE, E, SE, S, SW, W, NW
                n = sum(P)
                trans = sum((P[i] == 0) & (P[(i + 1) % 8] == 1) for i in range(8))
                c = (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0) if step == 0 else (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
                kill = (im == 1) & (n >= 2) & (n <= 6) & (trans == 1) & c
                if kill.any():
                    im[kill] = 0; changed = True
            if not changed:
                return im[1:-1, 1:-1].astype(bool)

from .common import FREE, OCC, UNK, align, cells_xy, colour, label, load_map, save_map, wall_scores, warp

PAD = 60          # cells (3 m) of margin around the reference map, so corridor stubs other maps saw are kept
MIN_FREE_M2 = 30  # a map with less floor than this is too partial to vote
MAX_MEDIAN = 0.12  # m: median wall distance to the reference after alignment
MIN_IN20 = 0.60   # share of wall cells within 20 cm of the reference


def _align_one(args):
    m, ref_pts = args
    A = cells_xy(m['cls'] == OCC, m['res'], m['origin'])
    if len(A) < 200:
        return m['name'], None
    R, t, d = align(A, ref_pts)
    return m['name'], {'yaw_deg': math.degrees(math.atan2(R[1, 0], R[0, 0])), 't': [float(t[0]), float(t[1])],
                       'median_m': float(np.median(d)), 'in10': float(np.mean(d < 0.10)), 'in20': float(np.mean(d < 0.20)),
                       'R': R.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='maps/*/maps_2d/*.yaml')
    ap.add_argument('--ref', default='lab2d_11_24_8')
    ap.add_argument('--out', default='data/map_cleaning/vote')
    ap.add_argument('--exclude', nargs='*', default=[], help='map names to leave out (e.g. evaluation maps)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    maps = [load_map(p) for p in sorted(glob.glob(a.maps))]
    maps = [m for m in maps if m['name'] not in a.exclude]
    ref = next(m for m in maps if m['name'] == a.ref)
    res = ref['res']
    assert all(abs(m['res'] - res) < 1e-9 for m in maps)
    ref_pts = cells_xy(ref['cls'] == OCC, res, ref['origin'])
    others = [m for m in maps if m is not ref]
    with Pool(min(16, len(others))) as pool:
        results = dict(pool.map(_align_one, [(m, ref_pts) for m in others]))

    rows, kept = [], [(ref, np.eye(2), np.zeros(2))]
    for m in others:
        r, free_m2 = results[m['name']], float((m['cls'] == FREE).sum() * res * res)
        why = ''
        if r is None:
            why = 'too few wall cells'
        elif free_m2 < MIN_FREE_M2:
            why = f'only {free_m2:.1f} m2 of floor'
        elif r['median_m'] > MAX_MEDIAN or r['in20'] < MIN_IN20:
            why = 'aligns badly'
        rows.append({'name': m['name'], 'free_m2': free_m2, 'kept': not why, 'why': why, **({} if r is None else {k: v for k, v in r.items() if k != 'R'})})
        if not why:
            kept.append((m, np.array(r['R']), np.array(r['t'])))
    rows.insert(0, {'name': ref['name'], 'free_m2': float((ref['cls'] == FREE).sum() * res * res), 'kept': True, 'why': 'reference',
                    'yaw_deg': 0.0, 't': [0.0, 0.0], 'median_m': 0.0, 'in10': 1.0, 'in20': 1.0})

    H, W = ref['cls'].shape[0] + 2 * PAD, ref['cls'].shape[1] + 2 * PAD
    origin = (ref['origin'][0] - PAD * res, ref['origin'][1] - PAD * res)
    stack = np.stack([warp(m['cls'], res, m['origin'], R, t, (H, W), origin) for m, R, t in kept])
    K = len(stack)
    k3, k21 = np.ones((3, 3), np.uint8), np.ones((21, 21), np.uint8)
    # "covered": the map looked at this place (known cells, grown by 0.5 m), so its unknown there is a real vote
    covered = np.stack([cv2.dilate((s != UNK).astype(np.uint8), k21) for s in stack]).astype(bool)
    n_cov = covered.sum(0)
    occ_sup = np.stack([cv2.dilate((s == OCC).astype(np.uint8), k3) for s in stack]).sum(0)   # a wall within 1 cell counts
    n_free = (stack == FREE).sum(0)
    enough = n_cov >= min(3, K)
    thick = (occ_sup >= np.maximum(2, np.ceil(0.5 * n_cov))) & enough
    wall = skeletonize(thick)                          # the smeared consensus wall, thinned to its centre line
    vote = np.full((H, W), UNK, np.uint8)
    vote[(n_free >= 0.4 * n_cov) & enough] = FREE
    vote[thick & ~wall & (n_free < 0.4 * n_cov)] = UNK
    vote[wall] = OCC
    strong = (occ_sup >= 0.8 * n_cov) & enough & (n_cov >= max(3, int(0.6 * K)))

    save_map(os.path.join(a.out, 'lab_vote'), vote, res, origin)
    cv2.imwrite(os.path.join(a.out, 'strong_wall.png'), strong.astype(np.uint8) * 255)
    np.savez_compressed(os.path.join(a.out, 'stack.npz'), stack=stack, covered=covered, names=np.array([m['name'] for m, _, _ in kept]),
                        vote=vote, strong=strong, origin=np.array(origin), res=res)
    for r, s in zip([x for x in rows if x['kept']], stack):   # how far each raw map is from the consensus
        r['wall_recall_vs_vote'], r['wall_precision_vs_vote'] = wall_scores(s == OCC, vote == OCC, tol=1)
    json.dump({'ref': ref['name'], 'origin': origin, 'res': res, 'shape': [H, W], 'maps': rows}, open(os.path.join(a.out, 'alignment.json'), 'w'), indent=1)

    with open(os.path.join(a.out, 'ALIGNMENT.md'), 'w') as f:
        f.write(f'# Vote baseline: alignment of {len(maps)} maps to `{ref["name"]}`\n\n')
        f.write(f'{K} maps voted. A map is dropped if it has under {MIN_FREE_M2} m2 of floor, a median wall distance to the reference over '
                f'{MAX_MEDIAN * 100:.0f} cm, or under {MIN_IN20 * 100:.0f} % of walls within 20 cm.\n\n')
        f.write('| map | floor m2 | yaw deg | shift m | median cm | within 10 cm | within 20 cm | voted | wall recall / precision vs vote (1 cell) |\n|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            if 'median_m' in r:
                rp = f"{r['wall_recall_vs_vote']:.2f} / {r['wall_precision_vs_vote']:.2f}" if r['kept'] else '-'
                f.write(f"| {r['name']} | {r['free_m2']:.1f} | {r['yaw_deg']:.1f} | ({r['t'][0]:.2f}, {r['t'][1]:.2f}) | {r['median_m'] * 100:.1f} | "
                        f"{r['in10'] * 100:.0f} % | {r['in20'] * 100:.0f} % | {'yes' if r['kept'] else 'no: ' + r['why']} | {rp} |\n")
            else:
                f.write(f"| {r['name']} | {r['free_m2']:.1f} | - | - | - | - | - | no: {r['why']} | - |\n")
        f.write(f'\nVote map: {int((vote == OCC).sum())} wall cells, {(vote == FREE).sum() * res * res:.1f} m2 floor, '
                f'{int(strong.sum())} cells in the strong-wall (guardrail) mask.\n')

    S = 3
    raw = stack[0]
    over = colour(vote, S)
    m_raw_only = cv2.resize(((raw == OCC) & (cv2.dilate((vote == OCC).astype(np.uint8), k3) == 0)).astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST) > 0
    m_vote_only = cv2.resize(((vote == OCC) & (cv2.dilate((raw == OCC).astype(np.uint8), k3) == 0)).astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST) > 0
    over[m_raw_only] = (60, 60, 230)
    over[m_vote_only] = (230, 140, 0)
    fig = np.vstack([label(colour(raw, S), f'raw map {ref["name"]}'), label(colour(vote, S), f'vote of {K} maps'),
                     label(over, 'vote map; red = only in the raw map (removed), blue = wall only in the vote (added)')])
    cv2.imwrite(os.path.join(a.out, 'vote_before_after.png'), fig)
    print(open(os.path.join(a.out, 'ALIGNMENT.md')).read())


if __name__ == '__main__':
    main()
