#!/usr/bin/env python3
"""Step 2: real-noise harvest. Aligned raw map minus the vote map = a library of real defects.

    python -m ml.map_clean.harvest [--vote data/map_cleaning/vote] [--out data/map_cleaning/noise]

For every map that voted:
  extra obstacle = occupied in the raw map with no vote wall within 2 cells   (chairs, carts, people, wall jitter)
  leak           = free in the raw map where the vote says unknown             (rays through doors and glass)
  missing wall   = vote wall with no raw obstacle within 2 cells, in an area the map covered
Writes noise_library.npz (64 x 64 obstacle patches, 96 x 96 leak patches), STATS.md with the measured
amounts (used to set the simulator's noise levels), and noise_examples.png.
"""
import argparse
import json
import os

import cv2
import numpy as np

from .common import FREE, OCC, UNK, colour, label

P_OCC, P_LEAK = 64, 96


def patches(mask, size, min_cells, stride):
    out = []
    H, W = mask.shape
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            p = mask[y:y + size, x:x + size]
            if p.sum() >= min_cells:
                out.append(p.copy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vote', default='data/map_cleaning/vote')
    ap.add_argument('--out', default='data/map_cleaning/noise')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(a.vote, 'stack.npz'), allow_pickle=True)
    stack, covered, names, vote, res = z['stack'], z['covered'], [str(n) for n in z['names']], z['vote'], float(z['res'])
    k5, k3 = np.ones((5, 5), np.uint8), np.ones((3, 3), np.uint8)
    wall = vote == OCC
    near_wall = cv2.dilate(wall.astype(np.uint8), k5) > 0
    vote_free_m2 = float((vote == FREE).sum() * res * res)
    d_wall = cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5)        # cells to the nearest vote wall

    occ_p, leak_p, rows, layers = [], [], [], []
    jitter = []
    for name, s, cov in zip(names, stack, covered):
        extra = (s == OCC) & ~near_wall
        leak = (s == FREE) & (vote == UNK)
        leak = cv2.morphologyEx(leak.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0       # drop single-cell slivers along walls
        missing = wall & ~(cv2.dilate((s == OCC).astype(np.uint8), k5) > 0) & cov
        # extra obstacles standing in the vote's floor (true clutter) vs at the edge of the map
        in_floor = extra & (vote == FREE)
        n, lab, st, _ = cv2.connectedComponentsWithStats(in_floor.astype(np.uint8), connectivity=8)
        sizes = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
        jitter.append(d_wall[(s == OCC) & near_wall])                                 # how far raw wall cells sit from the consensus wall
        rows.append({'map': name, 'extra_occ_cells': int(extra.sum()), 'clutter_cells_in_floor': int(in_floor.sum()),
                     'clutter_blobs': int(n - 1), 'blob_median_cells': float(np.median(sizes)), 'blob_p90_cells': float(np.percentile(sizes, 90)),
                     'clutter_cells_per_m2_floor': float(in_floor.sum() / vote_free_m2), 'leak_m2': float(leak.sum() * res * res),
                     'missing_wall_cells': int(missing.sum()), 'missing_wall_share': float(missing.sum() / max(1, (wall & cov).sum()))})
        occ_p += patches(extra, P_OCC, 4, 32)
        leak_p += patches(leak, P_LEAK, 60, 48)
        layers.append((name, s, extra, leak, missing))

    jit = np.concatenate(jitter) * res
    stats = {'maps': rows, 'n_occ_patches': len(occ_p), 'n_leak_patches': len(leak_p), 'vote_floor_m2': vote_free_m2,
             'wall_jitter_m': {'mean': float(jit.mean()), 'p50': float(np.median(jit)), 'p90': float(np.percentile(jit, 90))},
             'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'map'}}
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(a.out, 'noise_library.npz'), occ=np.array(occ_p, bool), leak=np.array(leak_p, bool))

    with open(os.path.join(a.out, 'STATS.md'), 'w') as f:
        f.write(f'# Real-noise harvest from {len(rows)} lab maps (development data)\n\n')
        f.write(f'Library: {len(occ_p)} obstacle patches ({P_OCC} x {P_OCC} cells), {len(leak_p)} leak patches ({P_LEAK} x {P_LEAK}).\n')
        f.write(f'Vote map floor: {vote_free_m2:.1f} m2. Raw wall cells sit {jit.mean() * 100:.1f} cm (mean), {np.percentile(jit, 90) * 100:.1f} cm (90th percentile) '
                f'from the consensus wall, among cells within 2 cells of it.\n\n')
        f.write('| map | extra obstacle cells | clutter cells in floor | clutter blobs | blob size median / p90 (cells) | leak m2 | missing wall cells (share) |\n|---|---|---|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r['map']} | {r['extra_occ_cells']} | {r['clutter_cells_in_floor']} | {r['clutter_blobs']} | {r['blob_median_cells']:.0f} / {r['blob_p90_cells']:.0f} | "
                    f"{r['leak_m2']:.1f} | {r['missing_wall_cells']} ({r['missing_wall_share'] * 100:.0f} %) |\n")
        m = stats['mean']
        f.write(f"\nMean per map: {m['clutter_cells_in_floor']:.0f} clutter cells in the floor in {m['clutter_blobs']:.0f} blobs "
                f"({m['clutter_cells_per_m2_floor']:.2f} cells per m2), {m['leak_m2']:.1f} m2 of leak, {m['missing_wall_share'] * 100:.0f} % of consensus wall missing.\n")

    S = 2
    figs = []
    for name, s, extra, leak, missing in layers[:3]:
        img = colour(s, S)
        for mask, col in ((extra, (60, 60, 230)), (leak, (0, 170, 255)), (missing, (230, 140, 0))):
            img[cv2.resize(mask.astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST) > 0] = col
        figs.append(label(img, f'{name}: red = extra obstacle, orange = leak, blue = missing wall'))
    cv2.imwrite(os.path.join(a.out, 'noise_examples.png'), np.vstack(figs))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(occ_p), min(48, len(occ_p)), replace=False)
    tiles = [cv2.copyMakeBorder(np.where(occ_p[i], 0, 255).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=150) for i in pick]
    rows_img = [np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles) - len(tiles) % 12, 12)]
    if rows_img:
        cv2.imwrite(os.path.join(a.out, 'obstacle_patches.png'), cv2.resize(np.vstack(rows_img), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print(open(os.path.join(a.out, 'STATS.md')).read())


if __name__ == '__main__':
    main()
