#!/usr/bin/env python3
"""Check clean targets for leaks: floor that the simulated lidar marked free outside the walls of the world.

    python -m ml.map_clean.check_leaks --n 120                       # fresh worlds, built exactly as make_pairs does
    python -m ml.map_clean.check_leaks --pairs "data/map_cleaning/pairs/v2/shard_*.npz"   # the saved pairs themselves

A world is enclosed by its walls, closed doors and glass. `inside` = every cell a ray could reach from the region
without crossing those (flood fill, diagonal moves allowed, because rays move diagonally). A clean-target cell that
is free but not inside is a leak. Saved pairs carry no world, so there the test is on the target alone: a free cell
is a leak cell if it is 8-connected to the patch border's unknown side... see `patch_leak`.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import ndimage

from .common import FREE, OCC, UNK, colour, label
from .floorplans import crop, houseexpo_world, kth_files, kth_load, kth_world, rotate, seal
from .simulate import simulate_pair

K3 = np.ones((3, 3), np.uint8)


def world_leak(world, clean):
    """Free cells of the clean map that lie outside the enclosure of the world. -> (n_leak_cells, mask)"""
    occ = world.wall | world.glass | world.furn
    for d in world.doors:
        occ = occ | d
    lab, _ = ndimage.label(~occ, structure=np.ones((3, 3)))
    ids = np.unique(lab[world.region])
    inside = np.isin(lab, ids[ids > 0])
    # the flood fill itself escapes if the enclosure has a hole; then "inside" touches the canvas border
    escaped = inside[0].any() or inside[-1].any() or inside[:, 0].any() or inside[:, -1].any()
    near = cv2.dilate((world.region | world.furn).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    leak = (clean == FREE) & ~near
    return int(leak.sum()), leak, bool(escaped)


def patch_leak(Y):
    """For saved targets (no world): free cells that touch unknown with no wall between, in runs longer than a
    doorway-sized frontier. -> number of free cells 4-adjacent to unknown."""
    f, u = Y == FREE, Y == UNK
    return int((f[1:, :] & u[:-1, :]).sum() + (f[:-1, :] & u[1:, :]).sum() + (f[:, 1:] & u[:, :-1]).sum() + (f[:, :-1] & u[:, 1:]).sum())


def _one(task):
    kind, src, seed = task
    rng = np.random.default_rng(seed)
    try:
        w = kth_world(kth_load(src), rng) if kind == 'kth' else houseexpo_world(src, rng)
        if w is None:
            return None
        rot = 0.0
        if rng.random() < 0.6:
            rot = float(rng.uniform(0, 360)); w = rotate(w, rot)
        elif rng.random() < 0.5:
            rot = float(rng.uniform(-6, 6)); w = rotate(w, rot)
        w = seal(crop(w))
        if w.region.sum() < 1600 or max(w.region.shape) > 1400:
            return None
        pair = simulate_pair(w, rng, None, level=1.0, use_real_noise=False)
        if pair is None:
            return None
        n, mask, escaped = world_leak(w, pair[1])
        return kind, rot, n, escaped, (pair[1], mask) if n > 30 else None
    except Exception as e:  # noqa: BLE001
        return kind, 0.0, -1, False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--data', default='data/map_cleaning')
    ap.add_argument('--pairs', default='')
    ap.add_argument('--out', default='data/map_cleaning/leak_check')
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.pairs:
        for p in sorted(glob.glob(a.pairs)):
            z = np.load(p)
            adj = np.array([patch_leak(y) for y in z['Y']])
            for k in np.unique(z['kind']):
                m = z['kind'] == k
                print(f'{os.path.basename(p)} {k}: {m.sum()} targets, free-touching-unknown cells median {int(np.median(adj[m]))}, p90 {int(np.percentile(adj[m], 90))}, '
                      f'over 40: {100 * np.mean(adj[m] > 40):.0f} %')
        return
    rng = np.random.default_rng(a.seed)
    kth = kth_files(os.path.join(a.data, 'kth'))
    he = sorted(glob.glob(os.path.join(a.data, 'houseexpo', 'json', '*.json')))
    tasks = [('kth', kth[rng.integers(len(kth))], int(rng.integers(2 ** 31))) for _ in range(a.n // 2)]
    tasks += [('houseexpo', he[rng.integers(len(he))], int(rng.integers(2 ** 31))) for _ in range(a.n // 2)]
    with Pool(a.workers) as pool:
        res = [r for r in pool.map(_one, tasks) if r is not None]
    shown = 0
    for kind in ('kth', 'houseexpo'):
        rr = [r for r in res if r[0] == kind and r[2] >= 0]
        err = [r for r in res if r[0] == kind and r[2] < 0]
        for name, sel in (('rotated', lambda r: abs(r[1]) > 7), ('square to the grid', lambda r: abs(r[1]) <= 7)):
            s = [r for r in rr if sel(r)]
            if s:
                print(f'{kind:10s} {name:19s}: {len(s):3d} worlds, leaking (over 30 cells): {sum(r[2] > 30 for r in s):3d}, '
                      f'enclosure open: {sum(r[3] for r in s):3d}, median leak cells {int(np.median([r[2] for r in s]))}, max {max(r[2] for r in s)}')
        if err:
            print(f'{kind}: {len(err)} errors, e.g. {err[0][4]}')
        for r in rr:
            if r[4] is not None and shown < 6:
                clean, mask = r[4]
                img = colour(clean, 2)
                img[cv2.resize(mask.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) > 0] = (40, 40, 230)
                cv2.imwrite(os.path.join(a.out, f'leak_{shown}_{kind}.png'), label(img, f'{kind}, rotated {r[1]:.0f} deg, {r[2]} leak cells (red)'))
                shown += 1


if __name__ == '__main__':
    main()
