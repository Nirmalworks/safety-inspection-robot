#!/usr/bin/env python3
"""Step 3: synthetic (dirty, clean) map pairs.

    python -m ml.map_clean.make_pairs --n 6000 --out data/map_cleaning/pairs/v1
    python -m ml.map_clean.make_pairs --n 48 --out /tmp/pairs_try --preview          # quick look

Each world (procedural lab / KTH rooms / HouseExpo house) is rotated by a random angle, driven through by the
simulator, and cut into 256 x 256 patches at 0.05 m per cell. Output: shard_XXX.npz with X (dirty) and Y (clean)
as uint8 class maps (0 free, 1 occupied, 2 unknown), kind and noise level per patch; preview.png;
real_vs_synthetic.png (real lab-map crops next to synthetic dirty crops, to judge the look).
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from .common import FREE, OCC, UNK, colour, label
from .floorplans import crop, houseexpo_world, kth_files, kth_load, kth_world, procedural_lab, rotate, seal
from .simulate import simulate_pair

PATCH = 256
_G = {}


def _init(noise_path, real_noise):
    _G['lib'] = np.load(noise_path)['occ'] if (noise_path and os.path.exists(noise_path)) else None
    _G['real'] = real_noise
    _G['kth'] = {}


def patches(dirty, clean, rng, n_max=4):
    """Random 256 x 256 windows over the mapped area (padded with unknown when the map is smaller)."""
    known = (dirty != UNK) | (clean != UNK)
    ys, xs = np.nonzero(known)
    if len(xs) == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    n = int(np.clip(round((y1 - y0) * (x1 - x0) / (PATCH * PATCH * 0.6)), 1, n_max))
    out = []
    for _ in range(n):
        cy = rng.integers(min(y0, y1 - PATCH), max(y0, y1 - PATCH) + 1) if (y1 - y0) > PATCH else y0 - rng.integers(0, PATCH - (y1 - y0) + 1)
        cx = rng.integers(min(x0, x1 - PATCH), max(x0, x1 - PATCH) + 1) if (x1 - x0) > PATCH else x0 - rng.integers(0, PATCH - (x1 - x0) + 1)
        d, c = (np.full((PATCH, PATCH), UNK, np.uint8) for _ in range(2))
        ya, xa, yb, xb = max(0, cy), max(0, cx), min(dirty.shape[0], cy + PATCH), min(dirty.shape[1], cx + PATCH)
        if yb <= ya or xb <= xa:
            continue
        d[ya - cy:yb - cy, xa - cx:xb - cx] = dirty[ya:yb, xa:xb]
        c[ya - cy:yb - cy, xa - cx:xb - cx] = clean[ya:yb, xa:xb]
        if (c != UNK).mean() >= 0.06:
            out.append((d, c))
    return out


def _one(task):
    kind, src, seed = task
    rng = np.random.default_rng(seed)
    try:
        if kind == 'procedural':
            w = procedural_lab(rng)
        elif kind == 'kth':
            if src not in _G['kth']:
                _G['kth'][src] = kth_load(src)
            sp = _G['kth'][src]
            w = kth_world(sp, rng) if sp else None
        else:
            w = houseexpo_world(src, rng)
        if w is None:
            return []
        if rng.random() < 0.6:
            w = rotate(w, rng.uniform(0, 360))
        elif rng.random() < 0.5:
            w = rotate(w, rng.uniform(-6, 6))                    # nearly square to the grid, like most of the lab maps
        w = seal(crop(w))
        if w.region.sum() < 1600 or max(w.region.shape) > 1400:
            return []
        level = float(rng.uniform(0.2, 2.0))
        pair = simulate_pair(w, rng, _G['lib'], level=level, use_real_noise=_G['real'])
        if pair is None:
            return []
        return [(d, c, kind, level) for d, c in patches(pair[0], pair[1], rng)]
    except Exception as e:  # noqa: BLE001  (one odd floor plan must not stop the run; counted by the caller)
        return [('error', f'{kind} {os.path.basename(str(src))}: {type(e).__name__}: {e}')]


def tile(rows, scale=1):
    return np.vstack([np.hstack([cv2.copyMakeBorder(colour(c, scale), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(120, 120, 120)) for c in r]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6000, help='number of worlds to simulate')
    ap.add_argument('--out', default='data/map_cleaning/pairs/v1')
    ap.add_argument('--data', default='data/map_cleaning')
    ap.add_argument('--mix', default='0.5,0.25,0.25', help='share of procedural, kth, houseexpo worlds')
    ap.add_argument('--no-real-noise', action='store_true', help='ablation: synthetic defects only, no harvested patches')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--shard', type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    kth = kth_files(os.path.join(a.data, 'kth'))
    he = sorted(glob.glob(os.path.join(a.data, 'houseexpo', 'json', '*.json')))
    mix = [float(v) for v in a.mix.split(',')]
    kinds = rng.choice(['procedural', 'kth', 'houseexpo'], a.n, p=np.array(mix) / sum(mix))
    tasks = [(k, None if k == 'procedural' else kth[rng.integers(len(kth))] if k == 'kth' else he[rng.integers(len(he))], int(rng.integers(2 ** 31))) for k in kinds]
    X, Y, K, L, errors, n_ok, shard, t0 = [], [], [], [], [], 0, 0, time.time()
    counts = {'procedural': 0, 'kth': 0, 'houseexpo': 0}
    first = []

    def flush():
        nonlocal X, Y, K, L, shard
        if X:
            np.savez_compressed(os.path.join(a.out, f'shard_{shard:03d}.npz'), X=np.array(X), Y=np.array(Y), kind=np.array(K), level=np.array(L, np.float32))
            shard += 1
            X, Y, K, L = [], [], [], []

    with Pool(a.workers, initializer=_init, initargs=(os.path.join(a.data, 'noise', 'noise_library.npz'), not a.no_real_noise)) as pool:
        for res in pool.imap_unordered(_one, tasks, chunksize=4):
            if res and isinstance(res[0][0], str):       # ('error', message) from a world that failed
                errors.append(res[0][1])
                continue
            n_ok += bool(res)
            for d, c, kind, level in res:
                X.append(d); Y.append(c); K.append(kind); L.append(level)
                counts[kind] += 1
                if len(first) < 64:
                    first.append((d, c, kind, level))
            if len(X) >= a.shard:
                flush()
    flush()
    total = sum(counts.values())
    meta = {'worlds_requested': a.n, 'worlds_ok': n_ok, 'patches': total, 'by_kind': counts, 'errors': len(errors), 'error_examples': errors[:8],
            'real_noise': not a.no_real_noise, 'seed': a.seed, 'seconds': round(time.time() - t0, 1), 'patch': PATCH, 'res_m': 0.05}
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))

    if first:
        rows = []
        for i in range(0, min(len(first), 24), 6):
            g = first[i:i + 6]
            if len(g) == 6:
                rows += [[x[0] for x in g], [x[1] for x in g]]
        if rows:
            cv2.imwrite(os.path.join(a.out, 'preview.png'), label(tile(rows), 'rows alternate: dirty input, then its clean target'))
        stack_p = os.path.join(a.data, 'vote', 'stack.npz')
        if os.path.exists(stack_p):                              # real lab-map crops next to synthetic dirty crops
            z = np.load(stack_p, allow_pickle=True)
            r2 = np.random.default_rng(1)
            real = []
            for s in z['stack'][r2.permutation(len(z['stack']))]:
                ys, xs = np.nonzero(s != UNK)
                s = np.pad(s[ys.min():ys.max() + 1, xs.min():xs.max() + 1], PATCH // 2, constant_values=UNK)   # the lab is only ~180 cells tall
                for _ in range(20):
                    cy, cx = r2.integers(0, s.shape[0] - PATCH + 1), r2.integers(0, s.shape[1] - PATCH + 1)
                    p = s[cy:cy + PATCH, cx:cx + PATCH]
                    if (p != UNK).mean() > 0.3:
                        real.append(p)
                        break
            syn = [x[0] for x in first if (x[0] != UNK).mean() > 0.3]
            if len(real) >= 6 and len(syn) >= 12:
                fig = np.vstack([label(tile([real[:6]]), 'REAL lab maps (random crops)'), label(tile([syn[:6]]), 'SYNTHETIC dirty maps'),
                                 label(tile([syn[6:12]]), 'SYNTHETIC dirty maps')])
                cv2.imwrite(os.path.join(a.out, 'real_vs_synthetic.png'), fig)


if __name__ == '__main__':
    main()
