#!/usr/bin/env python3
"""Phase 2 training data: simulated lidar maps of ALL KTH plans and ALL HouseExpo houses, with keep / erase labels.

    python -m ml.map_clean.make_pairs2 --out data/map_cleaning/pairs/v2 --part 0/2 --workers 16
    python -m ml.map_clean.make_pairs2 --out data/map_cleaning/pairs/v2 --part 1/2 --workers 16      # on another machine
    python -m ml.map_clean.make_pairs2 --out data/map_cleaning/pairs/v2 --count                       # how much there is to do

No generated rooms. Every HouseExpo house is driven once. Every KTH floor is covered room by room: clusters of
connected rooms are grown until each room has been in at least one cluster, `--kth-passes` times with different
clusters, rotations and noise. The simulated lidar has the robot's 205-degree field of view.
Each world gives up to 3 patches of 256 x 256 cells: X = dirty map (0 free, 1 occupied, 2 unknown) and
L = label of its occupied cells (1 keep: the rays hit structure; 2 erase: they hit movable clutter; 0 elsewhere).
The work is split into chunks of `--chunk` worlds, one file each; a chunk whose file exists is skipped, so an
interrupted run can simply be started again.
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from .common import UNK, colour, label
from .floorplans import crop, houseexpo_world, kth_cluster, kth_files, kth_load, kth_rooms, kth_world, rotate, seal, stata_load
from . import simulate

PATCH = 256
_KTH = {}


def _init(fov):
    simulate.set_fov(fov)


def patches(dirty, lab, rng, n_max=3):
    known = dirty != UNK
    ys, xs = np.nonzero(known)
    if len(xs) == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    n = int(np.clip(round((y1 - y0) * (x1 - x0) / (PATCH * PATCH * 0.6)), 1, n_max))
    out = []
    for _ in range(n):
        cy = rng.integers(min(y0, y1 - PATCH), max(y0, y1 - PATCH) + 1) if (y1 - y0) > PATCH else y0 - rng.integers(0, PATCH - (y1 - y0) + 1)
        cx = rng.integers(min(x0, x1 - PATCH), max(x0, x1 - PATCH) + 1) if (x1 - x0) > PATCH else x0 - rng.integers(0, PATCH - (x1 - x0) + 1)
        d, l = np.full((PATCH, PATCH), UNK, np.uint8), np.zeros((PATCH, PATCH), np.uint8)
        ya, xa, yb, xb = max(0, cy), max(0, cx), min(dirty.shape[0], cy + PATCH), min(dirty.shape[1], cx + PATCH)
        if yb <= ya or xb <= xa:
            continue
        d[ya - cy:yb - cy, xa - cx:xb - cx] = dirty[ya:yb, xa:xb]
        l[ya - cy:yb - cy, xa - cx:xb - cx] = lab[ya:yb, xa:xb]
        if (d != UNK).mean() >= 0.06 and (l > 0).sum() >= 40:
            out.append((d, l))
    return out


def _one(task):
    kind, src, seed, inc = task
    rng = np.random.default_rng(seed)
    try:
        if kind in ('kth', 'stata'):
            if src not in _KTH:
                _KTH[src] = kth_load(src) if kind == 'kth' else stata_load(src)
            w = kth_world(_KTH[src], rng, inc=list(inc))
            if w is not None:
                w.kind = kind
        else:
            w = houseexpo_world(src, rng)
        if w is None:
            return []
        if rng.random() < 0.6:
            w = rotate(w, rng.uniform(0, 360))
        elif rng.random() < 0.5:
            w = rotate(w, rng.uniform(-6, 6))
        w = seal(crop(w))
        if w.region.sum() < 1600 or max(w.region.shape) > 1400:
            return []
        pair = simulate.simulate_labeled(w, rng)
        if pair is None:
            return []
        return [(d, l, kind) for d, l in patches(pair[0], pair[1], rng)]
    except Exception as e:  # noqa: BLE001  (one odd plan must not stop the run; counted)
        return [('error', f'{kind} {os.path.basename(str(src))}: {type(e).__name__}: {e}', kind)]


def stata_tasks(plans, floors, passes, seed):
    """Room clusters of the MIT Stata Center floors named in `floors` (e.g. 1 3 7 9), every room covered `passes` times."""
    rng = np.random.default_rng(seed + 7)
    tasks, n_rooms = [], 0
    for fl in floors:
        one = os.path.join(plans, f'32-{fl}.xml')
        for p in ([one] if os.path.exists(one) else [os.path.join(plans, f'32-D{fl}.xml'), os.path.join(plans, f'32-G{fl}.xml')]):
            sp = stata_load(p)
            if not sp:
                continue
            rooms = kth_rooms(sp)
            n_rooms += len(rooms)
            for _ in range(passes):
                left = set(rooms)
                order = list(rooms)
                rng.shuffle(order)
                for start in order:
                    if start in left:
                        inc = kth_cluster(sp, rng, start)
                        left -= set(inc)
                        tasks.append(('stata', p, int(rng.integers(2 ** 31)), tuple(inc)))
    return tasks, n_rooms


def build_tasks(data, kth_passes, seed):
    """The full, fixed list of worlds: the same on every machine, so `--part k/n` splits it without overlap."""
    rng = np.random.default_rng(seed)
    tasks = []
    for p in sorted(glob.glob(os.path.join(data, 'houseexpo', 'json', '*.json'))):
        tasks.append(('houseexpo', p, int(rng.integers(2 ** 31)), ()))
    n_rooms = 0
    for p in kth_files(os.path.join(data, 'kth')):
        sp = kth_load(p)
        if not sp:
            continue
        rooms = kth_rooms(sp)
        n_rooms += len(rooms)
        for _ in range(kth_passes):
            left = list(rooms)
            rng.shuffle(left)
            left = set(left)
            order = [r for r in rooms if r in left]
            rng.shuffle(order)
            for start in order:
                if start not in left:
                    continue
                inc = kth_cluster(sp, rng, start)
                left -= set(inc)
                tasks.append(('kth', p, int(rng.integers(2 ** 31)), tuple(inc)))
    order = np.random.default_rng(seed + 1).permutation(len(tasks))      # mix the kinds, so every part and chunk has both
    return [tasks[i] for i in order], n_rooms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/map_cleaning/pairs/v2')
    ap.add_argument('--data', default='data/map_cleaning')
    ap.add_argument('--part', default='0/1', help='k/n: do every n-th chunk starting at k')
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--kth-passes', type=int, default=3)
    ap.add_argument('--fov', type=float, default=205.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='only the first N worlds (for a trial)')
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--stata-floors', nargs='*', default=[], help='make ONLY worlds from these MIT Stata floors (files stata_XXXX.npz); leave out the floors held back for testing')
    ap.add_argument('--stata-passes', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prefix = 'chunk'
    if a.stata_floors:
        tasks, n_rooms = stata_tasks(os.path.join(a.data, 'stata', 'floorplans'), a.stata_floors, a.stata_passes, a.seed)
        prefix = 'stata'
    else:
        tasks, n_rooms = build_tasks(a.data, a.kth_passes, a.seed)
    if a.limit:
        tasks = tasks[:a.limit]
    nk = sum(t[0] == 'kth' for t in tasks)
    chunks = [tasks[i:i + a.chunk] for i in range(0, len(tasks), a.chunk)]
    k, n = (int(v) for v in a.part.split('/'))
    print(f'{len(tasks)} worlds: {len(tasks) - nk} HouseExpo houses, {nk} KTH room clusters covering {n_rooms} rooms {a.kth_passes} times; '
          f'{len(chunks)} chunks of {a.chunk}; this part does chunks {k}, {k + n}, ...', flush=True)
    if a.count:
        return
    t0, done = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(a.fov,)) as pool:
        for ci in range(k, len(chunks), n):
            f = os.path.join(a.out, f'{prefix}_{ci:04d}.npz')
            if os.path.exists(f):
                continue
            X, L, K, errs, ok = [], [], [], [], 0
            for res in pool.imap_unordered(_one, chunks[ci], chunksize=2):
                if res and isinstance(res[0][0], str):
                    errs.append(res[0][1]); continue
                ok += bool(res)
                for d, l, kind in res:
                    X.append(d); L.append(l); K.append(kind)
            if not X:
                X, L = [np.full((PATCH, PATCH), UNK, np.uint8)][:0], []
            tmp = f + '.tmp.npz'
            np.savez_compressed(tmp, X=np.array(X, np.uint8).reshape(-1, PATCH, PATCH), L=np.array(L, np.uint8).reshape(-1, PATCH, PATCH), kind=np.array(K),
                                worlds=len(chunks[ci]), worlds_ok=ok, errors=np.array(errs[:20]), n_errors=len(errs))
            os.replace(tmp, f)
            done += 1
            print(f'chunk {ci:4d}: {len(X):5d} patches from {ok}/{len(chunks[ci])} worlds, {len(errs)} errors, {(time.time() - t0) / done:.0f} s per chunk', flush=True)
    files = sorted(glob.glob(os.path.join(a.out, prefix + '_*.npz')))
    tot = {'chunks_done': len(files), 'chunks_total': len(chunks), 'patches': 0, 'worlds_ok': 0, 'errors': 0, 'by_kind': {}}
    for f in files:
        z = np.load(f)
        tot['patches'] += len(z['X']); tot['worlds_ok'] += int(z['worlds_ok']); tot['errors'] += int(z['n_errors'])
        for kd, c in zip(*np.unique(z['kind'], return_counts=True)):
            tot['by_kind'][str(kd)] = tot['by_kind'].get(str(kd), 0) + int(c)
    tot.update(worlds=len(tasks), houseexpo_houses=len(tasks) - nk, kth_clusters=nk, kth_rooms=n_rooms, kth_passes=a.kth_passes, fov_deg=a.fov, seed=a.seed)
    json.dump(tot, open(os.path.join(a.out, f'meta_{prefix}_part{k}of{n}.json'), 'w'), indent=1)
    print(json.dumps(tot, indent=1))


def preview(pattern, out, n=12):
    """Picture of the first patches: dirty map above, and below it the same map with erase cells in red."""
    z = np.load(sorted(glob.glob(pattern))[0])
    tiles_a, tiles_b = [], []
    for x, l in zip(z['X'][:n], z['L'][:n]):
        tiles_a.append(cv2.copyMakeBorder(colour(x), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(120, 120, 120)))
        img = colour(x)
        img[l == 2] = (40, 40, 230)
        tiles_b.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(120, 120, 120)))
    rows = []
    for i in range(0, n, 6):
        rows += [np.hstack(tiles_a[i:i + 6]), np.hstack(tiles_b[i:i + 6])]
    cv2.imwrite(out, label(np.vstack(rows), 'rows alternate: simulated dirty map, then its labels (red = erase, black = keep)'))


if __name__ == '__main__':
    main()
