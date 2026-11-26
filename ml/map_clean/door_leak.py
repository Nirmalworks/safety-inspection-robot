#!/usr/bin/env python3
"""Door-leak rule (no ML): cut off free space that spills out of the room through an opening.

    python -m ml.map_clean.door_leak RAW.pgm CLEANED.pgm OUT.pgm [--min-enclosed 0.6]

Inside a room, a ray sent from a floor cell in any direction soon ends on something solid. In a leak (the fan of
free space a lidar draws through an open door) most rays run off into unknown space. For every free cell of the RAW
map, 16 rays are followed to the first cell that is not free; the cell's enclosure is the share that end on an
occupied cell within `--reach` metres. Free cells below `--min-enclosed`, and free space then cut off from the main
room, are turned to unknown in the map being cleaned. Occupied cells are never changed.
The leak is found on the raw map on purpose: once clutter is erased, rays escape through gaps in the outer wall.
(`close_leaks`, a width-based variant, is kept for reference; it misses doorways as wide as the aisles.)
"""
import argparse
import os

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, PGM, UNK, pgm_to_cls


def _first_hit_right(cls):
    """Per cell: (class of the first non-free cell to its right, distance in cells). Edge of the image = unknown."""
    H, W = cls.shape
    nonfree = cls != FREE
    idx = np.where(nonfree, np.arange(W)[None, :], W)                    # column of a non-free cell, else W
    nxt = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]           # nearest non-free column at or right of each cell
    hit = np.full((H, W), UNK, np.uint8)
    ok = nxt < W
    rows = np.repeat(np.arange(H)[:, None], W, 1)
    hit[ok] = cls[rows[ok], nxt[ok]]
    return hit, (nxt - np.arange(W)[None, :]).astype(np.float32)


def enclosure(cls, n_dirs=16, reach_cells=160):
    """Share of rays from each cell that end on an occupied cell within reach (0..1)."""
    H, W = cls.shape
    pad = int(0.25 * max(H, W)) + 4
    big = cv2.copyMakeBorder(cls, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=UNK)
    Hb, Wb = big.shape
    score = np.zeros((Hb, Wb), np.float32)
    for k in range(n_dirs // 2):
        ang = 180.0 * k / (n_dirs // 2)
        M = cv2.getRotationMatrix2D((Wb / 2, Hb / 2), ang, 1.0)
        r = cv2.warpAffine(big, M, (Wb, Hb), flags=cv2.INTER_NEAREST, borderValue=UNK)
        for flip in (False, True):
            rr = r[:, ::-1] if flip else r
            hit, dist = _first_hit_right(np.ascontiguousarray(rr))
            good = ((hit == OCC) & (dist <= reach_cells)).astype(np.float32)
            if flip:
                good = good[:, ::-1]
            score += cv2.warpAffine(np.ascontiguousarray(good), M, (Wb, Hb), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
    return (score / n_dirs)[pad:pad + H, pad:pad + W]


def close_leaks(cls, neck_m=1.1, min_walled=0.5, res=0.05, return_parts=False):
    """-> (cleaned class map, mask of free cells turned unknown).
    1. The floor is split at every passage narrower than `neck_m`: each body of floor that holds together when such
       passages are pinched shut is one piece, and every other free cell joins the piece it reaches first.
    2. The largest piece is the room. Any other piece stays only if it is walled in: at least `min_walled` of the
       cells around it are occupied (walls, furniture) and not unknown. A leak through a door is bordered by unknown.
    Pieces that go become unknown. Occupied cells are never changed."""
    free = cls == FREE
    r = int(round(neck_m / res / 2))
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents((dist > r).astype(np.uint8), connectivity=8)
    if n <= 1:
        return (cls.copy(), np.zeros(cls.shape, bool)) + (([],) if return_parts else ())
    lab = lab.astype(np.float32)                                         # grow every piece through the floor until all free cells are taken
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(4 * max(cls.shape)):
        g = cv2.dilate(lab, k3)
        new = free & (lab == 0) & (g > 0)
        if not new.any():
            break
        lab[new] = g[new]
    lab = lab.astype(np.int32)
    area = np.bincount(lab.ravel(), minlength=n)
    area[0] = 0
    main = int(np.argmax(area))
    occ, unk = cls == OCC, cls == UNK
    keep = lab == main
    parts = []
    for i in range(1, n):
        if i == main or area[i] == 0:
            continue
        piece = lab == i
        ring = (cv2.dilate(piece.astype(np.uint8), k3) > 0) & ~piece
        walled = occ[ring].sum() / max(1, occ[ring].sum() + unk[ring].sum())
        ok = walled >= min_walled
        parts.append((int(area[i]), float(walled), bool(ok)))
        if ok:
            keep |= piece
    out = cls.copy()
    cut = free & ~keep
    out[cut] = UNK
    return (out, cut, parts) if return_parts else (out, cut)


def leak_mask(raw, min_enclosed=0.6, reach_m=8.0, res=0.05):
    """Free cells of the RAW map that lie in a leak: fewer than `min_enclosed` of the 16 rays from them end on an
    occupied cell within reach, or they are cut off from the main room once such cells are removed. Use the raw map:
    clutter stops rays as well as walls do, and erasing it first opens gaps the rays escape through."""
    enc = cv2.blur(enclosure(raw, reach_cells=int(reach_m / res)), (5, 5))
    free = raw == FREE
    inside = free & (enc >= min_enclosed)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros(raw.shape, bool)
    main = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    grown = cv2.dilate(main.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # give back the thin rim along walls
    leak = free & ~(main | (grown & free))
    # A leak runs out to the unknown space around the map. Poorly seen floor between benches also scores low, but it
    # only touches the unknown inside furniture, so it is left alone.
    unk = (raw == UNK).astype(np.uint8)
    nu, lu = cv2.connectedComponents(unk, connectivity=8)
    edge = np.unique(np.r_[lu[0], lu[-1], lu[:, 0], lu[:, -1]])
    outside = np.isin(lu, edge[edge > 0])
    touch = cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    nl, ll = cv2.connectedComponents(leak.astype(np.uint8), connectivity=8)
    ids = np.unique(ll[leak & touch])
    return np.isin(ll, ids[ids > 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', help='the raw map: the leak is found here')
    ap.add_argument('inp', help='the map to cut (the raw map again, or its cleaned version on the same grid)')
    ap.add_argument('out')
    ap.add_argument('--min-enclosed', type=float, default=0.6)
    ap.add_argument('--reach', type=float, default=8.0, help='metres a ray may travel before it counts as escaped')
    a = ap.parse_args()
    raw = pgm_to_cls(cv2.imread(a.raw, cv2.IMREAD_GRAYSCALE))
    cls = pgm_to_cls(cv2.imread(a.inp, cv2.IMREAD_GRAYSCALE))
    y_in = os.path.splitext(a.raw)[0] + '.yaml'
    res = 0.05
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in)); res = float(y['resolution'])
    m = leak_mask(raw, a.min_enclosed, a.reach, res) & (cls == FREE)
    out = cls.copy(); out[m] = UNK
    cv2.imwrite(a.out, PGM[out])
    if os.path.exists(y_in):
        y['image'] = os.path.basename(a.out)
        yaml.safe_dump(y, open(os.path.splitext(a.out)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f'{a.inp} -> {a.out}: {m.sum() * res * res:.1f} m2 of leaked free space turned unknown, {(out == FREE).sum() * res * res:.1f} m2 of floor left')


if __name__ == '__main__':
    main()
