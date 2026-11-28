#!/usr/bin/env python3
"""Synthetic hazard frames by RGB-D cut-and-paste (after Dwibedi et al., "Cut, Paste and Learn", ICCV 2017).

    python3 ml/cutpaste.py <dataset> [--per-scene-pair 6] [--seed 0]

Cut: in a frame of a hazard burst, the obstacle is what stands ON the floor and IN FRONT of the background:
  height above the per-frame floor (ml/ground.py) 0.03-0.90 m, and at least 0.12 m nearer than the
  background of its image column (the door, wall or far end seen above it). The largest such region is
  kept; regions touching the top of the image (people, doors, benches) are rejected, so pasted objects
  are never people (team rule: a person in the zone is normal).
Depth and RGB of this camera are NOT registered (offset ~ -14 px x, +10..+40 px y), so each modality is cut in
  its own coordinates: the depth mask as above, the RGB mask by GrabCut started from the depth box moved by the
  median offset. Both are pasted with the same shift, which keeps the camera's own offset in the synthetic frame.
Paste: into a frame of a NORMAL burst, at the same image row (same distance and perspective; the camera height
  is the same on every frame) with a random horizontal shift; depth is composited by "nearer wins" and the paste
  is skipped when the object would be behind the target scene; RGB edges are feathered to avoid the boundary
  artefacts the paper warns about.
Writes <dataset>/synthetic/*.png and synthetic.csv (file_rgb, file_depth, source_scene, target_scene).
ml/loso_eval.py --synthetic uses a synthetic frame in a fold only if BOTH its source and its target scene
are training scenes of that fold.
"""
import argparse
import collections
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ground import DEFAULT_CAMERA, fit_floor_constrained, height_above_floor  # noqa: E402


def scene_of(r):
    return f'{r["session"]}|{r["checkpoint"]}|{r["hazard"]}'


def segment_object(depth_mm, floor):
    """Boolean mask of the obstacle standing in front of the background, or None."""
    z = depth_mm.astype(np.float32) / 1000.0
    valid = z > 0
    h = height_above_floor(depth_mm, floor)
    H, W = z.shape
    # depth edges separate an object from the door / wall behind it: a jump of > 6 cm + 4 % of the range
    zf = np.where(valid, z, 0)
    jump = np.zeros_like(valid)
    dv = np.abs(np.diff(zf, axis=0))
    dh = np.abs(np.diff(zf, axis=1))
    tol_v = 0.06 + 0.04 * np.maximum(zf[1:], zf[:-1])
    tol_h = 0.06 + 0.04 * np.maximum(zf[:, 1:], zf[:, :-1])
    jump[1:] |= dv > tol_v
    jump[:-1] |= dv > tol_v
    jump[:, 1:] |= dh > tol_h
    jump[:, :-1] |= dh > tol_h
    cand = valid & ~jump & (h > 0.04) & (z > 0.4) & (z < 3.0)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 4)
    rows_idx = np.arange(H)[:, None]
    top_valid = np.where(valid.any(0), np.argmax(valid, 0), H)          # first row with depth, per column
    best, score = None, 0
    for i in range(1, n):
        x, y, w, hh, a = stats[i]
        if a < 300:
            continue
        m = lab == i
        cols = np.nonzero(m.any(0))[0]
        top = np.argmax(m[:, cols], 0)
        if np.mean(top <= top_valid[cols] + 4) > 0.3:            # reaches the top of the depth image: door, wall, person
            continue
        hs = h[m]
        if np.percentile(hs, 95) > 0.95 or np.percentile(hs, 5) > 0.15:   # taller than ~1 m, or not on the floor
            continue
        if w > 0.7 * W:                                          # spans the view: a bench front or wall base
            continue
        if w * np.median(z[m]) / DEFAULT_CAMERA['fx'] > 0.9:     # wider than 0.9 m: bench or cabinet, not a movable obstacle
            continue
        if a > score:
            best, score = i, a
    if best is None:
        return None
    m = lab == best
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


RGB_OFFSET = (-14, 24)          # median depth -> RGB pixel offset measured on the 2026-11-24/25 data (not registered)


def rgb_mask(bgr, depth_mask, margin=28):
    """Object mask in RGB coordinates: GrabCut (Rother et al. 2004) started from the depth box moved by the
    measured offset and widened by `margin` px to cover the offset's spread. None if GrabCut finds nothing."""
    H, W = depth_mask.shape
    ys, xs = np.nonzero(depth_mask)
    x0, x1 = xs.min() + RGB_OFFSET[0] - margin, xs.max() + RGB_OFFSET[0] + margin
    y0, y1 = ys.min() + RGB_OFFSET[1] - margin, ys.max() + RGB_OFFSET[1] + margin
    x0, y0, x1, y1 = max(x0, 1), max(y0, 1), min(x1, W - 2), min(y1, H - 2)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    gc = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None
    m = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ratio = m.sum() / depth_mask.sum()
    return m if 0.4 < ratio < 2.5 else None                          # GrabCut grabbed far too much or too little


def paste(src_rgb, src_d, dmask, cmask, tgt_rgb, tgt_d, shift, flip):
    """Paste the object: depth pixels by the depth mask, colour pixels by the RGB mask, both moved by the same
    shift, so the synthetic frame keeps the camera's own depth/RGB offset. None when the object would be hidden."""
    if flip:   # mirror about the centre of each modality's own object, so both stay at the same scene position
        src_rgb, src_d, dmask, cmask = src_rgb[:, ::-1], src_d[:, ::-1], dmask[:, ::-1], cmask[:, ::-1]
    H, W = dmask.shape
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    mv = lambda im, interp=cv2.INTER_NEAREST: cv2.warpAffine(np.ascontiguousarray(im), M, (W, H), flags=interp)  # noqa: E731
    dm, cm = mv(dmask.astype(np.uint8)).astype(bool), mv(cmask.astype(np.uint8)).astype(bool)
    s_rgb, s_d = mv(src_rgb, cv2.INTER_LINEAR), mv(src_d)
    if dm.sum() < 300 or cm.sum() < 300:
        return None
    td, sd = tgt_d.astype(np.float32), s_d.astype(np.float32)
    if (dm & (td > 0) & (sd > td - 50)).sum() > 0.3 * dm.sum():     # behind the target scene: skip
        return None
    front = dm & (sd > 0) & ((td == 0) | (sd < td))
    depth = tgt_d.copy()
    depth[front] = s_d[front]
    alpha = cv2.GaussianBlur(cm.astype(np.float32), (5, 5), 1.5)[..., None]
    rgb = (alpha * s_rgb.astype(np.float32) + (1 - alpha) * tgt_rgb.astype(np.float32)).astype(np.uint8)
    return rgb, depth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--per-scene-pair', type=int, default=6, help='synthetic frames per (hazard scene, normal scene) pair')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--preview', help='save a contact sheet of examples here')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = list(csv.DictReader(open(os.path.join(a.dataset, 'splits.csv'))))
    by = collections.defaultdict(list)
    for r in rows:
        by[scene_of(r)].append(r)
    load = lambda r: (cv2.imread(os.path.join(a.dataset, r['file_rgb'])),  # noqa: E731
                      cv2.imread(os.path.join(a.dataset, r['file_depth']), cv2.IMREAD_UNCHANGED))
    # objects: from every 10th frame of each hazard burst (open door is not an object: skipped)
    objects = collections.defaultdict(list)
    for sc, rs in by.items():
        if rs[0]['scene'] != 'hazard' or rs[0]['hazard'] == 'open_door':
            continue
        for r in rs[::10]:
            bgr, d = load(r)
            f = fit_floor_constrained(d)
            if not f['ok']:
                continue
            m = segment_object(d, f)
            if m is None:
                continue
            cm = rgb_mask(bgr, m)
            if cm is not None:
                objects[sc].append((bgr, d, m, cm))
        print(f'objects cut from {sc}: {len(objects[sc])}')
    normals = [sc for sc, rs in by.items() if rs[0]['scene'] == 'normal']
    out_dir = os.path.join(a.dataset, 'synthetic')
    os.makedirs(out_dir, exist_ok=True)
    out, previews = [], []
    for src in objects:
        if not objects[src]:
            continue
        for tgt in normals:
            made, tries = 0, 0
            while made < a.per_scene_pair and tries < 6 * a.per_scene_pair:
                tries += 1
                bgr, d, m, cm = objects[src][rng.integers(len(objects[src]))]
                tr = by[tgt][rng.integers(len(by[tgt]))]
                tbgr, td = load(tr)
                res = paste(bgr, d, m, cm, tbgr, td, int(rng.integers(-50, 51)), False)
                if res is None:
                    continue
                rgb, depth = res
                name = f'{len(out):05d}'
                cv2.imwrite(os.path.join(out_dir, name + '_rgb.png'), rgb)
                cv2.imwrite(os.path.join(out_dir, name + '_depth.png'), depth)
                out.append({'file_rgb': f'synthetic/{name}_rgb.png', 'file_depth': f'synthetic/{name}_depth.png',
                            'source_scene': src, 'target_scene': tgt})
                if len(previews) < 24 and rng.random() < 0.1:
                    previews.append(np.hstack([cv2.resize(tbgr, (160, 120)), cv2.resize(rgb, (160, 120))]))
                made += 1
    with open(os.path.join(a.dataset, 'synthetic.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, ['file_rgb', 'file_depth', 'source_scene', 'target_scene'])
        w.writeheader()
        w.writerows(out)
    print(f'{len(out)} synthetic hazard frames from {sum(1 for s in objects if objects[s])} object scenes x '
          f'{len(normals)} normal scenes -> {a.dataset}/synthetic.csv')
    if a.preview and previews:
        while len(previews) % 3:
            previews.append(np.zeros_like(previews[0]))
        cv2.imwrite(a.preview, np.vstack([np.hstack(previews[i:i + 3]) for i in range(0, len(previews), 3)]))


if __name__ == '__main__':
    main()
