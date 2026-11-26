"""Shared helpers for the map-cleaning work: map I/O, the three classes, alignment, warping, colours.

A map is a dict: name, cls (H x W uint8 of FREE / OCC / UNK), res (m per cell), origin (x, y of the
lower-left corner in metres, as in a Nav2 map .yaml). Row 0 of `cls` is the TOP of the map.
"""
import math
import os

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

FREE, OCC, UNK = 0, 1, 2
PGM = np.array([254, 0, 205], np.uint8)             # class id -> Nav2 trinary grey value
RGB = np.array([[255, 255, 255], [0, 0, 0], [205, 205, 205]], np.uint8)


def pgm_to_cls(img):
    c = np.full(img.shape, UNK, np.uint8)
    c[img > 250] = FREE
    c[img < 100] = OCC
    return c


def load_map(yaml_path):
    m = yaml.safe_load(open(yaml_path))
    img = cv2.imread(os.path.join(os.path.dirname(yaml_path), m['image']), cv2.IMREAD_GRAYSCALE)
    return {'name': os.path.splitext(os.path.basename(yaml_path))[0], 'cls': pgm_to_cls(img),
            'res': float(m['resolution']), 'origin': (float(m['origin'][0]), float(m['origin'][1])), 'path': yaml_path}


def save_map(stem, cls, res, origin):
    """Write stem.pgm + stem.yaml (Nav2 map_server format)."""
    cv2.imwrite(stem + '.pgm', PGM[cls])
    y = {'image': os.path.basename(stem) + '.pgm', 'mode': 'trinary', 'resolution': float(res),
         'origin': [round(float(origin[0]), 4), round(float(origin[1]), 4), 0], 'negate': 0,
         'occupied_thresh': 0.65, 'free_thresh': 0.25}
    yaml.safe_dump(y, open(stem + '.yaml', 'w'), default_flow_style=None, sort_keys=False)


def cells_xy(mask, res, origin):
    """Centres (metres, map frame) of the True cells of a mask."""
    ys, xs = np.nonzero(mask)
    return np.c_[origin[0] + (xs + 0.5) * res, origin[1] + (mask.shape[0] - ys - 0.5) * res]


def rot(th):
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def align(A, B, search=5.0, n_start=3):
    """Rigid transform (R, t) taking wall points A onto wall points B.

    Same method as docs/progress/compare_maps.py (coarse search over rotation and translation, then
    trimmed point-to-point ICP on the best 80 % of matches), with a wider translation search and ICP
    started from the best few coarse candidates, because these maps cover different extents.
    """
    tb = cKDTree(B)
    sub, ca, cb = A[::4], A.mean(0), B.mean(0)
    cand = []
    shifts = np.arange(-search, search + 1e-6, 0.5)
    for deg in range(0, 360, 3):
        R = rot(math.radians(deg))
        base = sub @ R.T + (cb - R @ ca)
        for dx in shifts:
            for dy in shifts:
                s = float(np.mean(np.minimum(tb.query(base + [dx, dy])[0], 0.5)))
                cand.append((s, deg, dx, dy))
    cand.sort()
    picked = []                                      # best few that are not near-duplicates
    for s, deg, dx, dy in cand:
        if all(abs((deg - d + 180) % 360 - 180) > 9 or abs(dx - x) + abs(dy - y) > 1.0 for _, d, x, y in picked):
            picked.append((s, deg, dx, dy))
        if len(picked) == n_start:
            break
    best = None
    for _, deg, dx, dy in picked:
        R = rot(math.radians(deg))
        t = cb - R @ ca + [dx, dy]
        for _ in range(60):
            d, i = tb.query(A @ R.T + t)
            k = d < np.percentile(d, 80)
            P, M = A[k], B[i[k]]
            pm, mm = P.mean(0), M.mean(0)
            U, _, Vt = np.linalg.svd((P - pm).T @ (M - mm))
            R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            t = mm - R @ pm
        d = tb.query(A @ R.T + t)[0]
        score = float(np.mean(np.sort(d)[:int(0.8 * len(d))]))
        if best is None or score < best[0]:
            best = (score, R, t, d)
    return best[1], best[2], best[3]


def warp(cls, res, origin, R, t, shape, origin_to):
    """Redraw a class image on another grid (`shape`, `origin_to`, same resolution) after moving it by (R, t)."""
    H, W = shape
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([origin_to[0] + (cols + 0.5) * res, origin_to[1] + (H - rows - 0.5) * res], -1)
    Q = (P - t) @ R                                   # target frame -> source frame
    mx = ((Q[..., 0] - origin[0]) / res - 0.5).astype(np.float32)
    my = (cls.shape[0] - (Q[..., 1] - origin[1]) / res - 0.5).astype(np.float32)
    return cv2.remap(cls, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=UNK)


def colour(cls, scale=1):
    img = RGB[cls][..., ::-1].copy()                  # BGR for cv2
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale != 1 else img


def label(img, text, colour_bgr=(0, 0, 200), size=0.6):
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 14 + int(len(text) * 19 * size)), int(34 * size) + 6), (255, 255, 255), -1)
    cv2.putText(img, text, (6, int(26 * size) + 2), cv2.FONT_HERSHEY_SIMPLEX, size, colour_bgr, 2, cv2.LINE_AA)
    return img


def wall_scores(pred_occ, true_occ, tol=1):
    """Wall recall and precision with a tolerance of `tol` cells (a cell counts if the other map has a wall within tol)."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    near_true = cv2.dilate(true_occ.astype(np.uint8), k) > 0
    near_pred = cv2.dilate(pred_occ.astype(np.uint8), k) > 0
    rec = float((true_occ & near_pred).sum() / max(1, true_occ.sum()))
    prec = float((pred_occ & near_true).sum() / max(1, pred_occ.sum()))
    return rec, prec
