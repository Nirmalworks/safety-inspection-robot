"""Calibration-free geometry for the depth camera: per-frame floor fit and the HHA encoding.

The keep-clear rule needs the camera's pitch and height. A fixed hand-measured value breaks when the
camera is knocked or the robot rocks (2.5 deg of error -> 82 % false alarms on rendered scenes), so every
frame estimates its own floor instead:

* fit_floor_constrained(): Ground Plane Fitting in the style of Zermas et al. (ICRA 2017): seeds are
  the LOWEST points (the floor is the lowest large surface seen), then RANSAC (Fischler & Bolles 1981)
  restricted to near-horizontal planes at a plausible camera height, then a least-squares refit on the
  inliers. The restriction matters: at most checkpoints the robot faces an exit door from 1-2 m, and
  an unconstrained fit locks onto the door (a vertical plane) instead of the floor.
* height_above_floor(): every pixel's height above the fitted floor (metres).
* hha(): the geocentric HHA encoding of Gupta et al. (ECCV 2014), three 8-bit channels from depth:
  horizontal disparity, height above ground, angle of the surface normal with gravity. Gravity is the
  fitted floor normal, so the encoding does not depend on how the camera is mounted.
No ROS, NumPy + OpenCV only.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'ame505_bringup'))
from ame505_bringup.keep_clear import DEFAULT_CAMERA  # noqa: E402

PITCH_LIMIT_DEG = 20.0          # floor must be within this of level (camera mounted ~level)
HEIGHT_RANGE_M = (0.10, 0.35)   # camera height above the floor (URDF: 0.207 m)


def camera_points(depth_mm, camera=None):
    """Pixels -> 3D points in a level camera frame at the lens: fwd, lat (left), drop (down), and pixel rows/cols."""
    cam = {**DEFAULT_CAMERA, **(camera or {})}
    z = depth_mm.astype(np.float32) / 1000.0
    v, u = np.nonzero(z > 0)
    z = z[v, u]
    fwd = z
    lat = -(u - cam['cx']) / cam['fx'] * z
    drop = (v - cam['cy']) / cam['fy'] * z
    return np.stack([fwd, lat, drop], 1), v, u


def fit_floor_constrained(depth_mm, camera=None, iters=200, tol=0.015, seed_frac=0.25, seed=0):
    """Return dict(pitch_deg, height_m, inlier_frac, ok). ok=False -> the default mount is returned."""
    cam = {**DEFAULT_CAMERA, **(camera or {})}
    P, _, _ = camera_points(depth_mm, cam)
    near = (P[:, 0] > 0.3) & (P[:, 0] < 3.0) & (P[:, 2] > 0.02)      # in front, below the lens
    P = P[near]
    fallback = {'pitch_deg': cam['pitch_deg'], 'height_m': cam['height'], 'inlier_frac': 0.0, 'ok': False}
    if len(P) < 300:
        return fallback
    # lowest-point seeds (Zermas et al.): the floor is the lowest large surface; per forward-distance band,
    # keep the lowest points so a near wall or door cannot dominate the candidates
    bands = np.digitize(P[:, 0], np.arange(0.3, 3.0, 0.25))
    seeds = []
    for b in np.unique(bands):
        idx = np.nonzero(bands == b)[0]
        k = max(10, int(len(idx) * seed_frac))
        seeds.append(idx[np.argsort(-P[idx, 2])[:k]])                 # largest drop = lowest
    S = P[np.concatenate(seeds)]
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    cos_lim = math.cos(math.radians(PITCH_LIMIT_DEG))
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if n[2] < cos_lim:                                             # not near-horizontal
            continue
        h = float(n @ a)
        if not HEIGHT_RANGE_M[0] <= h <= HEIGHT_RANGE_M[1]:
            continue
        cnt = int((np.abs(P @ n - h) < tol).sum())
        if cnt > best_n:
            best, best_n = (n, h), cnt
    if best is None or best_n < 200:
        return fallback
    n, h = best
    Q = P[np.abs(P @ n - h) < tol]
    centre = Q.mean(0)
    n = np.linalg.svd(Q - centre, full_matrices=False)[2][-1]
    if n[2] < 0:
        n = -n
    h = float(n @ centre)
    pitch = math.degrees(math.atan2(n[0], n[2]))                      # > 0 = camera tilted down
    ok = n[2] >= cos_lim and HEIGHT_RANGE_M[0] <= h <= HEIGHT_RANGE_M[1]
    if not ok:
        return fallback
    return {'pitch_deg': pitch, 'height_m': h, 'inlier_frac': len(Q) / len(P), 'ok': True}


def floor_normal(pitch_deg):
    p = math.radians(pitch_deg)
    return np.array([math.sin(p), 0.0, math.cos(p)])                  # in the level camera frame (fwd, lat, drop)


def height_above_floor(depth_mm, floor, camera=None):
    """HxW float32 height above the fitted floor (m); 0 where depth is missing."""
    P, v, u = camera_points(depth_mm, camera)
    hmap = np.zeros(depth_mm.shape, np.float32)
    hmap[v, u] = floor['height_m'] - P @ floor_normal(floor['pitch_deg'])
    return hmap


def hha(depth_mm, floor=None, camera=None):
    """HxWx3 uint8 HHA image (Gupta et al. 2014): disparity, height above ground, angle with gravity."""
    cam = {**DEFAULT_CAMERA, **(camera or {})}
    floor = floor or fit_floor_constrained(depth_mm, cam)
    z = depth_mm.astype(np.float32) / 1000.0
    valid = z > 0
    # horizontal disparity ~ 1 / depth, scaled for 0.3-6 m
    disp = np.where(valid, 1.0 / np.maximum(z, 0.3), 0)
    c1 = np.clip(disp / (1 / 0.3) * 255, 0, 255)
    # height above the fitted floor, 0-2 m
    c2 = np.clip(height_above_floor(depth_mm, floor, cam) / 2.0 * 255, 0, 255)
    # surface normal from neighbouring 3D points (cross product of image-space tangents), angle with gravity
    H, W = z.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
    X = np.stack([z, -(uu - cam['cx']) / cam['fx'] * z, (vv - cam['cy']) / cam['fy'] * z], -1)
    zs = cv2.blur(z, (5, 5))
    Xs = np.stack([cv2.blur(X[..., i], (5, 5)) for i in range(3)], -1)
    dx = np.zeros_like(Xs)
    dy = np.zeros_like(Xs)
    dx[:, 1:-1] = Xs[:, 2:] - Xs[:, :-2]
    dy[1:-1] = Xs[2:] - Xs[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    g = floor_normal(floor['pitch_deg'])
    ang = np.degrees(np.arccos(np.clip(np.abs(n @ g), 0, 1)))        # 0 = horizontal surface, 90 = vertical
    c3 = np.clip(ang / 90.0 * 255, 0, 255)
    out = np.stack([c1, c2, c3], -1)
    out[~valid | (zs <= 0)] = 0
    return out.astype(np.uint8)
