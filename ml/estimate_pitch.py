#!/usr/bin/env python3
"""Estimate the depth camera's pitch and height from frames of EMPTY floor.

    python3 ml/estimate_pitch.py <depth.png> [<depth.png> ...]
    python3 ml/estimate_pitch.py --selftest

The keep-clear rule (keep_clear.py) assumes the camera pitch; a wrong value of about 2-3 degrees
makes the floor count as an object (docs/progress/figures/pitch_sensitivity.png). This fits a plane
to the floor pixels (RANSAC + least squares) and prints the pitch (> 0 = tilted down, the
convention of keep_clear.py) and the camera height above the floor, ready to paste into
src/ame505_bringup/config/inspection.yaml under `camera:`.

Use frames where at least the lower half of the image is bare floor (no objects within ~2 m).
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'ame505_bringup'))
from ame505_bringup.keep_clear import DEFAULT_CAMERA, points_in_robot_frame  # noqa: E402


def fit_floor(depth_mm, camera=None, iters=300, tol=0.01, seed=0):
    """Return (pitch_deg, height_m, inlier_fraction) for one depth image."""
    cam = {**DEFAULT_CAMERA, **(camera or {}), 'pitch_deg': 0.0, 'x': 0.0, 'height': 0.0}
    fwd, lat, up, _ = points_in_robot_frame(depth_mm, cam)       # level camera frame, origin at the lens
    drop = -up
    keep = (drop > 0.05) & (fwd > 0.3) & (fwd < 3.0)              # below the lens, within 3 m
    P = np.stack([fwd[keep], lat[keep], drop[keep]], 1)
    if len(P) < 200:
        raise ValueError(f'only {len(P)} candidate floor points; is the floor visible?')
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    centre = Q.mean(0)
    n = np.linalg.svd(Q - centre, full_matrices=False)[2][-1]     # least-squares refit on inliers
    if n[2] < 0:
        n = -n
    # floor: fwd*sin(p) + drop*cos(p) = h  (see keep_clear.points_in_robot_frame)
    pitch = math.degrees(math.atan2(n[0], n[2]))
    return pitch, float(n @ centre), float(best.mean())


def selftest():
    """Render bare floor seen by a camera pitched by a known angle and recover it."""
    cam, h = DEFAULT_CAMERA, 0.207
    v, _ = np.mgrid[0:240, 0:320].astype(np.float32)
    down = (v - cam['cy']) / cam['fy']
    worst = 0.0
    for true_pitch in (-3.0, -1.0, 0.0, 2.0, 5.0):
        p = math.radians(true_pitch)
        drop_per_z = math.sin(p) + down * math.cos(p)             # per metre of optical depth
        with np.errstate(divide='ignore'):
            z = np.where(drop_per_z > 0, h / drop_per_z, 0)
        z[z > 6] = 0
        est, height, _ = fit_floor((z * 1000).astype(np.uint16))
        worst = max(worst, abs(est - true_pitch))
        print(f'true pitch {true_pitch:+.1f} deg -> estimated {est:+.2f} deg, height {height:.3f} m')
    assert worst < 0.3, worst
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('depth', nargs='*', help='16-bit depth PNGs (mm) of bare floor')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.depth:
        ap.error('give depth images or --selftest')
    res = []
    for f in a.depth:
        try:
            p, h, frac = fit_floor(cv2.imread(f, cv2.IMREAD_UNCHANGED))
        except ValueError as e:
            print(f'{f}: skipped ({e})')
            continue
        res.append((p, h))
        print(f'{os.path.basename(f)}: pitch {p:+.2f} deg, height {h:.3f} m, floor inliers {100 * frac:.0f} %')
    if res:
        p, h = np.median(np.array(res), 0)
        spread = np.ptp([r[0] for r in res]) if len(res) > 1 else 0.0
        print(f'\nmedian of {len(res)}: pitch {p:+.2f} deg (spread {spread:.2f}), height {h:.3f} m')
        print(f'inspection.yaml ->  camera: {{pitch_deg: {p:.1f}, height: {h:.3f}}}')


if __name__ == '__main__':
    main()
