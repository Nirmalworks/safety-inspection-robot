#!/usr/bin/env python3
"""Figures and numbers for the progress report / presentation (no ROS needed).

    python3 docs/progress/make_figures.py <placeholder dataset dir>    # from ml/fake_dataset.py

Writes docs/progress/figures/:
  map_lab.png              the 2D lidar map of the lab (debug_logs/20261113_195519/maps_2d)
  keep_clear_example.png   depth-rule output on a placeholder normal / hazard frame
  pitch_sensitivity.png    depth-rule recall and false-alarm rate vs camera-pitch calibration error
and prints the numbers used in the report. Placeholder (rendered) frames verify the pipeline;
they are NOT a measure of real accuracy.
"""
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'ame505_bringup'))
sys.path.insert(0, os.path.join(ROOT, 'ml'))
from ame505_bringup.keep_clear import DEFAULT_CAMERA, DEFAULT_ZONE, check, points_in_robot_frame  # noqa: E402
from evaluate import metrics  # noqa: E402

OUT = os.path.join(HERE, 'figures')
MAP = os.path.join(ROOT, 'debug_logs', '20261113_195519', 'maps_2d', 'lab_map_new')


def fig_map():
    meta = yaml.safe_load(open(MAP + '.yaml'))
    img = cv2.imread(MAP + '.pgm', cv2.IMREAD_GRAYSCALE)
    res, (ox, oy, _) = meta['resolution'], meta['origin']
    h, w = img.shape
    occupied = int((img < 100).sum())
    free = int((img > 250).sum())          # 0 in lab_map_new: saved with no free cells, only walls + unknown
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(img, cmap='gray', extent=[ox, ox + w * res, oy, oy + h * res])
    ax.plot(0, 0, 'o', color='tab:red', ms=6)
    ax.annotate('map origin (start pose)', (0, 0), (8, 8), textcoords='offset points', color='tab:red', fontsize=8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'2D lidar map of the lab ({w * res:.1f} x {h * res:.1f} m, {res * 100:.0f} cm cells)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'map_lab.png'), dpi=160)
    plt.close(fig)
    print(f'map: {w}x{h} cells at {res} m = {w * res:.1f} x {h * res:.1f} m; '
          f'occupied cells {occupied}, free cells {free}, unknown cells {h * w - occupied - free}')


def fig_example(dataset):
    rows = list(csv.DictReader(open(os.path.join(dataset, 'index.csv'))))
    pick = {s: next(r for r in rows if r['scene'] == s and r['objects'] == ('box' if s == 'hazard' else ''))
            for s in ('normal', 'hazard')}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (scene, r) in zip(axes, pick.items()):
        depth = cv2.imread(os.path.join(dataset, r['file_depth']), cv2.IMREAD_UNCHANGED)
        res = check(depth)
        z = depth.astype(np.float32) / 1000
        v, u = np.nonzero(z > 0)
        x, y, zz, _ = points_in_robot_frame(depth, DEFAULT_CAMERA)
        zone = DEFAULT_ZONE
        inside = ((x >= zone['x_min']) & (x <= zone['x_max']) & (np.abs(y) <= zone['half_width'])
                  & (zz >= zone['z_min']) & (zz <= zone['z_max']))
        ax.imshow(np.where(z > 0, z, np.nan), cmap='viridis', vmin=0, vmax=4)
        ax.scatter(u[inside], v[inside], s=1, c='tab:red')
        ax.set_title(f'{scene}: {"HAZARD" if res["hazard"] else "normal"} '
                     f'({res["area_m2"]} m$^2$ in zone)', fontsize=9)
        ax.axis('off')
    fig.suptitle('Depth keep-clear rule on placeholder frames (red = points inside the zone)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'keep_clear_example.png'), dpi=160)
    plt.close(fig)


def fig_pitch(dataset):
    """The rule assumes the camera pitch; frames are rendered level. A pitch error tilts the floor."""
    rows = list(csv.DictReader(open(os.path.join(dataset, 'index.csv'))))
    depths = [cv2.imread(os.path.join(dataset, r['file_depth']), cv2.IMREAD_UNCHANGED) for r in rows]
    truth = [r['scene'] == 'hazard' for r in rows]
    errs = np.arange(-4, 4.5, 0.5)
    rec, far = [], []
    for e in errs:
        pred = [check(d, camera={'pitch_deg': float(e)})['hazard'] for d in depths]
        m = metrics(list(zip(truth, pred)))
        rec.append(m['recall'])
        far.append(m['false_alarm_rate'])
        print(f'pitch error {e:+.1f} deg: recall {m["recall"]:.2f}, false-alarm rate {m["false_alarm_rate"]:.2f}')
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(errs, rec, 'o-', label='hazard recall')
    ax.plot(errs, far, 's-', label='false-alarm rate (normal frames)')
    ax.axhline(0.9, ls='--', lw=0.8, color='gray')
    ax.set_xlabel('camera pitch calibration error [deg]  (+ = assumed tilted down)')
    ax.set_ylabel('rate')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Depth rule vs pitch error ({len(rows)} placeholder frames)', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pitch_sensitivity.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_map()
    if len(sys.argv) > 1:
        fig_example(sys.argv[1])
        fig_pitch(sys.argv[1])
