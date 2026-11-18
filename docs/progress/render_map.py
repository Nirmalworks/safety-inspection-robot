#!/usr/bin/env python3
"""Presentation renderings of a saved 2D lidar map (slam_toolbox / map_saver .pgm + .yaml).

    python3 docs/progress/render_map.py [<map.yaml>]     # default: the 2026-11-13 lab map
    python3 docs/progress/render_map.py maps/lab2d_11_18_2/maps_2d/lab2d_11_18_2.yaml \
        --checkpoints maps/lab2d_11_18_2/checkpoints/lab2d_11_18_2.yaml --name map_lab_1118 --date 2026-11-18

Writes to docs/progress/figures/ (<name> defaults to map_lab):
  <name>_dark.png    slide version: walls with a glow on a dark background, metric grid,
                      scale bar, start pose and robot footprint to scale
  map_lab_light.png   the same for the printed report
  map_lab_3d.png      wall cells extruded to a nominal height, perspective view
Only cells the lidar marked occupied are drawn; nothing is filled in or smoothed.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, '..', '..', 'debug_logs', '20261113_195519', 'maps_2d', 'lab_map_new.yaml')
OUT = os.path.join(HERE, 'figures')
FOOTPRINT = (0.33, 0.28)            # JetAuto base, m (ame505_description)
SUBTITLE = 'slam_toolbox · RPLidar A1 · AMCL start pose · mapped {date}'

THEMES = {
    'dark': dict(bg='#0b1020', unknown='#111a2e', grid='#23304d', wall='#5ee6ff', glow='#1fb6ff',
                 text='#e6edf7', muted='#8b9bb8', accent='#ff6b6b'),
    'light': dict(bg='#ffffff', unknown='#f3f5f9', grid='#dde3ec', wall='#0b3d91', glow='#5b8def',
                  text='#141a24', muted='#5d6b82', accent='#d62f2f'),
}


def load(path):
    meta = yaml.safe_load(open(path))
    img = cv2.imread(os.path.join(os.path.dirname(path), meta['image']), cv2.IMREAD_GRAYSCALE)
    occ = img < 255 * (1 - meta.get('occupied_thresh', 0.65))   # trinary: 0 = occupied
    res = meta['resolution']
    ox, oy = meta['origin'][:2]
    occ = occ[::-1]                                              # row 0 = lowest y
    ys, xs = np.nonzero(occ)
    pad = 10
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad, occ.shape[0] - 1)
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad, occ.shape[1] - 1)
    crop = occ[r0:r1 + 1, c0:c1 + 1]
    extent = [ox + c0 * res, ox + (c1 + 1) * res, oy + r0 * res, oy + (r1 + 1) * res]
    return crop, extent, res, int(occ.sum()), (ox + xs.min() * res, ox + (xs.max() + 1) * res,
                                               oy + ys.min() * res, oy + (ys.max() + 1) * res)


def flat(crop, extent, res, n_occ, bbox, theme, name, checkpoints=(), subtitle=SUBTITLE):
    t = THEMES[theme]
    w_m, h_m = extent[1] - extent[0], extent[3] - extent[2]
    fig = plt.figure(figsize=(13, 13 * h_m / w_m + 1.6), facecolor=t['bg'])
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.78], facecolor=t['unknown'])
    # glow: blurred copies of the wall mask under the sharp walls
    m = crop.astype(np.float32)
    for sigma, alpha in ((6, 0.35), (2.5, 0.55)):
        g = cv2.GaussianBlur(m, (0, 0), sigma)
        g = g / g.max()
        rgba = np.zeros(g.shape + (4,))
        rgba[..., :3] = matplotlib.colors.to_rgb(t['glow'])
        rgba[..., 3] = np.clip(g * 2.2, 0, 1) * alpha
        ax.imshow(rgba, extent=extent, origin='lower', interpolation='bilinear', zorder=2)
    walls = np.zeros(crop.shape + (4,))
    walls[..., :3] = matplotlib.colors.to_rgb(t['wall'])
    walls[..., 3] = crop
    ax.imshow(walls, extent=extent, origin='lower', interpolation='nearest', zorder=3)
    # 1 m grid
    for x in np.arange(np.ceil(extent[0]), extent[1], 1.0):
        ax.axvline(x, color=t['grid'], lw=0.6 if x % 5 else 1.2, zorder=1)
    for y in np.arange(np.ceil(extent[2]), extent[3], 1.0):
        ax.axhline(y, color=t['grid'], lw=0.6 if y % 5 else 1.2, zorder=1)
    # start pose + robot footprint to scale
    fx, fy = FOOTPRINT
    ax.add_patch(Rectangle((-fx / 2, -fy / 2), fx, fy, fc=t['accent'], ec='none', zorder=5))
    ax.add_patch(FancyArrow(0, 0, 0.9, 0, width=0.06, head_width=0.28, head_length=0.25,
                            color=t['accent'], zorder=5, lw=0))
    ax.annotate('start / home pose\n(robot footprint to scale)', (0, 0), (14, -34), textcoords='offset points',
                color=t['text'], fontsize=10, zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground=t['unknown'])])
    # patrol checkpoints: numbered disc + the direction the camera faces there
    for i, c in enumerate(checkpoints, 1):
        x, y, yaw = c['x'], c['y'], c.get('yaw', 0.0)
        ax.add_patch(FancyArrow(x, y, 0.75 * np.cos(yaw), 0.75 * np.sin(yaw), width=0.05, head_width=0.22,
                                head_length=0.2, color='#ffb020', zorder=6, lw=0))
        ax.scatter([x], [y], s=330, color='#ffb020', edgecolors=t['bg'], linewidths=1.5, zorder=7)
        ax.text(x, y, str(i), color='#111', ha='center', va='center', fontsize=10, weight='bold', zorder=8)
        ax.annotate(c.get('id', f'cp_{i:02d}'), (x, y), (12, 10), textcoords='offset points', color=t['text'],
                    fontsize=10, zorder=8, path_effects=[pe.withStroke(linewidth=3, foreground=t['unknown'])])
    # scale bar
    sx, sy = extent[0] + 0.6, extent[2] + 0.5
    ax.plot([sx, sx + 2], [sy, sy], color=t['text'], lw=3, solid_capstyle='butt', zorder=6)
    ax.text(sx + 1, sy + 0.18, '2 m', color=t['text'], ha='center', va='bottom', fontsize=10, zorder=6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.tick_params(colors=t['muted'], labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel('x [m]  (map frame)', color=t['muted'], fontsize=10)
    ax.set_ylabel('y [m]', color=t['muted'], fontsize=10)
    bx = bbox[1] - bbox[0]
    by = bbox[3] - bbox[2]
    fig.text(0.05, 0.955, 'Lab map from 2D lidar SLAM', color=t['text'], fontsize=22, weight='bold', va='top')
    fig.text(0.05, 0.905, subtitle + (f' · {len(checkpoints)} checkpoints'
                                      if checkpoints else ''), color=t['muted'], fontsize=12, va='top')
    fig.text(0.95, 0.955, f'{bx:.1f} m × {by:.1f} m', color=t['text'], fontsize=18, weight='bold',
             ha='right', va='top')
    fig.text(0.95, 0.905, f'{res * 100:.0f} cm cells · {n_occ:,} occupied cells', color=t['muted'],
             fontsize=12, ha='right', va='top')
    fig.text(0.05, 0.025, 'Only lidar-confirmed obstacles are drawn (walls, furniture, table and chair legs). '
             'Dark = not marked free in the saved map.', color=t['muted'], fontsize=9)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, facecolor=t['bg'])
    plt.close(fig)
    return path


def extruded(crop, extent, res, theme='dark', name='map_lab_3d.png', height=0.6, subtitle=SUBTITLE):
    t = THEMES[theme]
    ys, xs = np.nonzero(crop)
    x = extent[0] + xs * res
    y = extent[2] + ys * res
    fig = plt.figure(figsize=(13, 7.5), facecolor=t['bg'])
    ax = fig.add_axes([0, 0.02, 1, 0.86], projection='3d', facecolor=t['bg'])
    order = np.argsort(-(y - x * 0.2))                           # draw far cells first
    shade = 0.55 + 0.45 * (y - y.min()) / max(np.ptp(y), 1e-6)
    cols = [(*np.array(matplotlib.colors.to_rgb(t['wall'])) * s, 1.0) for s in shade[order]]
    ax.bar3d(x[order], y[order], np.zeros(len(order)), res, res, height, color=cols, shade=True,
             edgecolor='none', zsort='max')
    fx, fy = FOOTPRINT
    ax.bar3d(-fx / 2, -fy / 2, 0, fx, fy, 0.25, color=t['accent'], shade=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(0, height * 1.5)
    ax.set_box_aspect((extent[1] - extent[0], extent[3] - extent[2], height * 1.5), zoom=1.55)
    ax.view_init(elev=42, azim=-65)
    ax.set_axis_off()
    fig.text(0.04, 0.95, 'Lab map: lidar obstacles extruded', color=t['text'], fontsize=20, weight='bold', va='top')
    fig.text(0.04, 0.90, f'{subtitle} · walls drawn {height} m high for display · robot in red',
             color=t['muted'], fontsize=11, va='top')
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180, facecolor=t['bg'])
    plt.close(fig)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('map', nargs='?', default=DEFAULT, help='map .yaml (map_saver)')
    ap.add_argument('--checkpoints', help='checkpoints .yaml (dashboard / checkpoint_recorder)')
    ap.add_argument('--name', default='map_lab', help='output file prefix')
    ap.add_argument('--date', default='2026-11-13', help='mapping date for the subtitle')
    a = ap.parse_args()
    cps = (yaml.safe_load(open(a.checkpoints)) or {}).get('checkpoints', []) if a.checkpoints else []
    sub = SUBTITLE.format(date=a.date)
    os.makedirs(OUT, exist_ok=True)
    crop, extent, res, n_occ, bbox = load(a.map)
    print(flat(crop, extent, res, n_occ, bbox, 'dark', f'{a.name}_dark.png', cps, sub))
    print(flat(crop, extent, res, n_occ, bbox, 'light', f'{a.name}_light.png', cps, sub))
    try:
        print(extruded(crop, extent, res, name=f'{a.name}_3d.png', subtitle=sub))
    except ValueError as e:      # Jetson: the system mpl_toolkits shadows a pip matplotlib -> no '3d'
        print(f'3D view skipped ({e}); run on a laptop for {a.name}_3d.png')
