#!/usr/bin/env python3
"""Top-down lidar map of a speed-test run: every scan placed at the robot's position.

    python3 speed_map.py <run folder>      writes <run>/graphs/lidar_map.png

The robot only drives straight forward and back in this test, so its position along the
run is how much closer the wall straight ahead is than at the start. Each scan is shifted
by that amount, which stacks them into a map of what is in front of the robot.
"""
import math
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402

C = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speed_test.yaml')))['speed_test']
YAW, CONE = math.radians(C['lidar_yaw_deg']), math.radians(C['wall_cone_deg'])
CMD, MEAS, INK, MUTED, GRID = '#e8871e', '#1b8fa6', '#1d2430', '#6b7686', '#e6e9ee'


def main():
    run = os.path.abspath(sys.argv[1])
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.join(run, 'bag'), storage_id='sqlite3'), ConverterOptions('', ''))
    scans, events = [], []
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == '/scan':
            m = deserialize_message(data, LaserScan)
            a = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment + YAW
            a = np.arctan2(np.sin(a), np.cos(a))
            d = np.array(m.ranges)
            ok = np.isfinite(d) & (d > 0.15)
            ahead = ok & (np.abs(a) < CONE)
            if ahead.sum() >= 3:
                scans.append((t / 1e9, float(np.median(d[ahead])), d[ok] * np.cos(a[ok]), d[ok] * np.sin(a[ok])))
        elif topic == '/speed_test/event':
            events.append((t / 1e9, deserialize_message(data, String).data))
    if not scans:
        print('no scans')
        return
    start_wall = statistics.median(w for _, w, _, _ in scans[:10])
    xs, ys, ts = [], [], []
    path = []
    for i, (t, w, x, y) in enumerate(scans):
        pos = start_wall - w                        # how far the robot has moved forward
        path.append((t, pos))
        if i % 2 == 0:                              # every 2nd scan is plenty
            xs.append(x + pos); ys.append(y); ts.append(np.full(len(x), t - scans[0][0]))
    xs, ys, ts = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(ys, xs, c=ts, s=1.2, cmap='viridis', alpha=0.5, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('time in run (s)', color=MUTED, fontsize=8)
    # robot path and the stop points of each run
    pp = np.array(path)
    ax.plot(np.zeros(len(pp)), pp[:, 1], color=CMD, lw=6, alpha=0.25, solid_capstyle='round', label='robot path')
    stops = []
    for t, txt in events:
        m = re.match(r'rest v=([\d.]+)', txt)
        if m:
            pos = pp[np.argmin(np.abs(pp[:, 0] - t)), 1]
            stops.append((m.group(1), pos))
            ax.plot(0, pos, 'o', color=CMD, ms=7, mec='white')
    if stops:
        label = 'hard stops at\n' + '\n'.join(f'{p:.2f} m after {v} m/s' for v, p in stops)
        ax.annotate(label, (0, max(p for _, p in stops)), xytext=(-18, 6), textcoords='offset points',
                    fontsize=8, color=INK, ha='right', va='bottom')
    ax.add_patch(plt.Rectangle((-0.12, -0.15), 0.24, 0.30, color=CMD, alpha=0.9, label='robot at start (to scale)'))
    ax.annotate('', xy=(0, 0.45), xytext=(0, 0.15), arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
    ax.axhline(start_wall, color=MEAS, lw=1, ls='--', alpha=0.7)
    ax.text(ax.get_xlim()[0] if False else -0.05, start_wall, f'  wall {start_wall:.2f} m ahead at start', color=MEAS,
            fontsize=8, va='bottom', ha='right')
    ax.set_aspect('equal')
    ax.invert_xaxis()                                # robot's left on the left of the picture
    ax.set_xlabel('left  ←  sideways (m)  →  right', color=MUTED, fontsize=9)
    ax.set_ylabel('forward from the start (m)', color=MUTED, fontsize=9)
    ax.set_title(f'Lidar map of the test area · {os.path.basename(run)}', loc='left', fontsize=12, color=INK, fontweight='bold')
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(run, 'graphs', 'lidar_map.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, facecolor='white')
    print(out)


if __name__ == '__main__':
    main()
