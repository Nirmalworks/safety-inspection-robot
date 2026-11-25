#!/usr/bin/env python3
"""Graphs for one speed-test run, saved next to its data.

    python3 speed_plot.py <run folder>      e.g. CSCI-513/jetauto_recordings/speed_test_20261108_0902

Reads <run>/analysis/*.csv (from speed_analysis.py) and <run>/speed_test_log.txt; writes
<run>/graphs/speed_test.png (timeline: lidar distance, speed, IMU, ping) and, when the
run has finished trials, <run>/graphs/results.png (measured speed and braking per speed).
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CMD, MEAS, IMU, PING, GRID, INK, MUTED = '#e8871e', '#1b8fa6', '#6a4fb3', '#3a7d44', '#e6e9ee', '#1d2430', '#6b7686'


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rs, k):
    return [float(r[k]) if r[k] not in ('', 'nan') else float('nan') for r in rs]


def style(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=11, color=INK, fontweight='bold')
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c5ccd6')
    ax.tick_params(colors=MUTED, labelsize=8)


def empty(ax, text):
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color=MUTED, fontsize=10, wrap=True)


def main():
    run = os.path.abspath(sys.argv[1])
    an, out = os.path.join(run, 'analysis'), os.path.join(run, 'graphs')
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(run)
    wall, imu = rows(os.path.join(an, 'wall_timeline.csv')), rows(os.path.join(an, 'imu_timeline.csv'))
    ping, trials = rows(os.path.join(an, 'ping_rtt.csv')), rows(os.path.join(an, 'trials.csv'))
    log = open(os.path.join(run, 'speed_test_log.txt')).read() if os.path.exists(os.path.join(run, 'speed_test_log.txt')) else ''

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), gridspec_kw=dict(height_ratios=[1.2, 1, 0.9, 0.9]))
    fig.suptitle(f'Robot Shazam speed test · {name}', x=0.06, ha='left', fontsize=14, color=INK, fontweight='bold')
    skip = re.search(r'SKIP .*', log)
    fig.text(0.06, 0.955, skip.group(0) if skip else ('trials: ' + ', '.join(f"{t['cmd_mps']} m/s" for t in trials) if trials else 'no trials recorded'),
             fontsize=9, color='#b4372f' if skip else MUTED)

    a1, a2, a3, a4 = axes
    if wall:
        t, w, cmd, sp = col(wall, 't_s'), col(wall, 'wall_m'), col(wall, 'cmd_mps'), col(wall, 'lidar_speed_mps')
        a1.plot(t, w, color=MEAS, lw=1.6, label='distance to wall (lidar)')
        moving = [c > 0.01 for c in cmd]
        a1.fill_between(t, min(w) - 0.05, max(w) + 0.05, where=moving, color=CMD, alpha=0.12, step='post', label='driving forward')
        a1.legend(loc='upper right', fontsize=8, frameon=False)
        a2.step(t, cmd, where='post', color=CMD, lw=1.6, label='commanded')
        a2.plot(t, sp, color=MEAS, lw=1.4, label='measured (lidar)')
        a2.legend(loc='upper right', fontsize=8, frameon=False)
    else:
        empty(a1, 'no lidar data'); empty(a2, 'no lidar data')
    style(a1, 'Distance to the wall ahead', 'm')
    style(a2, 'Speed: commanded vs measured', 'm/s')

    if imu:
        ti, ax_, ay_ = col(imu, 't_s'), col(imu, 'acc_x'), col(imu, 'acc_y')
        a3.plot(ti, ax_, color=IMU, lw=0.9, label='acc x')
        a3.plot(ti, ay_, color='#b9a9df', lw=0.9, label='acc y')
        a3.legend(loc='upper right', fontsize=8, frameon=False)
    else:
        empty(a3, 'No IMU data in this run: the motor-board serial port was also held by\n'
                  "Hiwonder's ROS 1 stack, so our board driver couldn't read the IMU.")
    style(a3, 'IMU acceleration (start and braking jolts)', 'm/s²')
    if wall or imu:
        for a in (a1, a2, a3):
            a.set_xlabel('time in bag (s)', color=MUTED, fontsize=8)

    if ping:
        seq, rtt = col(ping, 'seq'), col(ping, 'rtt_ms')
        tp = [s * 0.1 for s in seq]
        a4.scatter(tp, rtt, s=4, color=PING, alpha=0.7, linewidths=0)
        srt = sorted(rtt)
        p50, p95 = srt[len(srt) // 2], srt[int(len(srt) * 0.95)]
        for v, lbl in ((p50, 'median'), (p95, '95 %')):
            a4.axhline(v, color=INK, lw=0.8, ls='--', alpha=0.6)
            a4.text(tp[-1], v, f' {lbl} {v:.0f} ms', va='bottom', ha='right', fontsize=8, color=INK)
        lost = sorted(set(range(1, int(max(seq)) + 1)) - set(int(s) for s in seq))
        if lost:
            a4.scatter([s * 0.1 for s in lost], [max(rtt) * 1.05] * len(lost), marker='x', color='#b4372f', s=18, label=f'lost ({len(lost)})')
            a4.legend(loc='upper right', fontsize=8, frameon=False)
        a4.set_yscale('log')
        a4.set_xlabel('time (s)', color=MUTED, fontsize=8)
        target = re.search(r'PING ([\d.]+)', open(os.path.join(run, 'ping.txt')).read()) if os.path.exists(os.path.join(run, 'ping.txt')) else None
        style(a4, f'Ping round-trip time to {target.group(1) if target else "?"} ({len(rtt)} replies)', 'ms (log scale)')
    else:
        empty(a4, 'no ping data'); style(a4, 'Ping round-trip time', 'ms')

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = os.path.join(out, 'speed_test.png')
    fig.savefig(p1, dpi=110, facecolor='white')
    print(p1)

    if trials:
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4))
        v = [t['cmd_mps'] for t in trials]
        b1.bar(v, col(trials, 'measured_mps'), color=MEAS, width=0.6, label='measured')
        b1.plot(v, [float(x) for x in v], 'o', color=CMD, label='commanded')
        b1.legend(fontsize=8, frameon=False)
        style(b1, 'Top speed reached', 'm/s')
        b2.bar(v, col(trials, 'braking_distance_m'), color=CMD, width=0.6)
        for i, d in enumerate(col(trials, 'braking_distance_m')):
            b2.text(i, d, f'{d * 100:.1f} cm', ha='center', va='bottom', fontsize=9, color=INK)
        style(b2, 'Braking distance after a hard stop', 'm')
        for b in (b1, b2):
            b.set_xlabel('commanded speed (m/s)', color=MUTED, fontsize=8)
        fig.tight_layout()
        p2 = os.path.join(out, 'results.png')
        fig.savefig(p2, dpi=110, facecolor='white')
        print(p2)


if __name__ == '__main__':
    main()
