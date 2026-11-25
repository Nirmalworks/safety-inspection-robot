#!/usr/bin/env python3
"""Measure speed, start delay and braking from a speed_test.py bag, plus a ping summary.

    python3 speed_analysis.py <bag dir> <output dir> [ping.txt]

Real motion comes from the lidar range to the wall ahead (median of a narrow cone) and
the IMU accelerometer. /odom and /odom_raw only add up the commanded speed, so they are
not used as evidence of motion. Writes trials.csv, wall_timeline.csv, ping_rtt.csv and
results.md into the output dir. Settings (cones, lidar yaw) come from speed_test.yaml.
"""
import csv
import math
import os
import re
import statistics
import sys

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

C = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speed_test.yaml')))['speed_test']
LIDAR_YAW = math.radians(C['lidar_yaw_deg'])
WALL_CONE = math.radians(C['wall_cone_deg'])


def read_bag(path):
    r = SequentialReader()
    r.open(StorageOptions(uri=path, storage_id='sqlite3'), ConverterOptions('', ''))
    cls = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
    d = dict(wall=[], imu=[], cmd=[], ev=[], batt=[])
    t0 = None
    while r.has_next():
        topic, data, t = r.read_next()
        t = t / 1e9
        t0 = t0 or t
        if topic not in ('/scan', '/imu', '/controller/cmd_vel', '/speed_test/event', '/ros_robot_controller/battery'):
            continue
        m = deserialize_message(data, cls[topic])
        if topic == '/scan':
            v = []
            for i, x in enumerate(m.ranges):
                a = m.angle_min + i * m.angle_increment + LIDAR_YAW
                if math.isfinite(x) and x > m.range_min and abs(math.atan2(math.sin(a), math.cos(a))) < WALL_CONE:
                    v.append(x)
            if v:
                d['wall'].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, statistics.median(v)))
        elif topic == '/imu':
            a = m.linear_acceleration
            d['imu'].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, a.x, a.y))
        elif topic == '/controller/cmd_vel':
            d['cmd'].append((t, m.linear.x))
        elif topic == '/speed_test/event':
            d['ev'].append((t, m.data))
        else:
            d['batt'].append((t, m.data / 1000))
    return t0, {k: np.array(v) if k != 'ev' else v for k, v in d.items()}


def first_cross(t, y, start, level, below=True):
    """First time after `start` that y crosses `level`, linearly interpolated."""
    for i in range(1, len(t)):
        if t[i] <= start:
            continue
        hit = y[i] <= level if below else y[i] >= level
        if hit:
            if y[i] == y[i - 1]:
                return t[i]
            return t[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1])
    return math.nan


def imu_metrics(imu, tg, ts, tr):
    """Start/stop delay and peak accel/decel from the IMU; NaN when the IMU recorded nothing."""
    out = dict(start_delay=math.nan, stop_delay=math.nan, peak_accel=math.nan, peak_decel=math.nan)
    if imu.ndim != 2 or len(imu) == 0:
        return out
    ax = imu[(imu[:, 0] > tg - 1) & (imu[:, 0] < tr)]
    if len(ax) < 20:
        return out
    col = 1 if np.ptp(ax[:, 1]) >= np.ptp(ax[:, 2]) else 2          # forward axis swings most
    before = ax[:, 0] < tg
    a = ax[:, col] - (np.median(ax[before, col]) if before.any() else 0)
    a_s = np.convolve(a, np.ones(5) / 5, mode='same')                 # ~70 ms smoothing
    early = (ax[:, 0] > tg) & (ax[:, 0] < tg + 1)
    a_s *= 1 if early.any() and a_s[early].mean() > 0 else -1
    acc = a_s[(ax[:, 0] > tg) & (ax[:, 0] < tg + 1.5)]
    dec = a_s[(ax[:, 0] > ts) & (ax[:, 0] < ts + 1.5)]
    noise = np.std(a_s[before]) if before.sum() > 10 else 0.05
    thr = max(4 * noise, 0.25)
    out.update(start_delay=round(first_cross(ax[:, 0], -np.abs(a_s), tg, -thr) - tg, 3),
               stop_delay=round(first_cross(ax[:, 0], a_s, ts, -thr) - ts, 3),
               peak_accel=round(float(acc.max()), 2) if len(acc) else math.nan,
               peak_decel=round(float(-dec.min()), 2) if len(dec) else math.nan)
    return out


def analyse(t0, d):
    wt, wd = d['wall'][:, 0], d['wall'][:, 1]
    # Speed toward the wall from the lidar: slope over a 5-scan window (~0.4 s).
    sp = np.full_like(wd, np.nan)
    for i in range(2, len(wd) - 2):
        sp[i] = -np.polyfit(wt[i - 2:i + 3], wd[i - 2:i + 3], 1)[0]
    trials = []
    evs = d['ev']
    for k, (tg, txt) in enumerate(evs):
        m = re.match(r'GO v=([\d.]+)', txt)
        if not m:
            continue
        v = float(m.group(1))
        ts = next((t for t, x in evs[k + 1:] if x.startswith('STOP')), math.nan)
        stop_txt = next((x for t, x in evs[k + 1:] if x.startswith('STOP')), '')
        tr = next((t for t, x in evs[k + 1:] if x.startswith('rest')), ts + C['rest_after'])
        cmd_on = [t for t, x in d['cmd'] if tg - 0.05 <= t <= ts and abs(x - v) < 1e-3]
        tg = cmd_on[0] if cmd_on else tg
        base = np.median(wd[(wt > tg - 1.5) & (wt < tg)])
        rest = np.median(wd[(wt > tr - 1.0) & (wt < tr)])
        hold = ts - tg
        # steady speed: fit over the last 40 % of the run
        sel = (wt > ts - 0.4 * hold) & (wt <= ts)
        v_meas = -np.polyfit(wt[sel], wd[sel], 1)[0] if sel.sum() >= 3 else math.nan
        run_sel = (wt >= tg) & (wt <= ts + 1.0)
        v_peak = np.nanmax(sp[run_sel]) if run_sel.any() else math.nan
        t_move = first_cross(wt, wd, tg, base - 0.015)                    # moved 1.5 cm
        t50 = first_cross(wt, -np.nan_to_num(sp, nan=0), tg, -0.5 * v_meas)  # half of steady speed
        t90 = first_cross(wt, -np.nan_to_num(sp, nan=0), tg, -0.9 * v_meas)
        d_stop = float(np.interp(ts, wt, wd))
        brake = d_stop - rest
        t_slow50 = first_cross(wt, np.nan_to_num(sp, nan=0), ts, 0.5 * v_meas)
        t_still = first_cross(wt, wd, ts, rest + 0.01)                    # within 1 cm of rest
        im = imu_metrics(d['imu'], tg, ts, tr)
        trials.append(dict(
            cmd_mps=v, aborted=('ABORT' in stop_txt), run_s=round(hold, 2),
            measured_mps=round(v_meas, 3), peak_mps=round(v_peak, 3),
            tracking_pct=round(100 * v_meas / v, 1),
            start_delay_lidar_s=round(t_move - tg, 3), start_delay_imu_s=im['start_delay'],
            t50_s=round(t50 - tg, 3), t90_s=round(t90 - tg, 3),
            accel_mps2=round(0.4 * v_meas / (t90 - t50), 2) if t90 > t50 else math.nan,  # 50 -> 90 %
            imu_peak_accel_mps2=im['peak_accel'],
            stop_delay_imu_s=im['stop_delay'], stop_t50_s=round(t_slow50 - ts, 3),
            stop_time_s=round(t_still - ts, 3), braking_distance_m=round(brake, 3),
            brake_decel_eff_mps2=round(v_meas ** 2 / (2 * brake), 2) if brake > 0 else math.nan,
            imu_peak_decel_mps2=im['peak_decel'],
            travel_m=round(base - rest, 3), commanded_m=round(v * hold, 3),
            wall_start_m=round(base, 3), wall_rest_m=round(rest, 3),
        ))
    return trials, sp


def ping_stats(path):
    rtt, sent, recv = [], None, None
    for line in open(path):
        m = re.search(r'icmp_seq=(\d+).*time=([\d.]+)', line)
        if m:
            rtt.append((int(m.group(1)), float(m.group(2))))
        m = re.search(r'(\d+) packets transmitted, (\d+) received', line)
        if m:
            sent, recv = int(m.group(1)), int(m.group(2))
    if not rtt:
        return None, rtt
    r = np.array([x for _, x in rtt])
    seqs = {s for s, _ in rtt}
    run = best = 0
    for s in range(1, (sent or max(seqs)) + 1):
        run = run + 1 if s not in seqs else 0
        best = max(best, run)
    p = lambda q: round(float(np.percentile(r, q)), 2)
    return dict(sent=sent, received=recv, loss_pct=round(100 * (sent - recv) / sent, 2) if sent else math.nan,
                min_ms=round(float(r.min()), 2), avg_ms=round(float(r.mean()), 2), p50_ms=p(50),
                p95_ms=p(95), p99_ms=p(99), max_ms=round(float(r.max()), 2),
                jitter_ms=round(float(np.mean(np.abs(np.diff(r)))), 2), longest_loss_burst=best,
                over_100ms=int((r > 100).sum())), rtt


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    bag, out = sys.argv[1], sys.argv[2]
    ping = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(out, exist_ok=True)
    t0, d = read_bag(bag)
    trials, sp = analyse(t0, d)
    if trials:
        write_csv(os.path.join(out, 'trials.csv'), trials)
    wt = d['wall'][:, 0]
    cmd_t = d['cmd'][:, 0] if len(d['cmd']) else np.array([0.0])
    cmd_v = d['cmd'][:, 1] if len(d['cmd']) else np.array([0.0])
    write_csv(os.path.join(out, 'wall_timeline.csv'), [
        dict(t_s=round(t - t0, 3), wall_m=round(w, 4), lidar_speed_mps=round(s, 3) if np.isfinite(s) else '',
             cmd_mps=round(float(cmd_v[max(np.searchsorted(cmd_t, t) - 1, 0)]), 3))
        for t, w, s in zip(wt, d['wall'][:, 1], sp)])
    imu = d['imu']
    if imu.ndim == 2 and len(imu):
        write_csv(os.path.join(out, 'imu_timeline.csv'), [
            dict(t_s=round(t - t0, 3), acc_x=round(x, 3), acc_y=round(y, 3)) for t, x, y in imu])
    ps, rtt = ping_stats(ping) if ping and os.path.exists(ping) else (None, [])
    if rtt:
        write_csv(os.path.join(out, 'ping_rtt.csv'), [dict(seq=s, rtt_ms=x) for s, x in rtt])
    batt = d['batt'][:, 1] if len(d['batt']) else []
    lines = ['## Speed and hard-stop results', '',
             '| cmd m/s | measured m/s | peak m/s | start delay (lidar / IMU) s | 50 % / 90 % speed at s | '
             'stop delay (IMU) s | braking distance m | time to standstill s | eff. decel m/s² | IMU peak decel m/s² |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in trials:
        lines.append(f"| {r['cmd_mps']}{' (aborted)' if r['aborted'] else ''} | {r['measured_mps']} | {r['peak_mps']} | "
                     f"{r['start_delay_lidar_s']} / {r['start_delay_imu_s']} | {r['t50_s']} / {r['t90_s']} | "
                     f"{r['stop_delay_imu_s']} | {r['braking_distance_m']} | {r['stop_time_s']} | "
                     f"{r['brake_decel_eff_mps2']} | {r['imu_peak_decel_mps2']} |")
    if not len(imu):
        lines += ['', 'IMU: no data recorded (usually the motor-board serial port was also opened by '
                  "Hiwonder's ROS 1 stack), so the IMU columns are blank."]
    if len(batt):
        lines += ['', f'Battery during the test: {min(batt):.2f}–{max(batt):.2f} V.']
    if ps:
        lines += ['', '## Ping', '', '| ' + ' | '.join(ps) + ' |', '|' + '---|' * len(ps),
                  '| ' + ' | '.join(str(v) for v in ps.values()) + ' |']
    open(os.path.join(out, 'results.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
