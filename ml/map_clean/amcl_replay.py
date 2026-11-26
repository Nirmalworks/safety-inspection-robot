#!/usr/bin/env python3
"""AMCL replay: does the robot localise as well on a cleaned map as on the raw one?

    source /opt/ros/humble/setup.bash
    python3 ml/map_clean/amcl_replay.py --bag BAG.db3 --maps name=MAP.yaml ... --out DIR [--repeats 3] [--rate 3]

For every map the recorded /scan and odometry (odom -> base_footprint) of one mapping run are played into the
real nav2_amcl with the project's settings (src/ame505_bringup/config/nav2_params.yaml). The map -> odom
transform that slam_toolbox published during the run is NOT passed on; it is the reference instead:
error = AMCL pose (/amcl_pose) minus the slam pose at the same time stamp.
Two starts: "exact" (AMCL starts at the slam pose) and "offset" (0.5 m and 20 degrees off).
Limit: the reference is the SLAM run that drew the raw map, so the raw map has a home advantage.
"""
import argparse
import bisect
import csv
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    """2D poses (x, y, yaw): a then b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


class Track:
    """A transform over time, linearly interpolated."""

    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        self.t.append(t); self.v.append(v)

    def at(self, t):
        i = min(max(bisect.bisect_left(self.t, t), 1), len(self.t) - 1)
        t0, t1, a, b = self.t[i - 1], self.t[i], self.v[i - 1], self.v[i]
        f = min(max((t - t0) / max(1e-9, t1 - t0), 0.0), 1.0)
        dy = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
        return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * dy


def reference(bag):
    """slam pose of base_footprint in the map frame, as a function of time (seconds), and the first scan time."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage
    c = sqlite3.connect(bag).cursor()
    tp = {n: i for i, n in c.execute('select id, name from topics')}
    mo, ob = Track(), Track()
    for (d,) in c.execute('select data from messages where topic_id=? order by timestamp', (tp['/tf'],)):
        for t in deserialize_message(d, TFMessage).transforms:
            st = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
            v = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
            if (t.header.frame_id, t.child_frame_id) == ('map', 'odom'):
                mo.add(st, v)
            elif (t.header.frame_id, t.child_frame_id) == ('odom', 'base_footprint'):
                ob.add(st, v)
    for tr in (mo, ob):
        o = np.argsort(tr.t); tr.t, tr.v = [tr.t[i] for i in o], [tr.v[i] for i in o]
    d = c.execute('select data from messages where topic_id=? order by timestamp limit 1', (tp['/scan'],)).fetchone()[0]
    s = deserialize_message(d, LaserScan).header.stamp
    t0 = max(s.sec + s.nanosec * 1e-9, mo.t[0], ob.t[0])
    return (lambda t: compose(mo.at(t), ob.at(t))), t0, min(mo.t[-1], ob.t[-1])


def relay_main(out_csv):
    """Node: forwards /tf_bag to /tf without map -> odom, and logs /amcl_pose."""
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    n = Node('replay_relay')
    pub = n.create_publisher(TFMessage, '/tf', QoSProfile(depth=100))
    f = open(out_csv, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['t', 'x', 'y', 'yaw', 'cov_xx', 'cov_yy', 'cov_aa'])

    def on_tf(m):
        keep = [t for t in m.transforms if not (t.header.frame_id == 'map' and t.child_frame_id == 'odom')]
        if keep:
            pub.publish(TFMessage(transforms=keep))

    def on_pose(m):
        p = m.pose.pose
        w.writerow([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.position.x, p.position.y, yaw_of(p.orientation),
                    m.pose.covariance[0], m.pose.covariance[7], m.pose.covariance[35]])
        f.flush()

    n.create_subscription(TFMessage, '/tf_bag', on_tf, QoSProfile(depth=1000))
    n.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', on_pose, QoSProfile(depth=100))
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception):
        pass


def one_run(job):
    name, map_yaml, start, rep, init, domain, a = job
    tag = f'{name}__{start}__{rep}'
    d = os.path.join(a.out, 'runs', tag)
    os.makedirs(d, exist_ok=True)
    amcl = yaml.safe_load(open(os.path.join(REPO, 'src/ame505_bringup/config/nav2_params.yaml')))['amcl']['ros__parameters']
    amcl.update({'use_sim_time': True, 'set_initial_pose': True, 'initial_pose': {'x': float(init[0]), 'y': float(init[1]), 'z': 0.0, 'yaw': float(init[2])}})
    params = {'amcl': {'ros__parameters': amcl},
              'map_server': {'ros__parameters': {'use_sim_time': True, 'yaml_filename': os.path.abspath(map_yaml)}},
              'lifecycle_manager': {'ros__parameters': {'use_sim_time': False, 'autostart': True, 'node_names': ['map_server', 'amcl'], 'bond_timeout': 0.0}}}
    pf = os.path.join(d, 'params.yaml')
    yaml.safe_dump(params, open(pf, 'w'))
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1', ROS_LOG_DIR=os.path.join(d, 'log'))
    log = open(os.path.join(d, 'out.log'), 'w')
    run = lambda cmd: subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    procs = [run(['ros2', 'run', 'nav2_map_server', 'map_server', '--ros-args', '--params-file', pf]),
             run(['ros2', 'run', 'nav2_amcl', 'amcl', '--ros-args', '--params-file', pf]),
             run([sys.executable, os.path.abspath(__file__), '--relay', os.path.join(d, 'amcl.csv')])]
    time.sleep(3)
    procs.append(run(['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '--params-file', pf]))
    time.sleep(8)
    play = run(['ros2', 'bag', 'play', a.bag, '-s', 'sqlite3', '--clock', '100', '--rate', str(a.rate), '--topics', '/scan', '/tf', '/tf_static',
                '--remap', '/tf:=/tf_bag', '--disable-keyboard-controls'])
    try:
        play.wait(timeout=600 / a.rate + 120)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(2)
    for p in procs + [play]:
        try:
            os.killpg(p.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for p in procs + [play]:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return tag


def score(csv_path, ref, t0, t1):
    rows = [r for r in csv.DictReader(open(csv_path))] if os.path.exists(csv_path) else []
    e, ey, ts = [], [], []
    for r in rows:
        t = float(r['t'])
        if not (t0 <= t <= t1):
            continue
        g = ref(t)
        e.append(math.hypot(float(r['x']) - g[0], float(r['y']) - g[1]))
        ey.append(abs((float(r['yaw']) - g[2] + math.pi) % (2 * math.pi) - math.pi))
        ts.append(t - t0)
    if len(e) < 20:
        return {'updates': len(e)}
    e, ey, ts = np.array(e), np.degrees(ey), np.array(ts)
    late = ts > ts[0] + 30                                      # 30 s after the first pose update, i.e. once a bad start had time to recover
    conv = next((float(ts[i]) for i in range(len(e)) if e[i] < 0.15 and np.all(e[i:i + 10] < 0.25)), float('nan'))
    return {'updates': len(e), 'covered_s': float(ts[-1]), 'pos_mean_cm': float(e.mean() * 100), 'pos_median_cm': float(np.median(e) * 100),
            'pos_p95_cm': float(np.percentile(e, 95) * 100), 'pos_max_cm': float(e.max() * 100), 'yaw_mean_deg': float(ey.mean()), 'yaw_p95_deg': float(np.percentile(ey, 95)),
            'within_10cm': float((e < 0.10).mean()), 'within_25cm': float((e < 0.25).mean()),
            'late_pos_mean_cm': float(e[late].mean() * 100) if late.any() else float('nan'), 'late_within_25cm': float((e[late] < 0.25).mean()) if late.any() else float('nan'),
            'lost_at_end': bool(e[-10:].mean() > 0.5), 'converged_after_s': conv}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--relay':
        return relay_main(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--maps', nargs='+', required=True, help='name=path/to/map.yaml')
    ap.add_argument('--out', default='data/map_cleaning/amcl')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--rate', type=float, default=3.0)
    ap.add_argument('--parallel', type=int, default=6)
    ap.add_argument('--offset', type=float, nargs=3, default=[0.5, 0.0, 20.0], help='start error for the "offset" runs: x m, y m, yaw deg')
    ap.add_argument('--score-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref, t0, t1 = reference(a.bag)
    p0 = ref(t0)
    starts = {'exact': p0, 'offset': (p0[0] + a.offset[0], p0[1] + a.offset[1], p0[2] + math.radians(a.offset[2]))}
    maps = dict(m.split('=', 1) for m in a.maps)
    jobs = [(n, y, s, r, starts[s], 0, a) for n, y in maps.items() for s in starts for r in range(a.repeats)]
    jobs = [j[:5] + (60 + i % a.parallel,) + j[6:] for i, j in enumerate(jobs)]
    if not a.score_only:
        lanes = [[j for j in jobs if j[5] == 60 + k] for k in range(a.parallel)]     # one ROS domain per lane, runs in a lane go one after another
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(lambda lane: [print('done', one_run(j), flush=True) for j in lane], lanes))
    res = {}
    for n, y, s, r, *_ in jobs:
        res.setdefault(n, {}).setdefault(s, []).append(score(os.path.join(a.out, 'runs', f'{n}__{s}__{r}', 'amcl.csv'), ref, t0, t1))
    md = [f'# AMCL replay on {os.path.basename(a.bag)} ({t1 - t0:.0f} s of driving, {a.repeats} repeats per cell, played at {a.rate} x)\n',
          'Error = AMCL pose minus the slam_toolbox pose recorded during mapping. Mean over repeats, range of the repeats in brackets.\n']
    for s in starts:
        md += [f'\n## Start: {s}' + ('' if s == 'exact' else f' ({a.offset[0]} m, {a.offset[1]} m, {a.offset[2]} deg off)') + '\n',
               '| map | position error mean (cm) | median (cm) | 95th pct (cm) | yaw error mean (deg) | time within 10 cm | time within 25 cm | 30 s after the first update: mean (cm) | runs lost at the end | pose updates |', '|---|---|---|---|---|---|---|---|---|---|']
        for n in maps:
            rr = [x for x in res[n][s] if 'pos_mean_cm' in x]
            if not rr:
                md.append(f'| {n} | no data | | | | | | | | |'); continue
            m = lambda k: np.mean([x[k] for x in rr])
            rg = lambda k: f"{m(k):.1f} ({min(x[k] for x in rr):.1f} to {max(x[k] for x in rr):.1f})"
            md.append(f"| {n} | {rg('pos_mean_cm')} | {m('pos_median_cm'):.1f} | {m('pos_p95_cm'):.1f} | {m('yaw_mean_deg'):.2f} | {m('within_10cm') * 100:.0f} % | {m('within_25cm') * 100:.0f} % | "
                      f"{rg('late_pos_mean_cm')} | {sum(x['lost_at_end'] for x in rr)} of {len(res[n][s])} | {m('updates'):.0f} |")
    json.dump({'t0': t0, 'duration_s': t1 - t0, 'start_pose': p0, 'offset': a.offset, 'results': res}, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'tables.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
