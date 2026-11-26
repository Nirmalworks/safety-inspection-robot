#!/usr/bin/env python3
"""CARMEN or Player laser log (Radish / Stachniss 2D laser datasets) -> ROS 2 bag that slam_toolbox can map.

    python -m ml.map_clean.carmen_convert IN.log.gz OUT_DIR

Reads FLASER (old CARMEN) or ROBOTLASER1 (new CARMEN) lines. Writes /scan and /tf (odom -> base_footprint, taken
from the laser's odometry pose in each line; the laser is treated as sitting at the robot's centre).
Readings beyond 12 m are dropped, like our robot's lidar. Time stamps that start near zero are shifted to a valid epoch.
"""
import gzip
import math
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

RMAX = 12.0


def lines(path):
    op = gzip.open if str(path).endswith('.gz') else open
    pose = None
    with op(path, 'rt', errors='ignore') as f:
        for ln in f:
            p = ln.split()
            if not p:
                continue
            try:
                if p[0] == 'FLASER':
                    n = int(p[1]); r = np.array(p[2:2 + n], np.float32); x, y, th = (float(v) for v in p[2 + n:5 + n]); ts = float(p[8 + n])
                    yield ts, r, -math.pi / 2, math.pi / (n - 1) if n in (180, 181, 360, 361) else math.pi / n, x, y, th
                elif p[0] == 'ROBOTLASER1':
                    start, fov, res = float(p[2]), float(p[3]), float(p[4]); n = int(p[8]); r = np.array(p[9:9 + n], np.float32)
                    k = 9 + n; nr = int(p[k]); k += 1 + nr
                    x, y, th = (float(v) for v in p[k:k + 3]); ts = float(p[k + 11])
                    yield ts, r, start, res, x, y, th
                else:                                                 # Player logs (USC, Stanford): "[time host port] laser|position idx time ..."
                    k = next((i for i, v in enumerate(p[:4]) if v in ('laser', 'position')), None)
                    if k is None or (p[k] == 'laser' and int(p[k + 1]) != 0):
                        continue
                    ts, q = float(p[k + 2]), p[k + 3:]
                    if p[k] == 'position':
                        pose = tuple(float(v) for v in q[:3])
                    elif pose is not None:
                        if '.' not in q[3]:                           # Player 1.6: min max step count, then range intensity pairs
                            n = int(q[3]); yield ts, np.array(q[4:4 + 2 * n:2], np.float32), float(q[0]), float(q[2]), *pose
                        else:                                         # older: range bearing intensity triples
                            yield ts, np.array(q[0::3], np.float32), float(q[1]), float(q[4]) - float(q[1]), *pose
            except (ValueError, IndexError):
                continue


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ts_ = get_typestore(Stores.ROS2_HUMBLE); T = ts_.types
    Time, Header = T['builtin_interfaces/msg/Time'], T['std_msgs/msg/Header']
    Scan, TFM, TS = T['sensor_msgs/msg/LaserScan'], T['tf2_msgs/msg/TFMessage'], T['geometry_msgs/msg/TransformStamped']
    Tr, V3, Q = T['geometry_msgs/msg/Transform'], T['geometry_msgs/msg/Vector3'], T['geometry_msgs/msg/Quaternion']
    n = 0; t0 = last = None; off = 0.0
    with Writer(dst, version=8) as w:
        cs = w.add_connection('/scan', 'sensor_msgs/msg/LaserScan', typestore=ts_)
        ct = w.add_connection('/tf', 'tf2_msgs/msg/TFMessage', typestore=ts_)
        for ts, r, a0, da, x, y, th in lines(src):
            if t0 is None:
                t0 = ts; off = 1.0e9 if ts < 1.0e8 else 0.0
            if last is not None and ts <= last:
                continue                                              # keep time strictly increasing
            last = ts
            t = ts + off; ns = int(t * 1e9); stamp = Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            rg = r.copy(); rg[(rg > RMAX) | (rg < 0.15)] = np.inf
            sc = Scan(header=Header(stamp=stamp, frame_id='base_footprint'), angle_min=float(a0), angle_max=float(a0 + da * (len(rg) - 1)), angle_increment=float(da),
                      time_increment=0.0, scan_time=0.2, range_min=0.15, range_max=RMAX, ranges=rg, intensities=np.zeros(0, np.float32))
            tf = TS(header=Header(stamp=stamp, frame_id='odom'), child_frame_id='base_footprint',
                    transform=Tr(translation=V3(x=x, y=y, z=0.0), rotation=Q(x=0.0, y=0.0, z=math.sin(th / 2), w=math.cos(th / 2))))
            w.write(ct, ns, ts_.serialize_cdr(TFM(transforms=[tf]), 'tf2_msgs/msg/TFMessage'))
            w.write(cs, ns + 1000, ts_.serialize_cdr(sc, 'sensor_msgs/msg/LaserScan')); n += 1
    print(f'{src.name}: {n} scans over {((last or 0) - (t0 or 0)) / 60:.1f} min, {len(r)} beams -> {dst}')


if __name__ == '__main__':
    main()
