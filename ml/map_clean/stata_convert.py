#!/usr/bin/env python3
"""MIT Stata Center bag (ROS 1, PR2 robot) -> small ROS 2 bag that looks like our robot's lidar.

    python -m ml.map_clean.stata_convert IN.bag OUT_DIR [SECONDS]

With SECONDS the drive is cut into pieces of that length (OUT_DIR_s00, _s01, ...): a short drive drifts less,
so its map fits the floor plan better.

Kept: /base_scan as /scan, cut to the robot's lidar (205-degree field of view, readings beyond 12 m dropped,
every third scan = about 13 Hz), and the two transforms slam_toolbox needs (odom_combined -> base_footprint ->
base_laser_link), with the leading slashes of the old frame names removed.
"""
import math
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

FOV, RMAX, EVERY = math.radians(205.0), 12.0, 3
KEEP_TF = {('odom_combined', 'base_footprint'), ('base_footprint', 'base_laser_link')}


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    seg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0       # seconds per output bag; 0 = one bag for the whole drive
    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    Time, Header = T['builtin_interfaces/msg/Time'], T['std_msgs/msg/Header']
    Scan, TFM, TS = T['sensor_msgs/msg/LaserScan'], T['tf2_msgs/msg/TFMessage'], T['geometry_msgs/msg/TransformStamped']
    Tr, V3, Q = T['geometry_msgs/msg/Transform'], T['geometry_msgs/msg/Vector3'], T['geometry_msgs/msg/Quaternion']
    hdr = lambda h, f: Header(stamp=Time(sec=int(h.stamp.sec), nanosec=int(h.stamp.nanosec)), frame_id=f)
    n_scan = n_tf = k = n_seg = 0
    w = c_scan = c_tf = None
    t_seg = None

    def open_next():
        nonlocal w, c_scan, c_tf, n_seg
        if w is not None:
            w.close()
        path = Path(f'{dst}_s{n_seg:02d}') if seg > 0 else dst
        n_seg += 1
        w = Writer(path, version=8)
        w.open()
        c_scan = w.add_connection('/scan', 'sensor_msgs/msg/LaserScan', typestore=ts)
        c_tf = w.add_connection('/tf', 'tf2_msgs/msg/TFMessage', typestore=ts)

    with AnyReader([src]) as r:
        for con, t, raw in r.messages(connections=[c for c in r.connections if c.topic in ('/base_scan', '/tf')]):
            if w is None or (seg > 0 and t - t_seg >= seg * 1e9):
                open_next(); t_seg = t
            m = r.deserialize(raw, con.msgtype)
            if con.topic == '/tf':
                out = []
                for tr in m.transforms:
                    a, b = tr.header.frame_id.lstrip('/'), tr.child_frame_id.lstrip('/')
                    if (a, b) in KEEP_TF:
                        tt, rr = tr.transform.translation, tr.transform.rotation
                        out.append(TS(header=hdr(tr.header, a), child_frame_id=b,
                                      transform=Tr(translation=V3(x=float(tt.x), y=float(tt.y), z=float(tt.z)), rotation=Q(x=float(rr.x), y=float(rr.y), z=float(rr.z), w=float(rr.w)))))
                if out:
                    w.write(c_tf, t, ts.serialize_cdr(TFM(transforms=out), 'tf2_msgs/msg/TFMessage')); n_tf += 1
            else:
                k += 1
                if k % EVERY:
                    continue
                ang = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
                sel = np.abs(ang) <= FOV / 2
                rg = np.asarray(m.ranges, np.float32)[sel].copy()
                rg[(rg > RMAX) | (rg < 0.15)] = np.inf
                inten = np.asarray(m.intensities, np.float32)
                inten = inten[sel] if len(inten) == len(sel) else np.zeros(int(sel.sum()), np.float32)
                sc = Scan(header=hdr(m.header, m.header.frame_id.lstrip('/')), angle_min=float(ang[sel][0]), angle_max=float(ang[sel][-1]),
                          angle_increment=float(m.angle_increment), time_increment=float(m.time_increment), scan_time=float(m.scan_time) * EVERY,
                          range_min=0.15, range_max=RMAX, ranges=rg, intensities=inten)
                w.write(c_scan, t, ts.serialize_cdr(sc, 'sensor_msgs/msg/LaserScan')); n_scan += 1
    if w is not None:
        w.close()
    print(f'{src.name}: {n_scan} scans, {n_tf} transform messages -> {n_seg} bag(s) at {dst}')


if __name__ == '__main__':
    main()
