#!/usr/bin/env python3
"""Straight-line speed + hard-stop test for the JetAuto. Record a rosbag while it runs.

For each speed (default 0.2 0.3 0.5 m/s): stand still, drive forward RUN metres at that
speed, then a hard stop (command jumps straight to zero, no ramp), then reverse slowly
back to the start. Face the robot at a flat wall the lidar can see; the wall distance
is what speed_analysis.py uses to measure real speed, delay and braking distance.

Safety: a trial only starts with RUN + braking + MARGIN of free space ahead; any trial
hard-stops if something comes within ABORT_DIST, if the wall gets further away instead
of closer (lidar orientation wrong), or if /estop goes true (B on the Xbox controller).
Phase markers are published on /speed_test/event so they land in the bag.
Settings: speed_test.yaml next to this file.
Usage: python3 speed_test.py [speeds...]   (speeds override the YAML)
"""
import math
import os
import signal
import statistics
import sys
import time

import yaml

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speed_test.yaml')))
C = CFG['speed_test']
RUN = C['run_distance']
MARGIN = C['margin']
ABORT_DIST = C['abort_distance']
REAR_CLEAR = C['rear_clearance']
BACK_SPEED = C['return_speed']
LIDAR_YAW = math.radians(C['lidar_yaw_deg'])
WALL_CONE = math.radians(C['wall_cone_deg'])    # cone used for the wall distance (median)
LANE = C['lane_half_width_m']                   # obstacle checks: only points inside the driving lane


class SpeedTest(Node):
    def __init__(self):
        super().__init__('speed_test')
        self.pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.ev = self.create_publisher(String, '/speed_test/event', 10)
        self.wall = self.front = self.rear = None
        self.estop = False
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        # /scan is cropped to the front half by Hiwonder's filter; /scan_raw is the full 360 deg.
        self.create_subscription(LaserScan, '/scan_raw', self.on_scan_raw, qos_profile_sensor_data)
        self.create_subscription(Bool, '/estop', self.on_estop,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def on_estop(self, m):
        if m.data and not self.estop:
            self.send()
            self.event('ESTOP received, aborting')
        self.estop = m.data

    def on_scan(self, m):
        wall, front = [], []
        for i, d in enumerate(m.ranges):
            if not math.isfinite(d) or d < max(m.range_min, 0.12):
                continue
            a = m.angle_min + i * m.angle_increment + LIDAR_YAW
            a = math.atan2(math.sin(a), math.cos(a))
            x, y = d * math.cos(a), d * math.sin(a)
            if abs(a) < WALL_CONE: wall.append(d)
            if x > 0 and abs(y) < LANE: front.append(x)   # distance along the path, inside the lane
        self.wall = statistics.median(wall) if wall else float('inf')
        self.front = min(front) if front else float('inf')

    def on_scan_raw(self, m):
        rear, seen_behind = [], 0
        for i, d in enumerate(m.ranges):
            if not math.isfinite(d) or d < max(m.range_min, 0.2):
                continue
            a = m.angle_min + i * m.angle_increment + LIDAR_YAW
            a = math.atan2(math.sin(a), math.cos(a))
            x, y = d * math.cos(a), d * math.sin(a)
            if abs(a) > math.radians(150):
                seen_behind += 1          # the lidar really sees the area straight behind
            if x < 0 and abs(y) < LANE:
                rear.append(-x)
        # Too few returns straight behind = unknown, not clear (e.g. a scan cropped to the front
        # half): rear stays None and the robot does not reverse to the start.
        self.rear = (min(rear) if rear else float('inf')) if seen_behind >= 20 else None

    def event(self, text):
        self.ev.publish(String(data=text))
        print(f'{time.strftime("%H:%M:%S")} {text}', flush=True)

    def send(self, vx=0.0):
        t = Twist()
        t.linear.x = float(vx)
        self.pub.publish(t)

    def wait(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def hard_stop(self):
        for _ in range(10):  # zero at 50 Hz for 0.2 s; the motor node keeps the last command
            self.send()
            self.wait(0.02)

    def trial(self, v):
        need = RUN + 0.5 * v + MARGIN  # generous braking allowance
        self.wait(0.3)
        if self.front < need:
            self.event(f'SKIP v={v} only {self.front:.2f} m free ahead, need {need:.2f}')
            return False
        start = self.wall
        self.event(f'still v={v} wall={start:.3f}')
        self.hard_stop()
        self.wait(C['still_before'])
        self.event(f'GO v={v}')
        t0 = time.time()
        hold = RUN / v
        reason = None
        while time.time() - t0 < hold:
            self.send(v)
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.estop:
                reason = 'estop'
            elif self.front < ABORT_DIST:
                reason = f'obstacle {self.front:.2f} m'
            elif time.time() - t0 > 0.8 and self.wall > start + 0.05:
                reason = 'wall getting further: lidar front is not the robot front'
            if reason:
                break
        self.send()
        self.event(f'STOP v={v}' + (f' ABORT {reason}' if reason else ''))
        self.hard_stop()
        self.wait(C['rest_after'])
        self.event(f'rest v={v} wall={self.wall:.3f}')
        if self.estop or (reason and 'lidar' in reason):
            return False
        self.ret(start)
        return not reason

    def ret(self, start):
        self.event(f'RETURN to wall={start:.3f}')
        t0 = time.time()
        while self.wall < start - 0.02 and time.time() - t0 < 15:
            if self.estop or self.rear is None or self.rear < REAR_CLEAR:
                why = 'estop' if self.estop else 'no lidar data behind' if self.rear is None else f'rear {self.rear:.2f} m'
                self.event(f'return stopped ({why}): move the robot back by hand or with the controller')
                break
            self.send(-BACK_SPEED)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.hard_stop()
        self.wait(1.5)
        self.event(f'back at wall={self.wall:.3f}')


def main():
    speeds = [float(s) for s in sys.argv[1:]] or [float(v) for v in C['speeds']]
    # Keep the context alive on Ctrl+C / TERM so the final hard stop still goes out.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.default_int_handler)
    n = SpeedTest()
    t = time.time()
    while n.wall is None and time.time() - t < 8:
        rclpy.spin_once(n, timeout_sec=0.1)
    if n.wall is None:
        print('no /scan: is the lidar running?')
        return
    n.event(f'START speeds={speeds} wall={n.wall:.2f} lane_clear_ahead={n.front:.2f} lane_clear_behind={n.rear}')
    try:
        for v in speeds:
            if n.estop or not n.trial(v):
                break
    except KeyboardInterrupt:
        n.send()
        n.event('INTERRUPTED')
    finally:
        n.hard_stop()
        n.event('END stopped')
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
