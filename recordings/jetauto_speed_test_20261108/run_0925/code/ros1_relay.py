#!/usr/bin/env python3
"""Use Hiwonder's ROS 1 stack from ROS 2: its lidar, IMU and battery in, our drive commands out.

Hiwonder's stack on the robot's host owns the lidar and the motor board and restarts itself
when stopped, so our own drivers can't open those ports while it runs. This relay
republishes its sensors in ROS 2 and passes /controller/cmd_vel to its motor node.
Safety: its motor node keeps the last command, so after drive.timeout_s without a ROS 2
command (or on exit) the relay sends a stop. Settings: ros1_relay.yaml.
"""
import os
import signal
import threading
import time

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import UInt16

from ros1_sub import Publisher, Subscriber

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ros1_relay.yaml')))
DRV = CFG['drive']
TYPES = {LaserScan: 'sensor_msgs/LaserScan', Imu: 'sensor_msgs/Imu', UInt16: 'std_msgs/UInt16'}
ROS2_TYPE = {'/scan': LaserScan, '/imu': Imu, '/ros_robot_controller/battery': UInt16}


class Relay(Node):
    def __init__(self):
        super().__init__('ros1_relay')
        self.counts = {}
        for src, outs in CFG['ros1_to_ros2'].items():
            cls = ROS2_TYPE[src]
            pubs = [self.create_publisher(cls, t, qos_profile_sensor_data if cls is not UInt16 else 1) for t in outs]
            Subscriber(CFG['master'], src, TYPES[cls], lambda m, c=cls, p=pubs, s=src: self.forward(s, c, p, m))
        self.out = Publisher(CFG['master'], DRV['ros1_cmd'], 'geometry_msgs/Twist')
        self.last_cmd, self.moving = 0.0, False
        self.cmd_lock = threading.Lock()
        self.create_subscription(Twist, DRV['ros2_cmd'], self.on_cmd, 10)
        self.create_timer(0.1, self.watchdog)
        self.create_timer(10.0, self.report)
        self.get_logger().info(f'relaying {list(CFG["ros1_to_ros2"])} from ROS 1; '
                               f'{DRV["ros2_cmd"]} -> ROS 1 {DRV["ros1_cmd"]}')

    def forward(self, src, cls, pubs, m):
        msg = cls()
        if cls is LaserScan:
            msg.header.frame_id = m['frame']
            for k in ('angle_min', 'angle_max', 'angle_increment', 'time_increment', 'scan_time', 'range_min', 'range_max'):
                setattr(msg, k, float(m[k]))
            msg.ranges, msg.intensities = list(m['ranges']), list(m['intensities'])
        elif cls is Imu:
            msg.header.frame_id = m['frame']
            q, w, a = m['orientation'], m['angular_velocity'], m['linear_acceleration']
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = w
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a
        else:
            msg.data = int(m['data'])
        if hasattr(msg, 'header'):
            msg.header.stamp = self.get_clock().now().to_msg()   # same clock as everything else here
        for p in pubs:
            p.publish(msg)
        self.counts[src] = self.counts.get(src, 0) + 1

    def on_cmd(self, m):
        clamp = lambda v, lim: max(-lim, min(lim, float(v)))
        vx, vy = clamp(m.linear.x, DRV['max_linear']), clamp(m.linear.y, DRV['max_linear'])
        wz = clamp(m.angular.z, DRV['max_angular'])
        with self.cmd_lock:
            self.out.publish_twist(vx, vy, wz)
            self.last_cmd = time.time()
            self.moving = bool(vx or vy or wz)

    def watchdog(self):
        with self.cmd_lock:
            if self.moving and time.time() - self.last_cmd > DRV['timeout_s']:
                self.out.publish_twist()
                self.moving = False
                self.get_logger().warn('no drive command for a while: sent stop')

    def report(self):
        self.get_logger().info(f'last 10 s: {self.counts} | ROS 1 motor node connected: {self.out.subscribers() > 0}')
        self.counts = {}

    def stop(self):
        for _ in range(5):
            self.out.publish_twist()
            time.sleep(0.02)
        self.out.close()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.default_int_handler)
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
