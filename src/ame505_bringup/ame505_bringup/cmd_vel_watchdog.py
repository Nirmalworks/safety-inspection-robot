#!/usr/bin/env python3
"""Stop the wheels when velocity commands stop arriving.

Hiwonder's odom_publisher sets the motor speeds once per Twist and HOLDS them: it has no timeout.
So whenever the last command any sender published was not zero, the robot kept driving forever:
a Nav2 node that crashed or stalled under load, the dashboard keyboard after a Wi-Fi drop, a
killed teleop. This node watches both inputs of the motor driver and, if the last command was
non-zero and nothing new came for `timeout` seconds, sends zero velocity.

    /controller/cmd_vel  (Xbox / keyboard teleop, E-STOP)   -> odom_publisher
    /cmd_vel             (Nav2 collision_monitor output)    -> odom_publisher

Senders that want continuous motion must keep publishing (Xbox teleop 20 Hz, Nav2 ~8 Hz,
dashboard keyboard 10 Hz). The timeout (0.8 s) is longer than a few missed Nav2 cycles on the Nano.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def is_zero(t, eps=1e-3):
    return (abs(t.linear.x) < eps and abs(t.linear.y) < eps and abs(t.angular.z) < eps)


class CmdVelWatchdog(Node):
    def __init__(self):
        super().__init__('cmd_vel_watchdog')
        self.timeout = float(self.declare_parameter('timeout', 0.8).value)
        self.pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.last = {}          # topic -> (time, moving)
        for topic in ('/controller/cmd_vel', '/cmd_vel'):
            self.create_subscription(Twist, topic, lambda m, t=topic: self.on_cmd(t, m), 10)
        self.create_timer(0.1, self.check)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, topic, m):
        self.last[topic] = (self.now(), not is_zero(m))

    def check(self):
        now = self.now()
        # the newest command (from either input) is what the motors are doing now
        if not self.last:
            return
        topic, (t, moving) = max(self.last.items(), key=lambda kv: kv[1][0])
        if moving and now - t > self.timeout:
            self.get_logger().warn(f'no velocity command on {topic} for {now - t:.1f} s -> stopping the wheels')
            for _ in range(3):
                self.pub.publish(Twist())
            self.last[topic] = (now, False)


def main():
    rclpy.init()
    node = CmdVelWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
