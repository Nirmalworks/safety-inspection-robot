#!/usr/bin/env python3
"""Xbox controller -> /controller/cmd_vel for the mecanum JetAuto, with a latching e-stop.

Hold LB (deadman) to move. Left stick: forward/back + strafe. Right stick X: rotate.
RB: turbo. D-pad up/down: speed level. B: E-STOP (latches). Start: release e-stop.
Parameters come from xbox_teleop.yaml (see teleop_xbox.sh).

The motor node keeps the last command it got, so this node always sends an explicit
zero: on LB release, on e-stop, when the controller is silent for `timeout`, and on exit.
E-stop state is published latched on /estop (std_msgs/Bool); publishing true there
from anywhere (e.g. a laptop) also triggers it.
"""
import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# joy_node indices for the Xbox Series pad (see xbox_teleop.yaml); only used for printing.
BUTTON_NAMES = ['A', 'B', 'X', 'Y', 'Back', 'Guide', 'Start', 'LStick', 'RStick',
                'LB', 'RB', 'DpadUp', 'DpadDown', 'DpadLeft', 'DpadRight']


class XboxTeleop(Node):
    def __init__(self):
        super().__init__('xbox_teleop')
        p = self.declare_parameters('', [
            ('cmd_topic', '/controller/cmd_vel'), ('estop_topic', '/estop'),
            ('axis_forward', 1), ('axis_strafe', 0), ('axis_turn', 2),
            ('button_deadman', 9), ('button_turbo', 10), ('button_estop', 1), ('button_reset', 6),
            ('button_speed_up', 11), ('button_speed_down', 12),
            ('speed_levels', [0.1, 0.2, 0.3]), ('start_level', 1),
            ('max_angular', 0.5), ('turbo_scale', 1.5), ('max_linear_limit', 0.5), ('timeout', 0.5),
        ])
        self.cfg = {x.name: x.value for x in p}
        c = self.cfg
        self.level = min(max(c['start_level'], 0), len(c['speed_levels']) - 1)
        self.pub = self.create_publisher(Twist, c['cmd_topic'], 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.estop_pub = self.create_publisher(Bool, c['estop_topic'], latched)
        self.create_subscription(Bool, c['estop_topic'], self.on_estop_topic, latched)
        self.create_subscription(Joy, 'joy', self.on_joy, 10)
        self.estop = False
        self.prev = []
        self.last_msg = self.get_clock().now()
        self.moving = False
        self.estop_pub.publish(Bool(data=False))
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info(
            f'Hold LB to drive at {self.speed():.1f} m/s. B = E-STOP, Start = release. Ctrl+C to quit.')

    def speed(self):
        return self.cfg['speed_levels'][self.level]

    def set_estop(self, on, why):
        if on == self.estop:
            return
        self.estop = on
        if on:
            self.halt()
            self.get_logger().error(f'E-STOP ({why}). Press Start to release.')
        else:
            self.get_logger().warn(f'E-stop released ({why}). Hold LB to drive.')
        self.estop_pub.publish(Bool(data=on))

    def on_estop_topic(self, msg):
        self.set_estop(msg.data, 'estop topic')

    def on_joy(self, msg):
        self.last_msg = self.get_clock().now()
        c = self.cfg
        btn = lambda i: i < len(msg.buttons) and msg.buttons[i] == 1
        prev, self.prev = self.prev, list(msg.buttons)
        pressed = lambda i: btn(i) and not (i < len(prev) and prev[i] == 1)
        for i, b in enumerate(msg.buttons):
            was = i < len(prev) and prev[i] == 1
            if b != was:
                name = BUTTON_NAMES[i] if i < len(BUTTON_NAMES) else '?'
                self.get_logger().info(f'{name} (button {i}) {"pressed" if b else "released"}')

        if pressed(c['button_estop']):
            self.set_estop(True, 'B button')
        elif pressed(c['button_reset']) and not btn(c['button_deadman']):
            self.set_estop(False, 'Start button')
        if self.estop:
            self.halt()
            return

        if pressed(c['button_speed_up']) or pressed(c['button_speed_down']):
            step = 1 if pressed(c['button_speed_up']) else -1
            self.level = min(max(self.level + step, 0), len(c['speed_levels']) - 1)
            self.get_logger().info(f'Speed {self.speed():.1f} m/s')

        ax = lambda i: msg.axes[i] if i < len(msg.axes) else 0.0
        if not btn(c['button_deadman']):
            if any(abs(ax(c[a])) > 0.2 for a in ('axis_forward', 'axis_strafe', 'axis_turn')):
                self.get_logger().warn('Stick moved but LB is not held: hold LB to drive',
                                       throttle_duration_sec=2.0)
            self.stop()
            return
        scale = c['turbo_scale'] if btn(c['button_turbo']) else 1.0
        vmax = min(self.speed() * scale, c['max_linear_limit'])
        t = Twist()
        t.linear.x = ax(c['axis_forward']) * vmax
        t.linear.y = ax(c['axis_strafe']) * vmax
        t.angular.z = ax(c['axis_turn']) * c['max_angular'] * scale
        self.pub.publish(t)
        self.moving = True
        self.get_logger().info(
            f'drive x={t.linear.x:+.2f} y={t.linear.y:+.2f} m/s  turn={t.angular.z:+.2f} rad/s',
            throttle_duration_sec=0.5)

    def watchdog(self):
        if self.moving and (self.get_clock().now() - self.last_msg).nanoseconds > self.cfg['timeout'] * 1e9:
            self.get_logger().warn('Controller silent, stopping')
            self.stop()

    def stop(self):
        if self.moving:
            self.halt()

    def halt(self):
        self.pub.publish(Twist())
        self.moving = False


def main():
    # Keep the context alive on Ctrl+C so the final zero command still goes out.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):  # also when started in the background
        signal.signal(sig, signal.default_int_handler)
    node = XboxTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.halt()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
