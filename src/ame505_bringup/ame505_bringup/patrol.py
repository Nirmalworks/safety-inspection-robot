#!/usr/bin/env python3
"""Autonomous patrol: visit every recorded checkpoint with Nav2, stop, dwell.

    ros2 run ame505_bringup patrol --map lab_map --loops 1 --dwell 5
    ros2 run ame505_bringup patrol --map lab_map --loops 0 --follow     # tour forever, pick up new checkpoints (dashboard Tour button)

At each checkpoint it publishes the checkpoint id on /patrol/at_checkpoint
(std_msgs/String) and waits --dwell seconds. That is the hook for the
inspection node (capture RGB-D -> YOLO -> keep-clear check) from the proposal.
"""
import argparse
import math
import os
import sys
import time

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import String

from ame505_bringup.paths import checkpoints_file


def make_pose(nav, frame, cp):
    p = PoseStamped()
    p.header.frame_id = frame
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = float(cp['x'])
    p.pose.position.y = float(cp['y'])
    p.pose.orientation.z = math.sin(cp['yaw'] / 2.0)
    p.pose.orientation.w = math.cos(cp['yaw'] / 2.0)
    return p


def wait_for_nav2(nav, timeout):
    """True once bt_navigator is active. Asks again every few seconds instead of using
    BasicNavigator.waitUntilNav2Active: that waits forever when a single reply is lost (seen on a
    loaded machine), and with a localizer name it waits for 'robot_localization/get_state', which
    nothing provides, or re-publishes an initial pose at the origin ('amcl')."""
    client = nav.create_client(GetState, 'bt_navigator/get_state')
    end = time.time() + timeout
    while time.time() < end:
        if client.wait_for_service(timeout_sec=1.0):
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(nav, future, timeout_sec=3.0)
            if future.done() and future.result() and future.result().current_state.label == 'active':
                return True
            future.cancel()
        time.sleep(1.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='lab_map', help='map name (selects checkpoint file)')
    ap.add_argument('--file', default='', help='explicit checkpoint yaml (overrides --map)')
    ap.add_argument('--loops', type=int, default=1, help='patrol loops (0 = forever)')
    ap.add_argument('--dwell', type=float, default=5.0, help='seconds to stop at each checkpoint')
    ap.add_argument('--follow', action='store_true',
                    help='re-read the checkpoint file before every leg, so checkpoints added (or removed) '
                         'while patrolling are followed; waits if there are none yet')
    ap.add_argument('--return-home', action='store_true', help='go back to cp_01 at the end')
    ap.add_argument('--nav2-timeout', type=float, default=180.0, help='seconds to wait for Nav2 to be active')
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    path = args.file or checkpoints_file(args.map)
    def load():
        """(checkpoints, frame) from the file; ([], 'map') when it is missing, empty or half-written."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return list(data.get('checkpoints') or []), data.get('frame_id', 'map')
        except (OSError, yaml.YAMLError):
            return [], 'map'

    cps, frame = load()
    if not args.follow:                                   # --follow waits for checkpoints instead
        if not os.path.exists(path):
            sys.exit(f'No checkpoint file {path}. Record checkpoints first (Y button during navigation).')
        if not cps:
            sys.exit(f'No checkpoints in {path}')

    rclpy.init()
    nav = BasicNavigator()
    pub = nav.create_publisher(String, 'patrol/at_checkpoint', 10)
    missed_pub = nav.create_publisher(String, 'patrol/not_reached', 10)   # after two attempts + a later return
    nav.get_logger().info(f'Loaded {len(cps)} checkpoints from {path}')
    if not wait_for_nav2(nav, args.nav2_timeout):
        sys.exit(f'Nav2 (bt_navigator) is not active after {args.nav2_timeout:.0f} s. Is navigation running?')
    nav.get_logger().info('Nav2 is ready')

    def drive_to(cp):
        nav.goToPose(make_pose(nav, frame, cp))
        while not nav.isTaskComplete():
            rclpy.spin_once(nav, timeout_sec=0.1)
        return nav.getResult() == TaskResult.SUCCEEDED

    def visit(cp, label):
        """Drive to a checkpoint (one retry) and stop there. False = not reached."""
        nav.get_logger().info(f'[{label}] -> {cp["id"]} ({cp["x"]}, {cp["y"]})')
        if not drive_to(cp):
            nav.get_logger().warn(f'{cp["id"]} not reached, trying once more')
            if not drive_to(cp):
                return False
        pub.publish(String(data=cp['id']))
        nav.get_logger().info(f'Arrived at {cp["id"]}, inspecting for {args.dwell}s')
        time.sleep(args.dwell)
        return True

    loop = 0
    try:
        waiting = False
        while args.loops == 0 or loop < args.loops:
            if args.follow:
                cps, frame = load()
                if not cps:                               # nothing to visit yet: wait for the first checkpoint
                    if not waiting:
                        nav.get_logger().info('No checkpoints yet. Add one ("+ Checkpoint here" or Y) and the robot will go.')
                    waiting = True
                    time.sleep(2.0)
                    continue
                waiting = False
            loop += 1
            if args.follow:
                # one leg at a time, re-reading the file: a checkpoint added during this loop is visited in it
                done, missed = set(), []
                while True:
                    cps, frame = load()
                    cp = next((c for c in cps if c['id'] not in done), None)
                    if cp is None:
                        break
                    done.add(cp['id'])
                    if not visit(cp, f'loop {loop}'):
                        missed.append(cp)
            else:
                missed = [cp for cp in cps if not visit(cp, f'loop {loop}')]
            # Something was in the way (people, a cart). Come back at the end of the loop: it may
            # have moved, and the robot now arrives from a different direction.
            missed = [cp for cp in missed if not visit(cp, f'loop {loop}, second attempt')]
            total = len(done) if args.follow else len(cps)   # with --follow the list can change during the loop
            for cp in missed:
                missed_pub.publish(String(data=cp['id']))
            nav.get_logger().info(
                f'Loop {loop}: {total - len(missed)}/{total} checkpoints inspected'
                + (f', NOT reached: {", ".join(cp["id"] for cp in missed)}' if missed else ''))
        if args.return_home and cps:
            drive_to(cps[0])
        nav.get_logger().info('Patrol finished')
    except KeyboardInterrupt:
        nav.cancelTask()
        nav.get_logger().warn('Patrol cancelled')
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
