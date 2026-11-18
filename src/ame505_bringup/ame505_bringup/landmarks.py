#!/usr/bin/env python3
"""Camera landmarks (printed ArUco markers) that tell the robot WHICH table it is at.

The 2D lidar sees one horizontal plane (~15 cm up): four identical tables in a row look the
same, so slam_toolbox / AMCL can confuse them. A marker id is unique, so seeing marker 3 means
"I am near table 3" regardless of what the lidar pattern says.

What it does (only while the robot is STANDING STILL, because the camera arrives 1-7 s late
over the ROS 1 bridge and its timestamps are receipt times):
  * detects ArUco markers (DICT_4X4_50, big cells: readable at 320x240) and their 3D pose
    (solvePnP, known printed size) in the camera frame
  * LEARN: unknown marker + AMCL confident  -> store its map pose in landmarks/<map>.yaml
           (averaged over several sightings). Or force it with the dashboard ("Save markers in view").
  * CHECK: known marker -> robot pose implied by the marker vs AMCL. If AMCL is off by more
           than correct_dist / correct_yaw, re-seed AMCL there (/initialpose with modest
           covariance); the lidar then refines it on the walls.
Status for the dashboard: /landmarks/status (JSON String).

    ros2 run ame505_bringup landmarks --ros-args -p map:=lab2d -p marker_size:=0.15
Print markers: ros2 run ame505_bringup make_markers  (or docs/markers/*.png)
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from ame505_bringup.paths import data_dir

DICT = cv2.aruco.DICT_4X4_50


def landmarks_file(map_name):
    return os.path.join(data_dir('landmarks'), f'{map_name}.yaml')


def tf_matrix(t):
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def planar(T):
    """4x4 -> (x, y, yaw) of the frame's x axis projected on the floor."""
    return float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0])


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def mean_angle(a):
    return math.atan2(sum(math.sin(x) for x in a), sum(math.cos(x) for x in a))


class Landmarks(Node):
    def __init__(self):
        super().__init__('landmarks')
        p = self.declare_parameter
        self.map = p('map', 'lab2d').value
        self.size = p('marker_size', 0.15).value            # black square side, metres
        self.max_range = p('max_range', 2.5).value          # farther = too few pixels at 320x240
        self.still_v = p('still_speed', 0.02).value         # m/s
        self.still_w = p('still_turn', 0.04).value          # rad/s
        self.latency = p('camera_latency', 2.5).value       # s: image must be newer than still-start + this
        self.correct_dist = p('correct_dist', 0.30).value    # m: AMCL further off -> re-seed
        self.correct_yaw = math.radians(p('correct_yaw_deg', 12.0).value)
        self.learn_std = p('learn_max_std', 0.12).value     # AMCL must be at least this sure to learn
        self.learn_n = p('learn_sightings', 3).value
        self.camera_frame = p('camera_frame', 'depth_cam_frame').value
        self.period = p('period', 0.5).value                # s between detections (CPU)

        self.file = landmarks_file(self.map)
        self.db = {}
        if os.path.exists(self.file):
            self.db = {int(k): v for k, v in ((yaml.safe_load(open(self.file)) or {}).get('markers') or {}).items()}
        self.pending = {}                                   # id -> list of (x, y, yaw) sightings
        self.K = None
        self.still_since = None
        self.amcl_std = (9.0, 9.0)
        self.last_fix = {}
        self.force_learn_until = 0.0
        self.status = {'map': self.map, 'known': len(self.db), 'seen': [], 'last': '', 'file': self.file}
        self.last_proc = 0.0

        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        det_params = cv2.aruco.DetectorParameters()
        det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), det_params)
        sd = qos_profile_sensor_data
        self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.on_image, sd)
        self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.on_info, sd)
        self.create_subscription(Odometry, '/odom', self.on_odom, sd)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl, latched)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        self.status_pub = self.create_publisher(String, '/landmarks/status', 1)
        self.create_service(Trigger, '/landmarks/learn_now', self.on_learn_now)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info(f'{len(self.db)} known markers for map "{self.map}" ({self.file}); '
                               f'marker size {self.size} m, dictionary DICT_4X4_50')

    # ---------- inputs ----------
    def on_info(self, m):
        self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)

    def on_odom(self, m):
        t = m.twist.twist
        moving = math.hypot(t.linear.x, t.linear.y) > self.still_v or abs(t.angular.z) > self.still_w
        now = time.time()
        if moving:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now

    def on_amcl(self, m):
        c = m.pose.covariance
        self.amcl_std = (math.sqrt(max(c[0], 0) + max(c[7], 0)), math.sqrt(max(c[35], 0)))

    def on_learn_now(self, req, res):
        self.force_learn_until = time.time() + 6.0
        res.success, res.message = True, 'Hold the robot still: markers in view are saved in the next few seconds.'
        return res

    # ---------- main ----------
    def on_image(self, m):
        now = time.time()
        if self.K is None or now - self.last_proc < self.period:
            return
        # still long enough that this (late) image was certainly taken while standing still
        if self.still_since is None or now - self.still_since < self.latency:
            self.status['seen'] = []
            return
        self.last_proc = now
        if m.encoding not in ('bgr8', 'rgb8'):
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if m.encoding == 'bgr8' else cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            self.status['seen'] = []
            return
        try:
            T_map_cam = tf_matrix(self.tf.lookup_transform('map', self.camera_frame, Time()))
            T_map_base = tf_matrix(self.tf.lookup_transform('map', 'base_footprint', Time()))
        except Exception:  # noqa: BLE001  (not localized yet)
            return
        h = self.size / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        seen, fixes = [], []
        for c, i in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), self.K, None,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            dist = float(np.linalg.norm(tvec))
            if dist > self.max_range:
                continue
            T_cam_mk = np.eye(4)
            T_cam_mk[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cam_mk[:3, 3] = tvec.flatten()
            T_map_mk = T_map_cam @ T_cam_mk
            if abs(T_map_mk[2, 2]) > 0.5:      # marker lying flat: no usable facing direction
                continue
            mk = planar_marker(T_map_mk)
            seen.append({'id': int(i), 'dist': round(dist, 2)})
            if int(i) in self.db:
                fixes.append(self.robot_from_marker(int(i), T_map_mk, T_map_base))
            else:
                self.learn(int(i), mk)
        self.status['seen'] = seen
        fixes = [f for f in fixes if f is not None]
        if fixes:
            self.check(fixes, T_map_base)

    def robot_from_marker(self, i, T_map_mk_obs, T_map_base):
        """Where the robot must be if marker i is really at its stored pose."""
        s = self.db[i]
        # The observed marker sits at T_map_mk_obs given the CURRENT localization; the stored one
        # is the truth. Shift the robot by the 2D transform that moves observed -> stored.
        ox, oy, oyaw = planar_marker(T_map_mk_obs)
        dyaw = wrap(s['yaw'] - oyaw)
        c, sn = math.cos(dyaw), math.sin(dyaw)
        bx, by, byaw = planar(T_map_base)
        rx, ry = bx - ox, by - oy
        return (s['x'] + c * rx - sn * ry, s['y'] + sn * rx + c * ry, wrap(byaw + dyaw), i)

    def check(self, fixes, T_map_base):
        x = float(np.mean([f[0] for f in fixes]))
        y = float(np.mean([f[1] for f in fixes]))
        yaw = mean_angle([f[2] for f in fixes])
        bx, by, byaw = planar(T_map_base)
        err, eyaw = math.hypot(x - bx, y - by), abs(wrap(yaw - byaw))
        ids = sorted({f[3] for f in fixes})
        self.status['last_check'] = {'ids': ids, 'err_m': round(err, 2), 'err_deg': round(math.degrees(eyaw), 1),
                                     't': time.strftime('%H:%M:%S')}
        if err < self.correct_dist and eyaw < self.correct_yaw:
            return
        if time.time() - self.last_fix.get(tuple(ids), 0) < 5.0:
            return
        self.last_fix[tuple(ids)] = time.time()
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.15 ** 2          # marker pose at 320x240: ~10-15 cm
        cov[35] = math.radians(8) ** 2
        msg.pose.covariance = cov
        self.init_pub.publish(msg)
        self.status['last'] = (f'{time.strftime("%H:%M:%S")} marker {ids}: AMCL was {err:.2f} m / '
                               f'{math.degrees(eyaw):.0f} deg off -> re-seeded at ({x:.2f}, {y:.2f})')
        self.get_logger().warn(self.status['last'])

    def learn(self, i, mk):
        forced = time.time() < self.force_learn_until
        if not forced and (self.amcl_std[0] > self.learn_std or self.amcl_std[1] > math.radians(6)):
            return
        obs = self.pending.setdefault(i, [])
        obs.append(mk)
        if len(obs) < (1 if forced else self.learn_n):
            return
        xs, ys, yws = zip(*obs)
        spread = math.hypot(np.std(xs), np.std(ys))
        if spread > 0.15:                     # inconsistent sightings: start over
            self.pending[i] = obs[-1:]
            return
        self.db[i] = {'x': round(float(np.mean(xs)), 3), 'y': round(float(np.mean(ys)), 3),
                      'yaw': round(mean_angle(yws), 3), 'n': len(obs), 'saved': time.strftime('%Y-%m-%d %H:%M')}
        self.pending.pop(i, None)
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, 'w') as f:
            yaml.safe_dump({'map': self.map, 'dictionary': 'DICT_4X4_50', 'marker_size': self.size,
                            'markers': self.db}, f, sort_keys=False)
        self.status['last'] = f'{time.strftime("%H:%M:%S")} learned marker {i} at ({self.db[i]["x"]}, {self.db[i]["y"]})'
        self.get_logger().info(self.status['last'])

    def publish_status(self):
        self.status['known'] = len(self.db)
        self.status['markers'] = {str(k): {'x': v['x'], 'y': v['y'], 'yaw': v['yaw']} for k, v in self.db.items()}
        self.status['still'] = self.still_since is not None and time.time() - self.still_since >= self.latency
        self.status_pub.publish(String(data=json.dumps(self.status)))


def planar_marker(T):
    """Marker 4x4 in map -> (x, y, yaw) where yaw is the direction the marker FACES (its +z)."""
    n = T[:3, 2]
    return float(T[0, 3]), float(T[1, 3]), math.atan2(n[1], n[0])


def main():
    rclpy.init()
    node = Landmarks()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
