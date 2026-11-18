"""Offline check of the landmark geometry (no ROS needed): python3 -m pytest test/test_landmarks_math.py

Renders a DICT_4X4_50 marker into a synthetic 320x240 camera image at a known pose, detects it,
and checks (1) the detected marker pose and (2) that a robot pose that is off by a whole table
(1.8 m, AMCL aliasing) is corrected back by robot_from_marker().
"""
import math

import cv2
import numpy as np

from ame505_bringup import landmarks as L

K = np.array([[285.0, 0, 160.0], [0, 285.0, 120.0], [0, 0, 1]])   # ~ Astra colour scaled to 320x240
SIZE = 0.15


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


# camera optical frame w.r.t. base (URDF: depth_cam_link at x=0.029, z=0.202; rpy -pi/2, 0, -pi/2)
R_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical z fwd, x right, y down
T_base_cam = trans(0.029, 0, 0.202)
T_base_cam[:3, :3] = R_opt


def render(T_cam_mk):
    """Synthetic image of the marker at T_cam_mk (marker frame: OpenCV convention)."""
    px = 400
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 3, px)
    mk = cv2.copyMakeBorder(mk, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    h = SIZE / 2 * 600 / 400                                     # white border is part of the plane
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0], [0, 0, 0]])
    pts = (T_cam_mk[:3, :3] @ obj.T).T + T_cam_mk[:3, 3]
    uv = (K @ pts.T).T
    uv = uv[:, :2] / uv[:, 2:]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [600, 0], [600, 600], [0, 600]]), np.float32(uv[:4]))
    img = cv2.warpPerspective(mk, H, (320, 240), borderValue=200)
    return img


def test_detect_and_correct():
    # marker on a table side at map (3.0, 1.0), facing -x (towards the robot), 0.35 m high
    # OpenCV marker frame: x right, y UP, z OUT of the marker. Facing -x in the map:
    # z = (-1,0,0), y = (0,0,1), x = y cross z = (0,-1,0)
    T_map_mk = trans(3.0, 1.0, 0.35)
    T_map_mk[:3, :3] = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    true_base = trans(1.6, 1.1, 0) @ rot_z(math.radians(5))
    T_cam_mk = np.linalg.inv(true_base @ T_base_cam) @ T_map_mk

    img = render(T_cam_mk)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(img)
    assert ids is not None and ids.flatten().tolist() == [3]
    h = SIZE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, corners[0].reshape(4, 2).astype(float), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    assert ok
    T_cam_obs = np.eye(4)
    T_cam_obs[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_obs[:3, 3] = tvec.flatten()
    assert np.linalg.norm(T_cam_obs[:3, 3] - T_cam_mk[:3, 3]) < 0.05      # distance right to 5 cm

    # AMCL is aliased one table further (+1.8 m in x): the marker "appears" 1.8 m off its stored spot
    wrong_base = trans(1.6 + 1.8, 1.1, 0) @ rot_z(math.radians(5))
    T_map_mk_obs = wrong_base @ T_base_cam @ T_cam_obs

    node = L.Landmarks.__new__(L.Landmarks)
    mx, my, myaw = L.planar_marker(T_map_mk)
    node.db = {3: {'x': mx, 'y': my, 'yaw': myaw}}
    x, y, yaw, i = L.Landmarks.robot_from_marker(node, 3, T_map_mk_obs, wrong_base)
    assert i == 3
    assert math.hypot(x - 1.6, y - 1.1) < 0.12, (x, y)                     # back at the true table
    assert abs(L.wrap(yaw - math.radians(5))) < math.radians(6), math.degrees(yaw)


if __name__ == '__main__':
    test_detect_and_correct()
    print('ok')
