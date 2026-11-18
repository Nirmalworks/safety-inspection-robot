#!/usr/bin/env python3
"""Live COCO object detection (Ultralytics YOLO26n) on the robot camera, for the dashboard.

    ros2 run ame505_bringup yolo_live                                  # dashboard starts it with "YOLO live"
    ros2 run ame505_bringup yolo_live --ros-args -p model:=yolo26s.pt -p period:=1.0

Subscribes  /depth_cam/rgb/image_raw, /depth_cam/depth/image_raw (aligned, mm)
Publishes   /yolo/image       annotated BGR image (boxes, class, confidence, distance)
            /yolo/detections  JSON String: {"stamp", "infer_s", "fps", "dets": [{cls, conf, box, dist_m, relevant}]}

It only looks at the NEWEST frame and runs at most every `period` seconds, so it never queues up
work: on the Jetson Nano CPU one 320x240 frame takes ~0.55 s. It uses ~400-500 MB of RAM (torch),
so the dashboard starts it on demand and stopping it gives the memory back. Detection only: it does
NOT drive or stop the robot (the lidar collision monitor does that).
Install once on the robot: scripts/setup_yolo.sh
"""
import json
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ame505_bringup.detection import DEFAULT_MODEL, RELEVANT, box_distance_m, import_yolo, weights_path

COLOR_REL = (40, 170, 255)      # orange (BGR): hazard-relevant classes
COLOR_OTHER = (200, 200, 200)


def to_np(m):
    buf = np.frombuffer(m.data, dtype=np.uint8)
    if m.encoding in ('bgr8', 'rgb8'):
        img = buf.reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        return img if m.encoding == 'bgr8' else np.ascontiguousarray(img[:, :, ::-1])
    if m.encoding in ('16UC1', 'mono16'):
        return buf.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
    return None


def draw(img, dets, fps):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d['box'])
        c = COLOR_REL if d['relevant'] else COLOR_OTHER
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        txt = f"{d['cls']} {d['conf']:.2f}" + (f" {d['dist_m']:.1f}m" if d['dist_m'] is not None else '')
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        y = max(th + 4, y1)
        cv2.rectangle(out, (x1, y - th - 4), (x1 + tw + 4, y), c, -1)
        cv2.putText(out, txt, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f'YOLO26n COCO  {fps:.1f} fps', (6, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class YoloLive(Node):
    def __init__(self):
        super().__init__('yolo_live')
        self.model_name = self.declare_parameter('model', DEFAULT_MODEL).value
        self.period = float(self.declare_parameter('period', 0.5).value)
        self.conf = float(self.declare_parameter('conf', 0.30).value)
        self.imgsz = int(self.declare_parameter('imgsz', 320).value)
        self.rgb = self.depth = None
        self.lock = threading.Lock()
        self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, '/depth_cam/depth/image_raw', self.on_depth, qos_profile_sensor_data)
        self.pub_img = self.create_publisher(Image, '/yolo/image', 1)
        self.pub_det = self.create_publisher(String, '/yolo/detections', 1)
        self.get_logger().info(f'loading {self.model_name} ...')
        self.model = import_yolo()(weights_path(self.model_name))
        self.model.predict(np.zeros((240, 320, 3), np.uint8), imgsz=self.imgsz, verbose=False)   # warm-up
        self.get_logger().info('YOLO ready')

    def on_rgb(self, m):
        with self.lock:
            self.rgb = m

    def on_depth(self, m):
        with self.lock:
            self.depth = m

    def loop(self):
        last_id, fps = None, 0.0
        while rclpy.ok():
            t0 = time.time()
            with self.lock:
                m, dm = self.rgb, self.depth
            if m is None or m is last_id:
                time.sleep(0.05)
                continue
            last_id = m
            img = to_np(m)
            depth = to_np(dm) if dm is not None else None
            if img is None:
                time.sleep(0.2)
                continue
            if depth is not None and depth.shape[:2] != img.shape[:2]:
                depth = None
            res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            dt = time.time() - t0
            names = self.model.names
            dets = []
            for b in res.boxes:
                cid = int(b.cls)
                box = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                dist = box_distance_m(depth, box)
                dets.append({'cls': names[cid], 'conf': round(float(b.conf), 3), 'box': box,
                             'dist_m': None if math.isnan(dist) else round(dist, 2),
                             'relevant': names[cid] in RELEVANT})
            fps = 0.7 * fps + 0.3 / max(dt, 1e-3) if fps else 1.0 / max(dt, 1e-3)
            out = draw(img, dets, fps)
            msg = Image(header=m.header, height=out.shape[0], width=out.shape[1], encoding='bgr8',
                        is_bigendian=0, step=out.shape[1] * 3, data=out.tobytes())
            self.pub_img.publish(msg)
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            self.pub_det.publish(String(data=json.dumps({'stamp': stamp, 'infer_s': round(dt, 3),
                                                         'fps': round(fps, 2), 'dets': dets})))
            time.sleep(max(0.0, self.period - (time.time() - t0)))


def main():
    rclpy.init()
    node = YoloLive()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        node.loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
