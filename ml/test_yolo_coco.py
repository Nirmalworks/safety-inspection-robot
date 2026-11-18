"""pytest ml/test_yolo_coco.py   (no ultralytics, no ROS needed)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_coco import box_distance_m, yolo_line  # noqa: E402


def test_box_distance_uses_the_box_centre_and_ignores_missing_depth():
    d = np.full((240, 320), 3000, np.uint16)       # wall at 3 m
    d[100:200, 100:200] = 1200                      # object at 1.2 m
    d[140:150, 140:150] = 0                         # a hole in the depth
    assert abs(box_distance_m(d, (100, 100, 200, 200)) - 1.2) < 1e-6
    assert math.isnan(box_distance_m(np.zeros((240, 320), np.uint16), (0, 0, 50, 50)))
    assert math.isnan(box_distance_m(None, (0, 0, 50, 50)))


def test_tiny_box_at_the_image_edge_still_reads_a_pixel():
    d = np.full((240, 320), 800, np.uint16)
    assert abs(box_distance_m(d, (319, 239, 320, 240)) - 0.8) < 1e-6


def test_yolo_line_is_normalized_centre_and_size():
    assert yolo_line(2, (80, 60, 240, 180), 320, 240) == '2 0.500000 0.500000 0.500000 0.500000'
