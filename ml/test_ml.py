"""pytest ml/test_ml.py   (no ROS, no robot, no real data needed)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ame505_bringup'))
sys.path.insert(0, HERE)
from ame505_bringup.keep_clear import check  # noqa: E402
from evaluate import metrics  # noqa: E402
from fake_dataset import render  # noqa: E402
from split import assign  # noqa: E402


def test_empty_floor_is_not_a_hazard():
    r = check(render())
    assert not r['hazard'] and r['area_m2'] == 0


def test_box_in_zone_is_a_hazard_with_the_right_size_and_distance():
    r = check(render(box=(1.0, 0.0, 0.3, 0.3, 0.4)))       # 0.3 m wide, 0.4 m high, front face at 0.85 m
    assert r['hazard']
    assert 0.08 < r['area_m2'] < 0.12                       # 0.3 x (0.4 - z_min 0.06) = 0.10 m2
    assert abs(r['nearest_m'] - 0.85) < 0.03


def test_box_beside_or_beyond_the_zone_is_not_a_hazard():
    assert not check(render(box=(1.0, 0.9, 0.3, 0.3, 0.4)))['hazard']     # 0.9 m to the left
    assert not check(render(box=(2.2, 0.0, 0.3, 0.3, 0.4)))['hazard']     # behind x_max 1.5


def test_small_object_under_the_area_threshold_is_ignored():
    assert not check(render(box=(1.0, 0.0, 0.05, 0.05, 0.12)))['hazard']  # 5 cm wide: ~0.003 m2


def test_tilted_camera_needs_its_pitch_configured():
    cam = {'fx': 285.0, 'fy': 285.0, 'cx': 160.0, 'cy': 120.0, 'x': 0.029, 'height': 0.207, 'pitch_deg': 0.0}
    r = check(render(), camera={**cam, 'pitch_deg': -4.0})  # rule believes the camera looks 4 deg up
    assert r['hazard']                                      # ...so the floor is taken for an object


def test_split_keeps_groups_whole_and_holdout_out_of_training():
    rows = [{'session': s, 'checkpoint': f'cp_{c:02d}'} for s in ('s1', 's2', 's3') for c in range(1, 11) for _ in range(48)]
    split = assign(rows, {'cp_10'})
    assert all(v == 'test' for g, v in split.items() if g[1] == 'cp_10')
    share = {k: sum(48 for v in split.values() if v == k) / len(rows) for k in ('train', 'val', 'test')}
    assert 0.6 < share['train'] < 0.75 and share['val'] >= 0.15 and share['test'] >= 0.15


def test_metrics():
    m = metrics([(True, True)] * 9 + [(True, False)] + [(False, False)] * 8 + [(False, True)] * 2)
    assert m['recall'] == 0.9 and round(m['precision'], 3) == 0.818 and m['fp'] == 2


def test_scores_and_labels():
    from data import CLASSES, label, scores
    assert label({'scene': 'normal', 'hazard': 'none'}) == 0
    assert label({'scene': 'hazard', 'hazard': 'open_door'}) == CLASSES.index('open_door')
    assert label({'scene': 'hazard', 'hazard': 'multiple'}) is None
    s = scores([0, 0, 1, 2, 3], [0, 1, 1, 2, 0])       # one false alarm, one missed hazard
    assert s['hazard_precision'] == 2 / 3 and s['hazard_recall'] == 2 / 3 and s['accuracy'] == 3 / 5


def test_hog_shape_and_response():
    import numpy as np
    from classic_baselines import hog
    flat = hog(np.zeros((96, 128), np.uint8))
    edge = np.zeros((96, 128), np.uint8)
    edge[:, 64:] = 255                                   # one vertical edge
    h = hog(edge)
    assert flat.shape == h.shape == (11 * 15 * 2 * 2 * 9,)
    assert h.max() > 0 and flat.max() == 0
