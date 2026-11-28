"""Burst mode: take a large set of RGB-D photos (default 120) at a spot, nudging the robot between
groups of photos (turns of 8 / 16 deg, slides and backs of 5 cm) so the frames are not all the same
view. Runs inside the dashboard (Burst mode card); photos go straight to the SSD
($AME505_DATA/dataset/<session>/<spot>/ + index.csv, same format as capture_dataset).

Two ways to use it:
  * "Burst here" (mode 'here'): drive the robot to the spot yourself (Xbox or the dashboard keys),
    then press the button. No map or localization needed: works in drive, mapping or navigation.
    The pose is still written to index.csv (map frame when navigation is localized, else odom).
  * "All checkpoints" (mode 'route'): Nav2 drives to each saved checkpoint and bursts there.

Localization for the route is judged by the lidar, not by AMCL's own confidence (AMCL standing
still stays "unsure" for a long time even when it is right): the scan must line up with the
saved map's walls. If it doesn't, the robot locates itself on the whole map by scan matching
(locate.py, like the dashboard's Auto-locate) and tries again. AMCL jumps that odometry doesn't
explain stop the route (wrong aisle among the identical tables).

Nudges are closed-loop on odometry (EKF: wheels + IMU), each skipped if the lidar sees something
close in the direction of motion. Missing photos are made up at the spot at the end.
"""
import csv
import json
import math
import os
import threading
import time

import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from ame505_bringup import locate
from ame505_bringup.capture_dataset import FIELDS, to_np
from ame505_bringup.paths import data_dir

# Nudge positions relative to the spot, in the robot's frame: (forward, left, turn) in units of the
# run's nudge_cm / nudge_deg. Position k uses entry k; the first is the spot itself.
NUDGES = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, 0, 2), (0, 0, -2), (0, 1, 0), (0, -1, 0),
          (-1, 0, 0), (0, 1, 1), (0, -1, -1), (-1, 1, -1), (-1, -1, 1)]
NUDGE_NAMES = ['at the spot', 'turn left', 'turn right', 'turn left x2', 'turn right x2', 'slide left',
               'slide right', 'back', 'slide left + turn left', 'slide right + turn right',
               'back + left, turn right', 'back + right, turn left']

# Localization gates (route mode only)
SCAN_MATCH_MIN = 0.65       # share of lidar points within 10 cm of a wall on the saved map. Tuned
                            # on lab2d_11_23_1: correct pose >= 81 %, one table-aisle over <= 55 %.
POS_STD_MAX = 0.5           # m: AMCL may be "fair", only a wide-spread cloud fails
JUMP_POS = 0.5              # m: AMCL moved this much more than odometry did -> it jumped
JUMP_YAW = math.radians(20)
# Sensors: the minimum for a burst (the Nano under load is slow, this only catches dead sensors)
NEED_HZ = {'/depth_cam/rgb/image_raw': 0.2, '/depth_cam/depth/image_raw': 0.2, '/scan': 1.0, '/odom': 1.0}
# Nudge motion limits
V_MAX, W_MAX = 0.06, 0.25   # m/s, rad/s
V_MIN, W_MIN = 0.025, 0.12  # m/s, rad/s: slowest command sent while not yet there
TOL_XY, TOL_YAW = 0.015, math.radians(2.0)
CLEAR_AHEAD = 0.25          # m from the robot's edge in the direction of motion
CLEAR_ROTATE = 0.30         # m from the centre (footprint corner radius is ~0.22 m)
HALF_L, HALF_W = 0.165, 0.14
SETTLE_S = 3.0              # after a nudge: the camera arrives 1-3 s late through the ROS 1 bridge
FRAME_GAP = 0.25            # s between photos (each must be a new camera frame)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def slow_floor(cmd, vmax, vmin, deadzone, err):
    """Clamp |cmd| to [vmin, vmax] (sign of err); zero once |err| is inside the dead zone."""
    if abs(err) < deadzone:
        return 0.0
    return math.copysign(max(vmin, min(vmax, abs(cmd))), err)


def compose(base, rel):
    """Pose `rel` (x, y, yaw in base's frame) expressed in base's parent frame."""
    c, s = math.cos(base[2]), math.sin(base[2])
    return base[0] + c * rel[0] - s * rel[1], base[1] + s * rel[0] + c * rel[1], wrap(base[2] + rel[2])


def relative(a, b):
    """Pose b expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])


def relocate(node, prior=None):
    """Find the robot on the whole saved map with the lidar and give AMCL that pose.
    (ok, message, result dict or None). Used by the dashboard's Auto-locate and by the route."""
    with node.lock:
        grid, info = getattr(node, 'map_grid', None), dict(node.map_info) if node.map_info else None
    if grid is None or info is None:
        return False, 'No map yet (start navigation with a saved map).', None
    pts = node.scan_in_map(step=1, frame='base_footprint')
    if len(pts) < 30:
        return False, 'No lidar scan (is the lidar running?).', None
    r = locate.locate(grid, info, pts, prior=prior)
    if r is None:
        return False, 'Lidar scan too thin to locate (too close to a wall or blocked?).', None
    x, y, yaw = r['pose']
    if r['score'] < SCAN_MATCH_MIN:
        return False, (f'No place on the map fits the lidar well (best {r["score"]:.0%} at '
                       f'{x:.2f}, {y:.2f}). Clutter around the robot? Use Set pose.'), r
    node.set_initial_pose(x, y, yaw, tight=True)
    msg = f'Located at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg), {r["score"]:.0%} of lidar points on walls.'
    if r['ambiguous']:
        ru = r['runner_up']['pose']
        msg += (f' Another place looks similar ({ru[0]:.2f}, {ru[1]:.2f}): check that the red lidar dots '
                'sit on the walls; if not, Set pose.')
    return True, msg, r


class BurstRun:
    def __init__(self, mgr, cps, session, scene, hazard, objects, photos, positions, nudge_cm, nudge_deg,
                 mode='route'):
        self.mgr, self.node = mgr, mgr.node
        self.mode = mode
        self.cps, self.session, self.scene, self.hazard, self.objects = cps, session, scene, hazard, objects
        self.photos = max(1, min(int(photos), 500))                  # per spot
        self.positions = max(1, min(int(positions), len(NUDGES)))
        self.per_pos = math.ceil(self.photos / self.positions)
        self.nudge_m, self.nudge_rad = nudge_cm / 100.0, math.radians(nudge_deg)
        self.allowed = ('navigation',) if mode == 'route' else ('drive', 'mapping', 'navigation')
        self.stop_flag = False
        self.out = data_dir(os.path.join('bursts', f'{session}_{time.strftime("%Y%m%d_%H%M%S")}'))
        self.results = [{'id': c['id'], 'state': 'waiting', 'frames': 0, 'bursts': 0, 'note': ''} for c in cps]
        self.status = {'state': 'starting', 'msg': 'starting', 'mode': mode, 'cp': '', 'index': 0,
                       'total': len(cps), 'burst': 0, 'bursts': self.positions, 'target': self.photos,
                       'frames': 0, 'results': self.results, 'last_rgb': '', 'dir': self.out, 'loc': {}}
        self.prev = None                # (map pose, odom pose) at the previous check, for jump detection
        self._dist = (None, None)       # (map version, distance-to-wall field) for the scan match
        self.nomotion = self.node.create_client(Empty, '/request_nomotion_update')

    # ------------------------------------------------------------------ bookkeeping
    def say(self, state, msg):
        self.status.update(state=state, msg=msg)
        self.mgr.job = f'Burst mode: {msg}'
        with open(os.path.join(self.out, 'run.log'), 'a') as f:
            f.write(f'{time.strftime("%H:%M:%S")} [{state}] {msg}\n')

    def save(self):
        """run.json on the SSD after every step, so a crash keeps everything up to that point."""
        meta = dict(mode=self.mode, session=self.session, scene=self.scene, hazard=self.hazard,
                    objects=self.objects, photos=self.photos, positions=self.positions,
                    nudge_cm=self.nudge_m * 100, nudge_deg=round(math.degrees(self.nudge_rad), 1), map=self.mgr.map)
        tmp = os.path.join(self.out, 'run.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(dict(meta=meta, status=self.status), f, indent=1)
        os.replace(tmp, os.path.join(self.out, 'run.json'))

    def stopped(self):
        return self.stop_flag or self.mgr.mode not in self.allowed or not self.mgr.running()

    # ------------------------------------------------------------------ localization (route mode)
    def scan_match(self):
        """Share of lidar points (at the AMCL pose) that land within 10 cm of a wall on the map."""
        n = self.node
        with n.lock:
            grid, info = getattr(n, 'map_grid', None), dict(n.map_info) if n.map_info else None
        if grid is None or info is None:
            return None
        if self._dist[0] != info['ver']:
            self._dist = (info['ver'], locate.distance_field(grid, info['res']))
        pts = np.asarray(n.scan_in_map(step=1, frame='base_footprint'), dtype=np.float32)
        pose = n.robot_pose('map')
        if len(pts) < 30 or pose is None:
            return None
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[(r > 0.15) & (r < locate.MAX_RANGE)]
        if len(pts) < 30:
            return None
        return locate.hit_share(self._dist[1], info, pose, pts[:, 0], pts[:, 1])

    def refresh_amcl(self, times=3):
        """AMCL only re-matches after 10 cm / 6 deg of motion; ask it to update now."""
        if self.nomotion.service_is_ready():
            for _ in range(times):
                self.nomotion.call_async(Empty.Request())
                time.sleep(0.4)
        time.sleep(0.6)

    def check_localization(self, refresh=True):
        """(ok, reason): the lidar lines up with the map at the AMCL pose and AMCL didn't jump."""
        n = self.node
        if refresh:
            self.refresh_amcl()
        loc = n.localization() or {}
        mp, od = n.robot_pose('map'), n._lookup('odom', 'base_footprint')
        match = self.scan_match()
        self.status['loc'] = dict(loc, scan_match=None if match is None else round(match, 2))
        if mp is None or od is None or not loc:
            return False, 'no map pose yet (AMCL not running or not initialised)'
        # Jump: since the previous check (about a second ago), AMCL moved far more than wheels + IMU
        # say the robot did. Compared over short steps so ordinary odometry drift never adds up.
        prev, self.prev = self.prev, (mp, od)
        if prev is not None:
            predicted = compose(prev[0], relative(prev[1], od))
            err = relative(predicted, mp)
            if math.hypot(err[0], err[1]) > JUMP_POS or abs(err[2]) > JUMP_YAW:
                return False, (f'AMCL jumped {math.hypot(err[0], err[1]):.2f} m / {math.degrees(err[2]):+.0f} deg '
                               'away from what odometry says: possibly the wrong aisle')
        if match is None:
            return False, 'no lidar scan to check the pose against the map'
        if match < SCAN_MATCH_MIN:
            return False, (f'lidar does not line up with the map ({match:.0%} of points on walls, '
                           f'need {SCAN_MATCH_MIN:.0%})')
        if loc['pos_std_m'] > POS_STD_MAX:
            return False, f'AMCL particles spread out (+-{loc["pos_std_m"]} m)'
        return True, f'localized: {match:.0%} of lidar points on walls (AMCL +-{loc["pos_std_m"]} m)'

    def ensure_localized(self, tries=3, may_relocate=True):
        ok, why = self.check_localization()
        for k in range(tries - 1):
            if ok or self.stopped():
                break
            if may_relocate and k == 0 and 'jumped' not in why:
                self.say(self.status['state'], f'{why}: locating the robot on the map with the lidar')
                good, msg, _ = relocate(self.node, prior=self.prev[0] if self.prev else None)
                self.say(self.status['state'], msg)
                self.prev = None                          # the re-seed is not a jump
                time.sleep(1.5 if good else 0.5)
            else:
                time.sleep(1.0)
            ok, why = self.check_localization()
        return ok, why

    def sensors_ok(self):
        if self.node.rate['/depth_cam/rgb/image_raw'].hz() < 0.2:
            self.say('starting', 'camera is off: starting it (takes ~10 s)')
            ok, msg = self.mgr.ensure_camera()
            if not ok:
                return False, msg
        _, rates = self.node.snapshot()
        low = [f'{t} {rates.get(t, 0):.1f} Hz' for t, hz in NEED_HZ.items() if rates.get(t, 0) < hz]
        if low:
            return False, ('no data from: ' + ', '.join(low) + '. Is the robot software running? '
                           '(camera: tick "camera" before starting, or restart it)')
        return True, ''

    # ------------------------------------------------------------------ motion
    def drive_to(self, cp):
        """Nav2 goal; watches for AMCL jumps on the way. True when arrived."""
        n = self.node
        ok, msg = n.send_goal(cp['x'], cp['y'], cp['yaw'], cp['id'])
        if not ok:
            return False, msg
        t0, last_check = time.time(), 0.0
        while time.time() - t0 < 180:
            if self.stopped():
                n.cancel_goal()
                return False, 'stopped'
            st = n.nav_state.get('state')
            if st == 'arrived':
                return True, 'arrived'
            if st in ('failed', 'canceled', 'rejected'):
                return False, f'navigation {st}'
            if time.time() - last_check > 1.0:          # jump check only (AMCL updates as it moves)
                last_check = time.time()
                ok, why = self.check_localization(refresh=False)
                if not ok and 'jumped' in why:
                    n.cancel_goal()
                    return False, why
            time.sleep(0.3)
        n.cancel_goal()
        return False, 'did not arrive within 3 min'

    def clear_for(self, vx, vy, wz):
        """Lidar check before a nudge: nothing close in the direction of motion."""
        m = self.node.last_scan
        if m is None:
            return False
        r = np.asarray(m.ranges, dtype=np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment + math.pi   # lidar is mounted yawed by pi
        ok = np.isfinite(r) & (r > max(m.range_min, 0.12)) & (r < m.range_max)
        x, y = (r * np.cos(a))[ok], (r * np.sin(a))[ok]
        if abs(vx) > 1e-3 or abs(vy) > 1e-3:
            h = math.atan2(vy, vx)                                   # motion direction, robot frame
            along, across = x * math.cos(h) + y * math.sin(h), -x * math.sin(h) + y * math.cos(h)
            half_len = abs(HALF_L * math.cos(h)) + abs(HALF_W * math.sin(h))
            half_wid = abs(HALF_L * math.sin(h)) + abs(HALF_W * math.cos(h))
            box = (along > 0) & (along < half_len + CLEAR_AHEAD) & (np.abs(across) < half_wid + 0.05)
            if box.any():
                return False
        if abs(wz) > 1e-3 and (np.hypot(x, y) < CLEAR_ROTATE).any():
            return False
        return True

    def move_to(self, ref, rel, label):
        """Closed loop on odometry: reach pose `rel` (robot frame of `ref`, an odom pose)."""
        n = self.node
        goal = compose(ref, rel)
        start = n._lookup('odom', 'base_footprint')
        t0, next_check, ok = time.time(), 0.0, True
        while time.time() - t0 < 12:
            if self.stopped():
                ok = False
                break
            cur = n._lookup('odom', 'base_footprint')
            if cur is None:
                ok = False
                break
            ex, ey, eyaw = relative(cur, goal)
            if math.hypot(ex, ey) < TOL_XY and abs(eyaw) < TOL_YAW:
                break
            # Proportional, but never slower than V_MIN / W_MIN: tiny commands barely turn the
            # mecanum wheels (motor dead band), and the last few mm / degrees took most of the time.
            vx, vy = slow_floor(1.2 * ex, V_MAX, V_MIN, TOL_XY / 2, ex), slow_floor(1.2 * ey, V_MAX, V_MIN, TOL_XY / 2, ey)
            wz = slow_floor(1.5 * eyaw, W_MAX, W_MIN, TOL_YAW / 2, eyaw)
            if time.time() >= next_check:            # lidar clearance before moving, then every 0.5 s
                next_check = time.time() + 0.5
                if not self.clear_for(vx, vy, wz):
                    ok = False
                    self.say(self.status['state'], f'{label}: skipped, obstacle close in that direction')
                    break
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
            n.estop_pub.publish(t)
            time.sleep(0.05)
        else:
            # Timed out: good enough if it got most of the way (the aim is a different view, not
            # an exact spot); not if it hardly moved (blocked, slipping on the floor).
            cur = n._lookup('odom', 'base_footprint')
            if start is None or cur is None:
                ok = False
            else:
                want = relative(start, goal)
                left = relative(cur, goal)
                ok = (math.hypot(left[0], left[1]) <= 0.5 * math.hypot(want[0], want[1]) + TOL_XY and
                      abs(left[2]) <= 0.5 * abs(want[2]) + TOL_YAW)
        self.mgr.estop_twist()
        return ok

    # ------------------------------------------------------------------ capture (in this process)
    def capture(self, cp_id, n, note):
        """Save n new RGB-D frames from the dashboard's own camera subscription. (saved, error)"""
        node = self.node
        folder = data_dir(os.path.join('dataset', self.session, cp_id))
        index = os.path.join(data_dir('dataset'), 'index.csv')
        new_index = not os.path.exists(index)
        tag = f'{self.scene}_{self.hazard}_{time.strftime("%H%M%S")}'
        saved, last, t_new = 0, None, time.time()
        with open(index, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new_index:
                w.writeheader()
            while saved < n and not self.stopped():
                with node.lock:
                    rgb_m, dep_m = node.raw_rgb, node.raw_depth
                if rgb_m is None or dep_m is None or rgb_m is last:
                    if time.time() - t_new > 10:
                        return saved, 'camera stopped sending frames for 10 s'
                    time.sleep(0.05)
                    continue
                last, t_new = rgb_m, time.time()
                try:
                    rgb, dep = to_np(rgb_m), to_np(dep_m)
                except ValueError as e:
                    return saved, str(e)
                base = os.path.join(folder, f'{tag}_{saved:03d}')
                cv2.imwrite(base + '_rgb.png', rgb)
                cv2.imwrite(base + '_depth.png', dep.astype(np.uint16))
                frame = 'map' if node.robot_pose('map') is not None else 'odom'
                x, y, yaw = node.robot_pose(frame) or (float('nan'),) * 3
                stamp = rgb_m.header.stamp.sec + rgb_m.header.stamp.nanosec * 1e-9
                w.writerow({'file_rgb': os.path.relpath(base + '_rgb.png', data_dir('dataset')),
                            'file_depth': os.path.relpath(base + '_depth.png', data_dir('dataset')),
                            'stamp': f'{stamp:.3f}', 'session': self.session, 'checkpoint': cp_id,
                            'scene': self.scene, 'hazard': self.hazard, 'objects': self.objects,
                            'light': 'normal', 'occluded': '0', 'note': note, 'pose_frame': frame,
                            'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 3),
                            'depth_valid_pct': round(100.0 * float((dep > 0).mean()), 1),
                            'rgb_age_s': round(time.time() - stamp, 2)})
                f.flush()
                saved += 1
                self.status['frames'] += 1
                self.status['last_rgb'] = os.path.relpath(base + '_rgb.png', data_dir('dataset'))
                time.sleep(FRAME_GAP)
        return saved, ''

    # ------------------------------------------------------------------ one spot
    def burst_at_spot(self, cp_id, res, check_loc):
        """All nudge positions at the current spot until self.photos are saved. Returns False to abort."""
        self.say('settling', f'{cp_id}: holding still {SETTLE_S:.0f} s (camera is late)')
        time.sleep(SETTLE_S)
        ref = self.node._lookup('odom', 'base_footprint')
        if ref is None:
            res['note'] = 'no odometry: photos only at the spot, no nudges'
        for k in range(self.positions):
            if self.stopped():
                return False
            want = min(self.per_pos, self.photos - res['frames'])
            if want <= 0:
                break
            self.status['burst'] = k + 1
            fx, fy, fyaw = NUDGES[k]
            label = NUDGE_NAMES[k]
            if k > 0:
                if ref is None:
                    break
                self.say('nudging', f'{cp_id} position {k + 1}/{self.positions}: {label}')
                rel = (fx * self.nudge_m, fy * self.nudge_m, fyaw * self.nudge_rad)
                if not self.move_to(ref, rel, label):
                    if self.stopped():
                        return False
                    continue
                time.sleep(SETTLE_S)
            if check_loc:
                ok, why = self.ensure_localized(tries=2, may_relocate=False)
                if not ok:
                    res['note'] = f'position {k + 1} skipped: {why}'
                    self.say('capturing', f'{cp_id} position {k + 1}: skipped ({why})')
                    if 'jumped' in why:
                        break
                    continue
            self.say('capturing', f'{cp_id} position {k + 1}/{self.positions} ({label}): '
                                  f'{want} photos ({res["frames"]}/{self.photos} so far)')
            saved, err = self.capture(cp_id, want, f'burst {k + 1}/{self.positions} {label}'.replace('+', 'and'))
            res['frames'] += saved
            res['bursts'] += 1 if saved else 0
            self.save()
            if err:
                res['note'] = f'capture: {err}'
                self.say('capturing', f'{cp_id}: {err}')
                return False
        if ref is not None and self.positions > 1 and not self.stopped():
            self.say('nudging', f'{cp_id}: back to the spot')
            self.move_to(ref, (0.0, 0.0, 0.0), 'back to the spot')
        missing = self.photos - res['frames']
        if missing > 0 and not self.stopped():           # skipped nudges: make up the count here
            time.sleep(SETTLE_S)
            self.say('capturing', f'{cp_id}: {missing} more photos at the spot to reach {self.photos}')
            saved, err = self.capture(cp_id, missing, 'burst top-up at the spot')
            res['frames'] += saved
            if err:
                res['note'] = f'capture: {err}'
        return not self.stopped()

    # ------------------------------------------------------------------ the run
    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001  (burst mode must never take the dashboard down)
            self.say('failed', f'error: {e}')
        finally:
            self.mgr.estop_twist()
            self.save()

    def _run(self):
        ok, why = self.sensors_ok()
        if not ok:
            return self.say('failed', why)
        if self.mode == 'here':
            res = self.results[0]
            self.status.update(cp=res['id'], index=1)
            res['state'] = 'capturing'
            self.burst_at_spot(res['id'], res, check_loc=False)
            res['state'] = 'done' if res['frames'] >= self.photos else 'partial' if res['frames'] else 'no photos'
            self.save()
            if self.stopped():
                return self.say('stopped', f'stopped: {res["frames"]} photos saved')
            return self.say('done', f'done: {res["frames"]} photos of {res["id"]} saved in '
                                    f'{data_dir("dataset")}/{self.session}/{res["id"]}')

        self.say('checking', 'checking localization before moving')
        ok, why = self.ensure_localized(tries=3)
        if not ok:
            return self.say('failed', f'not starting: {why}. Press Auto-locate (or Set pose) on the map, '
                                      'or use "Burst here" after driving to the spot yourself.')
        for i, cp in enumerate(self.cps):
            res = self.results[i]
            self.status.update(cp=cp['id'], index=i + 1, burst=0)
            if self.stopped():
                return self.say('stopped', 'stopped')
            res['state'] = 'driving'
            self.say('driving', f'{cp["id"]} ({i + 1}/{len(self.cps)}): driving there')
            arrived, why = self.drive_to(cp)
            if not arrived and 'jumped' in why and not self.stopped():
                ok, why2 = self.ensure_localized(tries=3)
                if ok:
                    arrived, why = self.drive_to(cp)
            elif not arrived and not self.stopped():
                self.say('driving', f'{cp["id"]}: {why}, trying once more')
                arrived, why = self.drive_to(cp)
            if not arrived:
                res.update(state='not reached', note=why)
                self.save()
                if self.stopped():
                    return self.say('stopped', 'stopped')
                if 'jumped' in why:
                    return self.say('failed', f'stopped at {cp["id"]}: {why}. Auto-locate, then restart '
                                              'from this checkpoint.')
                continue
            res['state'] = 'capturing'
            if not self.burst_at_spot(cp['id'], res, check_loc=True):
                res['state'] = 'partial' if res['frames'] else 'no photos'
                self.save()
                if self.stopped():
                    return self.say('stopped', 'stopped')
                continue
            res['state'] = 'done' if res['frames'] >= self.photos else 'partial' if res['frames'] else 'no photos'
            self.save()
        done = sum(1 for r in self.results if r['state'] == 'done')
        self.say('done', f'done: {self.status["frames"]} photos at {done}/{len(self.cps)} checkpoints, '
                         f'saved in {data_dir("dataset")}/{self.session}')


def start(mgr, cps, **kw):
    run = BurstRun(mgr, cps, **kw)
    threading.Thread(target=run.run, daemon=True).start()
    return run
