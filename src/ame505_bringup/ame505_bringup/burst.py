"""Burst mode: visit the checkpoints with Nav2 and, at each one, take several bursts of RGB-D photos
with a small nudge of the robot between bursts (turn a few degrees, slide a few cm), so the frames
of one checkpoint are not near-duplicates. Runs inside the dashboard (Burst mode card,
POST /api/burst_start, scripts/burst_mode.sh); photos go to the SSD through capture_dataset.

Localization is checked instead of trusted, because AMCL can be confidently wrong in the lab (four
identical tables): before every drive and every burst it must be
  * confident: AMCL position std < 0.15 m and heading std < 9 deg ("good" on the dashboard),
  * consistent with the map: most lidar points land on walls of the saved map at the AMCL pose,
  * consistent with odometry: AMCL may not jump away from where wheels + IMU say the robot went.
If a check fails it asks AMCL to re-match a few times; if it still fails the run stops instead of
driving or capturing with a wrong pose. Nudges are closed-loop on odometry (EKF: wheels + IMU),
which is smooth and doesn't jump, and each one is skipped if the lidar sees anything close in the
direction of motion.
"""
import json
import math
import os
import re
import subprocess
import threading
import time

import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from ame505_bringup.paths import data_dir

# Nudge pattern, relative to the checkpoint pose in the robot's frame: (forward, left, turn), in
# units of the run's nudge_cm / nudge_deg. Burst k uses entry k; the first burst is at the checkpoint.
NUDGES = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 1), (0, -1, -1)]
NUDGE_NAMES = ['at checkpoint', 'turn left', 'turn right', 'slide left', 'slide right', 'back',
               'slide left + turn left', 'slide right + turn right']

# Localization gates
POS_STD_MAX = 0.15          # m
YAW_STD_MAX = math.radians(9)
# Share of lidar points within MATCH_DIST of a wall on the saved map. Tuned on lab2d_11_23_1 with
# simulated lidar (3 cm noise, 15 % of rays hitting unmapped clutter) at all 8 checkpoints: correct
# pose >= 81 %, 10 cm or 3 deg off >= 73 %, 25 cm off ~54 %, one table-aisle over <= 55 %.
SCAN_MATCH_MIN = 0.65
MATCH_DIST = 0.10           # m
JUMP_POS = 0.5              # m: AMCL moved this much more than odometry did -> it jumped
JUMP_YAW = math.radians(20)
# Nudge motion limits
V_MAX, W_MAX = 0.06, 0.25   # m/s, rad/s
TOL_XY, TOL_YAW = 0.015, math.radians(1.5)
CLEAR_AHEAD = 0.25          # m from the robot's edge in the direction of motion
CLEAR_ROTATE = 0.30         # m from the centre (footprint corner radius is ~0.22 m)
HALF_L, HALF_W = 0.165, 0.14


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def compose(base, rel):
    """Pose `rel` (x, y, yaw in base's frame) expressed in base's parent frame."""
    c, s = math.cos(base[2]), math.sin(base[2])
    return base[0] + c * rel[0] - s * rel[1], base[1] + s * rel[0] + c * rel[1], wrap(base[2] + rel[2])


def relative(a, b):
    """Pose b expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])


class BurstRun:
    def __init__(self, mgr, cps, session, scene, hazard, objects, bursts, photos, nudge_cm, nudge_deg):
        self.mgr, self.node = mgr, mgr.node
        self.cps, self.session, self.scene, self.hazard, self.objects = cps, session, scene, hazard, objects
        self.bursts, self.photos = max(1, min(bursts, len(NUDGES))), max(1, min(photos, 20))
        self.nudge_m, self.nudge_rad = nudge_cm / 100.0, math.radians(nudge_deg)
        self.stop_flag = False
        self.out = data_dir(os.path.join('bursts', f'{session}_{time.strftime("%Y%m%d_%H%M%S")}'))
        self.results = [{'id': c['id'], 'state': 'waiting', 'frames': 0, 'bursts': 0, 'note': ''} for c in cps]
        self.status = {'state': 'starting', 'msg': 'starting', 'cp': '', 'index': 0, 'total': len(cps),
                       'burst': 0, 'bursts': self.bursts, 'frames': 0, 'results': self.results,
                       'last_rgb': '', 'dir': self.out, 'loc': {}}
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
        meta = dict(session=self.session, scene=self.scene, hazard=self.hazard, objects=self.objects,
                    bursts=self.bursts, photos=self.photos, nudge_cm=self.nudge_m * 100,
                    nudge_deg=round(math.degrees(self.nudge_rad), 1), map=self.mgr.map)
        tmp = os.path.join(self.out, 'run.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(dict(meta=meta, status=self.status), f, indent=1)
        os.replace(tmp, os.path.join(self.out, 'run.json'))

    def stopped(self):
        return self.stop_flag or self.mgr.mode != 'navigation' or not self.mgr.running()

    # ------------------------------------------------------------------ localization checks
    def scan_match(self):
        """Share of lidar points (at the AMCL pose) that land within MATCH_DIST of a wall on the map."""
        n = self.node
        with n.lock:
            grid, info = getattr(n, 'map_grid', None), dict(n.map_info) if n.map_info else None
        if grid is None or info is None:
            return None
        if self._dist[0] != info['ver']:
            free = (grid < 50).astype(np.uint8)        # unknown (-1) and free count as not-wall
            self._dist = (info['ver'], cv2.distanceTransform(free, cv2.DIST_L2, 5) * info['res'])
        dist = self._dist[1]
        pts = np.asarray(n.scan_in_map(step=2), dtype=np.float32)
        if len(pts) < 30:
            return None
        pose = n.robot_pose('map')
        if pose is not None:
            near = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1]) < 6.0   # far points: fewer, noisier
            pts = pts[near] if near.sum() >= 30 else pts
        u = ((pts[:, 0] - info['ox']) / info['res']).astype(int)
        v = ((pts[:, 1] - info['oy']) / info['res']).astype(int)
        inside = (u >= 0) & (u < info['w']) & (v >= 0) & (v < info['h'])
        if inside.sum() < 30:
            return 0.0
        return float((dist[v[inside], u[inside]] <= MATCH_DIST).mean())

    def refresh_amcl(self, times=3):
        """AMCL only re-matches after 10 cm / 6 deg of motion; ask it to update now."""
        if self.nomotion.service_is_ready():
            for _ in range(times):
                self.nomotion.call_async(Empty.Request())
                time.sleep(0.4)
        time.sleep(0.6)

    def check_localization(self, refresh=True):
        """(ok, reason): confident, lines up with the map, and no jump since the previous check."""
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
        if loc['pos_std_m'] > POS_STD_MAX or loc['yaw_std_deg'] > math.degrees(YAW_STD_MAX):
            return False, (f'AMCL unsure: position +-{loc["pos_std_m"]} m, heading +-{loc["yaw_std_deg"]} deg '
                           f'(need < {POS_STD_MAX} m, < {math.degrees(YAW_STD_MAX):.0f} deg)')
        if match is not None and match < SCAN_MATCH_MIN:
            return False, (f'lidar does not line up with the map ({match:.0%} of points on walls, '
                           f'need {SCAN_MATCH_MIN:.0%})')
        return True, f'localized (+-{loc["pos_std_m"]} m, scan match {match:.0%})' if match is not None else 'localized'

    def ensure_localized(self, tries=4):
        ok, why = self.check_localization()
        for _ in range(tries - 1):
            if ok or self.stopped():
                break
            self.say(self.status['state'], f'checking localization again: {why}')
            time.sleep(1.0)
            ok, why = self.check_localization()
        return ok, why

    def sensors_ok(self):
        _, rates = self.node.snapshot()
        need = {'/scan': 4.0, '/odom': 4.0, '/imu': 8.0, '/depth_cam/rgb/image_raw': 0.2}
        low = [f'{t} {rates.get(t, 0):.1f} Hz' for t, hz in need.items() if rates.get(t, 0) < hz]
        if low:
            return False, ('sensors too slow or silent: ' + ', '.join(low) +
                           '. If /imu is 0 Hz, Hiwonder\'s ROS 1 board driver may hold the serial port.')
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
        t0, next_check, ok = time.time(), 0.0, True
        while time.time() - t0 < 10:
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
            vx = max(-V_MAX, min(V_MAX, 1.2 * ex))
            vy = max(-V_MAX, min(V_MAX, 1.2 * ey))
            wz = max(-W_MAX, min(W_MAX, 1.5 * eyaw))
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
            ok = False
        self.mgr.estop_twist()
        return ok

    # ------------------------------------------------------------------ capture
    def capture(self, cp, k, label):
        cmd = ['ros2', 'run', 'ame505_bringup', 'capture_dataset', '--session', self.session,
               '--checkpoint', cp['id'], '--scene', self.scene, '--hazard', self.hazard,
               '--n', str(self.photos), '--note', f'burst {k + 1}/{self.bursts} {label}'.replace('+', 'and')]
        if self.objects:
            cmd += ['--objects', self.objects]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        lines = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode != 0:
            return 0, lines[-1] if lines else f'exit {r.returncode}'
        saved = sum(1 for ln in lines if re.match(r'^\s*\d+/\d+\s', ln))   # '  3/6  <file> ...'
        folder = data_dir(os.path.join('dataset', self.session, cp['id']))
        pngs = sorted((p for p in os.listdir(folder) if p.endswith('_rgb.png')),
                      key=lambda p: os.path.getmtime(os.path.join(folder, p)))
        if pngs:
            self.status['last_rgb'] = os.path.join(self.session, cp['id'], pngs[-1])
        return saved, ''

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
        self.say('checking', 'checking localization before moving')
        ok, why = self.ensure_localized()
        if not ok:
            return self.say('failed', f'not starting: {why}. Use "Set pose" on the map (or drive a '
                                      'little) until localization shows good, then start again.')
        for i, cp in enumerate(self.cps):
            res = self.results[i]
            self.status.update(cp=cp['id'], index=i + 1, burst=0)
            if self.stopped():
                return self.say('stopped', 'stopped')
            res['state'] = 'driving'
            self.say('driving', f'{cp["id"]} ({i + 1}/{len(self.cps)}): driving there')
            arrived, why = self.drive_to(cp)
            if not arrived and 'jumped' not in why and not self.stopped():
                self.say('driving', f'{cp["id"]}: {why}, trying once more')
                arrived, why = self.drive_to(cp)
            if not arrived:
                res.update(state='not reached', note=why)
                self.save()
                if 'jumped' in why:
                    return self.say('failed', f'stopped at {cp["id"]}: {why}. Set the pose again, then restart '
                                              'from this checkpoint.')
                if self.stopped():
                    return self.say('stopped', 'stopped')
                continue
            self.say('settling', f'{cp["id"]}: arrived, holding still 4 s (camera is 1-7 s late)')
            time.sleep(4)
            ref = self.node._lookup('odom', 'base_footprint')
            for k in range(self.bursts):
                if self.stopped():
                    return self.say('stopped', 'stopped')
                self.status['burst'] = k + 1
                fx, fy, fyaw = NUDGES[k]
                label = NUDGE_NAMES[k]
                if k > 0:
                    self.say('nudging', f'{cp["id"]} burst {k + 1}/{self.bursts}: {label}')
                    rel = (fx * self.nudge_m, fy * self.nudge_m, fyaw * self.nudge_rad)
                    if not self.move_to(ref, rel, label):
                        continue
                    time.sleep(3)                     # let a fresh camera frame arrive
                ok, why = self.ensure_localized(tries=3)
                if not ok:
                    res['note'] = f'burst {k + 1} skipped: {why}'
                    self.say('capturing', f'{cp["id"]} burst {k + 1}: skipped ({why})')
                    if 'jumped' in why:
                        break
                    continue
                self.say('capturing', f'{cp["id"]} burst {k + 1}/{self.bursts} ({label}): {self.photos} photos')
                saved, err = self.capture(cp, k, label)
                res['frames'] += saved
                res['bursts'] += 1 if saved else 0
                self.status['frames'] += saved
                if err:
                    res['note'] = f'capture: {err}'
                self.save()
            if self.bursts > 1 and ref is not None and not self.stopped():
                self.move_to(ref, (0.0, 0.0, 0.0), 'back to the checkpoint')
            res['state'] = 'done' if res['frames'] else 'no photos'
            self.save()
        done = sum(1 for r in self.results if r['state'] == 'done')
        self.say('done', f'done: {self.status["frames"]} photos at {done}/{len(self.cps)} checkpoints, '
                         f'saved in {data_dir("dataset")}/{self.session}')


def start(mgr, cps, **kw):
    run = BurstRun(mgr, cps, **kw)
    threading.Thread(target=run.run, daemon=True).start()
    return run
