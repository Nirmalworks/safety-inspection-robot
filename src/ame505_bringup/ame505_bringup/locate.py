"""Global localization by lidar scan matching ("Auto-locate" on the dashboard).

AMCL only tracks: it starts from the map's home pose (or a "Set pose" click) and refines it. If the
robot is somewhere else, AMCL stays unsure or locks onto the wrong spot. This searches the whole
saved map instead: for every free cell (every 15 cm) and heading (every 5 deg) it projects the
current lidar scan and scores how many points land on walls, then refines the best candidates to
3 cm / 1 deg. The winner is handed to AMCL as its new start pose.

The lab has four identical tables, so two places can score almost the same. Then the result is
reported as ambiguous and, if AMCL already has a guess, the candidate closest to it is used.
"""
import math

import cv2
import numpy as np

MATCH_DIST = 0.10       # m: a lidar point "hits a wall" if the map has a wall this close
COARSE_STEP = 0.15      # m between candidate positions
COARSE_YAWS = 72        # 5 deg
MIN_CLEARANCE = 0.15    # m: the robot centre can't be closer than this to a mapped wall
MAX_POINTS = 90
MAX_RANGE = 6.0         # m: far points are few and noisy
AMBIGUOUS = 0.06        # best and runner-up (somewhere else) closer than this -> ambiguous


def distance_field(grid, res):
    """Metres from every cell to the nearest wall (occupied >= 50); unknown counts as not-wall."""
    free = (grid < 50).astype(np.uint8)
    return cv2.distanceTransform(free, cv2.DIST_L2, 5) * res


def _scores(dist, info, cx, cy, yaws, px, py, sigma):
    """Score for each (candidate i, yaw j): mean over points of exp(-d^2 / 2 sigma^2)."""
    res, h, w = info['res'], info['h'], info['w']
    out = np.empty((len(cx), len(yaws)), np.float32)
    for j, yaw in enumerate(yaws):
        c, s = math.cos(yaw), math.sin(yaw)
        rx, ry = c * px - s * py, s * px + c * py                     # points rotated (P,)
        u = ((cx[:, None] + rx[None, :] - info['ox']) / res).astype(np.int32)
        v = ((cy[:, None] + ry[None, :] - info['oy']) / res).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.where(inside, dist[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)], 1.0)
        out[:, j] = np.exp(-(d * d) / (2 * sigma * sigma)).mean(1)
    return out


def hit_share(dist, info, pose, px, py):
    """Share of points within MATCH_DIST of a wall at `pose` (same measure burst mode gates on)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    u = ((pose[0] + c * px - s * py - info['ox']) / info['res']).astype(np.int32)
    v = ((pose[1] + s * px + c * py - info['oy']) / info['res']).astype(np.int32)
    inside = (u >= 0) & (u < info['w']) & (v >= 0) & (v < info['h'])
    if inside.sum() < 20:
        return 0.0
    hit = np.zeros(len(px), bool)
    hit[inside] = dist[v[inside], u[inside]] <= MATCH_DIST
    return float(hit.mean())


def locate(grid, info, pts, prior=None):
    """grid: OccupancyGrid data (h, w); info: res/w/h/ox/oy; pts: lidar points (N, 2) in the robot
    frame (base_footprint). prior: current AMCL (x, y, yaw) or None.
    Returns dict(pose, score, runner_up, ambiguous, candidates) or None if the scan is too thin."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 30:
        return None
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(r > 0.15) & (r < MAX_RANGE)]
    if len(pts) < 30:
        return None
    pts = pts[np.linspace(0, len(pts) - 1, min(MAX_POINTS, len(pts))).astype(int)]
    px, py = pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
    res = info['res']
    dist = distance_field(grid, res)

    # Coarse candidates: known-free cells away from walls, every COARSE_STEP
    step = max(1, int(round(COARSE_STEP / res)))
    ok = (grid == 0) & (dist >= MIN_CLEARANCE)
    sub = np.zeros_like(ok)
    sub[::step, ::step] = True
    vv, uu = np.nonzero(ok & sub)
    if len(uu) == 0:
        return None
    cx = (info['ox'] + (uu + 0.5) * res).astype(np.float32)
    cy = (info['oy'] + (vv + 0.5) * res).astype(np.float32)
    yaws = np.linspace(-math.pi, math.pi, COARSE_YAWS, endpoint=False)
    sc = _scores(dist, info, cx, cy, yaws, px, py, sigma=0.15)

    # Best few distinct peaks (>= 0.6 m apart, or same place but >= 30 deg apart)
    order = np.argsort(sc, axis=None)[::-1]
    peaks = []
    for flat in order[:4000]:
        i, j = divmod(int(flat), COARSE_YAWS)
        cand = (float(cx[i]), float(cy[i]), float(yaws[j]))
        if all(math.hypot(cand[0] - p[0], cand[1] - p[1]) > 0.6 or
               abs(math.atan2(math.sin(cand[2] - p[2]), math.cos(cand[2] - p[2]))) > math.radians(30)
               for p in peaks):
            peaks.append(cand)
        if len(peaks) >= 8:
            break

    # Refine each peak: +-15 cm in 3 cm, +-6 deg in 1 deg, then +-3 cm in 1 cm, +-1 deg in 0.25 deg
    refined = []
    for p in peaks:
        best = p
        for span, n_xy, yspan, n_yaw, sigma in ((0.15, 11, 6, 13, 0.08), (0.03, 7, 1, 9, 0.05)):
            g = np.linspace(-span, span, n_xy, dtype=np.float32)
            gx, gy = np.meshgrid(g, g)
            ccx, ccy = (best[0] + gx.ravel()).astype(np.float32), (best[1] + gy.ravel()).astype(np.float32)
            yy = best[2] + np.radians(np.linspace(-yspan, yspan, n_yaw))
            s = _scores(dist, info, ccx, ccy, yy, px, py, sigma)
            i, j = np.unravel_index(int(np.argmax(s)), s.shape)
            best = (float(ccx[i]), float(ccy[i]), float(math.atan2(math.sin(yy[j]), math.cos(yy[j]))))
        refined.append((hit_share(dist, info, best, px, py), best))
    refined.sort(key=lambda t: -t[0])
    best_score, best = refined[0]
    others = [(s, p) for s, p in refined[1:]
              if math.hypot(p[0] - best[0], p[1] - best[1]) > 0.5 or
              abs(math.atan2(math.sin(p[2] - best[2]), math.cos(p[2] - best[2]))) > math.radians(25)]
    runner = others[0] if others else (0.0, None)
    ambiguous = runner[1] is not None and best_score - runner[0] < AMBIGUOUS
    if ambiguous and prior is not None:
        # Two places look alike: take the one (of the close contenders) nearest AMCL's current guess
        close = [(s, p) for s, p in refined if best_score - s < AMBIGUOUS]
        best_score, best = min(close, key=lambda t: math.hypot(t[1][0] - prior[0], t[1][1] - prior[1]))
    return {'pose': best, 'score': round(best_score, 3), 'ambiguous': bool(ambiguous),
            'runner_up': None if runner[1] is None else {'pose': [round(v, 3) for v in runner[1]],
                                                         'score': round(runner[0], 3)},
            'candidates': [{'pose': [round(v, 3) for v in p], 'score': round(s, 3)} for s, p in refined[:5]]}
