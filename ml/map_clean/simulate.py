"""Mapping simulator: drive a robot with a front-180-degree lidar through a World and build an occupancy map.

The clean target is built from noise-free scans at the true poses, with every doorway closed and no
movable clutter. The dirty input is built from the same drive with: pose drift (walls smear and double),
range noise and dropouts, movable clutter in the room, some doorways open and some glass transparent
(leak fans), and real defect patches harvested from the lab maps pasted on top.
All distances here are in cells (0.05 m). x = column, y = row, yaw in image coordinates.
"""
import math

import cv2
import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

from .common import FREE, OCC, UNK

BEAMS = np.deg2rad(np.arange(-90, 90.01, 0.75))      # front half only, like the laser filter on the robot
RMIN = 4                                             # 0.2 m


def set_fov(deg, step=0.75):
    """Change the simulated lidar's field of view (the robot's bags show about 205 degrees, not 180)."""
    global BEAMS
    BEAMS = np.deg2rad(np.arange(-deg / 2, deg / 2 + 0.01, step))


def tighten(occ):
    """Fill the corner gap between two diagonally touching wall cells, so that no ray can slip between them."""
    o = occ.copy()
    a = occ[:-1, :-1] & occ[1:, 1:] & ~occ[:-1, 1:] & ~occ[1:, :-1]
    b = occ[:-1, 1:] & occ[1:, :-1] & ~occ[:-1, :-1] & ~occ[1:, 1:]
    o[:-1, 1:] |= a
    o[:-1, :-1] |= b
    return o


def plan_poses(world, occ, rng, spin_p):
    """A drive that covers the region: waypoints on a jittered lattice, visited nearest-first, with spins. -> (P, 3) x, y, yaw or None."""
    clear = world.region & (ndimage.distance_transform_edt(~occ) >= 6)          # 0.3 m from anything
    lab, n = ndimage.label(clear)
    if n == 0:
        return None
    comp = lab == (1 + int(np.argmax(ndimage.sum(clear, lab, range(1, n + 1)))))
    if comp.sum() < 400:
        return None
    ys, xs = np.nonzero(comp)
    s = rng.uniform(30, 55)
    gx, gy = np.meshgrid(np.arange(xs.min() + rng.uniform(0, s), xs.max(), s), np.arange(ys.min() + rng.uniform(0, s), ys.max(), s))
    pts = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.3 * s, 0.3 * s, (gx.size, 2))
    pts = np.round(pts).astype(int)
    ok = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < comp.shape[1]) & (pts[:, 1] < comp.shape[0])
    pts = pts[ok]
    pts = pts[comp[pts[:, 1], pts[:, 0]]]
    if len(pts) < 3:
        i = rng.choice(len(xs), min(4, len(xs)), replace=False)
        pts = np.vstack([pts.reshape(-1, 2), np.c_[xs[i], ys[i]]])
    order, left = [int(rng.integers(len(pts)))], set(range(len(pts)))
    left.discard(order[0])
    while left:
        cur = pts[order[-1]]
        nxt = min(left, key=lambda j: (pts[j, 0] - cur[0]) ** 2 + (pts[j, 1] - cur[1]) ** 2)
        order.append(nxt)
        left.discard(nxt)
    if rng.random() < 0.7:
        order.append(order[0])                                                   # come back to the start (a closed loop)
    cost = np.where(comp, 1.0, -1.0)
    step = rng.uniform(5, 9)
    poses = []
    for a, b in zip(order[:-1], order[1:]):
        p, q = pts[a], pts[b]
        d = max(1.0, float(np.hypot(*(q - p))))
        line = np.round(p + (q - p) * np.linspace(0, 1, int(d) + 1)[:, None]).astype(int)
        if comp[line[:, 1], line[:, 0]].all():
            path = line.astype(float)
        else:
            try:
                idx, _ = route_through_array(cost, (p[1], p[0]), (q[1], q[0]), fully_connected=True, geometric=True)
            except Exception:  # noqa: BLE001  (unreachable pair: skip the leg)
                continue
            path = np.array(idx, float)[:, ::-1]
        path = path[::max(1, int(step))]
        if len(path) < 2:
            continue
        tang = np.gradient(ndimage.uniform_filter1d(path, 3, axis=0, mode='nearest'), axis=0)
        yaw = np.arctan2(tang[:, 1], tang[:, 0])
        poses += [(x, y, th) for (x, y), th in zip(path, yaw)]
        if rng.random() < spin_p:                                                # look around at the waypoint
            k = int(rng.integers(4, 9))
            poses += [(q[0], q[1], yaw[-1] + 2 * math.pi * j / k) for j in range(1, k + 1)]
    if len(poses) < 8:
        return None
    P = np.array(poses, float)
    if len(P) > 600:
        P = P[np.linspace(0, len(P) - 1, 600).astype(int)]
    return P


def cast(occ, poses, rmax, step=0.5):
    """True range (cells) of every beam of every pose; inf where nothing is hit within rmax."""
    H, W = occ.shape
    dist = np.arange(RMIN, rmax, step)
    out = np.empty((len(poses), len(BEAMS)), np.float32)
    for i in range(0, len(poses), 32):
        p = poses[i:i + 32]
        ang = p[:, 2:3] + BEAMS[None, :]
        xs = np.rint(p[:, 0, None, None] + np.cos(ang)[..., None] * dist).astype(np.int32)
        ys = np.rint(p[:, 1, None, None] + np.sin(ang)[..., None] * dist).astype(np.int32)
        inside = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        hit = np.zeros(xs.shape, bool)
        hit[inside] = occ[ys[inside], xs[inside]]
        first = hit.argmax(-1)
        out[i:i + 32] = np.where(hit.any(-1), dist[first], np.inf)
    return out


def integrate(shape, poses, ranges, rmax, occ_ratio=0.1, clut=None):
    """Occupancy map from scans, karto / slam_toolbox style: a cell is occupied when more than `occ_ratio` of the
    beams that reached it ended there. Beams with no return clear the floor out to rmax and mark nothing."""
    H, W = shape
    dist = np.arange(0, rmax, 1.0)
    passes, hits, hits_c = np.zeros(H * W, np.int32), np.zeros(H * W, np.int32), np.zeros(H * W, np.int32)
    for i in range(0, len(poses), 64):
        p, r = poses[i:i + 64], ranges[i:i + 64]
        valid = ~np.isnan(r)
        ang = p[:, 2:3] + BEAMS[None, :]
        cs, sn = np.cos(ang), np.sin(ang)
        rr = np.where(np.isfinite(r), r, rmax)
        xs = np.rint(p[:, 0, None, None] + cs[..., None] * dist).astype(np.int32)
        ys = np.rint(p[:, 1, None, None] + sn[..., None] * dist).astype(np.int32)
        m = (dist[None, None, :] < rr[..., None] - 0.7) & valid[..., None] & (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        passes += np.bincount(ys[m] * W + xs[m], minlength=H * W).astype(np.int32)
        hb = valid & np.isfinite(r)
        hx = np.rint(p[:, 0, None] + cs * rr)[hb].astype(np.int32)
        hy = np.rint(p[:, 1, None] + sn * rr)[hb].astype(np.int32)
        k = (hx >= 0) & (hy >= 0) & (hx < W) & (hy < H)
        hits += np.bincount(hy[k] * W + hx[k], minlength=H * W).astype(np.int32)
        if clut is not None:
            kc = k & clut[i:i + 64][hb]
            hits_c += np.bincount(hy[kc] * W + hx[kc], minlength=H * W).astype(np.int32)
    seen = passes + hits
    cls = np.full(H * W, UNK, np.uint8)
    cls[seen > 0] = FREE
    cls[(hits > 0) & (hits > occ_ratio * seen)] = OCC
    if clut is not None:
        return cls.reshape(H, W), hits.reshape(H, W), hits_c.reshape(H, W)
    return cls.reshape(H, W)


def drift(n, rng, level):
    """Pose error along the drive: a slow random walk partly pulled back at the end (loop closure), plus at times one
    stretch of the drive that sits a few cm / a degree or two off (the doubled walls seen in the lab maps)."""
    s_xy, s_yaw = level * rng.uniform(0, 1.6), level * math.radians(rng.uniform(0, 1.0))
    e = np.cumsum(rng.normal(0, 1, (n, 3)), 0) / math.sqrt(n)
    e -= np.linspace(0, 1, n)[:, None] * e[-1] * rng.uniform(0.5, 1.0)
    e *= [s_xy, s_xy, s_yaw]
    if rng.random() < 0.35 * level:
        a = int(rng.integers(0, n - 2))
        b = min(n, a + int(n * rng.uniform(0.1, 0.3)) + 1)
        e[a:b] += [rng.uniform(-3, 3), rng.uniform(-3, 3), math.radians(rng.uniform(-2, 2))]
    return e


def clutter(world, rng, density):
    """Movable things on the floor: specks, chair bases (stars), boxes, pairs of legs. `density` = blobs per m2."""
    H, W = world.region.shape
    floor = cv2.erode(world.region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = np.zeros((H, W), np.uint8)
    ys, xs = np.nonzero(floor)
    if len(xs) == 0:
        return out > 0
    near = cv2.dilate((world.furn | world.wall).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0   # things gather around benches and along walls
    w = np.where(near[ys, xs], 4.0, 1.0)
    n = int(rng.poisson(density * floor.sum() * 0.0025))
    for i in rng.choice(len(xs), min(n, len(xs)), replace=False, p=w / w.sum()):
        x, y, u = int(xs[i]), int(ys[i]), rng.random()
        if u < 0.65:                                             # speck: a leg, a cable, a bag
            cv2.circle(out, (x, y), int(rng.integers(0, 2)), 1, -1)
        elif u < 0.90:                                           # chair: five-spoke base
            a0, L = rng.uniform(0, 2 * math.pi), rng.uniform(3, 6.5)
            for k in range(5):
                a = a0 + 2 * math.pi * k / 5
                cv2.line(out, (x, y), (int(round(x + L * math.cos(a))), int(round(y + L * math.sin(a)))), 1, 1)
        elif u < 0.95:                                           # box or cart
            bw, bh, ang = rng.uniform(5, 16), rng.uniform(5, 16), rng.uniform(0, 180)
            pts = cv2.boxPoints(((x, y), (bw, bh), ang)).astype(np.int32)
            cv2.fillPoly(out, [pts], 1)
        else:                                                    # a person standing: two legs
            a = rng.uniform(0, 2 * math.pi)
            for sg in (-1, 1):
                cv2.circle(out, (int(round(x + sg * 3 * math.cos(a))), int(round(y + sg * 3 * math.sin(a)))), 1, 1, -1)
    return (out > 0) & world.region


def paste_noise(dirty, clean, lib, rng, n):
    """Real obstacle patches harvested from the lab maps, pasted where the clean map has floor."""
    if lib is None or len(lib) == 0:
        return dirty
    H, W = dirty.shape
    ys, xs = np.nonzero(clean == FREE)
    if len(xs) == 0:
        return dirty
    for _ in range(n):
        p = lib[rng.integers(len(lib))]
        p = np.rot90(p, rng.integers(4))
        if rng.random() < 0.5:
            p = p[:, ::-1]
        i = rng.integers(len(xs))
        y0, x0 = ys[i] - p.shape[0] // 2, xs[i] - p.shape[1] // 2
        ya, xa, yb, xb = max(0, y0), max(0, x0), min(H, y0 + p.shape[0]), min(W, x0 + p.shape[1])
        sub = p[ya - y0:yb - y0, xa - x0:xb - x0] & (clean[ya:yb, xa:xb] == FREE)
        dirty[ya:yb, xa:xb][sub] = OCC
    return dirty


def simulate_pair(world, rng, noise_lib=None, level=None, use_real_noise=True):
    """-> (dirty, clean) class maps on the world's grid, or None if no drive could be planned.
    `level` scales every synthetic defect at once (0 = none, 1 = as measured on the lab maps on average, up to 2)."""
    level = rng.uniform(0.2, 2.0) if level is None else level
    shape = world.wall.shape
    occ_clean = world.wall | world.glass | world.furn
    for d in world.doors:
        occ_clean = occ_clean | d
    rmax = int(rng.uniform(160, 240))                           # 8 - 12 m
    base = plan_poses(world, occ_clean, rng, spin_p=1.0)        # the clean target uses a full-coverage drive: a spin at every waypoint
    if base is None:
        return None
    clean = integrate(shape, base, cast(occ_clean, base, rmax), rmax)

    # the dirty drive: the same route with fewer spins, so some walls are seen from fewer places
    keep = np.ones(len(base), bool)
    same = np.r_[False, (np.abs(np.diff(base[:, :2], axis=0)).sum(1) < 1e-6)]          # spin poses repeat the position
    keep[same & (rng.random(len(base)) < rng.uniform(0.0, 0.6))] = False
    poses = base[keep]
    occ_dirty = world.wall | world.furn
    if rng.random() > 0.3 * min(level, 1.0):
        occ_dirty = occ_dirty | world.glass                      # glass mostly reads as a wall
    for d in world.doors:
        if rng.random() > 0.5:
            occ_dirty = occ_dirty | d                            # this doorway is closed
    occ_dirty = occ_dirty | clutter(world, rng, density=level * rng.uniform(0.4, 1.2))   # lab maps: about 1.4 blobs per m2
    r = cast(occ_dirty, poses, rmax)
    sig = level * rng.uniform(0.15, 0.6)
    r = r + rng.normal(0, 1, r.shape) * (sig + 0.004 * np.where(np.isfinite(r), r, 0))
    r[rng.random(r.shape) < 0.02 * level] = np.nan              # dropouts
    est = poses + drift(len(poses), rng, level)
    dirty = integrate(shape, est, r, rmax, occ_ratio=rng.uniform(0.08, 0.2))
    # gaps in walls: the lab maps miss about a fifth of the consensus wall (dark or shiny surfaces, sparse views)
    wy, wx = np.nonzero(dirty == OCC)
    for i in rng.choice(len(wx), min(len(wx), int(rng.poisson(level * rng.uniform(0, 0.03) * len(wx)))), replace=False):
        k = int(rng.integers(1, 5))
        sub = dirty[max(0, wy[i] - k):wy[i] + k + 1, max(0, wx[i] - k):wx[i] + k + 1]
        sub[sub == OCC] = FREE
    if use_real_noise:
        dirty = paste_noise(dirty, clean, noise_lib, rng, int(rng.integers(0, 1 + int(4 * level))))
    return dirty, clean


def simulate_labeled(world, rng, level=None):
    """-> (dirty class map, label map) or None. Label of every occupied cell of the dirty map: 1 = keep (the rays that
    drew it hit structure: a wall, glass, a closed door, fixed furniture), 2 = erase (they hit movable clutter); 0 elsewhere.
    The label comes from which thing each simulated ray hit, so it stays right however the poses drift. No clean map is built."""
    level = rng.uniform(0.2, 2.0) if level is None else level
    shape = world.wall.shape
    closed = world.wall | world.glass | world.furn
    for d in world.doors:
        closed = closed | d
    rmax = int(rng.uniform(160, 240))                           # 8 - 12 m
    base = plan_poses(world, closed, rng, spin_p=1.0)
    if base is None:
        return None
    keep = np.ones(len(base), bool)
    same = np.r_[False, (np.abs(np.diff(base[:, :2], axis=0)).sum(1) < 1e-6)]          # spin poses repeat the position
    keep[same & (rng.random(len(base)) < rng.uniform(0.0, 0.6))] = False
    poses = base[keep]
    struct = world.wall | world.furn
    if rng.random() > 0.3 * min(level, 1.0):
        struct = struct | world.glass                            # glass mostly reads as a wall
    for d in world.doors:
        if rng.random() > 0.5:
            struct = struct | d                                  # this doorway is closed
    struct = tighten(struct)
    everything = struct | clutter(world, rng, density=level * rng.uniform(0.4, 1.2))
    r_s, r = cast(struct, poses, rmax), cast(everything, poses, rmax)
    from_clutter = r < r_s - 0.75                                # the ray stopped on clutter before reaching structure
    sig = level * rng.uniform(0.15, 0.6)
    r = r + rng.normal(0, 1, r.shape) * (sig + 0.004 * np.where(np.isfinite(r), r, 0))
    r[rng.random(r.shape) < 0.02 * level] = np.nan              # dropouts
    est = poses + drift(len(poses), rng, level)
    dirty, hits, hits_c = integrate(shape, est, r, rmax, occ_ratio=rng.uniform(0.08, 0.2), clut=from_clutter)
    wy, wx = np.nonzero(dirty == OCC)                            # gaps in walls, as in the lab maps
    for i in rng.choice(len(wx), min(len(wx), int(rng.poisson(level * rng.uniform(0, 0.03) * len(wx)))), replace=False):
        k = int(rng.integers(1, 5))
        sub = dirty[max(0, wy[i] - k):wy[i] + k + 1, max(0, wx[i] - k):wx[i] + k + 1]
        sub[sub == OCC] = FREE
    lab = np.zeros(shape, np.uint8)
    occ = dirty == OCC
    lab[occ] = 1
    lab[occ & (hits_c * 2 > hits)] = 2
    return dirty, lab
