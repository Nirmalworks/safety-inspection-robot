"""Clean floor plans as "worlds" for the mapping simulator: procedural lab rooms, KTH campus plans, HouseExpo.

A World holds boolean grids at 0.05 m per cell (row 0 = top):
  wall    static walls                           region  the floor the robot drives on and maps
  doors   list of masks, one per doorway on the edge of the region (closed in the clean target, maybe open in the dirty map)
  glass   windows (a wall, unless the dirty map lets the lidar through)
  furn    static furniture (benches, cabinets, pillars): solid in both the clean and the dirty map
"""
import glob
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

RES = 0.05
MARGIN = 120                      # cells (6 m) around the mapped region, so leaks through doors have somewhere to go
K3, K5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)


@dataclass
class World:
    wall: np.ndarray
    region: np.ndarray
    doors: list = field(default_factory=list)
    glass: np.ndarray = None
    furn: np.ndarray = None
    kind: str = ''

    def __post_init__(self):
        z = np.zeros(self.wall.shape, bool)
        self.glass = z.copy() if self.glass is None else self.glass
        self.furn = z.copy() if self.furn is None else self.furn


def ring(region, k=K5):
    """The wall around a region: a band 2 cells thick just outside it."""
    return (cv2.dilate(region.astype(np.uint8), k) > 0) & ~region


def _rect(mask, x0, y0, x1, y1, v=True):
    mask[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = v


def add_furniture(region, wall, rng, n):
    """A few solid boxes (desks, cabinets) on the floor of an arbitrary room."""
    furn = np.zeros(region.shape, bool)
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return furn
    inner = cv2.erode(region.astype(np.uint8), K3) > 0
    for _ in range(n):
        i = rng.integers(len(xs))
        w, h = rng.integers(8, 32), rng.integers(8, 32)           # 0.4 - 1.6 m
        x0, y0 = xs[i] - w // 2, ys[i] - h // 2
        box = np.zeros(region.shape, bool)
        _rect(box, x0, y0, x0 + w, y0 + h)
        if box.sum() == w * h and inner[box].all() and not (cv2.dilate(box.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0)[furn].any():
            furn |= box
    return furn


# ------------------------------------------------------------------ procedural lab rooms
def procedural_lab(rng):
    """A lab-like room: rectangle 8-30 m by 5-12 m, wall notches, 1-4 doorways of about 0.9 m, rows of benches, cabinets, pillars."""
    wc, hc = int(rng.uniform(8, 30) / RES), int(rng.uniform(5, 12) / RES)
    H, W = hc + 2 * MARGIN, wc + 2 * MARGIN
    region = np.zeros((H, W), bool)
    x0, y0, x1, y1 = MARGIN, MARGIN, MARGIN + wc, MARGIN + hc
    _rect(region, x0, y0, x1, y1)
    for _ in range(rng.integers(0, 5)):                           # notches: small jogs pushed in or out of a wall
        span, depth, out = int(rng.uniform(1, 4) / RES), int(rng.uniform(0.3, 1.2) / RES), rng.random() < 0.5
        side = rng.integers(4)
        if side < 2:
            a = rng.integers(x0, max(x0 + 1, x1 - span))
            yy = (y0 - depth, y0) if (side == 0 and out) else (y0, y0 + depth) if side == 0 else (y1, y1 + depth) if out else (y1 - depth, y1)
            _rect(region, a, yy[0], a + span, yy[1], out)
        else:
            a = rng.integers(y0, max(y0 + 1, y1 - span))
            xx = (x0 - depth, x0) if (side == 2 and out) else (x0, x0 + depth) if side == 2 else (x1, x1 + depth) if out else (x1 - depth, x1)
            _rect(region, xx[0], a, xx[1], a + span, out)
    wall = ring(region)
    doors = []
    for _ in range(rng.integers(1, 5)):
        dw = int(rng.uniform(0.8, 1.05) / RES)
        side = rng.integers(4)
        box = np.zeros((H, W), bool)
        if side < 2:
            a = rng.integers(x0 + 10, max(x0 + 11, x1 - dw - 10))
            yc = y0 if side == 0 else y1
            _rect(box, a, yc - 30, a + dw, yc + 30)
        else:
            a = rng.integers(y0 + 10, max(y0 + 11, y1 - dw - 10))
            xc = x0 if side == 2 else x1
            _rect(box, xc - 30, a, xc + 30, a + dw)
        d = wall & box
        if d.sum() >= 2 * dw * 0.8 and not any((d & o).any() for o in doors):
            doors.append(d)
    for d in doors:
        wall &= ~d
    if rng.random() < 0.7:                                        # a corridor outside one long wall: its far wall catches the leaks
        c = int(rng.uniform(1.6, 3.2) / RES)
        yy = y0 - c - 2 if rng.random() < 0.5 else y1 + c
        wall[max(0, yy):yy + 2, :] = True
    for _ in range(rng.integers(0, 4)):                           # stray walls of neighbouring rooms
        if rng.random() < 0.5:
            xx = rng.choice([rng.integers(5, x0 - 20), rng.integers(x1 + 20, W - 5)])
            wall[rng.integers(0, H // 2):rng.integers(H // 2, H), xx:xx + 2] = True
        else:
            yy = rng.choice([rng.integers(5, y0 - 20), rng.integers(y1 + 20, H - 5)])
            wall[yy:yy + 2, rng.integers(0, W // 2):rng.integers(W // 2, W)] = True
    wall &= ~region
    furn = np.zeros((H, W), bool)
    n = rng.integers(0, 7)                                        # rows of benches across the long axis
    bw, bl = int(rng.uniform(0.7, 1.6) / RES), int(rng.uniform(0.35, 0.7) * hc)
    if n and wc > n * (bw + 26) + 40:
        gap = (wc - n * bw) / (n + 1)
        cy = y0 + hc // 2 + int(rng.uniform(-0.12, 0.12) * hc)
        for i in range(n):
            bx = int(x0 + gap * (i + 1) + bw * i + rng.uniform(-4, 4))
            _rect(furn, bx, cy - bl // 2, bx + bw, cy + bl // 2)
    for _ in range(rng.integers(0, 6)):                           # cabinets against a wall
        d, l = int(rng.uniform(0.4, 0.65) / RES), int(rng.uniform(0.8, 2.2) / RES)
        side = rng.integers(4)
        if side < 2:
            a = rng.integers(x0, max(x0 + 1, x1 - l))
            _rect(furn, a, y0 if side == 0 else y1 - d, a + l, y0 + d if side == 0 else y1)
        else:
            a = rng.integers(y0, max(y0 + 1, y1 - l))
            _rect(furn, x0 if side == 2 else x1 - d, a, x0 + d if side == 2 else x1, a + l)
    for _ in range(rng.integers(0, 3)):                           # pillars
        s = int(rng.uniform(0.3, 0.6) / RES)
        px, py = rng.integers(x0 + 20, x1 - 20 - s), rng.integers(y0 + 20, y1 - 20 - s)
        _rect(furn, px, py, px + s, py + s)
    near_door = np.zeros((H, W), bool)
    for d in doors:
        near_door |= cv2.dilate(d.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    furn &= region & ~near_door                                  # keep doorways clear
    return World(wall=wall, region=region & ~furn, doors=doors, furn=furn, kind='procedural')


# ------------------------------------------------------------------ KTH campus plans
def kth_files(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, '**', '*.xml'), recursive=True)):
        b = os.path.basename(p)
        if b.startswith('._') or 'conflicted' in b or '_tmp_' in b:
            continue
        out.append(p)
    return out


def kth_load(path):
    """-> {space name: {'poly': N x 2 metres, 'segs': [(x1, y1, x2, y2, type, target)]}} or None if unusable."""
    try:
        root = ET.parse(path).getroot()
        sc = root.find('Scale')
        k = float(sc.get('RealDistance')) / float(sc.get('PixelDistance'))
    except Exception:  # noqa: BLE001  (one file in the set does not parse)
        return None
    if k <= 0:
        return None
    spaces = {}
    for sp in root.findall('space'):
        segs = [(float(s.get('x1')) * k, float(s.get('y1')) * k, float(s.get('x2')) * k, float(s.get('y2')) * k, s.get('type'), s.get('target') or '')
                for s in sp.iter('linesegment')]
        if len(segs) >= 3:
            spaces[sp.get('name')] = {'poly': np.array([[s[0], s[1]] for s in segs]), 'segs': segs}
    return spaces or None


def _area(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def kth_rooms(spaces):
    """Names of the spaces big enough to start a world from."""
    return [n for n, s in spaces.items() if 6 <= _area(s['poly']) <= 600]


def kth_cluster(spaces, rng, start):
    """A few connected rooms grown from `start` through its doors."""
    target = rng.uniform(25, 220)
    inc, queue, area = [start], [start], _area(spaces[start]['poly'])
    while queue and area < target and len(inc) < 10:
        cur = queue.pop(0)
        nbrs = [s[5] for s in spaces[cur]['segs'] if s[4] == 'Portal' and s[5] in spaces and s[5] not in inc]
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb not in inc and 2 <= _area(spaces[nb]['poly']) <= 600:      # tiny 'spaces' are door sills, not rooms
                inc.append(nb)
                queue.append(nb)
                area += _area(spaces[nb]['poly'])
                if area >= target:
                    break
    return inc


def kth_world(spaces, rng, inc=None):
    """A few connected rooms of one KTH floor: doors between them open, doors to the rest of the floor are the doorways."""
    if inc is None:
        names = kth_rooms(spaces)
        if not names:
            return None
        inc = kth_cluster(spaces, rng, names[rng.integers(len(names))])
    pts = np.vstack([spaces[n]['poly'] for n in inc])
    lo, hi = pts.min(0), pts.max(0)
    if max(hi - lo) > 40:
        return None
    off = lo - MARGIN * RES
    W, H = int((hi[0] - lo[0]) / RES) + 2 * MARGIN, int((hi[1] - lo[1]) / RES) + 2 * MARGIN
    c = lambda x, y: (int(round((x - off[0]) / RES)), int(round((y - off[1]) / RES)))
    wall, glass, region = (np.zeros((H, W), np.uint8) for _ in range(3))
    doors, incset = [], set(inc)
    for n in inc:
        cv2.fillPoly(region, [np.array([c(x, y) for x, y in spaces[n]['poly']], np.int32)], 1)
    for n, s in spaces.items():
        for x1, y1, x2, y2, typ, tgt in s['segs']:
            if max(x1, x2) < off[0] or max(y1, y2) < off[1] or min(x1, x2) > off[0] + W * RES or min(y1, y2) > off[1] + H * RES:
                continue
            a, b = c(x1, y1), c(x2, y2)
            if typ == 'Wall':
                cv2.line(wall, a, b, 1, 2)
            elif typ == 'Window':
                cv2.line(glass, a, b, 1, 2)
            elif typ == 'Portal':
                if n in incset and tgt in incset:
                    continue                                       # a door inside the mapped rooms: open
                if n in incset or tgt in incset:
                    d = np.zeros((H, W), np.uint8)
                    cv2.line(d, a, b, 1, 2)
                    doors.append(d > 0)
                elif rng.random() < 0.5:
                    cv2.line(wall, a, b, 1, 2)                     # some other room's door, closed
    wall, glass = wall > 0, (glass > 0) & ~(wall > 0)
    merged = []                                                   # the same doorway is listed by both rooms: merge overlapping masks
    for d in doors:
        d = d & ~wall
        for m in merged:
            if (cv2.dilate(m.astype(np.uint8), K5) > 0)[d].any():
                m |= d
                break
        else:
            if d.sum() >= 6:
                merged.append(d)
    blocked = wall | glass
    for d in merged:
        blocked |= d
    region = (region > 0) & ~blocked
    furn = add_furniture(cv2.erode(region.astype(np.uint8), K5) > 0, wall, rng, rng.integers(0, 7))
    return World(wall=wall, region=region & ~furn, doors=merged, glass=glass, furn=furn, kind='kth')


# ------------------------------------------------------------------ MIT Stata Center (MITquest XML, coordinates in feet)
FT = 0.3048


def stata_load(path, door_m=0.9):
    """-> the same structure as kth_load: {space: {'poly': N x 2 metres, 'segs': [(x1, y1, x2, y2, type, target)]}}.
    Each room is an outline; a portal sits on one edge of it: 'implicit' = an open stretch of that edge (no wall),
    anything else = a door at one point of the edge (drawn `door_m` wide)."""
    try:
        root = ET.parse(path).getroot()
    except Exception:  # noqa: BLE001
        return None
    spaces = {}
    for sp in root.findall('space'):
        con = sp.find('contour')
        if con is None:
            continue
        P = np.array([[float(q.get('x')) * FT, float(q.get('y')) * FT] for q in con.findall('point')])
        if len(P) < 3:
            continue
        cuts = {}                                              # edge index -> [(t0, t1, target)]
        for po in sp.findall('portal'):
            e = po.find('edge')
            if e is None:
                continue
            i = int(e.get('index')) % len(P)
            L = float(np.hypot(*(P[(i + 1) % len(P)] - P[i]))) or 1e-6
            if e.get('param') is not None:
                t = float(e.get('param')); h = 0.5 * door_m / L
                cuts.setdefault(i, []).append((max(0.0, t - h), min(1.0, t + h), po.get('target') or ''))
            else:
                def num(v, d):
                    try:
                        return float(v)
                    except (TypeError, ValueError):               # some files write 'False' for a missing bound
                        return d
                cuts.setdefault(i, []).append((num(e.get('minparam'), 0.0), num(e.get('maxparam'), 1.0), po.get('target') or ''))
        segs = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            pos = 0.0
            for t0, t1, tgt in sorted(cuts.get(i, [])):
                if t0 > pos + 1e-6:
                    segs.append((*(a + (b - a) * pos), *(a + (b - a) * t0), 'Wall', ''))
                segs.append((*(a + (b - a) * max(pos, t0)), *(a + (b - a) * t1), 'Portal', tgt))
                pos = max(pos, t1)
            if pos < 1 - 1e-6:
                segs.append((*(a + (b - a) * pos), *b, 'Wall', ''))
        spaces[sp.get('name')] = {'poly': P, 'segs': [tuple(float(v) if k < 4 else v for k, v in enumerate(s)) for s in segs]}
    return spaces or None


def stata_raster(paths, res=RES, margin=40):
    """Walls and doorways of whole floors on one grid. -> wall, door, inside (bool H x W, row 0 = largest y), origin (x, y of the lower-left corner, metres)."""
    sps = {}
    for p in paths:
        sps.update(stata_load(p) or {})
    pts = np.vstack([s['poly'] for s in sps.values()])
    lo, hi = pts.min(0) - margin * res, pts.max(0) + margin * res
    W, H = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    c = lambda x, y: (int(round((x - lo[0]) / res)), int(round(H - 1 - (y - lo[1]) / res)))
    wall, door, inside = (np.zeros((H, W), np.uint8) for _ in range(3))
    for s in sps.values():
        cv2.fillPoly(inside, [np.array([c(x, y) for x, y in s['poly']], np.int32)], 1)
        for x1, y1, x2, y2, typ, _ in s['segs']:
            cv2.line(wall if typ == 'Wall' else door, c(x1, y1), c(x2, y2), 1, 1)
    return wall > 0, (door > 0) & ~(wall > 0), inside > 0, (float(lo[0]), float(lo[1]))


# ------------------------------------------------------------------ HouseExpo
def houseexpo_world(path, rng):
    d = json.load(open(path))
    v = np.array(d['verts'], float)
    lo, hi = v.min(0), v.max(0)
    if len(v) < 4 or max(hi - lo) > 35 or min(hi - lo) < 3:
        return None
    W, H = int((hi[0] - lo[0]) / RES) + 2 * MARGIN, int((hi[1] - lo[1]) / RES) + 2 * MARGIN
    region = np.zeros((H, W), np.uint8)
    cv2.fillPoly(region, [np.round((v - lo) / RES + MARGIN).astype(np.int32)], 1)
    region = region > 0
    if region.sum() < 1600:                                        # under 4 m2
        return None
    wall = ring(region)
    doors = []
    ys, xs = np.nonzero(wall)
    for _ in range(rng.integers(0, 3)):                            # this set has no outside doors: cut a few
        i = rng.integers(len(xs))
        disc = np.zeros((H, W), np.uint8)
        cv2.circle(disc, (int(xs[i]), int(ys[i])), int(rng.uniform(0.4, 0.5) / RES), 1, -1)
        dm = wall & (disc > 0)
        if dm.sum() >= 20 and not any((dm & o).any() for o in doors):
            doors.append(dm)
    for dm in doors:
        wall &= ~dm
    furn = add_furniture(cv2.erode(region.astype(np.uint8), K5) > 0, wall, rng, rng.integers(0, 8))
    return World(wall=wall, region=region & ~furn, doors=doors, furn=furn, kind='houseexpo')


# ------------------------------------------------------------------ rotation (real maps are rarely square to the grid)
def rotate(world, deg):
    H, W = world.wall.shape
    cs, sn = abs(math.cos(math.radians(deg))), abs(math.sin(math.radians(deg)))
    W2, H2 = int(W * cs + H * sn) + 2, int(W * sn + H * cs) + 2
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    M[0, 2] += W2 / 2 - W / 2
    M[1, 2] += H2 / 2 - H / 2

    def r(m, thr):
        return cv2.warpAffine(m.astype(np.float32), M, (W2, H2), flags=cv2.INTER_LINEAR) > thr
    wall, glass, furn = r(world.wall, 0.35), r(world.glass, 0.35), r(world.furn, 0.5)
    doors = [r(d, 0.35) for d in world.doors]
    blocked = wall | glass | furn
    for d in doors:
        blocked |= d
    return World(wall=wall, region=r(world.region, 0.5) & ~blocked, doors=[d & ~wall for d in doors], glass=glass & ~wall, furn=furn & ~wall, kind=world.kind)


def seal(world):
    """Close every unrecorded gap in the boundary of the region. Some plans have an opening in a room outline that is
    not listed as a door; the clean target would then leak through it. Floor space that escapes the region is cut
    off in a band just outside the walls, and each cut becomes one more doorway (closed in the clean map)."""
    occ = world.wall | world.glass | world.furn
    for d in world.doors:
        occ = occ | d
    fill = (world.region | world.furn).astype(np.uint8)
    band = (cv2.dilate(fill, np.ones((11, 11), np.uint8)) > 0) & ~(cv2.dilate(fill, np.ones((5, 5), np.uint8)) > 0)
    lab, n = ndimage.label(~occ)
    ids = np.unique(lab[world.region])
    ids = ids[ids > 0]
    cut = np.isin(lab, ids) & band
    if not cut.any():
        return world
    cl, m = ndimage.label(cut, structure=np.ones((3, 3)))
    world.doors = world.doors + [cl == i for i in range(1, m + 1)]
    return world


def crop(world, pad=MARGIN):
    """Trim the canvas to the region plus `pad` cells."""
    ys, xs = np.nonzero(world.region)
    if len(xs) == 0:
        return world
    H, W = world.region.shape
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    s = (slice(y0, y1), slice(x0, x1))
    return World(wall=world.wall[s], region=world.region[s], doors=[d[s] for d in world.doors], glass=world.glass[s], furn=world.furn[s], kind=world.kind)
