#!/usr/bin/env python3
"""The best map of one room from everything available: all sessions, the model, the hand labels, the clean-up steps.

    python -m ml.map_clean.best_map lab2d_11_25_5 --out data/map_cleaning/best

1. Decide every occupied cell of the target map:
     erase when the model says erase and fewer than 85 % of the other days' sessions have a wall there,
     or when fewer than 20 % of them have one (rule tested with a day held out, see fuse.py);
     the model is the mean of the fine-tuned copies that never saw the target's day;
     hand labels overrule this: the target's own, then those of other labelled sessions carried to the same place.
2. Cut the floor that leaked through the door.
3. Use what the other sessions saw where the target saw nothing (inside the room only): floor that most sessions
   show becomes floor, and a wall that most sessions have at exactly that cell becomes wall,
   unless a hand label of another session calls it clutter.
4. `faithful` stops here, with the remaining unseen patches inside the room turned into floor.
5. `straight` goes on: desks as identical rectangles, the outer wall regularised, gaps in it closed, the inside known.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import yaml

from .common import FREE, OCC, PGM, UNK, load_map, warp
from .connect_walls import connect_walls
from .door_leak import leak_mask
from .erase import apply_erase, load_model, predict
from .fill_unknown import fill_unknown
from .fuse import LB, day, persistence, stack
from .outline import outline
from .straighten import straighten

LO, HI = 0.20, 0.85


def carried_labels(target, tm, own):
    """Hand labels of other sessions at the same place (within 2 cells), for cells the target's own labels leave open."""
    pj = json.load(open(f'{LB}/pairs.json')); pairs, res = pj['pairs'][target], pj['res']
    occ = tm['cls'] == OCC
    dk = np.full(occ.shape, 1e9, np.float32); de = dk.copy(); used = []
    for s, p in pairs.items():
        f = f'{LB}/{s}_label.png'
        if s == target or not os.path.exists(f) or p['median_m'] > 0.12:
            continue
        sm = load_map(f'maps/{s}/maps_2d/{s}.yaml'); sl = cv2.imread(f, -1)
        w = warp((sl + 10).astype(np.uint8), res, sm['origin'], np.array(p['R']), np.array(p['t']), occ.shape, tm['origin'])
        dk = np.minimum(dk, cv2.distanceTransform((w != 11).astype(np.uint8), cv2.DIST_L2, 3))
        de = np.minimum(de, cv2.distanceTransform((w != 12).astype(np.uint8), cv2.DIST_L2, 3)); used.append(s)
    out = np.zeros(occ.shape, np.uint8)
    out[occ & (dk <= 2.5) & (dk <= de)] = 1
    out[occ & (de <= 2.5) & (de < dk)] = 2
    out[(own == 1) | (own == 2)] = 0
    return out, used, de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target'); ap.add_argument('--out', default='data/map_cleaning/best')
    ap.add_argument('--models', nargs='*', default=None, help='fine-tuned copies that never saw the target day; default: by day')
    ap.add_argument('--desk-size', type=float, nargs=2, default=[3.45, 1.25]); ap.add_argument('--desks', type=int, default=4)
    a = ap.parse_args()
    T = a.target; os.makedirs(a.out, exist_ok=True)
    models = a.models or (['ftfold_A1', 'ftfold_A2'] if day(T) == '25' else ['ftfold_B1', 'ftfold_B2'] if day(T) == '24' else ['general_ft'])
    tm, other_days, _ = stack(T, other_days_only=True); _, everyone, _ = stack(T)
    raw = tm['cls']; occ = raw == OCC; res = tm['res']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ps, thrs = [], []
    for m in models:
        G, ck = load_model(f'data/map_cleaning/models2/{m}.pt', dev)
        with torch.no_grad():
            ps.append(predict(raw, G, dev)); thrs.append(ck['thr'])
    p, thr = np.mean(ps, 0), float(np.mean(thrs))
    pers, _, cov = persistence(other_days)
    erase = occ & (((p > thr) & (pers < HI)) | ((pers < LO) & (cov >= 3)))
    log = [f'{T}: {int(occ.sum())} occupied cells; {len(other_days)} sessions from other days, {len(everyone)} sessions in all; models {models}, threshold {thr:.2f}',
           f'1. model + other days: erase {int(erase.sum())} (model alone would erase {int((occ & (p > thr)).sum())})']
    own = cv2.imread(f'{LB}/{T}_label.png', -1) if os.path.exists(f'{LB}/{T}_label.png') else np.zeros(raw.shape, np.uint8)
    car, used, d_erase = carried_labels(T, tm, own)
    before = erase.copy()
    erase[(own == 1) | (car == 1)] = False; erase[occ & ((own == 2) | (car == 2))] = True
    log.append(f'   hand labels: own keep {int((own == 1).sum())} / erase {int((own == 2).sum())}, carried from {used}: keep {int((car == 1).sum())} / erase {int((car == 2).sum())}; '
               f'they changed the decision on {int((before != erase).sum())} cells -> erase {int(erase.sum())}')
    out = apply_erase(raw, erase.astype(np.float32), 0.5)
    lm = leak_mask(raw, 0.6, 8.0, res) & (out == FREE); out[lm] = UNK
    log.append(f'2. door leak: {lm.sum() * res * res:.1f} m2 of floor cut off')
    # 3. what the other sessions saw where this one saw nothing, inside the room
    free = (out == FREE).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), connectivity=4)
    room = cv2.morphologyEx((cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnt, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    room = cv2.drawContours(np.zeros_like(room), cnt, -1, 1, cv2.FILLED) > 0
    ncov = np.maximum(np.stack([cv2.dilate((m != UNK).astype(np.uint8), np.ones((21, 21), np.uint8)) for _, m in everyone]).sum(0), 1)
    occ_x = np.stack([(m == OCC) for _, m in everyone]).sum(0) / ncov
    fre_x = np.stack([(m == FREE) for _, m in everyone]).sum(0) / ncov
    unk = (out == UNK) & room & (ncov >= 4)
    add_wall = unk & (occ_x >= 0.5) & (d_erase > 1.5)          # not where another session's hand label says clutter
    add_free = unk & (fre_x >= 0.6) & ~add_wall
    out[add_free] = FREE; out[add_wall] = OCC
    log.append(f'3. from the other sessions, where this one saw nothing: {add_free.sum() * res * res:.2f} m2 of floor, {int(add_wall.sum())} wall cells')
    lab = np.zeros(raw.shape, np.uint8); lab[out == OCC] = 1                 # everything left is structure: the desk finder reads this
    faithful, filled, kept_unk = fill_unknown(out, lab, a.desks, a.desk_size)
    log.append(f'4. faithful: {filled * res * res:.2f} m2 of remaining unseen patches turned into floor, {kept_unk * res * res:.2f} m2 left unknown (desks, openings)')
    desks, info = straighten(out, a.desks, a.desk_size, min_len=8, gap=4, snap=2.0, lab=lab)
    st_, oinfo = outline(out, desks)
    straight, cinfo = connect_walls(st_, lab, a.desks, a.desk_size)
    log.append(f"5. straight: main direction {info['angle_deg']:.1f} deg, {len(info['desks'])} desks redrawn at {a.desk_size[0]} x {a.desk_size[1]} m, {cinfo['gaps_closed']} wall gaps closed with "
               f"{cinfo['wall_cells_drawn']} drawn cells, openings left {cinfo['openings_left_m'] or 'none'}, {cinfo['filled_m2']:.2f} m2 inside made floor")
    y = yaml.safe_load(open(f'maps/{T}/maps_2d/{T}.yaml'))
    for name, m in (('raw', raw), ('decided', apply_erase(raw, erase.astype(np.float32), 0.5)), ('faithful', faithful), ('straight', straight)):
        cv2.imwrite(f'{a.out}/{T}_{name}.pgm', PGM[m]); y2 = dict(y); y2['image'] = f'{T}_{name}.pgm'
        yaml.safe_dump(y2, open(f'{a.out}/{T}_{name}.yaml', 'w'), default_flow_style=None, sort_keys=False)
    for k, m in (('faithful', faithful), ('straight', straight)):
        log.append(f'   {k}: {int((m == OCC).sum())} occupied cells, {(m == FREE).sum() * res * res:.1f} m2 of floor, {int(((m == UNK) & room).sum())} unknown cells inside the room')
    open(f'{a.out}/{T}_log.txt', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))


if __name__ == '__main__':
    main()
