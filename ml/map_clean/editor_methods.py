#!/usr/bin/env python3
"""The cleaning methods behind the buttons of scripts/map_labeler.py.

propose(): a method marks every occupied cell of a map keep (1) or erase (2). Without ML: speckle filter, agreement
between sessions, ROSE2 (loaded from a file made beforehand, see ROSE_DIR). Everything combined: the best map of best_map.py. With ML: the erase-only models, and the
earlier model that redraws the map. The vote of the other sessions and the earlier model make a new map, so they
return that map as well (the editor shows it in the result view) next to keep / erase marks derived from it.
result(): the map that the current marks give, then the clean-up steps: cut the door leak, straighten, connect the
walls (close gaps in the outer wall and make everything inside known except the desks), fill unseen patches.
Class codes here are the labeler's: 0 free, 1 unknown, 2 occupied.
"""
import os

import cv2
import numpy as np

from .common import FREE, OCC, PGM, UNK

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, 'data', 'map_cleaning', 'models2')
OLD_MODEL = os.path.join(ROOT, 'data', 'map_cleaning', 'models', 'unet_ce.pt')
ROSE_DIR = os.path.join(ROOT, 'data', 'map_cleaning', 'rose2')            # <name>.png, 255 where ROSE2 keeps an occupied cell
EXPORT_DIR = os.path.join(ROOT, 'data', 'map_cleaning', 'exports')
# erase thresholds: set on a hand-labelled validation map to keep 95 % of structure (general model: its own, from simulation)
MODELS = {'ft': ('Fine-tuned general model', 'general_ft.pt', 0.21), 'lab': ('Lab model', 'lab.pt', 0.26), 'general': ('General model (simulated only)', 'general_sim.pt', None)}
_cache = {}


def to_common(c):                                           # labeler codes -> common.py codes
    out = np.full(c.shape, UNK, np.uint8); out[c == 0] = FREE; out[c == 2] = OCC
    return out


def to_labeler(c):
    out = np.ones(c.shape, np.uint8); out[c == FREE] = 0; out[c == OCC] = 2
    return out


def _torch():
    import torch
    return torch, ('cuda' if torch.cuda.is_available() else 'cpu')


def _pairs(name, labels_dir):
    f = os.path.join(labels_dir, 'pairs.json')
    if not os.path.exists(f):
        return None, 0.05
    import json
    pj = json.load(open(f))
    return pj['pairs'].get(name), pj['res']


def vote_map(name, raw, labels_dir, maps_dir):
    """A new map from the cells most other usable sessions share, on this map's grid (this map left out). -> (map, sessions used)"""
    import yaml
    from .common import pgm_to_cls, warp
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        from .vote import skeletonize
    pairs, res = _pairs(name, labels_dir)
    y = yaml.safe_load(open(os.path.join(maps_dir, name, 'maps_2d', name + '.yaml'))); origin = y['origin'][:2]
    stack = []
    for s, p in (pairs or {}).items():
        f = os.path.join(maps_dir, s, 'maps_2d', s)
        if p['median_m'] > 0.12 or not os.path.exists(f + '.yaml'):
            continue
        sy = yaml.safe_load(open(f + '.yaml')); sc = pgm_to_cls(cv2.imread(f + '.pgm', cv2.IMREAD_GRAYSCALE))
        w = warp((sc + 10).astype(np.uint8), res, sy['origin'][:2], np.array(p['R']), np.array(p['t']), raw.shape, origin)
        o = np.full(w.shape, UNK, np.uint8); m = w >= 10; o[m] = w[m] - 10
        stack.append(o)
    if len(stack) < 3:
        raise ValueError('the vote needs at least 3 other sessions that line up with this map')
    stack = np.stack(stack); K = len(stack)
    k3, k21 = np.ones((3, 3), np.uint8), np.ones((21, 21), np.uint8)
    n_cov = np.stack([cv2.dilate((s != UNK).astype(np.uint8), k21) for s in stack]).astype(bool).sum(0)
    occ_sup = np.stack([cv2.dilate((s == OCC).astype(np.uint8), k3) for s in stack]).sum(0)
    n_free = (stack == FREE).sum(0)
    enough = n_cov >= min(3, K)
    thick = (occ_sup >= np.maximum(2, np.ceil(0.5 * n_cov))) & enough
    wall = skeletonize(thick)
    vote = np.full(raw.shape, UNK, np.uint8)
    vote[(n_free >= 0.4 * n_cov) & enough] = FREE
    vote[thick & ~wall & (n_free < 0.4 * n_cov)] = UNK
    vote[wall] = OCC
    return vote, K


def best_reference(labels_dir, maps_dir):
    """The session a place's best map is drawn on: the one that has checkpoints, else the one most other sessions line up with."""
    f = os.path.join(labels_dir, 'pairs.json')
    if not os.path.exists(f):
        return None
    import json
    pairs = json.load(open(f))['pairs']
    names = [n for n in pairs if os.path.exists(os.path.join(maps_dir, n, 'maps_2d', n + '.pgm'))]
    with_cp = [n for n in names if os.path.isdir(os.path.join(maps_dir, n, 'checkpoints'))]
    if with_cp:
        return sorted(with_cp)[-1]
    fit = lambda n: [p['median_m'] for o, p in pairs[n].items() if o in names and p['median_m'] <= 0.12]
    names = [n for n in names if len(fit(n)) >= 2]
    return max(names, key=lambda n: (len(fit(n)), -float(np.mean(fit(n))))) if names else None


def available(name, labels_dir):
    """Buttons to offer for this map: [{id, label, group, note}]"""
    out = [{'id': 'raw', 'label': 'Raw (keep all)', 'group': 'noml'}, {'id': 'speckle', 'label': 'Speckle filter', 'group': 'noml'}]
    if os.path.exists(os.path.join(labels_dir, name + '_prelabel.png')) and _pairs(name, labels_dir)[0]:   # only where the file really is session agreement
        out += [{'id': 'agree_c', 'label': 'Session agreement, cautious', 'group': 'noml'}, {'id': 'agree_a', 'label': 'Session agreement, aggressive', 'group': 'noml'}]
    if _pairs(name, labels_dir)[0]:
        out.append({'id': 'vote', 'label': 'Vote of the other sessions', 'group': 'noml'})
    if os.path.exists(os.path.join(ROSE_DIR, name + '.png')):
        out.append({'id': 'rose', 'label': 'ROSE2 (from file)', 'group': 'noml'})
    try:
        _torch()
        if os.path.exists(OLD_MODEL):
            out.append({'id': 'earlier', 'label': 'Earlier model (redraws)', 'group': 'ml'})
        for k, (lbl, f, _) in MODELS.items():
            if os.path.exists(os.path.join(MODEL_DIR, f)):
                out.append({'id': k, 'label': lbl, 'group': 'ml'})
    except ImportError:
        pass
    if any(m['group'] == 'ml' for m in out) and os.path.exists(os.path.join(labels_dir, 'pairs.json')):   # one best map per place
        out += [{'id': 'best_f', 'label': 'Best map of this place, faithful', 'group': 'best'}, {'id': 'best_s', 'label': 'Best map of this place, straight', 'group': 'best'}]
    return out


def _erase_model(key):
    if key not in _cache:
        from .erase import load_model
        _, dev = _torch()
        G, ck = load_model(os.path.join(MODEL_DIR, MODELS[key][1]), dev)
        _cache[key] = (G, MODELS[key][2] if MODELS[key][2] is not None else ck['thr'], dev)
    return _cache[key]


def propose(name, method, cls_lab, labels_dir, maps_dir=None, n_desks=4, desk_size=None):
    """-> (marks: 0 not occupied, 1 keep, 2 erase, 3 unsure; message; the method's own map, or None when the marks say it all)"""
    raw = to_common(cls_lab); occ = raw == OCC
    lab = np.zeros(raw.shape, np.uint8); lab[occ] = 1
    view = None
    if method == 'raw':
        msg = 'Raw map: every occupied cell kept.'
    elif method == 'speckle':
        n, cc, st, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
        small = np.isin(cc, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < 8])
        lab[small & occ] = 2
        msg = 'Speckle filter: occupied blobs smaller than 8 cells erased.'
    elif method in ('agree_c', 'agree_a'):
        pre = cv2.imread(os.path.join(labels_dir, name + '_prelabel.png'), cv2.IMREAD_UNCHANGED)
        if pre is None or pre.shape != raw.shape:
            raise ValueError('no session-agreement file for this map')
        lab[occ & ((pre == 2) | ((pre == 3) if method == 'agree_a' else False))] = 2
        msg = 'Session agreement: erased what the other sessions ' + ('clearly lack.' if method == 'agree_c' else 'do not clearly share.')
    elif method == 'vote':
        view, K = vote_map(name, raw, labels_dir, maps_dir)
        near = cv2.dilate((view == OCC).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        lab[occ & ~near] = 2
        msg = (f'Vote of {K} other sessions: a new map, shown in the result view. The marks say erase where the vote has no wall within 10 cm. '
               f'{int(((view == OCC) & (cv2.dilate(occ.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)).sum())} of its wall cells have no wall of this map within 10 cm.')
    elif method == 'rose':
        keep = cv2.imread(os.path.join(ROSE_DIR, name + '.png'), cv2.IMREAD_GRAYSCALE)
        if keep is None or keep.shape != raw.shape:
            raise ValueError('no ROSE2 result for this map')
        lab[occ & (keep == 0)] = 2
        msg = 'ROSE2, default settings. Loaded from a result made beforehand (ROSE2 runs in its own container).'
    elif method == 'earlier':
        torch, dev = _torch()
        from .clean_map import clean, load_model
        if 'earlier' not in _cache:
            _cache['earlier'] = load_model(OLD_MODEL, dev)
        with torch.no_grad():
            out = clean(raw, _cache['earlier'], dev)
        near = cv2.dilate((out == OCC).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        lab[occ & ~near] = 2; view = out
        msg = f'Earlier model (redraws the map): its own map is shown in the result view. The marks say erase where it left no wall within 10 cm. It drew {int(((out == OCC) & ~occ).sum())} wall cells of its own.'
    elif method in ('best_f', 'best_s'):
        from .best_map import build
        ref = best_reference(labels_dir, maps_dir)
        if ref is None:
            raise ValueError('fewer than 3 sessions of this place line up, so it has no best map')
        if ref != name:
            raise LookupError(ref)                           # the place has one best map, drawn on its reference session
        key = ('best', labels_dir, name, n_desks, tuple(desk_size) if desk_size else None)
        if key not in _cache:                               # everything combined: see best_map.py
            _cache[key] = build(name, None, n_desks, desk_size or ((3.45, 1.25) if n_desks else None), None, labels_dir, maps_dir)
        b = _cache[key]
        lab[occ & b['erase']] = 2; view = b['faithful'] if method == 'best_f' else b['straight']
        msg = (f'Best map of this place, drawn on its reference session {name}, ' + ('faithful (nothing drawn that no session saw)' if method == 'best_f' else 'straight (identical desks, regular outline, walls connected)')
               + ': the model and the other days decide together, the saved hand labels overrule, other sessions fill in what this one did not see. ' + b['summary'] + '.')
    elif method in MODELS:
        torch, _ = _torch()
        from .erase import predict
        G, thr, dev = _erase_model(method)
        with torch.no_grad():
            p = predict(raw, G, dev)
        lab[occ & (p > thr)] = 2
        msg = f'{MODELS[method][0]}: erase where the score is above {thr:.2f} ({dev}).'
    else:
        raise ValueError('unknown method ' + method)
    return lab, msg + f' Keep {int((lab == 1).sum())}, erase {int((lab == 2).sum())}.', view


def result(cls_lab, lab, steps, n_desks=4, desk_size=None, res=0.05):
    """The map the marks give (erase cells removed, unsure cells kept), then the chosen clean-up steps. -> (labeler-coded map, message)"""
    from .erase import apply_erase
    raw = to_common(cls_lab)
    out = apply_erase(raw, (lab == 2).astype(np.float32), 0.5)
    notes = [f'{int(((lab == 2) & (raw == OCC)).sum())} cells erased']
    if 'leak' in steps:
        from .door_leak import leak_mask
        m = leak_mask(raw, 0.6, 8.0, res) & (out == FREE)
        out[m] = UNK; notes.append(f'door leak: {m.sum() * res * res:.1f} m2 of floor cut off')
    if 'straighten' in steps:
        from .outline import outline
        from .straighten import straighten
        desks, info = straighten(out, n_desks, desk_size, min_len=8, gap=4, snap=2.0, lab=lab)
        out, _ = outline(out, desks)
        notes.append(f"straightened: {len(info['desks'])} desks redrawn, outer wall regularised")
    if 'connect' in steps:
        from .connect_walls import connect_walls
        out, info = connect_walls(out, lab, n_desks, desk_size)
        left = info['openings_left_m']
        notes.append(f"walls connected: {info['gaps_closed']} gaps closed with {info['wall_cells_drawn']} drawn wall cells, {info['filled_m2']:.1f} m2 inside made floor"
                     + (f", {len(left)} opening(s) over 1.5 m left open" if left else ', room enclosed'))
    if 'fill' in steps and 'connect' not in steps:
        from .fill_unknown import fill_unknown
        out, filled, _ = fill_unknown(out, lab, n_desks, desk_size)
        notes.append(f'unseen patches: {filled * res * res:.1f} m2 turned into floor')
    return out, '; '.join(notes) + '.'


def score_text(name, out_common, cls_lab, labels_dir):
    """Scores against the saved hand labels of this map, if there are any."""
    f = os.path.join(labels_dir, name + '_label.png')
    if not os.path.exists(f):
        return 'No saved labels for this map, so no score.'
    from .score2 import score2
    hand = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    s = score2(out_common, to_common(cls_lab), hand)
    return (f"Against the saved labels: structure kept {s['kept'] * 100:.1f} %, clutter removed {s['removed'] * 100:.1f} %, "
            f"invented wall cells {s['invented']}, added floor {s['added_floor_m2']:.1f} m2.")


def export(name, out_common, map_yaml):
    import time
    import yaml
    os.makedirs(EXPORT_DIR, exist_ok=True)
    stem = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    cv2.imwrite(os.path.join(EXPORT_DIR, stem + '.pgm'), PGM[out_common])
    if os.path.exists(map_yaml):
        y = yaml.safe_load(open(map_yaml)); y['image'] = stem + '.pgm'
        yaml.safe_dump(y, open(os.path.join(EXPORT_DIR, stem + '.yaml'), 'w'), default_flow_style=None, sort_keys=False)
    return os.path.join(EXPORT_DIR, stem + '.pgm')
