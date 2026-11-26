#!/usr/bin/env python3
"""The cleaning methods behind the buttons of scripts/map_labeler.py.

propose(): a method marks every occupied cell of a map keep (1) or erase (2). Without ML: speckle filter, agreement
between sessions, ROSE2 (loaded from a file made beforehand, see ROSE_DIR). With ML: the erase-only models, and the
earlier model that redraws the map (shown as keep / erase marks; the walls it adds are not shown).
result(): the map that the current marks give, then the clean-up steps: cut the door leak, straighten, fill unseen patches.
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


def available(name, labels_dir):
    """Buttons to offer for this map: [{id, label, group, note}]"""
    out = [{'id': 'raw', 'label': 'Raw (keep all)', 'group': 'noml'}, {'id': 'speckle', 'label': 'Speckle filter', 'group': 'noml'}]
    if os.path.exists(os.path.join(labels_dir, name + '_prelabel.png')):
        out += [{'id': 'agree_c', 'label': 'Session agreement, cautious', 'group': 'noml'}, {'id': 'agree_a', 'label': 'Session agreement, aggressive', 'group': 'noml'}]
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
    return out


def _erase_model(key):
    if key not in _cache:
        from .erase import load_model
        _, dev = _torch()
        G, ck = load_model(os.path.join(MODEL_DIR, MODELS[key][1]), dev)
        _cache[key] = (G, MODELS[key][2] if MODELS[key][2] is not None else ck['thr'], dev)
    return _cache[key]


def propose(name, method, cls_lab, labels_dir):
    """-> (marks: 0 not occupied, 1 keep, 2 erase, 3 unsure; message)"""
    raw = to_common(cls_lab); occ = raw == OCC
    lab = np.zeros(raw.shape, np.uint8); lab[occ] = 1
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
        lab[occ & ~near] = 2
        msg = f'Earlier model (redraws the map): marked erase where it left no wall within 10 cm. It also drew {int(((out == OCC) & ~occ).sum())} wall cells of its own, not shown here.'
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
    return lab, msg + f' Keep {int((lab == 1).sum())}, erase {int((lab == 2).sum())}.'


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
    if 'fill' in steps:
        from .fill_unknown import fill_unknown
        out, filled, _ = fill_unknown(out, lab, n_desks, desk_size)
        notes.append(f'unseen patches: {filled * res * res:.1f} m2 turned into floor')
    return out, '; '.join(notes) + '.'


def score_text(name, out_common, cls_lab, labels_dir):
    """Scores against the saved hand labels of this map, if there are any."""
    f = os.path.join(labels_dir, name + '_label.png')
    if not os.path.exists(f):
        return 'No saved hand labels for this map, so no score.'
    from .score2 import score2
    hand = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    s = score2(out_common, to_common(cls_lab), hand)
    return (f"Against the saved hand labels: structure kept {s['kept'] * 100:.1f} %, clutter removed {s['removed'] * 100:.1f} %, "
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
