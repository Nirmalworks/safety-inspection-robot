#!/usr/bin/env python3
"""Erase-only map cleaner: for every occupied cell of a raw map, decide keep or erase. It never draws or moves a wall.

  train on the lab maps (phase 1):
    python -m ml.map_clean.erase train --name lab --lab-val lab2d_11_25_5 --lab-test lab2d_11_25_7
  train on simulated KTH + HouseExpo pairs (phase 2), then fine-tune on the lab:
    python -m ml.map_clean.erase train --name general --sim "data/map_cleaning/pairs/v2/chunk_*.npz" --lab-share 0
    python -m ml.map_clean.erase train --name general_ft --init data/map_cleaning/models2/general.pt --lr 5e-5 --epochs 10
  clean a map:
    python -m ml.map_clean.erase clean IN.pgm OUT.pgm --model data/map_cleaning/models2/lab.pt

Labels: 1 keep, 2 erase, 3 unsure (ignored), 0 not occupied. Lab labels come from <labels>/<name>_label.png if it
exists (hand-corrected), else <name>_proplabel.png (hand corrections carried over from another map), else
<name>_prelabel.png (agreement between sessions).
The erase threshold is chosen on validation data as the one that keeps at least `--keep-target` of the keep cells.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .common import FREE, OCC, PGM, UNK, pgm_to_cls
from .model import UNet

P = 256


def lab_pair(name, labels):
    cls = pgm_to_cls(cv2.imread(f'maps/{name}/maps_2d/{name}.pgm', cv2.IMREAD_GRAYSCALE))
    for suf in ('_label', '_proplabel', '_prelabel'):
        f = os.path.join(labels, name + suf + '.png')
        if os.path.exists(f):
            lab = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if lab.shape == cls.shape:
                lab = lab.copy(); lab[cls != OCC] = 0
                return cls, lab, suf[1:]
    raise FileNotFoundError(name)


def lab_crop(cls, lab, rng):
    """A random 256 x 256 window of a lab map, turned by a random angle and flipped, so position in the room carries no information."""
    H, W = cls.shape
    ang = rng.uniform(0, 360)
    cx, cy = rng.uniform(0.15 * W, 0.85 * W), rng.uniform(0.15 * H, 0.85 * H)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += P / 2 - cx; M[1, 2] += P / 2 - cy
    x = cv2.warpAffine(cls, M, (P, P), flags=cv2.INTER_NEAREST, borderValue=UNK)
    l = cv2.warpAffine(lab, M, (P, P), flags=cv2.INTER_NEAREST, borderValue=0)
    l[x != OCC] = 0
    if rng.random() < 0.5:
        x, l = x[:, ::-1], l[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(l)


def real_crop(cls, lab, rng):
    """Like lab_crop, for large maps: the window is centred near a random labelled cell, so it is never empty."""
    ys, xs = np.nonzero((lab == 1) | (lab == 2))
    i = rng.integers(len(xs))
    cx, cy = xs[i] + rng.uniform(-70, 70), ys[i] + rng.uniform(-70, 70)
    M = cv2.getRotationMatrix2D((cx, cy), rng.uniform(0, 360), 1.0)
    M[0, 2] += P / 2 - cx; M[1, 2] += P / 2 - cy
    x = cv2.warpAffine(cls, M, (P, P), flags=cv2.INTER_NEAREST, borderValue=UNK)
    l = cv2.warpAffine(lab, M, (P, P), flags=cv2.INTER_NEAREST, borderValue=0)
    l[x != OCC] = 0
    if rng.random() < 0.5:
        x, l = x[:, ::-1], l[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(l)


def load_real(pattern, skip=(), only=None):
    out = []
    for f in sorted(glob.glob(pattern)):
        n = os.path.basename(f)[:-4]
        if any(k in n for k in skip) or (only is not None and not any(k in n for k in only)):
            continue
        z = np.load(f)
        if ((z['L'] == 1) | (z['L'] == 2)).sum() >= 500:
            out.append((n, z['X'], z['L']))
    return out


def onehot(x):
    return F.one_hot(x.long(), 3).permute(0, 3, 1, 2).float()


@torch.no_grad()
def predict(cls, G, dev):
    """Erase probability for every cell of a whole map (sliding 256 windows, averaged)."""
    H, W = cls.shape
    pad = P // 2
    Hp, Wp = int(np.ceil((H + 2 * pad - P) / 128)) * 128 + P, int(np.ceil((W + 2 * pad - P) / 128)) * 128 + P
    big = np.full((Hp, Wp), UNK, np.uint8); big[pad:pad + H, pad:pad + W] = cls
    x = onehot(torch.from_numpy(big)[None])[0]
    acc, cnt = torch.zeros(Hp, Wp), torch.zeros(Hp, Wp)
    win = torch.hann_window(P, periodic=False).clamp_min(0.05); win = win[:, None] * win[None, :]
    tiles = [(y, xx) for y in range(0, Hp - P + 1, 128) for xx in range(0, Wp - P + 1, 128)]
    for i in range(0, len(tiles), 16):
        b = torch.stack([x[:, y:y + P, xx:xx + P] for y, xx in tiles[i:i + 16]]).to(dev)
        pr = torch.sigmoid(G(b).float())[:, 0].cpu()
        for (y, xx), q in zip(tiles[i:i + 16], pr):
            acc[y:y + P, xx:xx + P] += q * win; cnt[y:y + P, xx:xx + P] += win
    return (acc / cnt).numpy()[pad:pad + H, pad:pad + W]


def apply_erase(cls, p_erase, thr):
    """Erased cells become floor where floor touches them, otherwise unknown. Nothing else changes."""
    out = cls.copy()
    kill = (cls == OCC) & (p_erase > thr)
    near_free = cv2.dilate((cls == FREE).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    grow = near_free.copy()
    for _ in range(12):                                   # floor reaches the inside of an erased blob through its erased neighbours
        grow = grow | ((cv2.dilate(grow.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & kill)
    out[kill & grow] = FREE
    out[kill & ~grow] = UNK
    return out


def scores(p, lab, thr):
    k, e = lab == 1, lab == 2
    return float((p[k] <= thr).mean()) if k.any() else float('nan'), float((p[e] > thr).mean()) if e.any() else float('nan')


def pick_threshold(ps, labs, keep_target):
    pk = np.concatenate([p[l == 1] for p, l in zip(ps, labs)])
    pe = np.concatenate([p[l == 2] for p, l in zip(ps, labs)])
    best = (1.0, 0.0, 1.0)
    for thr in np.linspace(0.02, 0.98, 97):
        kr, er = float((pk <= thr).mean()), float((pe > thr).mean()) if len(pe) else 0.0
        if kr >= keep_target and er > best[1]:
            best = (float(thr), er, kr)
    return best                                           # threshold, clutter removed, structure kept


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    G = UNet(ck['width'], n_out=1).to(dev); G.load_state_dict(ck['model'])
    return G.eval(), ck


def train(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    names = sorted(f[:-len('_prelabel.png')] for f in os.listdir(a.labels) if f.endswith('_prelabel.png'))
    held = set(a.lab_val + a.lab_test)
    lab_tr = [lab_pair(n, a.labels) for n in names if n not in held] if a.lab_share > 0 else []
    lab_va = [lab_pair(n, a.labels) for n in a.lab_val]
    real_tr = load_real(a.real, skip=a.real_val + a.real_test) if a.real else []
    real_va = load_real(a.real, only=a.real_val) if (a.real and a.real_val) else []
    Xs = Ls = None
    if a.sim:
        files = sorted(glob.glob(a.sim))
        va_files, tr_files = files[::25], [f for i, f in enumerate(files) if i % 25]
        ld = lambda fs: (torch.from_numpy(np.concatenate([np.load(f)['X'] for f in fs])), torch.from_numpy(np.concatenate([np.load(f)['L'] for f in fs])))
        Xs, Ls = ld(tr_files); Xv, Lv = ld(va_files)
        print(f'simulated: {len(Xs)} training patches from {len(tr_files)} chunks, {len(Xv)} validation patches', flush=True)
    if real_tr:
        print(f'real labelled maps: {len(real_tr)} for training ({[n for n, _, _ in real_tr]}), validation {[n for n, _, _ in real_va]}, held out {a.real_test}', flush=True)
    print(f'{a.name}: lab training maps {len(lab_tr)} ({sum(s == "label" for *_, s in lab_tr)} hand-corrected), lab validation {a.lab_val}, test (untouched) {a.lab_test}, device {dev}', flush=True)
    G = UNet(a.width, n_out=1).to(dev).to(memory_format=torch.channels_last)
    if a.init:
        G.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)['model'])
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    steps = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    n_real = int(round(a.bs * a.real_share)) if real_tr else 0
    n_lab = int(round(a.bs * a.lab_share)) if (lab_tr and (Xs is not None or real_tr)) else (a.bs if lab_tr else 0)
    n_lab = min(n_lab, a.bs - n_real)
    if Xs is None:
        n_real = a.bs - n_lab if real_tr else 0
    log, best, t0 = [], None, time.time()
    for ep in range(1, a.epochs + 1):
        G.train(); tot = 0.0
        for _ in range(a.steps):
            xb, lb = [], []
            for _ in range(n_lab):
                c, l, _s = lab_tr[rng.integers(len(lab_tr))]
                x, l = lab_crop(c, l, rng); xb.append(torch.from_numpy(x)); lb.append(torch.from_numpy(l))
            for _ in range(n_real):
                _n, c, l = real_tr[rng.integers(len(real_tr))]
                x, l = real_crop(c, l, rng); xb.append(torch.from_numpy(x)); lb.append(torch.from_numpy(l))
            if a.bs - n_lab - n_real > 0:
                idx = torch.from_numpy(rng.integers(0, len(Xs), a.bs - n_lab - n_real))
                k = int(rng.integers(4)); xs, ls = torch.rot90(Xs[idx], k, (1, 2)), torch.rot90(Ls[idx], k, (1, 2))
                if rng.random() < 0.5:
                    xs, ls = xs.flip(2), ls.flip(2)
                xb += list(xs); lb += list(ls)
            x, l = torch.stack(xb).to(dev), torch.stack(lb).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                logit = G(onehot(x).contiguous(memory_format=torch.channels_last))
            logit = logit.float()[:, 0]
            m = (l == 1) | (l == 2)
            w = torch.where(l == 1, a.keep_weight, 1.0)[m]          # wrongly erasing structure costs more than leaving clutter
            loss = (F.binary_cross_entropy_with_logits(logit[m], (l[m] == 2).float(), reduction='none') * w).sum() / w.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        G.eval(); row = {'epoch': ep, 'loss': tot / a.steps, 'minutes': (time.time() - t0) / 60}
        ps, labs = [], []
        if lab_va:
            for c, l, _s in lab_va:
                ps.append(predict(c, G, dev)); labs.append(l)
            thr, er, kr = pick_threshold(ps, labs, a.keep_target)
            row.update(lab_thr=thr, lab_clutter_removed=er, lab_structure_kept=kr)
        if real_va:
            pr = [predict(c, G, dev) for _, c, _ in real_va]
            thr_r, er_r, kr_r = pick_threshold(pr, [l for _, _, l in real_va], a.keep_target)
            row.update(real_thr=thr_r, real_clutter_removed=er_r, real_structure_kept=kr_r)
        if Xs is not None:
            pv = []
            with torch.no_grad():
                for i in range(0, len(Xv), 128):
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                        pv.append(torch.sigmoid(G(onehot(Xv[i:i + 128].to(dev))).float())[:, 0].cpu())
            pv = torch.cat(pv).numpy()
            thr_s, er_s, kr_s = pick_threshold([pv], [Lv.numpy()], a.keep_target)
            row.update(sim_thr=thr_s, sim_clutter_removed=er_s, sim_structure_kept=kr_s)
        key = row.get('lab_clutter_removed', row.get('real_clutter_removed', row.get('sim_clutter_removed', -row['loss'])))
        log.append(row)
        if best is None or key > best:
            best = key
            torch.save({'model': G.state_dict(), 'width': a.width, 'args': vars(a), 'epoch': ep, 'row': row,
                        'thr': row.get('lab_thr', row.get('real_thr', row.get('sim_thr', 0.5)))}, os.path.join(a.out, a.name + '.pt'))
        json.dump({'args': vars(a), 'log': log}, open(os.path.join(a.out, a.name + '.log.json'), 'w'), indent=1)
        print('ep %2d  loss %.4f | ' % (ep, row['loss']) + '  '.join(f'{k} {v:.3f}' for k, v in row.items() if k not in ('epoch', 'loss', 'minutes')) + ' | %.1f min' % row['minutes'], flush=True)
    print(f'{a.name}: best {best:.4f} (clutter removed at {a.keep_target * 100:.0f} % structure kept), {(time.time() - t0) / 60:.1f} min', flush=True)


def clean(a):
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    G, ck = load_model(a.model, dev)
    cls = pgm_to_cls(cv2.imread(a.inp, cv2.IMREAD_GRAYSCALE))
    thr = a.thr if a.thr >= 0 else ck['thr']
    out = apply_erase(cls, predict(cls, G, dev), thr)
    cv2.imwrite(a.outp, PGM[out])
    y_in = os.path.splitext(a.inp)[0] + '.yaml'
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in)); y['image'] = os.path.basename(a.outp)
        yaml.safe_dump(y, open(os.path.splitext(a.outp)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f'{a.inp} -> {a.outp}: erased {int(((cls == OCC) & (out != OCC)).sum())} of {int((cls == OCC).sum())} occupied cells at threshold {thr:.2f} ({dev})')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--name', required=True); t.add_argument('--out', default='data/map_cleaning/models2')
    t.add_argument('--labels', default='data/map_cleaning/labels')
    t.add_argument('--sim', default='', help='glob of simulated chunks; empty = lab maps only')
    t.add_argument('--lab-share', type=float, default=1.0, help='share of each batch drawn from the lab maps (0 = none)')
    t.add_argument('--lab-val', nargs='*', default=['lab2d_11_25_5']); t.add_argument('--lab-test', nargs='*', default=['lab2d_11_25_7'])
    t.add_argument('--real', default='', help='glob of real labelled maps (.npz with X, L), e.g. Semantic2D')
    t.add_argument('--real-share', type=float, default=0.3, help='share of each batch drawn from the real labelled maps')
    t.add_argument('--real-val', nargs='*', default=[]); t.add_argument('--real-test', nargs='*', default=[])
    t.add_argument('--init', default=''); t.add_argument('--epochs', type=int, default=20); t.add_argument('--steps', type=int, default=300)
    t.add_argument('--bs', type=int, default=32); t.add_argument('--lr', type=float, default=2e-4); t.add_argument('--width', type=int, default=48)
    t.add_argument('--keep-weight', type=float, default=3.0); t.add_argument('--keep-target', type=float, default=0.95); t.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('clean')
    c.add_argument('inp'); c.add_argument('outp'); c.add_argument('--model', required=True); c.add_argument('--thr', type=float, default=-1); c.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else clean(a)


if __name__ == '__main__':
    main()
