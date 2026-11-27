#!/usr/bin/env python3
"""A generative (redraw) model on the same data as the erase-only model, for a like-for-like comparison.

    python -m ml.map_clean.redraw --name redraw_sim --sim "data/map_cleaning/pairs/v2/chunk_*.npz" --lab-share 0 --lab-val
    python -m ml.map_clean.redraw --name redrawft_A1 --init data/map_cleaning/models2/redraw_sim.pt --labels data/map_cleaning/folds/A1 \
        --lab-val lab2d_11_24_3 --lab-test <the held-out day> --lr 5e-5 --epochs 10 --steps 300 --bs 12

The same U-Net with three outputs (free / occupied / unknown for every cell): it redraws the whole patch and is free to
change any cell. The target is the input with its erase-labelled cells removed (to floor where floor reaches them, else
to unknown), so it is trained on exactly the patches and labels the erase-only model sees (erase.py). Cells labelled
unsure are left out of the loss. Occupied cells weigh more than the rest (they are about 2 % of a map), and a keep cell
3 times an erase cell, as in erase.py. Checkpoints load with clean_map.load_model.
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

from .clean_map import clean
from .common import FREE, OCC, UNK
from .erase import lab_crop, lab_pair, onehot
from .model import UNet
from .score2 import score2


def target(x, l):
    """x, l: B x H x W on the device -> the map each patch should become; 255 where the cell is not scored (unsure)."""
    e = l == 2
    grow = F.max_pool2d((x == FREE).float()[:, None], 3, 1, 1)[:, 0] > 0
    for _ in range(12):                                   # floor reaches the inside of an erased blob through its erased neighbours
        grow = grow | ((F.max_pool2d(grow.float()[:, None], 3, 1, 1)[:, 0] > 0) & e)
    y = x.clone().long()
    y[e & grow] = FREE; y[e & ~grow] = UNK
    y[l == 3] = 255
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--out', default='data/map_cleaning/models2')
    ap.add_argument('--labels', default='data/map_cleaning/labels'); ap.add_argument('--sim', default='')
    ap.add_argument('--lab-share', type=float, default=1.0)
    ap.add_argument('--lab-val', nargs='*', default=['lab2d_11_25_5']); ap.add_argument('--lab-test', nargs='*', default=['lab2d_11_25_7'])
    ap.add_argument('--init', default=''); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--occ-weight', type=float, default=20.0); ap.add_argument('--keep-weight', type=float, default=3.0); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    names = sorted(f[:-len('_prelabel.png')] for f in os.listdir(a.labels) if f.endswith('_prelabel.png'))
    held = set(a.lab_val + a.lab_test)
    lab_tr = [lab_pair(n, a.labels) for n in names if n not in held] if a.lab_share > 0 else []
    lab_va = [lab_pair(n, a.labels) for n in a.lab_val]
    Xs = Ls = None
    if a.sim:
        files = sorted(glob.glob(a.sim)); tr_files = [f for i, f in enumerate(files) if i % 25]
        Xs = torch.from_numpy(np.concatenate([np.load(f)['X'] for f in tr_files])); Ls = torch.from_numpy(np.concatenate([np.load(f)['L'] for f in tr_files]))
        print(f'simulated: {len(Xs)} training patches', flush=True)
    print(f'{a.name}: lab training maps {len(lab_tr)}, validation {a.lab_val}, untouched {a.lab_test}, device {dev}', flush=True)
    G = UNet(a.width).to(dev).to(memory_format=torch.channels_last)
    if a.init:
        G.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)['model'])
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps, pct_start=0.05)
    n_lab = int(round(a.bs * a.lab_share)) if (lab_tr and Xs is not None) else (a.bs if lab_tr else 0)
    log, best, t0 = [], None, time.time()
    for ep in range(1, a.epochs + 1):
        G.train(); tot = 0.0
        for _ in range(a.steps):
            xb, lb = [], []
            for _ in range(n_lab):
                c, l, _s = lab_tr[rng.integers(len(lab_tr))]
                x, l = lab_crop(c, l, rng); xb.append(torch.from_numpy(x)); lb.append(torch.from_numpy(l))
            if a.bs - n_lab > 0:
                idx = torch.from_numpy(rng.integers(0, len(Xs), a.bs - n_lab)); k = int(rng.integers(4))
                xs, ls = torch.rot90(Xs[idx], k, (1, 2)), torch.rot90(Ls[idx], k, (1, 2))
                if rng.random() < 0.5:
                    xs, ls = xs.flip(2), ls.flip(2)
                xb += list(xs); lb += list(ls)
            x, l = torch.stack(xb).to(dev), torch.stack(lb).to(dev)
            y = target(x, l)
            w = torch.ones(l.shape, device=dev); w[x == OCC] = a.occ_weight; w[l == 1] = a.occ_weight * a.keep_weight
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                logit = G(onehot(x).contiguous(memory_format=torch.channels_last))
            ce = F.cross_entropy(logit.float(), y, ignore_index=255, reduction='none')
            m = y != 255
            loss = (ce * w)[m].sum() / w[m].sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        G.eval(); row = {'epoch': ep, 'loss': tot / a.steps, 'minutes': (time.time() - t0) / 60}
        if lab_va:
            s = [score2(clean(c, G, dev), c, l) for c, l, _s in lab_va]
            row.update(kept=float(np.mean([v['kept'] for v in s])), removed=float(np.mean([v['removed'] for v in s])), invented=float(np.mean([v['invented'] for v in s])))
            key = row['removed'] - 5 * max(0.0, 0.95 - row['kept'])          # clutter removed, with a penalty for structure lost below the mark
        else:
            key = -row['loss']
        log.append(row)
        if best is None or key > best:
            best = key
            torch.save({'model': G.state_dict(), 'width': a.width, 'args': vars(a), 'epoch': ep, 'row': row}, os.path.join(a.out, a.name + '.pt'))
        json.dump({'args': vars(a), 'log': log}, open(os.path.join(a.out, a.name + '.log.json'), 'w'), indent=1)
        print('ep %2d  loss %.4f | ' % (ep, row['loss']) + '  '.join(f'{k} {v:.3f}' for k, v in row.items() if k not in ('epoch', 'loss', 'minutes')) + ' | %.1f min' % row['minutes'], flush=True)
    print(f'{a.name}: done, best key {best:.4f}, {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
