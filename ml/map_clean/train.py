#!/usr/bin/env python3
"""Step 4: train the map cleaner on the synthetic pairs.

    python -m ml.map_clean.train --name unet_ce                      # (a) U-Net, cross-entropy only (the baseline)
    python -m ml.map_clean.train --name unet_gan --adv 0.05          # (b) the same U-Net as a conditional GAN

Loss (b) = cross-entropy + adv * adversarial term (least-squares GAN, patch discriminator that sees the dirty
map together with the clean map). Everything else is identical between (a) and (b): data, seed, U-Net, optimizer,
steps. The best checkpoint is chosen on the SYNTHETIC validation set by wall F1 with a 1-cell tolerance.
Writes <out>/<name>.pt (weights + settings) and <out>/<name>.log.json (one row per epoch).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import FREE, OCC, UNK
from .model import PatchD, UNet


def load(pattern, kinds=None):
    X, Y = [], []
    for p in sorted(glob.glob(pattern)):
        z = np.load(p)
        keep = np.ones(len(z['X']), bool) if kinds is None else np.isin(z['kind'], kinds)
        X.append(z['X'][keep]); Y.append(z['Y'][keep])
    return torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(Y))


def augment(x, y, g):
    """The same random quarter-turn and flip for the dirty map and its target (maps have no preferred direction)."""
    k = int(torch.randint(0, 4, (1,), generator=g))
    x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
    if torch.rand(1, generator=g) < 0.5:
        x, y = x.flip(2), y.flip(2)
    return x, y


@torch.no_grad()
def wall_prf(pred, tgt, tol=1):
    """Wall precision / recall counted with a tolerance of `tol` cells, summed over a batch. -> (tp_p, n_pred, tp_r, n_true)"""
    p, t = (pred == OCC).float()[:, None], (tgt == OCC).float()[:, None]
    k = 2 * tol + 1
    near_t, near_p = F.max_pool2d(t, k, 1, tol), F.max_pool2d(p, k, 1, tol)
    return float((p * near_t).sum()), float(p.sum()), float((t * near_p).sum()), float(t.sum())


@torch.no_grad()
def evaluate(G, X, Y, dev, bs=128):
    G.eval()
    s = np.zeros(4)
    inter, union, correct, n, ce = np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(dev).long(), Y[i:i + bs].to(dev).long()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            logit = G(F.one_hot(x, 3).permute(0, 3, 1, 2).float())
        logit = logit.float()
        ce += float(F.cross_entropy(logit, y, reduction='sum'))
        p = logit.argmax(1)
        s += wall_prf(p, y)
        for c in range(3):
            inter[c] += float(((p == c) & (y == c)).sum()); union[c] += float(((p == c) | (y == c)).sum())
        correct += float((p == y).sum()); n += y.numel()
    prec, rec = s[0] / max(1, s[1]), s[2] / max(1, s[3])
    return {'wall_precision': prec, 'wall_recall': rec, 'wall_f1': 2 * prec * rec / max(1e-9, prec + rec),
            'iou_free': inter[FREE] / max(1, union[FREE]), 'iou_wall': inter[OCC] / max(1, union[OCC]), 'iou_unknown': inter[UNK] / max(1, union[UNK]),
            'pixel_acc': correct / n, 'ce': ce / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--train', default='data/map_cleaning/pairs/v1/shard_*.npz')
    ap.add_argument('--val', default='data/map_cleaning/pairs/v1_val/shard_*.npz')
    ap.add_argument('--out', default='data/map_cleaning/models')
    ap.add_argument('--adv', type=float, default=0.0, help='weight of the adversarial term (0 = plain U-Net)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--w-occ', type=float, default=3.0, help='class weight of walls in the cross-entropy (walls are about 2 %% of cells)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='use only the first N training pairs (for quick tests)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr, Ytr = load(a.train)
    Xva, Yva = load(a.val)
    if a.limit:
        Xtr, Ytr = Xtr[:a.limit], Ytr[:a.limit]
    print(f'{a.name}: {len(Xtr)} training pairs, {len(Xva)} validation pairs, adv weight {a.adv}, device {dev}', flush=True)

    G = UNet(a.width).to(dev).to(memory_format=torch.channels_last)
    optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(optG, a.lr, total_steps=steps, pct_start=0.05)
    D = optD = None
    if a.adv > 0:
        D = PatchD().to(dev).to(memory_format=torch.channels_last)
        optD = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.999))
    wts = torch.tensor([1.0, a.w_occ, 1.0], device=dev)
    g = torch.Generator().manual_seed(a.seed)
    log, best, t0 = [], -1.0, time.time()
    n_params = sum(p.numel() for p in G.parameters())

    for ep in range(1, a.epochs + 1):
        G.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = {'ce': 0.0, 'adv_g': 0.0, 'd': 0.0}
        nb = len(Xtr) // a.bs
        for b in range(nb):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            x, y = augment(Xtr[idx], Ytr[idx], g)
            x, y = x.to(dev, non_blocking=True).long(), y.to(dev, non_blocking=True).long()
            xo = F.one_hot(x, 3).permute(0, 3, 1, 2).float().contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                logit = G(xo)
            logit = logit.float()
            ce = F.cross_entropy(logit, y, weight=wts)
            loss = ce
            if D is not None:
                yo = F.one_hot(y, 3).permute(0, 3, 1, 2).float()
                prob = logit.softmax(1)
                # discriminator: real pair -> 1, generated pair -> 0 (least squares)
                d_real, d_fake = D(xo, yo), D(xo, prob.detach())
                loss_d = 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
                optD.zero_grad(set_to_none=True); loss_d.backward(); optD.step()
                adv_g = ((D(xo, prob) - 1) ** 2).mean()               # generator: make the pair look real
                loss = ce + a.adv * adv_g
                tot['adv_g'] += adv_g.item(); tot['d'] += loss_d.item()
            optG.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            optG.step(); sched.step()
            tot['ce'] += ce.item()
        m = evaluate(G, Xva, Yva, dev)
        row = {'epoch': ep, 'train_ce': tot['ce'] / nb, 'train_adv_g': tot['adv_g'] / nb, 'train_d': tot['d'] / nb,
               'minutes': (time.time() - t0) / 60, **{'val_' + k: v for k, v in m.items()}}
        log.append(row)
        if m['wall_f1'] > best:
            best = m['wall_f1']
            torch.save({'model': G.state_dict(), 'width': a.width, 'args': vars(a), 'epoch': ep, 'val': m, 'params': n_params},
                       os.path.join(a.out, a.name + '.pt'))
        json.dump({'args': vars(a), 'n_train': len(Xtr), 'n_val': len(Xva), 'params': n_params, 'log': log}, open(os.path.join(a.out, a.name + '.log.json'), 'w'), indent=1)
        print(f"ep {ep:2d}  ce {row['train_ce']:.4f}  adv {row['train_adv_g']:.3f}  D {row['train_d']:.3f} | val wall P {m['wall_precision']:.3f} R {m['wall_recall']:.3f} "
              f"F1 {m['wall_f1']:.3f}  IoU free {m['iou_free']:.3f} wall {m['iou_wall']:.3f} unk {m['iou_unknown']:.3f} | {row['minutes']:.1f} min", flush=True)
    print(f'{a.name}: best validation wall F1 {best:.4f}, {n_params / 1e6:.1f} M parameters, {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
