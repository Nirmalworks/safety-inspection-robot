#!/usr/bin/env python3
"""Semantic2D (Xie et al., Temple / HKU, CC BY 4.0): its ready occupancy maps and their label images -> keep / erase labels.

    python -m ml.map_clean.semantic2d [--root data/map_cleaning/semantic2d/maps] [--out data/map_cleaning/semantic2d/pairs]

Each environment has a map (.pgm, 0.025 m per cell) and map_labelme/label.png, a colour image in which every labelled
region carries one of nine classes. Maps are brought to our 0.05 m grid (a 2 x 2 block is occupied if any of its cells
is, free if all are). Label of an occupied cell = the class under it:
  keep  (1): Wall, Pillar, Door, Elevator, Table, Sofa        erase (2): Chair, Person, Trash bin
  unsure (3): no class drawn there (background)
Writes <out>/<name>.npz (X classes, L labels, C class ids) and <name>_view.png, and prints a table.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml

from .common import FREE, OCC, UNK, colour, label

NAMES = ['background', 'Chair', 'Door', 'Elevator', 'Person', 'Pillar', 'Sofa', 'Table', 'Trash bin', 'Wall']
KEEP, ERASE = {'Wall', 'Pillar', 'Door', 'Elevator', 'Table', 'Sofa'}, {'Chair', 'Person', 'Trash bin'}


def voc_colour(i):
    """LabelMe / PASCAL VOC palette: class id -> (r, g, b)."""
    r = g = b = 0
    c = i
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j); c >>= 3
    return r, g, b


def decode(lab_bgr):
    out = np.zeros(lab_bgr.shape[:2], np.uint8)
    for i in range(1, len(NAMES)):
        r, g, b = voc_colour(i)
        out[(lab_bgr[..., 2] == r) & (lab_bgr[..., 1] == g) & (lab_bgr[..., 0] == b)] = i
    return out


def load_env(d):
    y = yaml.safe_load(open(glob.glob(d + '/*.yaml')[0]))
    pgm = cv2.imread(glob.glob(d + '/*.pgm')[0], cv2.IMREAD_GRAYSCALE)
    ld = [x for x in glob.glob(d + '/map_label*') if os.path.isdir(x)][0]
    cid = decode(cv2.imread(ld + '/label.png', cv2.IMREAD_COLOR))
    assert cid.shape == pgm.shape, (cid.shape, pgm.shape)
    k = int(round(0.05 / float(y['resolution'])))
    H, W = (pgm.shape[0] // k) * k, (pgm.shape[1] // k) * k
    blk = lambda a: a[:H, :W].reshape(H // k, k, W // k, k)
    occ, free = blk(pgm < 100).any((1, 3)), blk(pgm > 250).all((1, 3))
    cls = np.full(occ.shape, UNK, np.uint8); cls[free] = FREE; cls[occ] = OCC
    # class of a coarse cell: the most common non-background class among its occupied fine cells
    fine = blk(np.where(pgm < 100, cid, 0)).transpose(0, 2, 1, 3).reshape(H // k, W // k, k * k)
    C = np.zeros(occ.shape, np.uint8)
    for i in range(1, len(NAMES)):
        n_i = (fine == i).sum(2)
        C[(n_i > 0) & (n_i >= (fine == C[..., None]).sum(2) * (C > 0))] = i
    L = np.zeros(occ.shape, np.uint8)
    L[occ] = 3
    for i, n in enumerate(NAMES):
        if n in KEEP: L[occ & (C == i)] = 1
        if n in ERASE: L[occ & (C == i)] = 2
    return cls, L, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/map_cleaning/semantic2d/maps')
    ap.add_argument('--out', default='data/map_cleaning/semantic2d/pairs')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('| environment | size m | occupied cells | keep | erase | unsure | ' + ' | '.join(NAMES[1:]) + ' |\n|' + '---|' * (6 + len(NAMES) - 1))
    for d in sorted(glob.glob(a.root + '/*/*_map')):
        name = os.path.basename(d)[:-4]
        cls, L, C = load_env(d)
        np.savez_compressed(os.path.join(a.out, name + '.npz'), X=cls, L=L, C=C)
        occ = cls == OCC
        print(f"| {name} | {cls.shape[1] * 0.05:.0f} x {cls.shape[0] * 0.05:.0f} | {int(occ.sum())} | {int((L == 1).sum())} | {int((L == 2).sum())} | {int((L == 3).sum())} | "
              + ' | '.join(str(int((occ & (C == i)).sum())) for i in range(1, len(NAMES))) + ' |')
        ys, xs = np.nonzero(cls != UNK)
        s = (slice(max(0, ys.min() - 4), ys.max() + 5), slice(max(0, xs.min() - 4), xs.max() + 5))
        img = colour(np.where(occ, FREE, cls).astype(np.uint8)[s], 2)
        up = lambda m: cv2.resize(m[s].astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) > 0
        img[up(L == 1)] = (40, 40, 40); img[up(L == 3)] = (0, 150, 255); img[up(L == 2)] = (40, 40, 230)
        cv2.imwrite(os.path.join(a.out, name + '_view.png'), label(img, f'{name}: black keep, red erase, orange no class drawn'))


if __name__ == '__main__':
    main()
