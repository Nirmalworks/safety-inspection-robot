#!/usr/bin/env python3
"""Clean a saved 2D map with a trained model. Runs on CPU or GPU.

    python -m ml.map_clean.clean_map IN.pgm OUT.pgm --model data/map_cleaning/models/unet_gan.pt
    python -m ml.map_clean.clean_map IN.pgm OUT.pgm --model M.pt --guard strong_wall.png   # never remove these walls

The map is never resized: it is padded with "unknown", cleaned in overlapping 256 x 256 windows whose class
probabilities are averaged, and cropped back, so OUT has the size of IN and IN's .yaml stays valid
(a copy of it is written next to OUT with the image name changed).
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .common import OCC, PGM, UNK, pgm_to_cls
from .model import UNet

PATCH, STRIDE = 256, 128


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    G = UNet(ck['width']).to(dev)
    G.load_state_dict(ck['model'])
    return G.eval()


@torch.no_grad()
def clean(cls, G, dev='cpu', guard=None):
    """cls: H x W class map -> cleaned H x W class map. `guard`: bool mask of walls that must stay walls."""
    H, W = cls.shape
    pad = PATCH // 2
    Hp = int(np.ceil((H + 2 * pad - PATCH) / STRIDE)) * STRIDE + PATCH
    Wp = int(np.ceil((W + 2 * pad - PATCH) / STRIDE)) * STRIDE + PATCH
    big = np.full((Hp, Wp), UNK, np.uint8)
    big[pad:pad + H, pad:pad + W] = cls
    x = F.one_hot(torch.from_numpy(big).long(), 3).permute(2, 0, 1).float()
    acc, cnt = torch.zeros(3, Hp, Wp), torch.zeros(1, Hp, Wp)
    win = torch.hann_window(PATCH, periodic=False).clamp_min(0.05)
    win = (win[:, None] * win[None, :])[None]                    # trust the middle of each window more than its edges
    tiles = [(y, xx) for y in range(0, Hp - PATCH + 1, STRIDE) for xx in range(0, Wp - PATCH + 1, STRIDE)]
    for i in range(0, len(tiles), 16):
        batch = torch.stack([x[:, y:y + PATCH, xx:xx + PATCH] for y, xx in tiles[i:i + 16]]).to(dev)
        prob = G(batch).float().softmax(1).cpu()
        for (y, xx), p in zip(tiles[i:i + 16], prob):
            acc[:, y:y + PATCH, xx:xx + PATCH] += p * win
            cnt[:, y:y + PATCH, xx:xx + PATCH] += win
    out = (acc / cnt).argmax(0).numpy().astype(np.uint8)[pad:pad + H, pad:pad + W]
    if guard is not None:
        out[guard & (cls == OCC)] = OCC                          # guardrail: a wall nearly every map agrees on is never removed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--model', required=True)
    ap.add_argument('--guard', default='', help='PNG mask (same size as the map) of walls that must be kept')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    cls = pgm_to_cls(cv2.imread(a.inp, cv2.IMREAD_GRAYSCALE))
    guard = (cv2.imread(a.guard, cv2.IMREAD_GRAYSCALE) > 0) if a.guard else None
    out = clean(cls, load_model(a.model, dev), dev, guard)
    cv2.imwrite(a.out, PGM[out])
    y_in = os.path.splitext(a.inp)[0] + '.yaml'
    if os.path.exists(y_in):
        y = yaml.safe_load(open(y_in))
        y['image'] = os.path.basename(a.out)
        yaml.safe_dump(y, open(os.path.splitext(a.out)[0] + '.yaml', 'w'), default_flow_style=None, sort_keys=False)
    print(f'{a.inp} -> {a.out}: {int((out != cls).sum())} of {cls.size} cells changed ({dev})')


if __name__ == '__main__':
    main()
