#!/usr/bin/env python3
"""Classify NEW checkpoint photos with a saved model, frame by frame and as a burst.

    python3 ml/predict.py <model> <image_rgb.png | folder> [--min-confidence 0.6] [--out preds.csv] [--annotate DIR]

<model> is either
  * a .pkl from ml/loso_eval.py --save-final (recommended): frozen ImageNet ResNet-18 features of the
    RGB image and of the HHA encoding of its depth image, then a logistic-regression head that outputs
    hazard / normal. Needs the matching *_depth.png next to every *_rgb.png (capture_dataset saves both).
  * a .pt from ml/train_cnn.py (fine-tuned CNN, RGB only, four classes; add --arch if not resnet18).
For every frame: the decision, its confidence, and "check" when the confidence is below
--min-confidence (re-capture or show to a person). For a folder it also prints the BURST decision
(majority of the frames), which is how the robot decides at a checkpoint.
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import CLASSES  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def images(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '**', '*_rgb.png'), recursive=True))
        return files or sorted(glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
                               + glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True))
    return [path]


def prep(rgb):
    return ((cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('path')
    ap.add_argument('--arch', default='resnet18', choices=['resnet18', 'mobilenet_v3_small'])
    ap.add_argument('--min-confidence', type=float, default=0.6)
    ap.add_argument('--out')
    ap.add_argument('--annotate')
    a = ap.parse_args()
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    except ImportError:
        pass
    import torch
    import torchvision
    from torch import nn

    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    probe = a.model.endswith('.pkl')
    if probe:
        from ground import hha
        bundle = pickle.load(open(a.model, 'rb'))
        head, labels = bundle['model'], bundle['classes']
        net = torchvision.models.resnet18(weights='DEFAULT')
        net.fc = nn.Identity()
    else:
        labels = CLASSES
        if a.arch == 'resnet18':
            net = torchvision.models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
        else:
            net = torchvision.models.mobilenet_v3_small(weights=None)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(CLASSES))
        net.load_state_dict(torch.load(a.model, map_location='cpu'))
    net = net.to(dev).eval()

    files = images(a.path)
    if not files:
        raise SystemExit(f'no images found in {a.path}')
    if a.annotate:
        os.makedirs(a.annotate, exist_ok=True)
    out = []
    with torch.no_grad():
        for f in files:
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if probe:
                dpath = f.replace('_rgb.png', '_depth.png')
                depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED) if dpath != f else None
                if depth is None:
                    print(f'skip (no depth image {os.path.basename(dpath)}): {f}')
                    continue
                x = torch.from_numpy(np.stack([prep(rgb), prep(hha(depth))])).float().to(dev)
                feat = net(x).cpu().numpy().reshape(1, -1)                 # [RGB 512 | HHA 512]
                p = head.predict_proba(feat)[0]
            else:
                p = torch.softmax(net(torch.from_numpy(prep(rgb)).unsqueeze(0).float().to(dev)), 1)[0].cpu().numpy()
            k = int(p.argmax())
            sure = p[k] >= a.min_confidence
            print(f'{labels[k]:20s} {p[k]:.2f}{"" if sure else "  check"}  {f}')
            out.append({'file_rgb': f, 'pred': int(k != 0), 'class': labels[k], 'confidence': f'{p[k]:.3f}',
                        'certain': int(sure)})
            if a.annotate:
                col = (0, 160, 0) if k == 0 else (0, 0, 200)
                if not sure:
                    col = (0, 140, 255)
                cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 26), (255, 255, 255), -1)
                cv2.putText(bgr, f'{labels[k]} {p[k]:.2f}{"" if sure else " (check)"}', (6, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.imwrite(os.path.join(a.annotate, os.path.basename(f)), bgr)
    if not out:
        raise SystemExit('nothing classified')
    per_class = {}
    for o in out:
        per_class[o['class']] = per_class.get(o['class'], 0) + 1
    hazard_share = np.mean([o['pred'] for o in out])
    print(f'\n{len(out)} frames: ' + ', '.join(f'{k} {v}' for k, v in sorted(per_class.items()))
          + f'; {sum(1 for o in out if not o["certain"])} below confidence {a.min_confidence}')
    if len(out) > 1:
        print(f'BURST DECISION: {"HAZARD" if hazard_share > 0.5 else "NORMAL"} ({100 * hazard_share:.0f} % of frames hazard)')
    if a.out:
        with open(a.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f'predictions: {a.out}')


if __name__ == '__main__':
    main()
