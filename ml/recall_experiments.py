#!/usr/bin/env python3
"""Ways to raise hazard recall without new data, all scored leave-one-scene-out (as ml/loso_eval.py).

    python3 ml/recall_experiments.py <dataset>       # after ml/loso_eval.py (reuses results/features.npz)

Variants of the RGB+HHA linear head:
  * base                 frozen ResNet-18 RGB+HHA, threshold 0.5
  * + zone crop          also the features of the lower-centre crop (where the keep-clear zone is)
  * + flip augmentation  training set doubled with horizontally flipped images (test images unflipped)
  * DINOv2 ViT-S/14      a stronger frozen backbone (Oquab et al. 2023) on RGB, crop and HHA
  * nested threshold     per held-out scene, the threshold is chosen by an INNER leave-one-scene-out on the
                         training scenes only, as the highest one that still gives >= 0.90 recall there
Extra features are cached in results/features_extra.npz.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ground import fit_floor_constrained, hha  # noqa: E402
from loso_eval import MEAN, STD, frame_scores  # noqa: E402


def zone_crop(im):
    """Lower-centre part of the image: the floor and objects 0.5-2 m in front of the robot."""
    h, w = im.shape[:2]
    return im[int(0.35 * h):, int(0.15 * w):int(0.85 * w)]


def models():
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    except ImportError:
        pass
    import torch
    import torchvision
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    res = torchvision.models.resnet18(weights='DEFAULT')
    res.fc = torch.nn.Identity()
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    return {'resnet': res.eval().to(dev), 'dino': dino.eval().to(dev)}, dev


def embed(net, dev, ims, batch=64):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(ims), batch):
            x = np.stack([((cv2.resize(im, (224, 224)).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                          for im in ims[i:i + batch]])
            out.append(net(torch.from_numpy(x).float().to(dev)).cpu().numpy())
    return np.concatenate(out)


def extra_features(dataset, rows):
    cache = os.path.join(dataset, 'results', 'features_extra.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z['res_crop']) == len(rows):
            return {k: z[k] for k in z.files}
    print(f'computing extra features for {len(rows)} frames ...')
    t0 = time.time()
    rgb, hh = [], []
    for r in rows:
        bgr = cv2.imread(os.path.join(dataset, r['file_rgb']))
        d = cv2.imread(os.path.join(dataset, r['file_depth']), cv2.IMREAD_UNCHANGED)
        rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        hh.append(hha(d, fit_floor_constrained(d)))
    nets, dev = models()
    crop_rgb, crop_hha = [zone_crop(i) for i in rgb], [zone_crop(i) for i in hh]
    flip = lambda ims: [np.ascontiguousarray(i[:, ::-1]) for i in ims]  # noqa: E731
    out = {}
    for name, ims in (('crop', crop_rgb), ('crop_hha', crop_hha), ('flip_rgb', flip(rgb)), ('flip_hha', flip(hh)),
                      ('flip_crop', flip(crop_rgb)), ('flip_crop_hha', flip(crop_hha))):
        out['res_' + name] = embed(nets['resnet'], dev, ims)
    for name, ims in (('rgb', rgb), ('hha', hh), ('crop', crop_rgb)):
        out['dino_' + name] = embed(nets['dino'], dev, ims)
        print(f'  dino {name} done ({time.time() - t0:.0f} s)')
    np.savez_compressed(cache, **out)
    return out


def head():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))


def loso_proba(X, y, scene, Xaug=None):
    """Out-of-fold hazard probability per frame. Xaug: extra (flipped) copies of each frame for training."""
    p = np.zeros(len(y))
    for s in np.unique(scene):
        te = scene == s
        Xt, yt = X[~te], y[~te]
        if Xaug is not None:
            Xt, yt = np.vstack([Xt, Xaug[~te]]), np.concatenate([yt, yt])
        p[te] = head().fit(Xt, yt).predict_proba(X[te])[:, 1]
    return p


def nested_threshold(X, y, scene, target=0.90):
    """For each held-out scene: inner LOSO on the other scenes picks the threshold, then the outer model applies it."""
    p_out = np.zeros(len(y))
    thr_out = np.zeros(len(y))
    for s in np.unique(scene):
        te = scene == s
        inner = loso_proba(X[~te], y[~te], scene[~te])
        cands = np.linspace(0.05, 0.5, 46)
        ok = [t for t in cands if ((inner >= t) & (y[~te] == 1)).sum() / max((y[~te] == 1).sum(), 1) >= target]
        t = max(ok) if ok else cands[0]
        p_out[te] = head().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        thr_out[te] = t
    return p_out, thr_out


def summarize(name, y, scene, pred):
    s = frame_scores(y, pred)
    ok = sum(int(pred[scene == sc].mean() > 0.5) == int(y[scene == sc][0]) for sc in np.unique(scene))
    hz = [sc for sc in np.unique(scene) if y[scene == sc][0] == 1]
    caught = sum(int(pred[scene == sc].mean() > 0.5) for sc in hz)
    return (f'| {name} | {s["precision"]:.2f} | {s["recall"]:.2f} | {s["f1"]:.2f} | {s["false_alarm"]:.2f} | '
            f'{ok} / {len(np.unique(scene))} | {caught} / {len(hz)} |')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dataset, 'splits.csv'))))
    scene = np.array([f'{r["session"]}|{r["checkpoint"]}|{r["hazard"]}' for r in rows])
    y = np.array([0 if r['scene'] == 'normal' else 1 for r in rows])
    F = dict(np.load(os.path.join(a.dataset, 'results', 'features.npz')))
    E = extra_features(a.dataset, rows)
    base = np.hstack([F['rgb'], F['hha']])
    crop = np.hstack([F['rgb'], F['hha'], E['res_crop'], E['res_crop_hha']])
    crop_flip = np.hstack([E['res_flip_rgb'], E['res_flip_hha'], E['res_flip_crop'], E['res_flip_crop_hha']])
    dino = np.hstack([E['dino_rgb'], E['dino_hha'], E['dino_crop']])
    lines = ['| Variant (leave-one-scene-out) | Frame precision | Frame recall | Frame F1 | False-alarm rate | '
             'Bursts correct | Hazard bursts caught |', '|---|---|---|---|---|---|---|']
    runs = {}
    for name, X, aug in (('ResNet-18 RGB+HHA (base)', base, None),
                         ('+ zone crop', crop, None),
                         ('+ zone crop + flip augmentation', crop, crop_flip),
                         ('DINOv2 RGB+HHA+crop', dino, None),
                         ('DINOv2 + ResNet (all)', np.hstack([dino, crop]), None)):
        t0 = time.time()
        p = loso_proba(X, y, scene, aug)
        runs[name] = (X, aug, p)
        lines.append(summarize(name + ', threshold 0.5', y, scene, (p >= 0.5).astype(int)))
        print(f'{name}: {time.time() - t0:.0f} s')
    best = max(runs, key=lambda k: frame_scores(y, (runs[k][2] >= 0.5).astype(int))['f1'])
    X, _, _ = runs[best]
    p, thr = nested_threshold(X, y, scene)
    lines.append(summarize(f'{best}, nested threshold (median {np.median(thr):.2f})', y, scene, (p >= thr).astype(int)))
    table = '\n'.join(lines)
    print('\n' + table)
    with open(os.path.join(a.dataset, 'results', 'recall_experiments.md'), 'w') as f:
        f.write(table + '\n')


if __name__ == '__main__':
    main()
