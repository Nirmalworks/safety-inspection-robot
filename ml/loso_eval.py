#!/usr/bin/env python3
"""Leave-one-scene-out evaluation: every burst is scored by a model that never saw it.

    python3 ml/loso_eval.py <dataset> [--save-final results/probe_rgb_hha.pkl]

A scene is one burst (session, checkpoint, label): ~150 near-identical frames of one setup. With ~20
scenes, a single train/val/test split leaves 1-2 hazard scenes per split, and fine-tuning a whole CNN
on them memorises the backgrounds. So, for each scene in turn, every model is fitted on all OTHER
scenes and predicts this one; the scores are pooled over all scenes and also given per scene.

Task: hazard vs normal (blocked exit, obstructed walkway, open door all count as hazard).
Models (fixed settings, nothing tuned on the held-out scene):
  * Depth rule, fixed camera          keep_clear.py with the hand-set mount (pitch 0, 0.207 m)
  * Depth rule, per-frame floor       same rule, camera pitch/height from ml/ground.py on every frame
  * HOG + colour -> SVM               hand-crafted features (as in classic_baselines.py)
  * ResNet-18 RGB -> logistic reg.    frozen ImageNet ResNet-18, 512-d penultimate features, linear head
  * ResNet-18 HHA -> logistic reg.    same on the HHA encoding of depth (Gupta et al. 2014)
  * ResNet-18 RGB+HHA -> logistic reg. both feature vectors concatenated (1024-d)
Burst decision: a scene is called hazard when more than half of its frames are.
--save-final fits the RGB+HHA head on ALL scenes and saves it for ml/predict.py; --save-folds saves, for every
scene, the head trained WITHOUT it, so a demo can run a model that has never seen the burst it is shown.
Features are cached in <dataset>/results/features.npz.
"""
import argparse
import collections
import csv
import os
import pickle
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ame505_bringup'))
from ame505_bringup.keep_clear import check, load_config  # noqa: E402
from classic_baselines import hog  # noqa: E402
from ground import fit_floor_constrained, hha  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def backbone():
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    except ImportError:
        pass
    import torch
    import torchvision
    net = torchvision.models.resnet18(weights='DEFAULT')
    net.fc = torch.nn.Identity()
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return net.eval().to(dev), dev


def embed(net, dev, rgb_images, batch=64):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb_images), batch):
            x = np.stack([((cv2.resize(im, (224, 224)).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                          for im in rgb_images[i:i + batch]])
            out.append(net(torch.from_numpy(x).float().to(dev)).cpu().numpy())
    return np.concatenate(out)


def frame_inputs(dataset, r, config):
    bgr = cv2.imread(os.path.join(dataset, r['file_rgb']))
    depth = cv2.imread(os.path.join(dataset, r['file_depth']), cv2.IMREAD_UNCHANGED)
    floor = fit_floor_constrained(depth)
    zone = {**(config.get('zone') or {})}
    fixed = check(depth, zone, config.get('camera') or {}, config.get('min_area_m2', 0.01))
    auto = check(depth, zone, {'pitch_deg': floor['pitch_deg'], 'height': floor['height_m']}, config.get('min_area_m2', 0.01))
    grey = cv2.cvtColor(cv2.resize(bgr, (128, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    col = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    col /= col.sum() + 1e-9
    return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hha(depth, floor), np.concatenate([hog(grey), col]),
            int(fixed['hazard']), int(auto['hazard']), floor['pitch_deg'], floor['height_m'])


def features(dataset, rows, config):
    cache = os.path.join(dataset, 'results', 'features.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z['rgb']) == len(rows):
            return {k: z[k] for k in z.files}
    print(f'computing features for {len(rows)} frames (floor fit, HHA, HOG, ResNet-18) ...')
    t0 = time.time()
    rgbs, hhas, hogs, rf, ra, pitch, height = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        a, b, c, d, e, p, h = frame_inputs(dataset, r, config)
        rgbs.append(a), hhas.append(b), hogs.append(c), rf.append(d), ra.append(e), pitch.append(p), height.append(h)
        if i % 500 == 0:
            print(f'  {i}/{len(rows)}  {time.time() - t0:.0f} s')
    net, dev = backbone()
    out = {'rgb': embed(net, dev, rgbs), 'hha': embed(net, dev, hhas), 'hog': np.asarray(hogs, np.float32),
           'rule_fixed': np.asarray(rf), 'rule_auto': np.asarray(ra), 'pitch': np.asarray(pitch), 'height': np.asarray(height)}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f'features done in {time.time() - t0:.0f} s')
    return out


def learned(X):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))


def svm(X):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=0), SVC(C=1, class_weight='balanced'))


def frame_scores(y, p):
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    return {'precision': prec, 'recall': rec, 'f1': 2 * prec * rec / (prec + rec) if tp else 0.0,
            'false_alarm': fp / (fp + tn) if fp + tn else float('nan'), 'accuracy': (tp + tn) / len(y)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'src', 'ame505_bringup', 'config', 'inspection.yaml'))
    ap.add_argument('--save-final', help='path for the RGB+HHA linear head trained on all scenes')
    ap.add_argument('--save-folds', help='folder: one RGB+HHA head per scene, trained WITHOUT that scene (for honest demos)')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dataset, 'splits.csv'))))
    scene = np.array([f'{r["session"]}|{r["checkpoint"]}|{r["hazard"]}' for r in rows])
    y = np.array([0 if r['scene'] == 'normal' else 1 for r in rows])
    F = features(a.dataset, rows, load_config(a.config))
    print(f'floor fit: pitch median {np.median(F["pitch"]):+.1f} deg (IQR {np.percentile(F["pitch"], 25):+.1f} to '
          f'{np.percentile(F["pitch"], 75):+.1f}), height median {np.median(F["height"]):.3f} m')
    inputs = {'HOG + colour -> SVM': ('hog', svm),
              'ResNet-18 RGB -> logistic reg.': ('rgb', learned),
              'ResNet-18 HHA -> logistic reg.': ('hha', learned),
              'ResNet-18 RGB+HHA -> logistic reg.': ('rgb+hha', learned)}
    X = {'hog': F['hog'], 'rgb': F['rgb'], 'hha': F['hha'], 'rgb+hha': np.hstack([F['rgb'], F['hha']])}
    preds = {'Depth rule, fixed camera': F['rule_fixed'], 'Depth rule, per-frame floor': F['rule_auto']}
    scenes = sorted(set(scene))
    for name, (key, make) in inputs.items():
        p = np.zeros(len(y), int)
        t0 = time.time()
        for s in scenes:
            te = scene == s
            m = make(X[key]).fit(X[key][~te], y[~te])
            p[te] = m.predict(X[key][te])
            if a.save_folds and key == 'rgb+hha':
                os.makedirs(a.save_folds, exist_ok=True)
                with open(os.path.join(a.save_folds, s.replace('|', '__') + '.pkl'), 'wb') as f:
                    pickle.dump({'model': m, 'input': 'rgb+hha', 'classes': ['normal', 'hazard'], 'held_out_scene': s,
                                 'backbone': 'torchvision resnet18 IMAGENET1K_V1, fc removed'}, f)
        preds[name] = p
        print(f'  {name}: {len(scenes)} folds in {time.time() - t0:.0f} s')

    out = os.path.join(a.dataset, 'results')
    lines = ['| Model | Frame precision | Frame recall | Frame F1 | False-alarm rate | Bursts correct (vote) |',
             '|---|---|---|---|---|---|']
    per_scene = []
    for name, p in preds.items():
        s = frame_scores(y, p)
        ok = 0
        for sc in scenes:
            m = scene == sc
            vote = int(p[m].mean() > 0.5)
            truth = int(y[m][0])
            ok += vote == truth
            per_scene.append({'model': name, 'scene': sc, 'frames': int(m.sum()), 'truth': 'hazard' if truth else 'normal',
                              'hazard_fraction': f'{p[m].mean():.2f}', 'burst_decision': 'hazard' if vote else 'normal',
                              'correct': int(vote == truth)})
        lines.append(f'| {name} | {s["precision"]:.2f} | {s["recall"]:.2f} | {s["f1"]:.2f} | {s["false_alarm"]:.2f} | '
                     f'{ok} / {len(scenes)} |')
    n_h = len({sc for sc in scenes if y[scene == sc][0] == 1})
    lines.append(f'\nLeave-one-scene-out over {len(scenes)} bursts ({n_h} hazard, {len(scenes) - n_h} normal), '
                 f'{len(y)} frames ({int(y.sum())} hazard). Hazard = blocked exit, obstructed walkway or open door.')
    table = '\n'.join(lines)
    print('\n' + table)
    with open(os.path.join(out, 'loso_table.md'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(out, 'loso_per_scene.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(per_scene[0].keys()))
        w.writeheader()
        w.writerows(per_scene)
    best = 'ResNet-18 RGB+HHA -> logistic reg.'
    print(f'\nper-scene results of "{best}" (hazard fraction = share of the burst called hazard):')
    for r in per_scene:
        if r['model'] == best:
            print(f'  {"OK  " if r["correct"] else "MISS"} {r["scene"]:60s} truth {r["truth"]:6s} hazard fraction {r["hazard_fraction"]}')
    if a.save_final:
        m = learned(X['rgb+hha']).fit(X['rgb+hha'], y)
        with open(a.save_final, 'wb') as f:
            pickle.dump({'model': m, 'input': 'rgb+hha', 'backbone': 'torchvision resnet18 IMAGENET1K_V1, fc removed',
                         'classes': ['normal', 'hazard'], 'trained_on_scenes': scenes}, f)
        print(f'final model (all {len(scenes)} scenes): {a.save_final}')


if __name__ == '__main__':
    main()
