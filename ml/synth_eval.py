#!/usr/bin/env python3
"""Does cut-and-paste synthetic data (ml/cutpaste.py) raise hazard recall? Leave-one-scene-out, no leakage.

    python3 ml/synth_eval.py <dataset>      # after ml/loso_eval.py (features.npz) and ml/cutpaste.py (synthetic.csv)

For each held-out burst the head is trained on the real frames of the other bursts, plus (variant
"+ synthetic") the synthetic frames whose object source scene AND background target scene are both
training scenes, so nothing of the held-out burst enters training. Same frozen ResNet-18 RGB+HHA features
and logistic head as ml/loso_eval.py. Reports threshold 0.5 and a nested threshold (chosen by an inner
leave-one-scene-out on the training scenes to give >= 0.90 recall there).
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
from loso_eval import backbone, embed, frame_scores, learned  # noqa: E402


def synth_features(dataset):
    cache = os.path.join(dataset, 'results', 'features_synth.npz')
    rows = list(csv.DictReader(open(os.path.join(dataset, 'synthetic.csv'))))
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z['X']) == len(rows):
            return rows, z['X']
    rgb, hh = [], []
    for r in rows:
        rgb.append(cv2.cvtColor(cv2.imread(os.path.join(dataset, r['file_rgb'])), cv2.COLOR_BGR2RGB))
        d = cv2.imread(os.path.join(dataset, r['file_depth']), cv2.IMREAD_UNCHANGED)
        hh.append(hha(d, fit_floor_constrained(d)))
    net, dev = backbone()
    X = np.hstack([embed(net, dev, rgb), embed(net, dev, hh)])
    np.savez_compressed(cache, X=X)
    return rows, X


def fit_predict(X, y, scene, train_mask, test_mask, S=None, s_src=None, s_tgt=None, train_scenes=None):
    Xt, yt = X[train_mask], y[train_mask]
    if S is not None:
        ok = np.isin(s_src, train_scenes) & np.isin(s_tgt, train_scenes)
        Xt, yt = np.vstack([Xt, S[ok]]), np.concatenate([yt, np.ones(ok.sum(), int)])
    return learned(None).fit(Xt, yt).predict_proba(X[test_mask])[:, 1]


def run(X, y, scene, S=None, s_src=None, s_tgt=None, target=0.90):
    scenes = np.unique(scene)
    p, thr = np.zeros(len(y)), np.zeros(len(y))
    for s in scenes:
        te = scene == s
        tr_scenes = scenes[scenes != s]
        p[te] = fit_predict(X, y, scene, ~te, te, S, s_src, s_tgt, tr_scenes)
        inner = np.zeros((~te).sum())                           # inner LOSO on the training scenes only
        sc_in, y_in = scene[~te], y[~te]
        X_in = X[~te]
        for s2 in tr_scenes:
            te2 = sc_in == s2
            inner[te2] = fit_predict(X_in, y_in, sc_in, ~te2, te2, S, s_src, s_tgt, tr_scenes[tr_scenes != s2])
        cands = np.linspace(0.05, 0.5, 46)
        good = [t for t in cands if ((inner >= t) & (y_in == 1)).sum() / max((y_in == 1).sum(), 1) >= target]
        thr[te] = max(good) if good else cands[0]
    return p, thr


def row(name, y, scene, pred):
    s = frame_scores(y, pred)
    scenes = np.unique(scene)
    ok = sum(int(pred[scene == sc].mean() > 0.5) == int(y[scene == sc][0]) for sc in scenes)
    hz = [sc for sc in scenes if y[scene == sc][0] == 1]
    caught = sum(int(pred[scene == sc].mean() > 0.5) for sc in hz)
    nz = [sc for sc in scenes if y[scene == sc][0] == 0]
    false_b = sum(int(pred[scene == sc].mean() > 0.5) for sc in nz)
    return (f'| {name} | {s["precision"]:.2f} | {s["recall"]:.2f} | {s["f1"]:.2f} | {s["false_alarm"]:.2f} | '
            f'{caught} / {len(hz)} | {false_b} / {len(nz)} |')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dataset, 'splits.csv'))))
    scene = np.array([f'{r["session"]}|{r["checkpoint"]}|{r["hazard"]}' for r in rows])
    y = np.array([0 if r['scene'] == 'normal' else 1 for r in rows])
    F = np.load(os.path.join(a.dataset, 'results', 'features.npz'))
    X = np.hstack([F['rgb'], F['hha']])
    srows, S = synth_features(a.dataset)
    s_src = np.array([r['source_scene'] for r in srows])
    s_tgt = np.array([r['target_scene'] for r in srows])
    print(f'{len(y)} real frames in {len(np.unique(scene))} bursts; {len(S)} synthetic hazard frames '
          f'from {len(np.unique(s_src))} object bursts')
    lines = ['| Model (leave-one-scene-out) | Frame precision | Frame recall | Frame F1 | False-alarm rate | '
             'Hazard bursts caught | Normal bursts falsely flagged |', '|---|---|---|---|---|---|---|']
    for name, extra in (('RGB+HHA head, real data', None), ('RGB+HHA head, real + synthetic', (S, s_src, s_tgt))):
        t0 = time.time()
        p, thr = run(X, y, scene, *(extra or (None, None, None)))
        lines.append(row(name + ', threshold 0.5', y, scene, (p >= 0.5).astype(int)))
        lines.append(row(f'{name}, nested threshold (median {np.median(thr):.2f})', y, scene, (p >= thr).astype(int)))
        print(f'{name}: {time.time() - t0:.0f} s')
    table = '\n'.join(lines)
    print('\n' + table)
    with open(os.path.join(a.dataset, 'results', 'synth_eval.md'), 'w') as f:
        f.write(table + '\n')


if __name__ == '__main__':
    main()
