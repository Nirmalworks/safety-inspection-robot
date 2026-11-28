#!/usr/bin/env python3
"""Classifier + "does this spot look normal?" anomaly score, stacked. Nested leave-one-scene-out.

    python3 ml/anomaly_eval.py <dataset> [--save-final M.pkl] [--save-folds DIR]   # after ml/loso_eval.py

Two scores per frame, both from the frozen ResNet-18 RGB+HHA features (ml/loso_eval.py, 1024 values,
L2-normalised here):
  p  classifier: logistic-regression head, hazard probability (what obstacles look like)
  d  anomaly:    mean cosine distance to the k = 5 nearest CLEAR frames of the training bursts (what this
                 place normally looks like). The nearest clear frames are almost always of the same spot,
                 so the reference is effectively per checkpoint. With checkpoints fixed on one map, the
                 reference can be restricted to that checkpoint's clear frames (--by-checkpoint).
Combination (no fitted weights): each score becomes its percentile among the OUT-OF-FOLD scores of the
training bursts (inner leave-one-scene-out), and the two percentiles are averaged with equal weight (--weight
0.5, fixed in advance). Percentiles against training scores work frame by frame on the robot. A learned
combiner (logistic stacking, Wolpert 1992) is also reported; it did worse on 21 bursts (AUC 0.860).
Threshold: highest value that gives >= --target recall on the out-of-fold training scores.
For each held-out burst everything above uses only the other bursts.
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loso_eval import frame_scores, learned  # noqa: E402

K = 5


def knn_distance(Xq, ref):
    if len(ref) == 0:
        return np.ones(len(Xq))
    sims = Xq @ ref.T
    k = min(K, ref.shape[0])
    return 1.0 - np.sort(sims, 1)[:, -k:].mean(1)


def references(X, y, cp, mask, by_checkpoint, cps=None):
    """Clear training frames; per checkpoint if asked (falls back to all clear frames)."""
    if not by_checkpoint:
        return lambda q_cp: X[mask & (y == 0)]
    return lambda q_cp: X[mask & (y == 0) & (cp == q_cp)] if (mask & (y == 0) & (cp == q_cp)).sum() >= K else X[mask & (y == 0)]


def scores_for(X, y, cp, train, test, by_checkpoint):
    clf = learned(None).fit(X[train], y[train])
    p = clf.predict_proba(X[test])[:, 1]
    ref = references(X, y, cp, train, by_checkpoint)
    d = np.zeros(test.sum())
    idx = np.nonzero(test)[0]
    for c in np.unique(cp[test]):
        sel = cp[idx] == c
        d[sel] = knn_distance(X[idx[sel]], ref(c))
    return p, d, clf


def stacker():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(class_weight='balanced')


def oof(X, y, scene, cp, mask, by_checkpoint):
    """Out-of-fold (p, d) for the frames in mask, by leave-one-scene-out among them."""
    p, d = np.zeros(len(y)), np.zeros(len(y))
    for s in np.unique(scene[mask]):
        te = mask & (scene == s)
        tr = mask & (scene != s)
        p[te], d[te], _ = scores_for(X, y, cp, tr, te, by_checkpoint)
    return p, d


def pick_threshold(prob, y, target):
    cands = np.linspace(0.02, 0.9, 89)
    ok = [t for t in cands if ((prob >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target]
    return max(ok) if ok else cands[0]


def row(name, y, scene, pred):
    s = frame_scores(y, pred)
    scenes = np.unique(scene)
    hz = [sc for sc in scenes if y[scene == sc][0] == 1]
    nz = [sc for sc in scenes if y[scene == sc][0] == 0]
    caught = sum(int(pred[scene == sc].mean() > 0.5) for sc in hz)
    flagged = sum(int(pred[scene == sc].mean() > 0.5) for sc in nz)
    return (f'| {name} | {s["precision"]:.2f} | {s["recall"]:.2f} | {s["f1"]:.2f} | {s["false_alarm"]:.2f} | '
            f'{caught} / {len(hz)} | {flagged} / {len(nz)} |')


def pct(v, ref_sorted):
    return np.searchsorted(ref_sorted, v) / len(ref_sorted)


def combine(p, d, p_ref, d_ref, w):
    return w * pct(p, p_ref) + (1 - w) * pct(d, d_ref)


def fit_bundle(X, y, scene, cp, mask, by_checkpoint, target, w=0.5):
    """Everything the deployed model needs, fitted on the frames in mask only."""
    p_in, d_in = oof(X, y, scene, cp, mask, by_checkpoint)
    p_ref, d_ref = np.sort(p_in[mask]), np.sort(d_in[mask])
    thr = pick_threshold(combine(p_in[mask], d_in[mask], p_ref, d_ref, w), y[mask], target)
    clf = learned(None).fit(X[mask], y[mask])
    keep = mask & (y == 0)
    keep[np.nonzero(keep)[0][1::2]] = False                     # every 2nd clear frame is enough as reference
    return {'kind': 'classifier+anomaly', 'classifier': clf, 'weight': w, 'p_ref': p_ref.astype(np.float32),
            'd_ref': d_ref.astype(np.float32), 'threshold': float(thr), 'k': K,
            'reference': X[keep].astype(np.float16), 'reference_checkpoint': cp[keep],
            'by_checkpoint': by_checkpoint, 'classes': ['normal', 'hazard'],
            'anomaly_normal_range': [float(np.percentile(d_in[mask & (y == 0)], 5)), float(np.percentile(d_in[mask & (y == 0)], 95))],
            'backbone': 'torchvision resnet18 IMAGENET1K_V1, fc removed; RGB + HHA, L2-normalised'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--target', type=float, default=0.90, help='recall target for the threshold (inside training)')
    ap.add_argument('--by-checkpoint', action='store_true', help='reference = clear frames of the same checkpoint id')
    ap.add_argument('--weight', type=float, default=0.5, help='weight of the classifier percentile')
    ap.add_argument('--save-final')
    ap.add_argument('--save-folds')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dataset, 'splits.csv'))))
    scene = np.array([f'{r["session"]}|{r["checkpoint"]}|{r["hazard"]}' for r in rows])
    cp = np.array([r['checkpoint'] for r in rows])
    y = np.array([0 if r['scene'] == 'normal' else 1 for r in rows])
    F = np.load(os.path.join(a.dataset, 'results', 'features.npz'))
    X = np.hstack([F['rgb'], F['hha']]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    scenes = np.unique(scene)
    p_o, d_o, prob, thr, comb, thr_c = (np.zeros(len(y)) for _ in range(6))
    thr_clf, thr_d = np.zeros(len(y)), np.zeros(len(y))
    for i, s in enumerate(scenes):
        te, tr = scene == s, scene != s
        p_in, d_in = oof(X, y, scene, cp, tr, a.by_checkpoint)                 # inner: training bursts only
        st = stacker().fit(np.c_[p_in[tr], d_in[tr]], y[tr])
        thr[te] = pick_threshold(st.predict_proba(np.c_[p_in[tr], d_in[tr]])[:, 1], y[tr], a.target)
        thr_clf[te] = pick_threshold(p_in[tr], y[tr], a.target)
        dn = (d_in - d_in[tr].min()) / (np.ptp(d_in[tr]) + 1e-9)
        thr_d[te] = pick_threshold(dn[tr], y[tr], a.target)
        p_o[te], d_o[te], _ = scores_for(X, y, cp, tr, te, a.by_checkpoint)      # outer
        prob[te] = st.predict_proba(np.c_[p_o[te], d_o[te]])[:, 1]
        p_ref, d_ref = np.sort(p_in[tr]), np.sort(d_in[tr])
        comb[te] = combine(p_o[te], d_o[te], p_ref, d_ref, a.weight)
        thr_c[te] = pick_threshold(combine(p_in[tr], d_in[tr], p_ref, d_ref, a.weight), y[tr], a.target)
        d_o[te] = (d_o[te] - d_in[tr].min()) / (np.ptp(d_in[tr]) + 1e-9)       # same scaling as its threshold
        if a.save_folds:
            os.makedirs(a.save_folds, exist_ok=True)
            b = fit_bundle(X, y, scene, cp, tr, a.by_checkpoint, a.target, a.weight)
            b['held_out_scene'] = s
            pickle.dump(b, open(os.path.join(a.save_folds, s.replace('|', '__') + '.pkl'), 'wb'))
        print(f'  fold {i + 1}/{len(scenes)} {s}: threshold {thr_c[te][0]:.2f}', flush=True)
    from sklearn.metrics import roc_auc_score
    lines = [f'| Model (nested leave-one-scene-out, threshold for {a.target:.2f} recall chosen inside training) | '
             'Frame precision | Frame recall | Frame F1 | False-alarm rate | Hazard bursts caught | Normal bursts flagged |',
             '|---|---|---|---|---|---|---|',
             row('Classifier alone, threshold 0.5', y, scene, (p_o >= 0.5).astype(int)),
             row('Classifier alone, nested threshold', y, scene, (p_o >= thr_clf).astype(int)),
             row('Anomaly (kNN to clear frames) alone, nested threshold', y, scene, (d_o >= thr_d).astype(int)),
             row('Classifier + anomaly, learned stacking, nested threshold', y, scene, (prob >= thr).astype(int)),
             row(f'**Classifier + anomaly, percentile average (w = {a.weight}), nested threshold**', y, scene,
                 (comb >= thr_c).astype(int))]
    lines.append(f'\nROC AUC (threshold-free): classifier {roc_auc_score(y, p_o):.3f}, anomaly {roc_auc_score(y, d_o):.3f}, '
                 f'learned stacking {roc_auc_score(y, prob):.3f}, percentile average {roc_auc_score(y, comb):.3f}. {len(scenes)} bursts, {len(y)} frames, '
                 f'reference = {"same checkpoint" if a.by_checkpoint else "all clear training frames (k = 5 nearest)"}.')
    table = '\n'.join(lines)
    print('\n' + table)
    out = os.path.join(a.dataset, 'results')
    with open(os.path.join(out, 'anomaly_eval.md'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(out, 'anomaly_per_frame.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_rgb', 'scene', 'truth', 'p_classifier', 'anomaly', 'combined', 'threshold'])
        for r, sc, t, a1, a2, a3, a4 in zip(rows, scene, y, p_o, d_o, comb, thr_c):
            w.writerow([r['file_rgb'], sc, t, f'{a1:.3f}', f'{a2:.3f}', f'{a3:.3f}', f'{a4:.2f}'])
    if a.save_final:
        b = fit_bundle(X, y, scene, cp, np.ones(len(y), bool), a.by_checkpoint, a.target, a.weight)
        pickle.dump(b, open(a.save_final, 'wb'))
        print(f'final model (all {len(scenes)} bursts, threshold {b["threshold"]:.2f}): {a.save_final}')


if __name__ == '__main__':
    main()
