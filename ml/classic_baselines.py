#!/usr/bin/env python3
"""Classical models on hand-crafted features, on the same split as the CNN.

    python3 ml/classic_baselines.py <dataset dir>            # needs splits.csv (ml/split.py)

Features per frame (feature extraction):
  * HOG on the grey image (edge directions = shape), 128 x 96 px
  * HSV colour histogram (8 x 4 x 4 bins)
  * depth geometry from the keep-clear rule: occupied area, point count, nearest distance, valid %
Pipeline: standardize -> PCA (dimensionality reduction, keeps 95 % of the variance) -> model.
Models: kNN, decision tree, SVM (RBF), logistic regression, random forest. Each model's
hyperparameters are chosen on the VALIDATION split (macro F1); the TEST split is scored once.
The non-learned depth rule is scored on the same test frames as the baseline.
Results: <dataset>/results/metrics.csv (+ confusion matrices). Needs: pip install scikit-learn
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ame505_bringup'))
sys.path.insert(0, HERE)
from ame505_bringup.keep_clear import check_checkpoint, load_config  # noqa: E402
from data import CLASSES, by_split, confusion, line, load, print_confusion, save_result, scores  # noqa: E402

def hog(grey, cell=8, bins=9, block=2):
    """Histogram of Oriented Gradients (Dalal & Triggs): unsigned gradient directions in `bins` bins per
    cell x cell pixels, L2-Hys normalized over block x block cells. Plain NumPy, so it does not depend
    on the OpenCV build (OpenCV 5 dropped cv2.HOGDescriptor from the Python package)."""
    g = grey.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    ch, cw = g.shape[0] // cell, g.shape[1] // cell
    b = np.minimum((ang / (180.0 / bins)).astype(int), bins - 1)
    hist = np.zeros((ch, cw, bins), np.float32)
    for k in range(bins):
        m = np.where(b == k, mag, 0.0)[:ch * cell, :cw * cell]
        hist[..., k] = m.reshape(ch, cell, cw, cell).sum((1, 3))
    out = []
    for i in range(ch - block + 1):
        for j in range(cw - block + 1):
            v = hist[i:i + block, j:j + block].ravel()
            v = v / np.sqrt((v ** 2).sum() + 1e-6)
            v = np.minimum(v, 0.2)
            out.append(v / np.sqrt((v ** 2).sum() + 1e-6))
    return np.concatenate(out)


def features(dataset, rows, config):
    X, rule = [], []
    for r in rows:
        bgr = cv2.imread(os.path.join(dataset, r['file_rgb']))
        depth = cv2.imread(os.path.join(dataset, r['file_depth']), cv2.IMREAD_UNCHANGED)
        grey = cv2.cvtColor(cv2.resize(bgr, (128, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        h = hog(grey)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        col = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        col /= col.sum() + 1e-9
        k = check_checkpoint(depth, config, r['checkpoint'])
        geo = [k['area_m2'], k['points'], k['nearest_m'] if k['nearest_m'] is not None else 5.0, k['valid_pct']]
        X.append(np.concatenate([h, col, np.asarray(geo, np.float32)]))
        rule.append(1 if k['hazard'] else 0)
    return np.asarray(X, np.float32), np.asarray(rule)


def models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    return {
        'kNN': (lambda p: KNeighborsClassifier(n_neighbors=p), [1, 3, 5, 7, 9], 'k'),
        'Decision tree': (lambda p: DecisionTreeClassifier(max_depth=p, class_weight='balanced', random_state=0),
                          [3, 5, 8, 12, None], 'max_depth'),
        'SVM (RBF)': (lambda p: SVC(C=p, kernel='rbf', class_weight='balanced'), [0.1, 1, 10, 100], 'C'),
        'Logistic regression': (lambda p: LogisticRegression(C=p, max_iter=2000, class_weight='balanced'),
                                [0.01, 0.1, 1, 10], 'C'),
        'Random forest': (lambda p: RandomForestClassifier(n_estimators=300, max_depth=p, class_weight='balanced',
                                                           random_state=0), [None, 8, 16], 'max_depth'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'src', 'ame505_bringup', 'config', 'inspection.yaml'))
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    config = load_config(a.config)
    sp = by_split(load(a.dataset))
    for s in ('train', 'val', 'test'):
        print(f'{s:5s} {len(sp[s]):5d} frames, classes {[CLASSES[c] for c in sorted({r["y"] for r in sp[s]})]}')
    if not sp['train'] or not sp['val'] or not sp['test']:
        raise SystemExit('need frames in train, val and test')
    print('extracting features (HOG + colour + depth geometry) ...')
    data = {s: features(a.dataset, sp[s], config) for s in sp}
    y = {s: np.array([r['y'] for r in sp[s]]) for s in sp}
    if len(set(y['train'])) < 2:
        raise SystemExit('train split has only one class')

    # baseline: the non-learned depth rule (hazard / normal only, so the multi-class columns are not meaningful)
    rs = scores((y['test'] != 0).astype(int), data['test'][1])
    rs['accuracy'] = float(((y['test'] != 0).astype(int) == data['test'][1]).mean())
    rs['macro_f1'] = float('nan')
    print('\nTEST (held-out checkpoint and groups)')
    print(line('Depth rule (baseline)', rs) + '   [hazard vs normal only]')
    save_result(a.dataset, 'Depth rule (baseline)', 'test', rs, 'zone area >= min_area_m2')

    for name, (make, grid, pname) in models().items():
        best = None
        for p in grid:
            clf = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=0), make(p))
            clf.fit(data['train'][0], y['train'])
            v = scores(y['val'], clf.predict(data['val'][0]))
            if best is None or v['macro_f1'] > best[0]['macro_f1']:
                best = (v, p, clf)
        v, p, clf = best
        pred = clf.predict(data['test'][0])
        t = scores(y['test'], pred)
        params = f'{pname}={p}; PCA {clf.named_steps["pca"].n_components_} comps; val macro F1 {v["macro_f1"]:.2f}'
        print(line(name, t) + f'   [{params}]')
        save_result(a.dataset, name, 'test', t, params, confusion(y['test'], pred))
        if name == 'SVM (RBF)':
            print_confusion(confusion(y['test'], pred), '  SVM confusion matrix (rows = true, columns = predicted)')
    print(f'\nresults: {os.path.join(a.dataset, "results", "metrics.csv")}   (table: python3 ml/results_table.py {a.dataset})')


if __name__ == '__main__':
    main()
