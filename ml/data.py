"""Shared dataset loading, labels and metrics for the model comparison (no ROS, no torch).

Labels come straight from the capture tool's index.csv: scene=normal -> 'normal', else the hazard type.
Every model is scored the same way and its test result is appended to <dataset>/results/metrics.csv,
so ml/results_table.py can print one comparison table.
"""
import csv
import os

import numpy as np

CLASSES = ['normal', 'blocked_exit', 'obstructed_walkway', 'open_door']
RESULT_FIELDS = ['model', 'split', 'n', 'accuracy', 'hazard_precision', 'hazard_recall', 'hazard_f1',
                 'false_alarm_rate', 'macro_f1', 'params']


def label(row):
    """Class index for a dataset row, or None if it cannot be used (e.g. hazard=multiple)."""
    name = 'normal' if row['scene'] == 'normal' else row['hazard']
    return CLASSES.index(name) if name in CLASSES else None


def load(dataset):
    """Rows of splits.csv (run ml/split.py first) with 'y' added; rows without a usable label are dropped."""
    path = os.path.join(dataset, 'splits.csv')
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found: run  python3 ml/split.py {dataset} --holdout <checkpoint ...>  first')
    rows = []
    skipped = 0
    for r in csv.DictReader(open(path)):
        y = label(r)
        if y is None:
            skipped += 1
            continue
        r['y'] = y
        rows.append(r)
    if skipped:
        print(f'note: {skipped} frames skipped (hazard label not in {CLASSES})')
    return rows


def by_split(rows):
    return {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}


def present_classes(rows):
    return sorted({r['y'] for r in rows})


def scores(y_true, y_pred):
    """Multi-class accuracy and macro F1, plus hazard-vs-normal precision / recall / F1 / false-alarm rate."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ht, hp = y_true != 0, y_pred != 0
    tp, fp, fn, tn = (ht & hp).sum(), (~ht & hp).sum(), (ht & ~hp).sum(), (~ht & ~hp).sum()
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    f1s = []
    for c in sorted(set(y_true) | set(y_pred)):
        ctp = ((y_true == c) & (y_pred == c)).sum()
        cp = (y_pred == c).sum()
        ct = (y_true == c).sum()
        f1s.append(2 * ctp / (cp + ct) if cp + ct else 0.0)
    return {'n': int(len(y_true)), 'accuracy': float((y_true == y_pred).mean()) if len(y_true) else float('nan'),
            'hazard_precision': float(prec), 'hazard_recall': float(rec), 'hazard_f1': float(f1),
            'false_alarm_rate': float(fp / (fp + tn)) if fp + tn else float('nan'),
            'macro_f1': float(np.mean(f1s)) if f1s else float('nan')}


def confusion(y_true, y_pred):
    m = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def print_confusion(m, title=''):
    used = [i for i in range(len(CLASSES)) if m[i].sum() or m[:, i].sum()]
    w = max(len(CLASSES[i]) for i in used)
    print(title or 'confusion matrix (rows = true, columns = predicted)')
    print(' ' * (w + 2) + ''.join(f'{CLASSES[i][:10]:>12}' for i in used))
    for i in used:
        print(f'{CLASSES[i]:>{w}}  ' + ''.join(f'{m[i, j]:>12}' for j in used))


def save_result(dataset, model, split, s, params='', cm=None):
    """Append (or replace) this model's row in <dataset>/results/metrics.csv."""
    out = os.path.join(dataset, 'results')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'metrics.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    rows = [r for r in rows if not (r['model'] == model and r['split'] == split)]
    rows.append({'model': model, 'split': split, 'params': params,
                 **{k: (f'{s[k]:.3f}' if isinstance(s[k], float) else s[k]) for k in RESULT_FIELDS if k in s}})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    if cm is not None:
        np.savetxt(os.path.join(out, f'confusion_{model.replace(" ", "_").replace("/", "-")}_{split}.csv'),
                   cm, fmt='%d', delimiter=',', header=','.join(CLASSES))
    return path


def line(name, s):
    return (f'{name:<24} acc {s["accuracy"]:.2f}  hazard P {s["hazard_precision"]:.2f}  R {s["hazard_recall"]:.2f}  '
            f'F1 {s["hazard_f1"]:.2f}  false alarms {100 * s["false_alarm_rate"]:.0f} %  macro F1 {s["macro_f1"]:.2f}')
