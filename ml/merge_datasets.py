#!/usr/bin/env python3
"""Merge burst datasets from several days into one training set, applying a label review.

    python3 ml/merge_datasets.py OUT --review ml/labels/review_2026-11-24_25.csv \\
        --day 2026-11-24=~/data/burst_dataset_2026-11-24 --day 2026-11-25=~/data/burst_dataset_2026-11-25

* each --day is an unzipped release folder (the one holding index.csv); OUT gets a symlink per day,
  so no image is copied
* the review file decides, per burst (day, session, checkpoint, label as captured): keep, relabel or
  exclude, and which split the burst goes to (train / val / test). A whole burst always stays in one
  split: its frames are near-duplicates of each other
* frames with less than --min-depth % valid depth are dropped (camera looking at the ceiling etc.)
Writes OUT/index.csv and OUT/splits.csv (same columns as capture_dataset + split), ready for
ml/classic_baselines.py and ml/train_cnn.py. Bursts missing from the review file stop the merge.
"""
import argparse
import collections
import csv
import os


def captured(r):
    return 'normal' if r['scene'] == 'normal' else r['hazard']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--day', action='append', required=True, help='DATE=folder with index.csv')
    ap.add_argument('--review', required=True)
    ap.add_argument('--min-depth', type=float, default=20.0)
    a = ap.parse_args()

    review = {}
    for r in csv.DictReader(open(a.review)):
        review[(r['day'], r['session'], r['checkpoint'], r['labelled_as'])] = r
    os.makedirs(a.out, exist_ok=True)
    rows, missing, counts = [], set(), collections.Counter()
    for spec in a.day:
        day, folder = spec.split('=', 1)
        folder = os.path.abspath(os.path.expanduser(folder))
        link = os.path.join(a.out, day)
        if not os.path.lexists(link):
            os.symlink(folder, link)
        for r in csv.DictReader(open(os.path.join(folder, 'index.csv'))):
            key = (day, r['session'], r['checkpoint'], captured(r))
            rv = review.get(key)
            if rv is None:
                missing.add(key)
                continue
            if rv['action'] == 'exclude':
                counts['excluded (review)'] += 1
                continue
            if float(r['depth_valid_pct'] or 0) < a.min_depth:
                counts[f'dropped (< {a.min_depth:.0f} % valid depth)'] += 1
                continue
            label = rv['label']
            r = dict(r, file_rgb=f'{day}/{r["file_rgb"]}', file_depth=f'{day}/{r["file_depth"]}',
                     session=f'{day}_{r["session"]}', scene='normal' if label == 'normal' else 'hazard',
                     hazard='none' if label == 'normal' else label, split=rv['split'],
                     note=(r['note'] + f' | review: {rv["action"]} from {captured(r)}' if rv['action'] == 'relabel' else r['note']))
            rows.append(r)
            counts[f'{rv["split"]} {label}'] += 1
    if missing:
        raise SystemExit('bursts not in the review file (add a row for each):\n  ' + '\n  '.join(map(str, sorted(missing))))
    fields = list(rows[0].keys())
    for name, sub in (('index.csv', [{k: v for k, v in r.items() if k != 'split'} for r in rows]), ('splits.csv', rows)):
        with open(os.path.join(a.out, name), 'w', newline='') as f:
            w = csv.DictWriter(f, [k for k in fields if name == 'splits.csv' or k != 'split'])
            w.writeheader()
            w.writerows(sub)
    print(f'{len(rows)} frames -> {a.out}/splits.csv')
    for k, v in sorted(counts.items()):
        print(f'  {k:40s} {v:5d}')
    bursts = collections.Counter((r['split'], 'normal' if r['scene'] == 'normal' else r['hazard'])
                                 for r in {(r['session'], r['checkpoint'], r['hazard']): r for r in rows}.values())
    print('bursts (independent scenes) per split and class:', dict(sorted(bursts.items())))


if __name__ == '__main__':
    main()
