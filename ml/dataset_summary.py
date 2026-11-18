#!/usr/bin/env python3
"""Count the dataset by checkpoint, scene, hazard type, light and occlusion (standard library only).

    python3 ml/dataset_summary.py [<dataset dir>]        # default: $AME505_DATA/dataset

Run it after each checkpoint during a session to see what is still missing, and use its tables
in the report.
"""
import collections
import csv
import os
import sys


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get('AME505_DATA', '/home/ubuntu/share/tmp/ame505_data'), 'dataset')
    rows = list(csv.DictReader(open(os.path.join(root, 'index.csv'))))
    print(f'{len(rows)} frames in {root}')
    for col in ('session', 'scene', 'hazard', 'light', 'occluded'):
        c = collections.Counter(r[col] for r in rows)
        print(f'  {col:9s} ' + '  '.join(f'{k}={v}' for k, v in sorted(c.items())))
    objs = collections.Counter(o for r in rows for o in r['objects'].split(',') if o)
    print('  objects   ' + '  '.join(f'{k}={v}' for k, v in objs.most_common()))
    print(f'\n{"checkpoint":10s} {"normal":>6s} {"hazard":>6s} {"low":>5s} {"occl":>5s} {"person":>6s}  hazard types')
    for cp in sorted({r['checkpoint'] for r in rows}):
        rs = [r for r in rows if r['checkpoint'] == cp]
        types = sorted({r['hazard'] for r in rs if r['scene'] == 'hazard'})
        print(f'{cp:10s} {sum(r["scene"] == "normal" for r in rs):6d} {sum(r["scene"] == "hazard" for r in rs):6d} '
              f'{sum(r["light"] == "low" for r in rs):5d} {sum(r["occluded"] == "1" for r in rs):5d} '
              f'{sum("person" in r["objects"] for r in rs):6d}  {",".join(types)}')
    bad = [r for r in rows if float(r.get('depth_valid_pct') or 0) < 50]
    if bad:
        print(f'\nWARNING: {len(bad)} frames have < 50 % valid depth (camera blocked or too close?)')


if __name__ == '__main__':
    main()
