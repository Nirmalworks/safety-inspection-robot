#!/usr/bin/env python3
"""Print the model comparison (results/metrics.csv) as a Markdown table for the report and slides.

    python3 ml/results_table.py <dataset dir> [--split test]
"""
import argparse
import csv
import os

COLS = [('model', 'Model'), ('n', 'Frames'), ('accuracy', 'Accuracy'), ('hazard_precision', 'Precision'),
        ('hazard_recall', 'Recall'), ('hazard_f1', 'F1'), ('false_alarm_rate', 'False-alarm rate'),
        ('macro_f1', 'Macro F1 (4 classes)'), ('params', 'Chosen on validation')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(a.dataset, 'results', 'metrics.csv'))) if r['split'] == a.split]
    rows.sort(key=lambda r: -float(r['hazard_recall']) if r['hazard_recall'] not in ('', 'nan') else 0)
    print('| ' + ' | '.join(h for _, h in COLS) + ' |')
    print('|' + '|'.join('---' for _ in COLS) + '|')
    for r in rows:
        print('| ' + ' | '.join('–' if r[k] in ('', 'nan') else r[k] for k, _ in COLS) + ' |')
    print(f'\n{a.split} split; precision / recall / F1 / false alarms are for "hazard" vs "normal". '
          'Targets: recall >= 0.90, F1 >= 0.85.')


if __name__ == '__main__':
    main()
