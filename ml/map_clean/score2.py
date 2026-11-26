#!/usr/bin/env python3
"""Scoring that is fair to methods that redraw walls, with counters for what a tolerance could hide.

structure kept      share of hand-keep cells with an occupied output cell within 10 cm (2 cells); `exact` = on the same cell
clutter removed     share of hand-erase cells with no occupied output cell within 5 cm (1 cell); `exact` = that cell is not occupied
invented            occupied output cells with no raw occupied cell within 10 cm
added floor (m2)    free output cells that were unknown in the raw map
"""
import cv2
import numpy as np

from .common import FREE, OCC, UNK


def score2(out, raw, lab, res=0.05):
    k, e = (lab == 1) & (raw == OCC), (lab == 2) & (raw == OCC)
    d = cv2.distanceTransform((out != OCC).astype(np.uint8), cv2.DIST_L2, 5)
    dr = cv2.distanceTransform((raw != OCC).astype(np.uint8), cv2.DIST_L2, 5)
    return {'kept': float((d[k] <= 2.45).mean()), 'kept_exact': float((out[k] == OCC).mean()),
            'removed': float((d[e] > 1.45).mean()), 'removed_exact': float((out[e] != OCC).mean()),
            'invented': int(((out == OCC) & (dr > 2.45)).sum()), 'added_floor_m2': float(((out == FREE) & (raw == UNK)).sum() * res * res)}
