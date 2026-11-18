#!/usr/bin/env python3
"""Make printable ArUco markers (DICT_4X4_50) for the landmarks node.

    ros2 run ame505_bringup make_markers                 # ids 0-9 -> docs/markers/
    ros2 run ame505_bringup make_markers --ids 0-19 --out ~/markers

Print each PNG on A4 at 100 % ("actual size"), then MEASURE the black square and pass that
size to the landmarks node (marker_size, default 0.15 m). Tape markers flat and VERTICAL
(table sides, walls, cabinet doors) at camera height (~0.2-0.6 m), one id per place,
never the same id twice.
"""
import argparse
import os

import cv2
import numpy as np

DPI = 150
A4 = (int(8.27 * DPI), int(11.69 * DPI))      # px (w, h)


def page(i, size_m):
    side = int(size_m / 0.0254 * DPI)
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), i, side)
    pg = np.full((A4[1], A4[0]), 255, np.uint8)
    x0, y0 = (A4[0] - side) // 2, int(1.2 * DPI)
    pg[y0:y0 + side, x0:x0 + side] = mk
    for k, (txt, sc) in enumerate([(f'ID {i}', 3.0),
                                   (f'AME 505 landmark - DICT_4X4_50 - black square = {size_m * 100:.0f} cm', 0.9),
                                   ('Print at 100% (actual size). Mount VERTICAL, one id per place.', 0.9)]):
        cv2.putText(pg, txt, (int(0.6 * DPI), y0 + side + int((0.9 + 0.45 * k) * DPI)),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, 0, 4 if k == 0 else 2)
    return pg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ids', default='0-9', help='e.g. 0-9 or 3,5,7')
    ap.add_argument('--size', type=float, default=0.15, help='black square side in metres (max ~0.18 on A4)')
    ap.add_argument('--out', default=os.path.join(os.environ.get('AME505_ROOT', '.'), 'docs', 'markers'))
    a = ap.parse_args()
    ids = (range(int(a.ids.split('-')[0]), int(a.ids.split('-')[1]) + 1) if '-' in a.ids
           else [int(x) for x in a.ids.split(',')])
    os.makedirs(a.out, exist_ok=True)
    pages = []
    for i in ids:
        p = os.path.join(a.out, f'marker_{i:02d}.png')
        pg = page(i, a.size)
        cv2.imwrite(p, pg, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        pages.append(pg)
        print(p)
    try:                                   # one A4 PDF with the real physical size embedded
        from PIL import Image
        ims = [Image.fromarray(pg).convert('L') for pg in pages]
        pdf = os.path.join(a.out, 'markers_A4.pdf')
        ims[0].save(pdf, save_all=True, append_images=ims[1:], resolution=DPI)
        print(pdf, '<- print this at 100 % / actual size')
    except ImportError:
        print('(PIL not installed: no PDF; print the PNGs at 150 dpi)')


if __name__ == '__main__':
    main()
