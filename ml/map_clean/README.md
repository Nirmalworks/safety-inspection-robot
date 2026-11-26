# Map cleaning: raw 2D lidar map in, clean map out

A U-Net (11.0 M parameters) trained on simulated (dirty, clean) map pairs, with and without an
adversarial term (pix2pix style). Maps are Nav2 trinary maps at 0.05 m per cell and are never resized.

All commands run from the repo root. Data and results live in `data/map_cleaning/` (git-ignored).

## Setup
```
python3 -m venv .venv && .venv/bin/pip install torch numpy scipy pillow matplotlib opencv-python-headless pyyaml tqdm scikit-image
```

## Clean a map (CPU is enough)
```
python -m ml.map_clean.clean_map IN.pgm OUT.pgm --model data/map_cleaning/models/unet_ce.pt
python -m ml.map_clean.clean_map IN.pgm OUT.pgm --model M.pt --guard strong_wall.png   # optional guardrail
```
`OUT.pgm` has the size of `IN.pgm`; a copy of `IN.yaml` is written next to it. The guardrail mask must be
on the same grid as the map (the one in `data/map_cleaning/vote/` is on the grid of `lab_vote.pgm`).

## Reproduce
| step | command | output |
|---|---|---|
| 1. vote baseline | `python -m ml.map_clean.vote --ref lab2d_11_24_8` | `vote/lab_vote.pgm`, `strong_wall.png`, `stack.npz`, `ALIGNMENT.md` |
| 2. real-noise harvest | `python -m ml.map_clean.harvest` | `noise/noise_library.npz`, `STATS.md` |
| 3. training pairs | `python -m ml.map_clean.make_pairs --n 6000 --out data/map_cleaning/pairs/v1` | 15,930 pairs (about 32 min on 16 cores) |
| 3. validation pairs | `python -m ml.map_clean.make_pairs --n 600 --seed 777 --out data/map_cleaning/pairs/v1_val` | 1,581 pairs |
| 3. ablation pairs | `python -m ml.map_clean.make_pairs --n 6000 --no-real-noise --out data/map_cleaning/pairs/v1_synth` | same worlds, no harvested patches |
| 4a. U-Net | `python -m ml.map_clean.train --name unet_ce` | `models/unet_ce.pt` |
| 4b. conditional GAN | `python -m ml.map_clean.train --name unet_gan --adv 0.05` | `models/unet_gan.pt` |
| 5. ablation model | `python -m ml.map_clean.train --name unet_ce_synth --train "data/map_cleaning/pairs/v1_synth/shard_*.npz"` | `models/unet_ce_synth.pt` |
| 5-6. scores | `python -m ml.map_clean.evaluate --hand PATH/lab_common_clean_edit.yaml` | `eval/tables.md`, `results.json`, figures |

Pass `--exclude NAME ...` to `vote` to keep evaluation maps out of the vote map and the noise library.

## Files
| file | what it does |
|---|---|
| `common.py` | map I/O, the three classes (free 0, occupied 1, unknown 2), alignment (coarse search + trimmed ICP), wall scores |
| `vote.py` | aligns the lab maps and takes a per-cell majority vote; writes the guardrail mask |
| `harvest.py` | cuts real defects (clutter, door leaks) out of the lab maps and measures how often they occur |
| `floorplans.py` | clean worlds: procedural lab rooms, KTH campus plans, HouseExpo houses |
| `simulate.py` | drives a front-180-degree lidar through a world and builds the dirty and the clean map |
| `make_pairs.py` | step 3: worlds to 256 x 256 training patches |
| `model.py` | U-Net generator and PatchGAN discriminator |
| `train.py` | step 4: cross-entropy, plus an optional adversarial term |
| `clean_map.py` | inference on a whole map with overlapping windows |
| `show_real.py`, `evaluate.py` | figures and scores on development data |

## Data sources
- KTH campus floor plans: Aydemir, Jensfelt, Folkesson, IROS 2012.
- HouseExpo: Li et al., 2019.
- Lab maps: this repo, `maps/*/maps_2d/`.
See `data/map_cleaning/SOURCES.md` for where each copy came from.
