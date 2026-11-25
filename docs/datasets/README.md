# Datasets

Images are not stored in git (public repo, no LFS). Each dataset is a zip attached to a GitHub
pre-release, and a copy is kept on the robot SSD at `$AME505_DATA/exports/`.

| Date | Release | Frames (RGB+depth pairs) | Labels |
|---|---|---|---|
| 2026-11-24 | [dataset-2026-11-24](https://github.com/Nirmalworks/safety-inspection-robot/releases/tag/dataset-2026-11-24) | 2,433 | cleaned; run2 cp_01/cp_04 possibly mislabelled |
| 2026-11-25 | [dataset-2026-11-25](https://github.com/Nirmalworks/safety-inspection-robot/releases/tag/dataset-2026-11-25) | 2,554 | **not verified**, see [2026-11-25.md](2026-11-25.md) |

Use with `python3 ml/dataset_summary.py <unzipped folder>` (see `ml/README.md`).
