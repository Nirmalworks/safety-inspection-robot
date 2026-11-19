# Maps

Committed copies of the maps the robot uses, with what was placed on them: checkpoints, home, and landmarks.
The robot reads them from `$AME505_DATA` (the SSD). Use `scripts/backup_map.sh <name>` to copy SSD → here,
and `scripts/backup_map.sh <name> --restore` to copy here → SSD.

| Map | Made | Notes |
|---|---|---|
| `lab_new_maps` | 2026-11-13, slam_toolbox (lidar), 0.05 m/cell | The lab map used for the dataset. 4 identical tables in a row, so localize against the walls (see docs/DATA_COLLECTION.md). |
| `lab2d_11_18_2` | 2026-11-18, slam_toolbox (lidar), 0.05 m/cell, 24.7 × 10 m | Newest lab map. 3 checkpoints at the doorways: `cp_01` east door, `cp_02` south door, `cp_03` west end. Each faces its door. |
| `lab_common_clean_edit` | 2026-11-18, edited with `scripts/map_editor.py`, 0.05 m/cell | Hand-cleaned floor plan of the lab in the frame of `lab2d_11_18_2` (same home and the same 3 checkpoints): closed outline, 4 tables, cabinet, no chairs, doorways drawn closed. Started from `lab2d_11_18_2` + `lab_new_maps`. **Not yet driven on the robot.** |
