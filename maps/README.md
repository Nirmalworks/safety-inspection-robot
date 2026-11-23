# Maps

Committed copies of the maps the robot uses, with what was placed on them: checkpoints, home, and landmarks.
The robot reads them from `$AME505_DATA` (the SSD). Use `scripts/backup_map.sh <name>` to copy SSD → here,
and `scripts/backup_map.sh <name> --restore` to copy here → SSD.

| Map | Made | Notes |
|---|---|---|
| `lab_new_maps` | 2026-11-13, slam_toolbox (lidar), 0.05 m/cell | The lab map used for the dataset. 4 identical tables in a row, so localize against the walls (see docs/DATA_COLLECTION.md). |
| `lab2d_11_18_2` | 2026-11-18, slam_toolbox (lidar), 0.05 m/cell, 24.7 × 10 m | Lab map. 3 checkpoints at the doorways: `cp_01` east door, `cp_02` south door, `cp_03` west end. Each faces its door. |
| `lab2d_11_23_1` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 23.9 × 9.3 m | Newest lab map, 105 m² of free floor. 8 checkpoints `cp_01`–`cp_08` around the four tables, each 0.61–0.84 m from the nearest obstacle. No home pose saved: AMCL starts at the mapping start pose (0, 0). |
