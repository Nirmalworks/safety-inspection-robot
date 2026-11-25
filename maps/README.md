# Maps

Committed copies of the maps the robot uses, with what was placed on them: checkpoints, home, and landmarks.
The robot reads them from `$AME505_DATA` (the SSD). Use `scripts/backup_map.sh <name>` to copy SSD → here,
and `scripts/backup_map.sh <name> --restore` to copy here → SSD.

| Map | Made | Notes |
|---|---|---|
| `lab_new_maps` | 2026-11-13, slam_toolbox (lidar), 0.05 m/cell | The lab map used for the dataset. 4 identical tables in a row, so localize against the walls (see docs/DATA_COLLECTION.md). |
| `lab2d_11_18_2` | 2026-11-18, slam_toolbox (lidar), 0.05 m/cell, 24.7 × 10 m | Lab map. 3 checkpoints at the doorways: `cp_01` east door, `cp_02` south door, `cp_03` west end. Each faces its door. |
| `lab2d_11_23_1` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 23.9 × 9.3 m | Newest lab map, 105 m² of free floor. 8 checkpoints `cp_01`–`cp_08` around the four tables, each 0.61–0.84 m from the nearest obstacle. No home pose saved: AMCL starts at the mapping start pose (0, 0). |
| `23_11_map` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 6.1 × 7.3 m | Saved to SSD 2026-11-23. No checkpoints yet. |
| `lab2d` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 20.5 × 9.6 m | Saved to SSD 2026-11-23. No checkpoints yet. |
| `lab2d_11_23_3` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 24.7 × 10.0 m | Saved to SSD 2026-11-23. No checkpoints yet. |
| `lab2d_11_23_4` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 24.6 × 9.0 m | Saved to SSD 2026-11-23. No checkpoints yet. |
| `lab2d_11_23_5` | 2026-11-23, slam_toolbox (lidar), 0.05 m/cell, 39.7 × 17.2 m | Saved to SSD 2026-11-23. No checkpoints yet. |
| `lab2d_11_24_2` | 2026-11-24, slam_toolbox (lidar), 0.05 m/cell, 25.0 × 8.1 m | No checkpoints yet. |
| `lab2d_11_24_3` | 2026-11-24, slam_toolbox (lidar), 0.05 m/cell, 24.7 × 9.5 m | No checkpoints yet. |
| `lab2d_11_24_4` | 2026-11-24, slam_toolbox (lidar), 0.05 m/cell, 23.9 × 8.6 m | No checkpoints yet. |
| `lab2d_11_24_6` | 2026-11-24, slam_toolbox (lidar), 0.05 m/cell, 22.4 × 8.7 m | No checkpoints yet. |
| `lab2d_11_24_7` | 2026-11-24, slam_toolbox (lidar), 0.05 m/cell, 24.8 × 9.1 m | No checkpoints yet. |
| `lab2d_11_24_8` | 2026-11-24, slam_toolbox (lidar), 0.05 m/cell, 24.8 × 9.0 m | Newest map. No checkpoints yet. |
| `lab2d_11_25_2` | 2026-11-25, slam_toolbox (lidar), 0.05 m/cell, 23.6 × 12.1 m | No checkpoints yet. |
| `lab2d_11_25_3` | 2026-11-25, slam_toolbox (lidar), 0.05 m/cell, 8.7 × 24.1 m | No checkpoints yet. |
| `lab2d_11_25_4` | 2026-11-25, slam_toolbox (lidar), 0.05 m/cell, 23.7 × 9.6 m | No checkpoints yet. |
| `lab2d_11_25_5` | 2026-11-25, slam_toolbox (lidar), 0.05 m/cell, 23.7 × 12.1 m | 8 checkpoints `cp_01`–`cp_08`. |
| `lab2d_11_25_6` | 2026-11-25, slam_toolbox (lidar), 0.05 m/cell, 8.8 × 24.1 m | No checkpoints yet. |
| `lab2d_11_25_7` | 2026-11-25, slam_toolbox (lidar), 0.05 m/cell, 24.1 × 9.2 m | Newest map. No checkpoints yet. Rosbag in `recordings/mapping/lab2d_11_25_7_20261125_192436`. |
| `lab2d_11_18_3` | 2026-11-18, slam_toolbox (lidar), 0.05 m/cell, 18.3 × 10.9 m | 5 checkpoints, but all at the same pose (8.68, -1.48): recorded without moving the robot, so re-record before patrolling. |
| `lab_map_new` | 2026-11-09, 0.05 m/cell, 20.7 × 10.0 m | No checkpoints. |

## 3D and RTAB-Map data

Not restored by `backup_map.sh`; copy back by hand to `$AME505_DATA/maps_3d` and `$AME505_DATA/maps` if needed.

- `3d/` — point clouds and meshes (`.ply`) of `lab_map` and `lab_map_new`, 2026-11-09 and 2026-11-13.
- `rtabmap/` — RTAB-Map databases `lab_map.db` (2026-11-09) and `lab_map_new.db` (2026-11-12).
