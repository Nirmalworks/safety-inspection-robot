# Maps

Committed copies of the maps the robot uses, with what was placed on them: checkpoints, home, and landmarks.
The robot reads them from `$AME505_DATA` (the SSD). Use `scripts/backup_map.sh <name>` to copy SSD → here,
and `scripts/backup_map.sh <name> --restore` to copy here → SSD.

| Map | Made | Notes |
|---|---|---|
| `lab_new_maps` | 2026-11-13, slam_toolbox (lidar), 0.05 m/cell | The lab map used for the dataset. 4 identical tables in a row, so localize against the walls (see docs/DATA_COLLECTION.md). |
