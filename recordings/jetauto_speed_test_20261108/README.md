# JetAuto speed and hard-stop recordings (Nov 8 2026)

Two runs of straight-line driving at 0.2, 0.3 and 0.5 m/s, each ended by a hard stop, on
our mecanum JetAuto ("Robot Shazam") on an indoor laminate floor. Recorded for CSCI-513 and
copied here so the inspection-robot work can use the same measurements of how the robot moves.

| Run | Folder | Bag | Duration |
|---|---|---|---|
| 1 | [`run_0920/`](run_0920) | `bag/` (ROS 2, sqlite3, 14.3 MiB) | 58.8 s |
| 2 | [`run_0925/`](run_0925) | `bag/` (ROS 2, sqlite3, 9.5 MiB) | see `bag_info.txt` |

Topics: `/controller/cmd_vel`, `/imu`, `/scan`, `/scan_raw`, `/joy`, `/estop`,
`/ros_robot_controller/battery`, `/speed_test/event` (marks the start and end of each trial).

```bash
ros2 bag info recordings/jetauto_speed_test_20261108/run_0920/bag
ros2 bag play recordings/jetauto_speed_test_20261108/run_0920/bag
```

## Results (run 1 / run 2)

| Commanded | Measured speed | Braking distance | First motion after command |
|---|---|---|---|
| 0.2 m/s | 0.194 / 0.201 m/s | 5.0 / 4.4 cm | 0.16 / 0.17 s |
| 0.3 m/s | 0.291 / 0.307 m/s | 6.9 / 6.6 cm | 0.15 / 0.17 s |
| 0.5 m/s | 0.511 / 0.523 m/s | 10.9 / 9.7 cm | 0.16 / 0.15 s |

Braking starts 0.15–0.17 s after a stop command. Wi-Fi ping to the router: 0 % loss, p95 under 52 ms.

Each run folder has its own README with the full results, the analysis CSVs, graphs, logs,
and in `code/` the exact scripts and settings used to record and analyse it. Those READMEs
mention files in the CSCI-513 repo (`config/params.yaml`, `docs/HW_MEASUREMENTS.md`) that are
not part of this repository.
