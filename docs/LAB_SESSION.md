# Lab session runbook: map, checkpoints, calibration, dataset S1

Step-by-step plan for one robot session on the JetAuto. Work top to bottom and do not skip a gate
(a **✔ check** line). Write measured values into the `Result` blanks and commit this file at the end.
Priorities, if time runs out: sections 1–4 are required; section 5 (dataset) is as much as time allows.

| # | Section | Time | Output |
|---|---|---|---|
| 1 | Update the robot to `dev` | 15 min | robot builds and runs the `dev` code |
| 2 | Re-map the lab and save checkpoints | 45 min | `maps_2d/lab2d.*`, `checkpoints/lab2d.yaml`, taped floor marks |
| 3 | Camera pitch calibration | 10 min | `camera: {pitch_deg, height}` in `config/inspection.yaml` |
| 4 | First patrol with the camera (+ bag) | 20 min | `inspections/index.csv`, one row per checkpoint |
| 5 | Dataset session S1 | 10–12 min per checkpoint | `dataset/index.csv`, ~60 frames per checkpoint |
| 6 | Bring the results home | 15 min | debug bundle in git, dataset on the shared Drive |

Rules for this session (also for anyone operating the robot from a terminal):

* Do not change Nav2 / AMCL parameters today. If navigation misbehaves, use the fallback in that section.
* Never commit `data/`, images, bags or `*.db3`. Only `debug_logs/`, `checkpoints`/maps via the debug
  bundle, `config/inspection.yaml` and this file go into git, on branch `dev`.
* The Nano has 4 GB RAM. Close editors and other heavy tools on the robot before starting navigation
  ([LOCALIZATION.md](LOCALIZATION.md), finding 5). Use the dashboard from a laptop browser.
* Never block a real fire exit; remove each staged object right after its burst. Avoid faces in images.

## Bring to the lab

Floor tape in two colours (robot marks, zone corners) · tape measure · 3+ boxes of different sizes,
a chair, a cart or bin, a bag · a teammate to act as "person" · laptop on the same Wi-Fi ·
charged battery (≥ 10.5 V) · Xbox controller.

**AprilTags: not for this dataset.** A tag visible in the images lets a model recognise the
checkpoint by its tag instead of by the hazard, and it breaks the unseen-location test (the
held-out checkpoint would still show a tag). Floor tape gives repeatable positions, and every
dataset row already stores the AMCL map pose. Keep tags out of the camera view at checkpoints.

---

## 1. Update the robot to `dev` (15 min)

The repository was renamed and its history rewritten on 2026-11-18. An old clone must be reset,
not merged. On the robot (inside the ROS 2 container):

```bash
cd ~/ame505-safety-inspection-robot 2>/dev/null || cd ~/safety-inspection-robot
git status --short                      # must be empty; if not: git stash
git remote set-url origin https://github.com/Nirmalworks/safety-inspection-robot.git
git fetch origin --prune
git checkout -B dev origin/dev          # dev = integration branch with all features
source scripts/env.sh && colcon build --symlink-install
./scripts/dashboard.sh restart          # dashboard: http://<robot-ip>:8505
```

New clone instead (if the old one is broken):
`git clone -b dev https://github.com/Nirmalworks/safety-inspection-robot.git ~/safety-inspection-robot`
(`env.sh` finds either folder name).

✔ check: `git log --oneline -1` shows the latest `dev` commit; the dashboard page loads; battery ____ V.

Quick facts (one line each, paste the output here):

```bash
cat /proc/device-tree/model; free -m | head -2
```
Result: ______________________

## 2. Re-map the lab and save checkpoints (45 min)

The current map `lab_map_new` has walls but almost no cells marked as free space (1 of 82,187),
so it is replaced by a fresh map named **`lab2d`**. Mapping practice: [LOCALIZATION.md](LOCALIZATION.md) §3.

1. Mark the patrol **home** spot with tape. Put the robot there, facing the route.
2. Second terminal, record the sensors for the whole mapping drive (lets us re-run the mapper offline):
   ```bash
   source scripts/env.sh && mkdir -p $AME505_DATA/bags && cd $AME505_DATA/bags
   ros2 bag record -o mapping_lab2d /scan /scan_raw /odom /odom_raw /imu /tf /tf_static
   ```
3. Dashboard: **Start mapping**, name `lab2d`. Drive slowly with LB held, close loops, and when you
   return to a place already mapped, face the same way as the first time (the lidar sees only the front 180°).
   Finish at home. **Stop & Save**. Stop the bag (Ctrl+C).
   ✔ check: the preview shows white floor and single sharp walls.
4. Dashboard: **Start navigation** on `lab2d`. ✔ check: lidar dots lie on the walls (if not: **Set pose**).
5. Choose **8–10 checkpoints** and assign hazard types so each type has at least 2–3 checkpoints:

   | id | Place (describe) | Hazard type staged here | Zone size if not default (x_max, half_width) |
   |---|---|---|---|
   | cp_01 | | blocked_exit | |
   | cp_02 | | obstructed_walkway | |
   | cp_03 | | open_door | |
   | cp_04 | | blocked_exit | |
   | cp_05 | | obstructed_walkway | |
   | cp_06 | | open_door | |
   | cp_07 | | blocked_exit | |
   | cp_08 | | obstructed_walkway | |
   | cp_09 | | (optional) | |
   | cp_10 | | (optional) | |

   **Held-out checkpoint (never used for training):** ______ (pick one blocked_exit or walkway checkpoint).

6. Drive to each spot in order, face the exit / corridor / door, and press **+ Checkpoint here** (or Xbox **Y**).
   At each one:
   * tape the robot's footprint (front corners of the base) on the floor;
   * tape the keep-clear zone corners: default **0.5–1.5 m in front of the robot, 0.8 m wide** (centred).
     If the real zone must differ, write it in the table and in `config/inspection.yaml` under `checkpoints:`.
7. Measure the narrowest doorway on the route: ______ m.

✔ check: `cat $AME505_DATA/checkpoints/lab2d.yaml` lists every checkpoint with x, y, yaw.

**Fallback** if navigation is unreliable today: skip section 4, and in section 5 drive each checkpoint
by hand with the Xbox controller, using the floor tape to line up. The dataset does not need Nav2.

## 3. Camera pitch calibration (10 min)

The depth rule is very sensitive to the assumed camera tilt: a 2.5° error makes 82 % of clear
frames false alarms (`docs/progress/figures/pitch_sensitivity.png`). Measure it once:

1. Dashboard: tick **camera**, then **Start navigation** again (or **Start driving**).
2. Put the robot on a checkpoint mark with **bare floor for ~2 m in front** (no objects, no people).
3. Capture calibration frames (they go to the dataset as normal frames with a note):
   ```bash
   cd $AME505_ROOT
   ros2 run ame505_bringup capture_dataset --session calib --checkpoint cp_01 --scene normal --note calib --n 5
   python3 ml/estimate_pitch.py $AME505_DATA/dataset/calib/cp_01/*_depth.png
   ```
4. Copy the printed numbers into `src/ame505_bringup/config/inspection.yaml`:
   ```yaml
   camera: {pitch_deg: <value>, height: <value>}
   ```
   Then restart navigation so the inspection node reloads the file.

Result: pitch ______ deg, height ______ m, spread ______ deg.
✔ check: the spread is below 0.5° and the height is within 2 cm of 0.207 m. If not, the floor was not
bare or the camera mount is loose: fix it and repeat.

## 4. First patrol with the camera (20 min)

This is the evidence for the progress report: the full chain map → patrol → capture → depth rule → log.

1. Navigation running with **camera** ticked (the inspection node starts with it).
2. Second terminal, record the patrol for offline replay:
   ```bash
   cd $AME505_DATA/bags && ros2 bag record -o patrol_lab2d_1 /scan /odom /tf /tf_static /amcl_pose \
       /patrol/at_checkpoint /inspection/result /depth_cam/rgb/image_raw /depth_cam/depth/image_raw /depth_cam/rgb/camera_info
   ```
3. Stage **one** hazard (a box inside one checkpoint's taped zone). Note which: ______.
4. Dashboard: **Patrol** (or `ros2 run ame505_bringup patrol --map lab2d --dwell 6`). Keep a hand near E-STOP.
5. Stop the bag when the patrol ends.

✔ check:
```bash
cat $AME505_DATA/inspections/index.csv
```
One row per checkpoint; the staged checkpoint says `hazard`, the others `normal`.

| Checkpoints reached | Correct decisions | False alarms | Missed | Notes (stalls, recoveries, E-STOP) |
|---|---|---|---|---|
| __ / __ | __ / __ | __ | __ | |

If a clear checkpoint says `hazard`: check its `area_m2` and `nearest_m` in the CSV, re-check the
pitch (section 3), and look at the saved image before changing `min_area_m2`.

## 5. Dataset session S1 (10–12 min per checkpoint)

At each checkpoint, line the robot up on its tape and run the bursts below (6 frames each; nudge the
robot a few cm or degrees between bursts). Replace `cp_03` / `blocked_exit` with the checkpoint and its
hazard type. `S=s1` is today's session.

```bash
cd $AME505_ROOT
S=s1; CP=cp_03; HZ=blocked_exit; cap() { ros2 run ame505_bringup capture_dataset --session $S --checkpoint $CP --n 6 "$@"; }

cap --scene normal                                                  # 1. clear, normal light
cap --scene normal --note "clear 2"                                 # 2. clear, robot nudged
cap --scene normal --objects box --note "outside zone"              # 3. object beside the zone -> still normal
cap --scene normal --objects person --note "person in zone"         # 4. person standing in the zone -> NOT a hazard
cap --scene hazard --hazard $HZ --objects box                       # 5. hazard, object 1
cap --scene hazard --hazard $HZ --objects chair                     # 6. hazard, object 2
cap --scene hazard --hazard $HZ --objects cart --note "position 2"  # 7. hazard, object 3, other position
cap --scene hazard --hazard $HZ --objects box,person --occluded 1   # 8. hazard partly hidden by a person
cap --scene normal --light low                                      # 9. lights off/half: clear
cap --scene hazard --hazard $HZ --objects box --light low           # 10. lights off/half: hazard
```

For **open_door** checkpoints: "normal" means the door is closed; the hazard bursts are the door
open at different angles (`--objects door --note "open 90deg"`, `"open 45deg"`, `"ajar"`).
Burst 4 (person) is what teaches the system that a person in the zone is not a blocked exit.

After each checkpoint:
```bash
python3 ml/dataset_summary.py          # counts per checkpoint, scene, light, occlusion, person
```
✔ check: about 60 new frames for that checkpoint, no "< 50 % valid depth" warning.

Target for S1: 8 checkpoints × 60 ≈ **480 frames**. Done: ______ frames, checkpoints ______.

## 6. Bring the results home (15 min)

```bash
cd $AME505_ROOT
python3 ml/dataset_summary.py > /tmp/s1_summary.txt
./scripts/save_debug_bundle.sh 8 "lab session: lab2d map, checkpoints, calibration, patrol, S1"
B=$(ls -d debug_logs/*/ | tail -1)
cp /tmp/s1_summary.txt $B && cp $AME505_DATA/inspections/index.csv $B/inspections_index.csv
git add debug_logs docs/LAB_SESSION.md src/ame505_bringup/config/inspection.yaml
git commit -m "Lab session: lab2d map + checkpoints, camera calibration, first patrol, dataset S1" && git push origin dev

# images and bags -> shared Google Drive (too big for git)
cd $AME505_DATA && tar czf ~/dataset_s1.tgz dataset inspections && tar czf ~/bags_s1.tgz bags
```
✔ check: the push succeeded; both archives are uploaded to the Drive folder.

## After the session (laptop, no robot)

| Step | Command | Gives |
|---|---|---|
| Split by session + location | `python3 ml/split.py <dataset> --holdout <held-out cp>` | `splits.csv` |
| Baseline on real data | `python3 ml/depth_baseline.py <dataset>` | first real recall / precision / F1 / false alarms |
| Counts for the report | `python3 ml/dataset_summary.py <dataset>` | dataset table |
| Labels | CVAT: exit door, open door, box, chair, cart, person | YOLO training set |
