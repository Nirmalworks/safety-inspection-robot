# Dataset collection plan (AME 505 proposal, Group 5)

From the proposal (*Project_proposal_Group5*, Oct 20 2026):

* **1,000–2,000 RGB-D images** from **8–10 checkpoints** in campus hallways / labs
* scenes: **normal** and **staged hazards**: *blocked exit*, *obstructed walkway*, *open door*
* conditions: **normal and low light**, **partial occlusion**, **different days (sessions)**
* split **by location + session**, ≈ **70 / 15 / 15** train/val/test, **≥ 1 checkpoint never in training**
* later: 5 live-patrol scenarios × ≥ 5 patrols each (S1 none, S2 single, S3 multiple, S4 low light, S5 occlusion).
  That's evaluation, not this dataset.

## Session runbook on the map `lab_new_maps` (mapped 2026-11-13)

The map is in the repo at `maps/lab_new_maps/`. On a fresh robot or container, run
`scripts/backup_map.sh lab_new_maps --restore`.

1. **Free RAM**: `./scripts/free_memory.sh --editor`. Close VS Code Remote and use plain SSH.
   You need more than 800 MB available.
2. **Dashboard**: `./scripts/dashboard.sh`. Open `http://<robot-ip>:8505`.
3. **Map panel**: pick `lab_new_maps`, tick **camera**, then **Start navigation**.
4. **Localize**: put the robot on the taped home spot. Use **Set pose** until the lidar dots sit
   on the walls. The map has 4 identical tables in a row, and AMCL can pick the wrong one. Check
   the dots against the walls and the room ends, not only the tables.
5. **Home and checkpoints, once**: press **Set home here** at the start spot. Then drive (LB) to
   each of the 8–10 spots and press **+ Checkpoint here**. Ids are `cp_01`… in the order you add
   them.
   Good spots: the lidar sees a wall corner or a doorway, the camera looks at the exit, door or
   aisle from 1–2 m, and the spot is not in the middle of the table row (aliasing).
6. **Push the checkpoints right away**, so they are safe and on `dev`:
   ```bash
   scripts/backup_map.sh lab_new_maps && git add maps/lab_new_maps && git commit -m "lab_new_maps checkpoints" && git push origin dev
   ```
7. **Tape** the robot footprint at every checkpoint, and the corners of the keep-clear zone.
8. **Capture** at each checkpoint with the commands under *Collecting* below.

Constraints to check on every burst. The capture tool prints them, and they also go into `index.csv`:

| Check | Why | Fix |
|---|---|---|
| robot **still for ≥ 3 s** before capturing | the camera arrives 1–7 s late over the ROS 1 bridge | wait, then run the command |
| `pose map` in the printout (not `odom`) | the pose is the checkpoint's ground truth | navigation running + localized (step 4) |
| pose within ~0.3 m of the checkpoint | same view across sessions S1–S3 | park on the tape |
| `depth_valid_pct` > 50 in `index.csv` | the depth rule needs depth | no glass or black surfaces right in front; 0.6–4 m range |
| hazard object **inside** the taped zone → `--scene hazard`; outside → `--scene normal --note "outside zone"` | label = the rule | |
| each checkpoint: normal + hazard + low light + occluded | proposal conditions | table below |
| never block a real fire exit, no faces | proposal / safety | |

After the session, run the COCO detector on what you captured. Do this on a laptop, or on the
robot with the robot software stopped (`./scripts/free_memory.sh --robot`):
```bash
scripts/setup_yolo.sh                 # robot only, once; laptop: pip install ultralytics
python3 ml/yolo_coco.py $AME505_DATA/dataset --save-images --prelabels
```
It runs YOLO26n (the newest Ultralytics COCO model) on every frame and writes:
* `yolo_coco/frames.csv` and `detections.csv`: classes, confidence and distance from depth,
  next to your labels
* `annotated/`: copies of the images with the boxes drawn
* `cvat_prelabels/`: CVAT "YOLO 1.1" pre-labels

It takes about 0.6 s per frame on the Nano CPU.

## Presentation pictures

Dashboard header: type a note (e.g. `cp_02 blocked exit, box`) and press **📸 Snapshot**. It saves
the camera, depth, YOLO view, map (robot, plan, checkpoints) and 3D robot view as they are on screen,
plus `state.json` (pose, localization, detections), to `$AME505_DATA/presentation/<time>/`.
Take one at each checkpoint for normal and for hazard, one during a patrol drive, and one with YOLO on.
**Demo at checkpoint** (dashboard card, or `scripts/demo_checkpoint.sh cp_01 hazard blocked_exit box`):
drives to the checkpoint, turns YOLO on, holds still 4 s, captures the photos into the dataset
(session `demo` unless set) and saves a snapshot. **⏺ Record screen** records the dashboard tab and
streams it to the robot SSD every second (`$AME505_DATA/recordings/dashboard_<time>.webm`, download
link when you stop), so a long recording can't fill the browser's memory and a closed tab keeps what
was recorded. It needs the page opened as `http://localhost:8505` (SSH tunnel), because browsers block
screen capture on plain `http://<robot-ip>`.
Also take **phone photos / short videos of the robot itself** next to the staged hazard. The dashboard
cannot show the robot from the outside.

## Target: ~1,500 images in 3 sessions (by Wednesday 2026-11-18)

Per checkpoint, per session (≈ 50 images, ~10 min per checkpoint):

| Scene | Light | Setups × frames | Images |
|---|---|---|---|
| normal (nothing staged) | normal | 2 × 6 | 12 |
| normal | low | 1 × 6 | 6 |
| hazard: the checkpoint's main type, 3 different objects/positions | normal | 3 × 6 | 18 |
| hazard | low | 1 × 6 | 6 |
| hazard partially hidden (person/object in front) | normal | 1 × 6 | 6 |
| **total** | | | **~48** |

10 checkpoints × 48 ≈ **480 per session** → 3 sessions ≈ **1,450 images** (≈ 40 % normal / 60 % hazard).
With 8 checkpoints it's ≈ 1,150, still inside the proposal's range.

Assign hazard types so each type has ≥ 3 checkpoints, e.g. 4 exits (blocked exit), 3 corridors/aisles
(obstructed walkway), 3 doors (open door). Use **varied objects**: boxes of different sizes, chairs,
carts, bags, trash can, cables/tripod. Keep the label "hazard" only when the object is **inside the
keep-clear zone** (proposal Fig. 2; the team fixes its size, e.g. door width × ~1 m in front of the
door; OSHA 1910.37 needs exit routes ≥ 28 in / 0.71 m wide); also take
some "object present but OUTSIDE the zone" shots labeled **normal**. These teach the
classifier the difference, and they're the interesting cases for the depth rule.

Sessions: **S1** (Fri/Sat) daytime · **S2** (Sun/Mon) different time of day · **S3** (Tue) evening /
different object set. Low light = ceiling lights off/half, not pitch black.
Hold out one checkpoint (e.g. `cp_10`) from training in all sessions.

## Before collecting (once)

1. Make the 2D map and set the 8–10 checkpoints (`docs/MAPPING_NAVIGATION.md`). Use the
   checkpoint ids (`cp_01`…) as the `--checkpoint` names below.
2. At each checkpoint mark the robot's floor position with tape (so S1–S3 views match), and mark the
   keep-clear zone corners on the floor with tape (for labeling + the depth rule).
3. Permission from the lab/building manager; never block a real fire exit (proposal). Avoid faces.

## Collecting

Start the robot **with the camera**: dashboard → tick **camera** → *Start navigation* (records the
map pose too), or *Start driving*. Drive / navigate to the checkpoint, then on the robot (ssh or a
terminal; `source scripts/env.sh` first):

```bash
# normal scene, 6 frames
ros2 run ame505_bringup capture_dataset --session s1 --checkpoint cp_03 --scene normal --n 6
# hazard: box blocking the exit
ros2 run ame505_bringup capture_dataset --session s1 --checkpoint cp_03 --scene hazard \
    --hazard blocked_exit --objects box --n 6
# low light, occluded
ros2 run ame505_bringup capture_dataset --session s1 --checkpoint cp_03 --scene hazard \
    --hazard blocked_exit --objects chair --light low --occluded 1 --n 6
# object present but outside the keep-clear zone -> normal
ros2 run ame505_bringup capture_dataset --session s1 --checkpoint cp_03 --scene normal \
    --objects box --note "outside zone" --n 6
```

Between bursts, nudge the robot a few cm / a few degrees (Xbox LB) so frames aren't identical.

### Burst mode: "Burst here" at each checkpoint (main way to collect)

1. Dashboard: **▶ Start driving** (or navigation). The camera starts by itself when a burst begins.
2. Put the robot on a checkpoint (carry it or drive it), facing the way the checkpoint looks.
3. **Burst mode (dataset)** card: pick the checkpoint (`cp_01` ... `cp_08`), the session (one per
   staging, e.g. `run1_normal`, `run2_obstructed`), scene `normal` / `hazard` + hazard type, objects.
4. **📸 Burst here.** The robot stays at the spot and takes 120 photos (setting: photos) over 12
   positions: at the spot, turn ±8° and ±16°, slide left/right 5 cm, back 5 cm and combinations, then
   returns. ~2 minutes. Nudges are closed-loop on odometry and skipped if the lidar sees something
   within 25 cm that way; skipped photos are made up at the spot. When done, the checkpoint field moves
   to the next one. Move the robot there and press again. E-STOP, ■ Stop or driving by hand stops it.

Terminal: `SESSION=run2_obstructed SCENE=hazard HAZARD=obstructed_walkway OBJECTS=box scripts/burst_mode.sh here cp_03`.
No map/localization is needed; the pose in `index.csv` is the map pose when navigation is localized,
else odometry. Run log: `$AME505_DATA/bursts/<session>_<time>/run.log`.

**Automatic route** (card → "Automatic", or `scripts/burst_mode.sh route all`): Nav2 drives to every
checkpoint and does the same there. Localization is judged by the lidar (≥ 65 % of points within 10 cm
of a wall of the saved map; one table-aisle over scores ≤ 55 %) and AMCL jumps vs odometry. If it
doesn't fit, the robot locates itself on the whole map by scan matching and checks again.

**🧭 Auto-locate** (map toolbar) and navigation start-up find the robot on the whole saved map with the
lidar (`locate.py`, ~3-5 s on the Nano; on `lab2d_11_23_1` correct within 5 cm at 15 of 16 test poses),
so it no longer has to start at home. If it says another place looks similar, check that the red
lidar dots sit on the walls; otherwise use 📍 Set pose.

Each frame = `*_rgb.png` + `*_depth.png` (16-bit, mm, aligned to RGB) under
`data/ssd/dataset/<session>/<checkpoint>/`, and one row in `data/ssd/dataset/index.csv` with all labels +
robot pose. That CSV is the proposal's "staging log" (checkpoint-level ground truth).

Note: the Astra comes through Hiwonder's ROS 1 bridge at **320×240**, a few frames/s. Fine for
YOLO at imgsz 320 and for the depth rule. The tool waits for a NEW frame for each image.

## After each session

```bash
du -sh data/ssd/dataset; tail -3 data/ssd/dataset/index.csv
# copy to the shared Google Drive (images are too big for git):
tar czf ~/dataset_s1.tgz -C /home/ubuntu/share/tmp/ame505_data dataset
```
Then: CVAT labeling (exit door, open door, box, chair, cart, …) → YOLO format; split by
`session` + `checkpoint` columns of `index.csv` (never random).

## Checklist for Wednesday

- [ ] 2D map + 8–10 checkpoints saved, floor marks + keep-clear zones taped
- [ ] S1 ≈ 480 images · S2 ≈ 480 · S3 ≈ 480 (≥ 1,000 minimum)
- [ ] every checkpoint has normal + hazard + low light + occluded
- [ ] index.csv backed up to Drive with the images
- [ ] held-out checkpoint chosen and written down
