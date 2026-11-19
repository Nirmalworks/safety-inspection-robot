# Progress report draft: Autonomous Safety Inspection Robot (Group 5)

Draft text for the course template (`Project progress report temp-2026`), section by section.
Items in `[brackets]` must be filled in or confirmed by the team before submission.
Every number below comes from the repository or a command in this file. Synthetic results are
labelled as such and are not real results.

Figures: `docs/progress/figures/` (regenerate with `docs/progress/make_figures.py`, see the end of this file).

---

## Executive summary (≈150 words)

Exit routes must stay clear, but they are checked only during periodic manual walk-throughs. We
are building a robot that patrols 8–10 checkpoints in a campus lab and flags three staged hazards:
blocked exit, obstructed walkway and open door. It runs on a Hiwonder JetAuto (Jetson Nano 4 GB,
RPLidar A1, Astra RGB-D). A YOLO detector fine-tuned on our own RGB-D images, combined with a
depth-based keep-clear check and a checkpoint classifier, will make the per-checkpoint decision.
To date the navigation stack works on the robot: a 24.6 × 9.8 m lidar map with 3 checkpoints saved, AMCL localization,
Nav2, a patrol node and a web dashboard. The non-ML depth-rule baseline, the dataset capture tool,
the location/session split and the evaluation scripts are implemented and verified end to end on
synthetic frames. No real images have been collected yet. That is the main risk, and the first
real baseline result is due **[Nov 24]**. We request feedback on reducing the dataset scope.

## 1 Introduction and motivation

OSHA 29 CFR 1910.37 requires exit routes to be kept free of obstructions, and obstructed exit
routes are among the most frequently cited violations of that standard. In practice, exits and
walkways are checked on manual checklist walk-throughs, so a hazard created after an inspection
stays unnoticed until the next one. Facilities and EHS staff would benefit from a robot that
inspects the same checkpoints more often and in a repeatable way, then points staff to flagged
locations with an image as evidence.

ML is appropriate because hazards look different each time: boxes, chairs and carts of any shape,
under changing light and partial occlusion. A fixed geometric rule can tell that *something* is in
a zone, but not *what* it is. It also cannot recognise an open door, because nothing is inside the
zone in that case. This report covers the work since the proposal (Oct 20): the robot platform,
the baseline, the data pipeline, and the plan to collect data and train the models.

## 2 Problem definition and project objective

* **Practical problem and intended use:** Facilities and EHS staff run the robot between manual
  inspections. Afterwards they open the dashboard, review flagged checkpoints and their images,
  dismiss false alarms or fix the hazard. The robot is an extra check, not a replacement for an
  inspector.
* **Unit of analysis:** one checkpoint inspection: an aligned RGB and depth frame captured while
  the robot is stopped at a known checkpoint, plus the checkpoint ID and its keep-clear zone. For
  detector training, the unit is one labelled bounding box.
* **Inputs, outputs, and ML task:**
  - Inputs: RGB frame, aligned 16-bit depth frame (320 × 240), robot pose from AMCL, and the
    checkpoint's keep-clear zone.
  - Outputs: a normal or hazard decision with the hazard type, a confidence score, bounding boxes
    and an annotated image. One CSV row per inspection, shown on the dashboard.
  - ML task: supervised multi-class object detection (YOLO), followed by binary checkpoint
    classification (normal vs. hazard).
* **Success criterion:** checkpoint-level hazard recall ≥ 0.90 in scenarios S1–S3, ≤ 1 false
  alarm per patrol in S1, ≥ 80 % of patrols completed without help, and ≤ 200 ms inference per
  image on the Jetson. These are targets, not results.
* **Changes since the proposal:** The problem and objective are unchanged. Platform facts now
  known: the Jetson is a **Nano 4 GB**, and the camera arrives in ROS 2 at **320 × 240, 3–7 Hz,
  1–7 s late** (it is bridged from the vendor's ROS 1 container). Both are acceptable because
  inspection happens while the robot is stopped. The schedule has slipped by about one week (§7).

## 3 Approach taken and system design

**Figure 1** (pipeline, status marked):

```
 [OPERATIONAL]                 [OPERATIONAL]         [IMPLEMENTED, not yet run on robot]
 2D lidar map ──► AMCL ──► Nav2 patrol ──► stop at checkpoint ──► RGB-D capture (settle 3 s)
                                                                          │
              ┌───────────────────────────────────────────────────────────┤
              ▼                                                           ▼
 [IMPLEMENTED] depth keep-clear rule (baseline)        [DESIGNED] YOLO detector (TensorRT)
              │                                                           │
              └──────────────► [DESIGNED] checkpoint classifier ◄─────────┘
                                          │
                     [IMPLEMENTED] CSV log + images ──► [OPERATIONAL] web dashboard
```

* **Data preparation:**
  - `capture_dataset` (implemented) saves RGB and 16-bit depth pairs aligned to RGB, along with
    an `index.csv` row: session, checkpoint, scene, hazard type, objects, light, occlusion and
    the robot's map pose. That CSV is the staging log that provides checkpoint-level labels.
  - `ml/split.py` (implemented) splits whole (session, checkpoint) groups into
    train/val/test ≈ 70/15/15. A hold-out checkpoint is only ever in test. Frames are never split
    randomly, because frames from one burst are near-duplicates.
  - Collection protocol: `docs/DATA_COLLECTION.md`. Per checkpoint and session, about 48 frames:
    normal, low light, 3 hazard set-ups, an occluded hazard, and objects present but outside the
    zone (labelled normal).
  - Not done yet: CVAT boxes (exit door, open door, box, chair, cart), depth hole-filling and
    median filtering, and removal of blurred or duplicate frames.
* **Baseline:**
  - The non-ML depth rule (`keep_clear.py`) is **implemented**. It projects every depth pixel to
    the robot frame and sums the frontal area of points above the floor inside the keep-clear zone
    (default 0.5–1.5 m ahead, 0.8 m wide, OSHA ≥ 0.71 m). It flags a hazard at ≥ 0.01 m².
  - The COCO-only YOLO baseline (YOLO26n) **runs live on the robot** (~1.5 fps on the Nano CPU, dashboard view); it has not been evaluated yet.
  - The manual walk-through is used for qualitative comparison only.
* **Selected approach:**
  - **Detector:** YOLO (small, or nano if over 200 ms) fine-tuned from COCO, then exported to
    TensorRT. COCO already has chair, suitcase and similar classes, which reduces the labelling
    needed. A detector can identify an open door; the depth rule cannot.
  - **Checkpoint classifier:** kNN, decision tree and SVM on colour histograms, HOG, and depth
    occupancy in the zone, compared with a fine-tuned MobileNet or ResNet-18. This tests whether
    learned features beat hand-crafted ones.
  - **Considered and rejected:** reinforcement learning (not needed for fixed checkpoints) and
    inspection while moving (camera latency of 1–7 s).
* **System operation:** One dashboard (`scripts/dashboard.sh`, http://robot:8505) starts
  mapping, navigation, checkpoint saving and the patrol. With the camera enabled it also starts
  the inspection node. At each checkpoint the node waits 3 s, takes a new frame, runs the rule,
  saves the images, appends to `inspections/index.csv`, publishes `inspection/result`, and shows
  the result in the dashboard's *Inspection results* card. Offline, the `ml/` scripts run on a
  laptop without ROS, and `scripts/sim_env.sh` runs mapping, Nav2 and the patrol against a
  simulated base and lidar.
* **Changes since the proposal:**
  1. Localization moved from RTAB-Map (3D, camera) to **slam_toolbox + AMCL (2D lidar)**.
     Camera-based relocalization never worked: the bridged images arrive up to 7 s late, and the
     start pose was off by 1.4 m. See `docs/LOCALIZATION.md`.
  2. The detector input will be **320 px**, matching the camera stream.
  3. The rule cannot detect an open door, so that hazard type depends entirely on the detector.

## 4 Models developed and implementation status

| Model or component | Design and reason for using it | Current status | Evidence of completed work | Next step |
|---|---|---|---|---|
| Baseline: depth keep-clear rule | Geometric check of the depth points inside the zone. Shows whether ML adds value beyond geometry. | **Implemented**. Run on 432 synthetic frames only. Not evaluated on real data. | `src/ame505_bringup/ame505_bringup/keep_clear.py`, `ml/depth_baseline.py`; 7 unit tests pass (`pytest ml/test_ml.py`); Fig. 3 and Fig. 4 | Tune `min_area_m2` and camera pitch on the real validation split (first run **[Nov 24]**) |
| Baseline: COCO-only YOLO | Pretrained detector without fine-tuning. Shows the value of our own data. | **Operational, not evaluated** | Runs live on the robot camera (`yolo_live.py`, ~1.5 fps on CPU); class mapping planned (chair, suitcase, …) | Run on the first real session **[Nov 25]** |
| Main model: fine-tuned YOLO + TensorRT | Single-stage real-time detector. COCO pretraining reduces label needs. TensorRT for the Nano. | **Designed** | Classes, input size 320, split procedure defined | CVAT labels, then training **[Nov 28–30]**; measure Jetson latency |
| Checkpoint classifier | kNN / DT / SVM on HOG, colour and depth features vs. a CNN. The best one is used in the final system. | **Designed** | Feature list defined; depth features come from `keep_clear.check()` | Train on the real split **[Dec 2]** |
| System integration | Map → AMCL → Nav2 patrol → capture → rule → CSV + dashboard | **Partially operational** | Lidar map (Fig. 2); AMCL + Nav2 run on the robot on Nov 13 (`debug_logs/20261113_195519`); inspection node and dashboard card implemented; laptop simulation | Save the remaining checkpoints (3 of 8–10 done) and run the first full patrol with the camera **[Nov 20–21]** |

**Most important open decision:** how to handle the camera pitch. The rule assumes a known
camera tilt. Fig. 4 shows that a calibration error of −2.5° makes 82 % of clear frames false
alarms, and −3° makes 100 %. Options: (a) measure the pitch once per robot and store it in
`config/inspection.yaml`, or (b) fit the floor plane in each frame (RANSAC), as the proposal's
rule describes. The team prefers (b), with (a) as the fallback.

## 5 Evidence of progress and evaluation

**No experiments on real data have been run yet.** No labelled RGB-D images exist; collection
starts once the checkpoints are saved (§7). The completed artifacts are:

1. **Robot navigation on a real map (Fig. 2).** The 2D lidar map of the lab is 413 × 199 cells
   at 5 cm (a 20.7 × 10.0 m image); the mapped obstacles span 19.7 × 9.0 m. AMCL localization and Nav2 goals were run on it on Nov 13. One
   live failure was diagnosed from logs: AMCL fell behind under CPU load (Jetson load average
   18–19), so map→odom was 0.32 s old against a 0.3 s tolerance and the controller aborted. The
   TF tolerances were fixed. This shows that the robot can localize and receive goals on the map.
   It does not yet show repeatable patrol completion: no full patrol has been run. The Nov 13 map
   marked only 1 cell free, so the lab was re-mapped on **Nov 18** (`maps/lab2d_11_18_2`, 493 × 200
   cells, 24.6 × 9.8 m of mapped obstacles) and the first **3 doorway checkpoints** were saved on it
   (`map_lab_1118_light.png`). The two maps, aligned by ICP, agree to a **median wall distance of
   7 cm** (84 % of walls within 20 cm, moved furniture included; `map_repeatability.png`).
2. **End-to-end baseline pipeline on synthetic data (Fig. 3).** `ml/fake_dataset.py` renders
   432 depth frames in the real file layout (3 sessions × 8 checkpoints; 216 hazard, 216 normal,
   including 72 normal frames with an object *outside* the zone). `ml/split.py --holdout cp_08`
   gives 288 / 72 / 72 frames, with cp_08 only in test.

| Model or configuration | Data split or scenario | Metric and measured value | What the result indicates |
|---|---|---|---|
| Depth rule, calibrated camera | **Synthetic** test split, 72 frames (36 hazard) | Recall 1.00, precision 1.00, F1 1.00, false alarms 0/36 | The capture → split → rule → metrics pipeline runs end to end and the geometry is correct. **Not** a measure of real accuracy (noise-free boxes, no clutter). |
| Depth rule, camera pitch error −2.5° / −3.0° | **Synthetic**, all 432 frames | False-alarm rate 0.82 / 1.00 (recall stays 1.00) | The rule is very sensitive to camera tilt. Calibration or per-frame floor fitting is required before the real evaluation (Fig. 4). |
| Fine-tuned YOLO / classifiers | — | Not yet run | — |

**Evaluation readiness:**
- Baselines: the depth rule (implemented) and COCO-only YOLO.
- Metrics, implemented in `ml/evaluate.py`: recall, precision, F1, and false-alarm rate, broken
  down by hazard type, light and occlusion.
- Success criterion: recall ≥ 0.90, ≤ 1 false alarm per patrol.
- Failure cases to analyse: camera tilt, low light, occlusion, objects just outside the zone, and
  the robot stopping off its taped mark.
- Still open: whether offline metrics are reported per frame or per checkpoint. We propose per
  frame offline and per checkpoint in live patrols.

## 6 Questions and feedback requested

| Question or unresolved issue | Options considered / work already attempted | Team's current position | Feedback needed |
|---|---|---|---|
| Data collection starts about one week late. Is a smaller dataset acceptable? | (a) the full 1,000–2,000 images across 3 sessions; (b) about 700 images, 8 checkpoints, 2 sessions; (c) drop to two hazard types (the proposal's contingency) | (b), keeping all three hazard types and the held-out checkpoint | Is (b) sufficient for a credible model comparison, or should we cut a hazard type? |
| Open door: detector class or image-level classifier? | The depth rule cannot see it. A detector needs box labels for "open door"; the checkpoint classifier could learn it from image labels alone | Detector class ("open door" boxes in CVAT), with the classifier as a second opinion | Is detecting the door state acceptable as a substitute for a zone check for this hazard? |

## 7 Remaining work, schedule, and contributions

### Work remaining

| Task | Tangible completion evidence | Responsible member(s) | Target date | Dependency or fallback |
|---|---|---|---|---|
| Add the remaining checkpoints (3 of 8–10 saved), tape keep-clear zones, first patrol | `checkpoints/<map>.yaml`, patrol log, `inspections/index.csv` from one patrol | Nirmal (lead), all | [Nov 20–21] | Fallback: robot placed by hand at each checkpoint |
| Data collection S1–S3 | `dataset/index.csv` with counts per checkpoint, scene and light | Ryan (lead), all | [Nov 21–25] | Fallback: 2 sessions (§6) |
| Real depth-rule baseline and pitch calibration | Metrics on the val/test split | [Soowhan / Leo] | [Nov 24] | Needs S1 |
| CVAT labels, then YOLO fine-tune and COCO-only baseline | Labelled set, mAP@0.5, Jetson latency | Soowhan (lead), Ryan | [Nov 30] | Nano variant + 320 px if > 200 ms |
| Checkpoint classifiers (kNN / DT / SVM / CNN) | Comparison table on the same split | Ryan, Soowhan | [Dec 2] | |
| Integrate detector into the inspection node, annotated images on the dashboard | Live patrol with detector output | Nirmal | [Dec 4] | Offline bag replay as fallback |
| Final evaluation S1–S5 (≥ 5 patrols each), failure analysis | Recall, false alarms per patrol, confusion matrix | Leo (lead), all | [Dec 8] | |
| Final presentation, live demo, report | Slides, rehearsed 3-patrol demo, report | All | [final dates] | Recorded patrols as backup |

Next concrete milestone: the **first full patrol with the camera on**. It is reached when
`inspections/index.csv` holds one row per checkpoint from a single patrol run.

### Team contributions to date

| Member | Work completed and associated artifact | Next responsibility |
|---|---|---|
| Nirmal Kumar Jagannathan | Robot bring-up and ROS 2 workspace; URDF; 2D lidar mapping (slam_toolbox) and AMCL localization; Nav2 tuning for the Nano (timeouts, TF tolerances, no-spin BT); dashboard; dataset capture tool; documentation (commits 2026-11-09, 11-13) | Checkpoints, first patrol, detector integration |
| Soowhan Yi | Keep-clear depth rule, inspection node, dataset split and evaluation scripts with tests; dashboard E-STOP and results card; laptop simulation; RPP controller option; stuck recovery (14 commits, 2026-11-13) | YOLO fine-tuning, CNN classifier |
| Ryan O'Gorman | [fill in] | Data collection lead, CVAT, feature classifiers |
| Leo Crampe | [fill in] | Evaluation lead, metrics, failure analysis |
| [Shreyash, if listed as a member] | GTSAM lidar/gyro pose-graph SLAM experiments (`gtsam_experiments/`, 2026-11-12) | [ ] |

Integration and review: all code goes through feature branches into `dev` (the integration
branch), then into `main` by pull request with one reviewer. Every component uses the same
inspection-result format (checkpoint, time, result, area, file paths) and the same split file.

## 8 References and supporting materials

- Repository: https://github.com/Nirmalworks/safety-inspection-robot (public; branch `dev`)
- OSHA 29 CFR 1910.37; NIOSH school safety checklist; Redmon et al., YOLO (CVPR 2016);
  Lin et al., COCO (ECCV 2014); Ultralytics YOLO and Jetson guide; NVIDIA TensorRT; CVAT;
  scikit-learn metrics (as in the proposal)
- ROS 2 Humble, slam_toolbox, Nav2, AMCL; Hiwonder JetAuto drivers (vendor, not in the repo)
- Figures: `docs/progress/figures/map_lab_1118_light.png` (Fig. 2; slide version `map_lab_1118_dark.png`), `map_repeatability.png`, `keep_clear_example.png` (Fig. 3),
  `pitch_sensitivity.png` (Fig. 4)

### Reproducing the numbers

```bash
python3 ml/fake_dataset.py /tmp/fake && python3 ml/split.py /tmp/fake --holdout cp_08
python3 ml/depth_baseline.py /tmp/fake                     # metrics per split
python3 -m pytest ml/test_ml.py                            # 7 tests
python3 docs/progress/make_figures.py /tmp/fake            # figures + pitch sweep
```
