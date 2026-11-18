# Progress presentation: 5 slides, 5 minutes

Same numbers and status words as `PROGRESS_REPORT.md`. Figures are in `docs/progress/figures/`.

---

## Slide 1: Problem and objective (45 s)

**Autonomous Safety Inspection Robot**, Group 5: Leo Crampe · Nirmal Kumar Jagannathan · Ryan O'Gorman · Soowhan Yi

- Problem: blocked exits are a top OSHA 1910.37 violation, yet they are checked only on periodic manual walk-throughs
- Robot patrols 8–10 checkpoints and flags **blocked exit · obstructed walkway · open door**
- Input: RGB + depth frame at a checkpoint, robot pose, keep-clear zone
- Output: normal / hazard + type, confidence, annotated image, shown on a dashboard
- User: facilities / EHS staff review the flagged checkpoints

*Visual:* robot photo + one-line pipeline

## Slide 2: Approach and data (60 s)

*Visual:* `figures/map_lab_dark.png` (or `map_lab_3d.png`) next to the Figure 1 pipeline from the report, with each box coloured **operational / implemented / designed**

- Platform: JetAuto, Jetson Nano 4 GB, RPLidar A1, Astra RGB-D (320 × 240, 1–7 s latency → inspect only while stopped)
- Data obtained: **lidar map of the lab, 19.7 × 9.0 m** (`map_lab_dark.png`). **No RGB-D images yet.**
- Preparation completed: capture tool with a labelled `index.csv` (staging log), split by session + location (70/15/15, held-out checkpoint), collection protocol (~48 frames per checkpoint per session)

## Slide 3: Models and implementation (90 s)

| Component | Status |
|---|---|
| Depth keep-clear rule (baseline) | **Implemented**: 7 unit tests, synthetic run only |
| COCO-only YOLO (baseline) | Designed |
| Fine-tuned YOLO → TensorRT, 320 px | Designed |
| Checkpoint classifier: kNN / DT / SVM vs CNN | Designed |
| Map → AMCL → Nav2 → dashboard | **Operational** on the robot |
| Patrol → capture → rule → log | Implemented, first robot run pending |

- Key design decision: 3D camera SLAM → **2D lidar + AMCL** (camera relocalization failed: frames up to 7 s late, start pose off by 1.4 m)
- The depth rule cannot see an open door, so that hazard type needs the detector

## Slide 4: Progress evidence (60 s)

*Visual:* `pitch_sensitivity.png` (main) + `keep_clear_example.png` (small)

- Baseline pipeline runs end to end: 432 **synthetic** frames → split 288/72/72 → rule → metrics
  (F1 1.00 on synthetic test, which checks the code, not real accuracy)
- Finding: a **−2.5° camera tilt error turns 82 % of clear frames into false alarms** (−3°: 100 %)
  → before the real evaluation, add per-frame floor-plane fitting (or calibrate the pitch)
- Robot: AMCL + Nav2 on the real map; TF-timing failure diagnosed from logs and fixed
- **Map repeatability** (`map_repeatability.png`): two independent lidar maps of the lab (Nov 13, Nov 18),
  aligned by ICP: **median wall distance 7 cm, 84 % of walls within 20 cm** (moved furniture included)
- Nov 18 map (`map_lab_1118_dark.png`, 24.6 × 9.8 m) with the first 3 patrol checkpoints at the doorways
- Dashboard: live YOLO26 (COCO) at ~1.5 fps on the Nano CPU, URDF robot view, keyboard drive, E-STOP + command watchdog
- Limitation: no real-image results yet

## Slide 5: Next steps and questions (45 s)

- [Nov 20–21] checkpoints + first patrol with the camera
- [Nov 21–25] data sessions S1–S3 · [Nov 24] first real baseline result
- [Nov 30] YOLO fine-tuned + Jetson latency · [Dec 2] classifier comparison · [Dec 8] live evaluation S1–S5
- **Q1:** collection is one week late: is ~700 images (8 checkpoints, 2 sessions) enough, or should we drop a hazard type?
- **Q2:** open door: is a detector class acceptable instead of a zone check?
