# Hazard detection: dataset tools and the depth-rule baseline

Runs on a laptop with Python 3, NumPy, OpenCV and PyYAML. No ROS, no robot.

| Step | Command | What it does |
|---|---|---|
| Split | `python3 ml/split.py <dataset> --holdout cp_10` | writes `splits.csv`: whole (session, checkpoint) groups go to train / val / test (~70/15/15); the hold-out checkpoint is only ever in test |
| Baseline | `python3 ml/depth_baseline.py <dataset>` | runs the keep-clear depth rule on every frame, writes `pred_depth_rule.csv`, prints recall / precision / F1 / false alarms per split |
| Score a model | `python3 ml/evaluate.py <dataset> <predictions.csv> --split test` | same metrics for any predictions file with columns `file_rgb`, `pred` (1 = hazard) |
| COCO detector | `python3 ml/yolo_coco.py <dataset> --save-images --prelabels` | pretrained YOLO26n on every frame: `frames.csv` / `detections.csv` (class, confidence, depth distance), annotated images, CVAT YOLO 1.1 pre-labels. Install: `pip install ultralytics` (robot: `scripts/setup_yolo.sh`) |
| Classic models | `python3 ml/classic_baselines.py <dataset>` | HOG + colour histogram + depth-geometry features → standardize → PCA → kNN, decision tree, SVM, logistic regression, random forest; hyperparameters chosen on **val**, scored once on **test**, next to the depth rule. Needs `scikit-learn` |
| CNN | `python3 ml/train_cnn.py <dataset> [--arch mobilenet_v3_small] [--onnx]` | transfer learning from ImageNet (ResNet-18 default), class-weighted loss, best epoch on **val**, scored once on **test**; `--onnx` exports opset 13 for TensorRT on the Jetson. Needs `torch torchvision` (+ `onnx onnxscript` for `--onnx`) |
| Comparison table | `python3 ml/results_table.py <dataset>` | every model's test row from `results/metrics.csv` as a Markdown table (confusion matrices in `results/`) |
| Merge + review | `python3 ml/merge_datasets.py OUT --review <csv> --day DATE=folder ...` | several burst datasets in one folder (symlinks), labels fixed per burst, whole bursts per split |
| Leave-one-scene-out | `python3 ml/loso_eval.py <dataset> [--save-final M.pkl] [--save-folds DIR]` | depth rule, HOG+SVM and frozen ResNet-18 (RGB, HHA, both) scored on bursts they never saw |
| Predict | `python3 ml/predict.py <model.pkl or .pt> <image or folder>` | new photos: decision, confidence, "check" when unsure, burst decision |
| Tests | `python3 -m pytest ml/` | geometry of the rule, split rules, metrics, HOG, floor fit |
| Placeholder data | `python3 ml/fake_dataset.py /tmp/fake` | rendered floor / wall / box scenes in the real file layout, to try the commands above |

`<dataset>` is the folder with `index.csv` that `capture_dataset` writes (`$AME505_DATA/dataset`).

## The depth rule

`src/ame505_bringup/ame505_bringup/keep_clear.py`: every depth pixel becomes a 3D point in the robot
frame; the frontal area of the points inside the keep-clear zone and above the floor is added up;
at least `min_area_m2` (default 0.01 m², a 10 x 10 cm object) means hazard. Zones and the threshold
are in `src/ame505_bringup/config/inspection.yaml`, with per-checkpoint overrides.

Limits:
* It needs the robot at the taped checkpoint position, since the zone is fixed in the robot frame.
* It cannot detect an **open door**: nothing is inside the zone. That hazard needs the trained model.
* Camera numbers are assumed (Astra at 320x240: fx = fy = 285; mount 0.207 m high, level, from the
  URDF). A camera tilted by a few degrees makes the floor count as an object: set `camera.pitch_deg`.
  On the robot the node takes the intrinsics from `/depth_cam/rgb/camera_info`.
* So far it has only been run on rendered placeholder scenes. Its real accuracy is unknown.

## On the robot

`navigation_2d.launch.py use_camera:=true` (dashboard: tick **camera**, Start navigation) starts the
`inspection` node. At each patrol checkpoint it waits 3 s, takes a new RGB-D frame, runs the rule
and writes `$AME505_DATA/inspections/index.csv` plus the image pair. The dashboard's
**Inspection results** card lists the latest 20; results are also published on `inspection/result`.

## Burst datasets of 2026-11-24/25: hazard detection that generalizes to unseen bursts

**Data.** The two releases (`docs/datasets/`) hold 4,987 RGB-D frames in 31 bursts. Labels were typed
during capture and contradict each other in places (the same clear closed door is "blocked_exit" one
day and "normal" the next), so every burst was reviewed: `ml/labels/review_2026-11-24_25.csv` keeps,
relabels or excludes it, with the reason. Result: 3,354 frames in **21 bursts** (9 hazard, 12 normal).
A burst is one scene (~150 near-identical frames), so the unit of evaluation is the burst, not the frame.

**Method.**
* `ground.py`: per-frame floor fit (lowest-point seeds as in Zermas et al., ICRA 2017; RANSAC limited to
  near-horizontal planes at 0.10-0.35 m) and the HHA depth encoding (Gupta et al., ECCV 2014: disparity,
  height above the fitted floor, angle with gravity). No hand-set camera pitch is needed: the fit found
  the camera tilted about 4 deg up (median -3.9 deg), 14 ms per frame.
* Features: frozen ImageNet ResNet-18 (torchvision, last layer removed) on the RGB image and on the HHA
  image, 512 + 512 values; a logistic-regression head (C = 0.1, balanced classes) says hazard / normal.
  Off-the-shelf CNN features with a linear head (Razavian et al., CVPRW 2014) instead of fine-tuning:
  fine-tuning all layers on 11 training scenes memorised them (training loss 0.001, test hazard recall 0.01).
* Burst decision: hazard when more than half of the burst's frames are hazard.

**Evaluation: leave-one-scene-out** (`loso_eval.py`): each of the 21 bursts is predicted by models fitted
on the other 20; nothing is tuned on the held-out burst.

| Model | Frame precision | Frame recall | Frame F1 | False-alarm rate | Bursts correct |
|---|---|---|---|---|---|
| Depth rule, fixed camera (baseline) | 0.48 | 0.98 | 0.64 | 0.85 | 11 / 21 |
| Depth rule, per-frame floor | 0.48 | 0.98 | 0.64 | 0.85 | 11 / 21 |
| HOG + colour -> SVM | 0.51 | 0.32 | 0.40 | 0.24 | 13 / 21 |
| ResNet-18 RGB -> logistic reg. | 0.79 | 0.61 | 0.69 | 0.13 | 17 / 21 |
| ResNet-18 HHA -> logistic reg. | 0.70 | 0.60 | 0.65 | 0.21 | 16 / 21 |
| **ResNet-18 RGB+HHA -> logistic reg.** | **0.91** | **0.66** | **0.76** | **0.05** | **18 / 21** |

The depth rule does not improve with the per-frame floor (4 of 3,354 frames change): its false alarms
come from the exit doors, benches and chair bases that stand inside the fixed 0.5-1.5 m zone, not from
the camera pitch. The learned model separates "the door" from "something in front of the door".
Misses of the best model: the only open-door burst (no other open door to learn from), chairs and wires
in the cp_08 aisle (2026-11-25 run1_obstruction_walkway), and people walking in the cp_07 hallway
(53 % of frames called hazard; 39 frames below 0.6 confidence are flagged "check").

**Reproduce** (laptop, ~3 min; datasets from the GitHub releases, unzipped):
```bash
pip install numpy opencv-python-headless pyyaml scikit-learn torch torchvision certifi
python3 ml/merge_datasets.py data/merged --review ml/labels/review_2026-11-24_25.csv \
    --day 2026-11-24=<unzipped 11-24 folder> --day 2026-11-25=<unzipped 11-25 folder>
python3 ml/loso_eval.py data/merged --save-final data/merged/results/probe_rgb_hha.pkl \
    --save-folds data/merged/results/fold_models          # table: data/merged/results/loso_table.md
```

**Live demo** (a model that has never seen the burst it is shown):
```bash
F=data/merged/results/fold_models
python3 ml/predict.py $F/2026-11-25_run6__cp_06__blocked_exit.pkl data/merged/2026-11-25/run6/cp_06            # hazard, unseen place
python3 ml/predict.py $F/2026-11-25_run8_run3_clean__cp_05__none.pkl data/merged/2026-11-25/run8_run3_clean/cp_05  # normal: closed door
python3 ml/predict.py $F/2026-11-25_run5_open_door__cp_05__open_door.pkl data/merged/2026-11-25/run5_open_door/cp_05  # failure: open door
python3 ml/predict.py $F/2026-11-25_run1__cp_07__none.pkl data/merged/2026-11-25/run1/cp_07                    # hard: people walking
```
New photos: `python3 ml/predict.py data/merged/results/probe_rgb_hha.pkl <folder with *_rgb.png + *_depth.png>`.

## Train and compare (after a data session)

```bash
python3 ml/dataset_summary.py <dataset>                              # enough frames per class / checkpoint?
python3 ml/split.py <dataset> --holdout cp_03 cp_05 cp_08            # one held-out checkpoint per hazard type
python3 ml/classic_baselines.py <dataset>                            # depth rule + 5 classical models (~1 min)
python3 ml/train_cnn.py <dataset> --onnx                             # CNN (~5 min on a laptop GPU)
python3 ml/train_cnn.py <dataset> --no-pretrained                    # optional: what transfer learning buys
python3 ml/results_table.py <dataset>                                # table for the report and slides
```

Classes: `normal`, `blocked_exit`, `obstructed_walkway`, `open_door` (from `scene` / `hazard` in
`index.csv`; frames labelled `multiple` are skipped). Precision, recall, F1 and the false-alarm rate are
for hazard vs normal (the proposal's metrics); accuracy and macro F1 are over the four classes.
A hazard type that appears at only one checkpoint cannot be told apart from that location: give each
hazard type at least two training checkpoints.

## Not here yet

YOLO fine-tuning on CVAT box labels, and the detector inside the inspection node.
