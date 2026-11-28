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
| Synthetic data | `python3 ml/cutpaste.py <dataset>` then `python3 ml/synth_eval.py <dataset>` | RGB-D cut-and-paste hazard frames; leak-free leave-one-scene-out comparison with and without them |
| Classifier + anomaly | `python3 ml/anomaly_eval.py <dataset> [--save-final M.pkl] [--save-folds DIR]` | adds "does this place look normal?" (kNN distance to clear frames); nested leave-one-scene-out; saved models for `predict.py` |
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

### Raising recall (2026-11-28), all leave-one-scene-out

Frame-level hazard recall / false-alarm rate of the RGB+HHA head. "Nested threshold": chosen inside the
training scenes only (inner leave-one-scene-out) to give >= 0.90 recall there. "Synthetic": RGB-D
cut-and-paste (`cutpaste.py`, after Dwibedi et al. 2017), used in a fold only when both the object's and the
background's burst are training bursts (`synth_eval.py`).

| Labels | Variant | Recall | False alarms | Hazard bursts caught | Normal bursts flagged |
|---|---|---|---|---|---|
| 21 bursts (people-only bursts excluded) | real, threshold 0.5 | 0.66 | 0.05 | 7 / 9 | 1 / 12 |
| | real, nested threshold | 0.85 | 0.34 | 8 / 9 | 3 / 12 |
| | real + 414 synthetic, threshold 0.5 | 0.86 | 0.40 | 8 / 9 | 4 / 12 |
| | real + synthetic, nested threshold | 0.89 | 0.57 | 8 / 9 | 6 / 12 |
| 28 bursts (person in the zone = normal) | real, threshold 0.5 | 0.40 | 0.14 | 4 / 9 | 2 / 19 |
| | real + 650 synthetic, nested threshold | 0.80 | 0.52 | 8 / 9 | 9 / 19 |

Findings: (1) adding people-only bursts as "normal" lowers recall (0.66 -> 0.40) because several hazard
bursts also show a person next to the obstacle, so the model learns "person -> normal"; (2) cut-and-paste
and the safety threshold both trade false alarms for recall; neither reaches 0.90 recall at an acceptable
false-alarm rate; (3) the usable cut-outs are mostly two objects (a grey trash can, a bin), so the synthetic
set is not diverse. Zone crop, flip augmentation and DINOv2 features did not help (`recall_experiments.py`).
The limit is the number of distinct hazard scenes (9), not the model. The camera's depth and RGB images are
not registered (offset about -14 px horizontally, +10 to +40 px vertically): enable registration in the
Astra driver before the next collection.

### Add-on: "does this checkpoint look normal?" (anomaly score), 2026-11-28

The robot knows which checkpoint it is at (map + localization), so besides "does this look like an
obstacle?" (classifier) it can ask "does this place look like it normally does?". `anomaly_eval.py`:
anomaly = mean cosine distance of the frame's RGB+HHA features to the 5 nearest CLEAR training frames (kNN;
the nearest ones are almost always of the same spot). Both scores become percentiles among the out-of-fold
training scores and are averaged with equal weight (fixed in advance; 0.3 / 0.7 give AUC 0.89 / 0.94).
Nested leave-one-scene-out, 21 bursts, threshold for 0.90 recall chosen inside training:

| Model | ROC AUC | Recall | False alarms | Hazard bursts caught | Normal bursts flagged |
|---|---|---|---|---|---|
| Classifier alone | 0.874 | 0.89 | 0.40 | 9 / 9 | 4 / 12 |
| Anomaly alone | 0.794 | 0.88 | 0.41 | 8 / 9 | 5 / 12 |
| Learned stacking (logistic on both scores) | 0.860 | 0.88 | 0.39 | 8 / 9 | 5 / 12 |
| **Classifier + anomaly, percentile average** | **0.948** | 0.80 | **0.27** | 8 / 9 | **3 / 12** |

The two scores fail differently: the open door is invisible to the classifier (0.04) but "unusual for this
place"; people walking look like obstacles to the classifier (0.72) but "normal for this place". The learned
combiner overfits the 20 training bursts; the parameter-free average does not. Checkpoint ids are not the same
place across the re-mapped sessions, so the reference is all clear frames (`--by-checkpoint` restricts it to
the same checkpoint once the checkpoints are fixed on one map).

```bash
python3 ml/anomaly_eval.py data/merged --save-final data/merged/results/classifier_anomaly.pkl --save-folds data/merged/results/ca_folds
F=data/merged/results/ca_folds        # demo: each burst judged by the model that never saw it
python3 ml/predict.py $F/2026-11-25_run5_open_door__cp_05__open_door.pkl data/merged/2026-11-25/run5_open_door/cp_05   # open door -> HAZARD
python3 ml/predict.py $F/2026-11-25_run1__cp_07__none.pkl data/merged/2026-11-25/run1/cp_07                        # people -> NORMAL
```

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
