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
| Tests | `python3 -m pytest ml/` | geometry of the rule, split rules, metrics, HOG |
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
