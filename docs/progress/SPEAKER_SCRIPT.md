# Progress presentation: speaker script

Deck: `AME505_Group5_Progress.pptx` (6 slides, a demo slide and 1 backup; no speaker notes inside).
Total 5:00, then Q&A. Numbers and status words match `PROGRESS_REPORT.md`. Read this aloud while
rehearsing; on the day, speak from the slides, not from this page.

Presenters: slides 1–2 ______ · slides 3–5 ______ (one or two people; everyone takes questions).

---

## Slide 1: Problem and Objective (0:00–0:45)

We are Group 5: Leo, Nirmal, Ryan and Soowhan.

Exit routes have to stay clear, and blocked exit routes are among the most-cited OSHA 1910.37
violations. But they are checked only on periodic walk-throughs, so a hazard that appears after an
inspection stays until the next one.

Our objective is an autonomous robot that patrols 8 to 10 checkpoints and detects three hazard
classes: a blocked exit, an obstructed walkway and an open door. Figure 1 shows the problem as a
classification task: the input at each checkpoint is an RGB and depth frame, the robot's pose and the
keep-clear zone; the output is a class label with a confidence; the user is facilities or safety staff,
who review the flags. It is an extra check, not a replacement for an inspector.

Figure 2 is our platform, rendered from our own URDF: a Hiwonder JetAuto with a Jetson Nano, a 2D lidar
and a depth camera. All experiments run in the BHE 301 senior laboratory.

## Slide 2: Approach and Data (0:45–1:45)

The workflow reads left to right. The robot builds a 2D lidar map, localizes on it with AMCL, and Nav2
drives it to each checkpoint; those three blocks are operational. At each stop it captures RGB-D
frames, and those frames become our labelled dataset. The decision block is where inference happens:
the rule, the CNN detector and the classifier. It is partial today: the rule is built, the learned
models are next.

Figures 3 and 4 are our two independent maps of the lab, Data 1 and Data 2, shown in 3D. Data 2
carries the first three checkpoints: the east door, the south door and the west end.

On the data side, the capture tool writes the label with every frame, so the dataset is labelled as
it is collected. We split it 70/15/15 by session and by location, with one whole checkpoint held out
for testing. A random split would put nearly identical frames from the same spot in both training and
test, which leaks information and inflates the score. Holding out a location measures generalization:
how the model does at a place it has never seen.

## Slide 3: Models and Implementation (1:45–3:15)

Figure 5 is the inspection pipeline, and each block maps to a method from the course.

**Input.** Real frames from our robot: a person standing in front of it, in colour and in depth. These
are the labelled examples a supervised model learns from.

**Features.** Two kinds. Depth geometry is a hand-crafted feature: we project the depth pixels into the
robot frame and measure how much surface sits inside the keep-clear zone. Image features come in two
flavours: classical ones such as HOG, which summarizes edge directions, and colour histograms; and a
CNN embedding, where a convolutional network learns its own features. Convolution shares the same
small filters across the whole image, so the network learns edges, then parts, then objects with far
fewer parameters than a fully connected layer.

**Models.** Three of them. First, the depth rule, a non-learned baseline: if more than 0.01 square
metres of surface is in the zone, it is a hazard. Every learned model has to beat it. Second, YOLO, a
CNN object detector. We use transfer learning: the network is pretrained on COCO, which already knows
people, chairs and boxes, so we only fine-tune it on a few hundred of our own frames instead of
training from scratch. The pretrained model already runs on the robot. Third, the checkpoint
classifier, where we compare classical models (kNN, a decision tree and an SVM on the hand-crafted
features) against a CNN. Hyperparameters such as k, the tree depth and the SVM kernel are chosen on the
validation split only, never on the test set.

**Output.** A class label with a confidence. We score it with precision and recall. For a safety check,
a missed hazard costs more than a false alarm, so recall is the main number.

The table shows where each component stands: designed, implemented, trained and evaluated. The key
design decision: we moved from 3D camera SLAM to 2D lidar with AMCL, because camera frames arrived up
to 7 seconds late and camera relocalization erred by 1.4 metres. And the reason we need learning at
all: depth alone cannot tell a person, who will move, from an obstruction, and an open door leaves the
zone empty.

## Slide 4: Progress Evidence (3:15–4:15)

Figure 6: the robot navigating autonomously from its start pose to checkpoint A. Figure 7: the operator
dashboard during a run; Nav2 plans the global path, in blue, to checkpoint 1 on the Data 2 map. Figure
8: the synchronised camera, depth and lidar views the robot records at a checkpoint.

Our strongest result: we mapped the lab twice, Data 1 and Data 2, starting from different places.
After aligning the two maps, the median distance between their walls is 7 centimetres. That includes
furniture that moved, so it is a pessimistic number. The map is stable enough to return to the same
checkpoints session after session, which is what a fixed-location dataset needs.

The current limitation: near the identical lab tables, AMCL's uncertainty grew from ±0.29 to ±0.78
metres and the robot had to replan. The lidar sees the same shape at several tables. Printed ArUco
markers to tell them apart are in progress.

No model has results on real images yet. The first experiment, the depth-rule baseline on real frames,
scored by precision and recall, is on November 24.

## Slide 5: Operator Dashboard (about 20 s; skip if short on time)

This is the dashboard the robot serves to a laptop browser. Panel 1, the status bar: run mode and map,
battery, localization confidence, detector state and the emergency stop. Panel 2: the map with the live
pose, lidar returns in red to check alignment, the Nav2 plan in blue, the checkpoints, and distance to
goal. Panel 3: the live camera, depth and lidar views. Panel 4: sensor health, with a dot that turns red
when a topic stops publishing. This is also where the inspection results will appear.

## Slide 6: Next Steps (4:15–5:00)

This week: the remaining checkpoints and the first patrol with the camera. November 21 to 25: data
sessions 1 to 3, with the first real baseline result on the 24th. November 30: the CNN detector
fine-tuned, with latency measured on the Jetson. December 2: kNN, decision tree and SVM against the CNN
on the same split. December 8: the live evaluation.

The targets come from our proposal: recall of at least 0.90, at most one false alarm per patrol, at
least 80 percent of patrols completed, and at most 200 milliseconds of inference per frame. Every model
is trained on the same split and compared on the held-out checkpoint, so the comparison is fair.

Thank you.

---

## Demo slide (after slide 5, only if time allows)

Two videos, click to play: the robot run in BHE 301, and the same dashboard and Nav2 stack touring the
checkpoints in simulation. The dashboard run is already on slide 4 as a GIF.

## Backup slide: Robot Platform (Q&A only)

Sensors (lidar, RGB-D camera, IMU and wheel encoders) feed the Jetson Nano, which runs ROS 2 in Docker:
slam_toolbox for mapping, AMCL and the EKF for the pose, Nav2 for paths, YOLO26n, the inspection node
and the dashboard server. Outputs are the mecanum base and the dashboard in a laptop browser. Camera
frames come through the vendor's ROS 1 container and arrive 1 to 7 seconds late, which is why the
robot inspects only while stopped.

## Feedback questions (not on the slides; ask only if invited in Q&A)

1. Collection starts about a week late. Are around 700 images (8 checkpoints, 2 sessions) enough for a
   fair comparison, or should we drop a hazard type?
2. The depth rule cannot see an open door. Is a detector class acceptable instead of a zone check?

## Likely questions

- **Why a CNN and not only kNN or an SVM?** The classical models need features we design (HOG, colour,
  depth area); a CNN learns its own features from pixels. With few images the classical models may do
  as well, which is exactly what the comparison on Dec 2 tests.
- **Why transfer learning?** We will have hundreds of images, not tens of thousands. Starting from COCO
  weights means the early layers already detect edges and common objects; we only adapt the last layers.
- **How do you avoid overfitting?** Small model (YOLO nano, 320 px input), pretrained weights, early
  stopping on the validation split, and a test checkpoint the model never sees.
- **How do you avoid data leakage?** Frames from one checkpoint and one session never appear in both
  train and test; one checkpoint is held out entirely.
- **Why recall first?** A missed blocked exit is worse than a false alarm that staff can dismiss.
- **What if a person stands in the zone?** The robot waits and re-checks. The detector's person class
  separates a person from an obstruction; the depth rule alone cannot.
- **Why 2D lidar instead of the camera for navigation?** Late camera frames (up to 7 s) and a 1.4 m
  relocalization error; the lidar maps agree to 7 cm.
- **Latency?** COCO YOLO26n runs at about 1.5 fps on the Nano CPU; the TensorRT export at 320 px is
  meant to reach the 200 ms target. To be measured on Nov 30.
