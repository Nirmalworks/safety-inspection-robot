# Progress presentation: speaker script

Deck: `AME505_Group5_Progress.pptx` (5 slides + 1 backup, no speaker notes inside). Total 5:00, then Q&A.
Numbers and status words match `PROGRESS_REPORT.md`. Read this aloud while rehearsing; on the day,
speak from the slides, not from this page.

Presenters: slides 1–2 ______ · slides 3–5 ______ (one or two people; everyone takes questions).

---

## Slide 1: Problem and Objective (0:00–0:45)

We are Group 5: Leo, Nirmal, Ryan and Soowhan.

Exit routes have to stay clear, and blocked exit routes are among the most-cited OSHA 1910.37
violations. But they are checked only on periodic walk-throughs, so a hazard that appears after an
inspection stays until the next one.

Our robot patrols 8 to 10 checkpoints on its own and flags three staged hazards: a blocked exit, an
obstructed walkway and an open door. At each checkpoint the input is an RGB and depth frame, the
robot's pose and the keep-clear zone for that spot. The output is normal or a hazard type, with a
confidence and an annotated image. Facilities or EHS staff review the flagged checkpoints. It is an
extra check, not a replacement for an inspector.

*(Gesture to the render:)* This is our platform, drawn from our own URDF: a JetAuto with a Jetson Nano,
a 2D lidar and a depth camera. All experiments run in the BHE 301 senior lab.

## Slide 2: Approach and Data (0:45–1:45)

Read the diagram left to right. The robot builds a 2D lidar map, localizes on it with AMCL, and Nav2
drives it to each checkpoint. Those three blocks are operational on the robot. At each stop it
captures RGB-D frames, which is implemented, and then decides. The decision step is partial: the
non-learned rule is built, and the learned models are next. Results go to a log and the web dashboard.

Data so far: the lidar map of BHE 301 on the left, 24.6 by 9.8 metres, mapped on November 18 with the first
three checkpoints. The camera streams live, but we have no labelled images yet. That is our main risk.

What is prepared: a capture tool that saves every frame with its labels, and a split by session and
location, 70/15/15, with one whole checkpoint held out for testing. We split by location rather than
at random so the test score measures how the model does at a place it has never seen, not how well it
remembers a background.

## Slide 3: Models and Implementation (1:45–3:15)

This is the inspection model as a block diagram. On the left is a real frame from our robot: a person
standing in front of it, in colour and in depth.

From each frame we compute two kinds of features: depth geometry, meaning how much surface sits above
the floor inside the keep-clear zone, and image features such as HOG, colour histograms and a CNN
embedding.

There are three models. First, the baseline: a depth rule with no learning. If more than 0.01 square
metres of surface is in the zone, it is a hazard. It is implemented and tested, but only on synthetic
frames. Second, a YOLO detector. The COCO-pretrained version already runs live on the robot at about
1.5 frames per second; fine-tuning on our own frames is designed. Third, a checkpoint classifier, where
we will compare kNN, a decision tree and an SVM on the hand-made features against a CNN, and use the best.

Why learn anything at all? Look at the depth image: the rule only knows that something is close. It
cannot tell a person who will walk away from a box that blocks the exit, and an open door leaves the
zone empty, so the rule cannot see it. Those cases need a learned model.

Key design decision: we moved from 3D camera SLAM to 2D lidar with AMCL. Camera frames arrived up to
7 seconds late and camera relocalization left the start pose 1.4 metres off.

Evaluation: supervised labels from staged scenes, tested on the held-out checkpoint, scored by recall,
false alarms per patrol and latency. For a safety check, a missed hazard costs more than a false alarm,
so recall is the main number.

## Slide 4: Progress Evidence (3:15–4:15)

Here is the robot working. On the left, a phone video: the robot drives from its start pose to
checkpoint A on its own. On the right, the dashboard during a run: Nav2 plans the blue path to
checkpoint 1 on the November 18 map, and the robot marker follows it.

Our strongest number: we mapped the lab twice, five days apart, starting from different places. After
aligning the two maps, the median distance between their walls is 7 centimetres. That includes
furniture that moved, so it is a pessimistic number. The map is stable enough to return to the same
checkpoints session after session.

The current limitation: near the identical lab tables, AMCL's uncertainty grew from about ±0.3 to
±0.78 metres and the robot had to replan. The lidar sees the same shape at several tables. Printed ArUco
markers to tell them apart are in progress.

No model has results on real images yet. The first experiment, the depth-rule baseline on real
frames, is on November 24.

## Slide 5: Next Steps and Questions (4:15–5:00)

This week: save the remaining checkpoints and run the first patrol with the camera. November 21 to
25: data sessions 1 to 3, and on the 24th the first real baseline result. November 30: YOLO
fine-tuned, with latency measured on the Jetson. December 2: the classifier comparison on the same
split. December 8: the live evaluation, against the targets from our proposal: recall of at least
0.90, at most one false alarm per patrol, at least 80 percent of patrols completed, and at most 200
milliseconds per frame.

Two questions for you. First, collection starts about a week late. Are around 700 images, from 8
checkpoints and 2 sessions, enough for a fair comparison, or should we drop a hazard type? Second, the
depth rule cannot see an open door. Is a detector class acceptable instead of a zone check?

Thank you.

---

## Backup slide: Robot Platform (Q&A only)

Use it if asked what runs where. Sensors (lidar, RGB-D camera, IMU and wheel encoders) feed the Jetson
Nano, which runs ROS 2 in Docker: slam_toolbox for mapping, AMCL and the EKF for the pose, Nav2 for
paths, YOLO26n, the inspection node and the dashboard server. Outputs are the mecanum base and the
dashboard in a laptop browser. Camera frames come through the vendor's ROS 1 container and arrive 1 to
7 seconds late, which is why the robot inspects only while stopped.

## Likely questions

- **Why not train YOLO now?** No labelled real frames yet; the first sessions are Nov 21–25.
- **How do you avoid data leakage?** Frames from one checkpoint and one session never appear in both
  train and test; one checkpoint is held out entirely.
- **What if a person stands in the zone?** The robot waits and re-checks. The detector's person class
  separates a person from an obstruction; the depth rule alone cannot.
- **Why 2D lidar instead of the camera for navigation?** Late camera frames (up to 7 s) and a 1.4 m
  relocalization error; the lidar map repeats to 7 cm.
- **Latency?** COCO YOLO26n runs at about 1.5 fps on the Nano CPU; the TensorRT export at 320 px is
  meant to reach the 200 ms target. To be measured on Nov 30.
