#!/bin/bash
# Map a room or a whole building with the Xbox controller, in one terminal:
# starts SLAM (robot base + lidar + slam_toolbox + RViz), then Xbox driving.
#   ./map_xbox.sh            drive around, then in another terminal: ./save_map.sh <name>
#   ./map_xbox.sh --bag      also records a rosbag to CSCI-513/jetauto_recordings/map_<date>/
# Ctrl+C stops driving, recording and SLAM. Controls: see teleop_xbox.sh.
source "$(dirname "$0")/env.sh"
# Background children ignore SIGINT, so stop them (and their children) with TERM.
stop_children() { kill_tree $(jobs -p); wait; }
trap stop_children EXIT
trap 'exit 130' INT TERM
"$TOOLKIT/scripts/slam_start.sh" > /tmp/slam_start.log 2>&1 &
echo "Starting SLAM (log: /tmp/slam_start.log)..."
sleep 15
if [ "$1" = --bag ]; then
    shift
    out="$TOOLKIT/CSCI-513/jetauto_recordings/map_$(date +%Y%m%d_%H%M)"
    ros2 bag record -o "$out" /scan /scan_raw /odom /odom_raw /odom_rf2o /imu /tf /tf_static /map /pose \
        /controller/cmd_vel /joy /estop /ros_robot_controller/battery > /tmp/map_bag.log 2>&1 &
    echo "Recording to $out"
fi
"$TOOLKIT/scripts/teleop_xbox.sh" "$@"
