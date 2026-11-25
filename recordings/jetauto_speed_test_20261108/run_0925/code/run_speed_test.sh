#!/bin/bash
# Speed + hard-stop test at the speeds in speed_test.yaml, recorded to a rosbag, with a
# ping running alongside. Everything ends up in one zip in record.out_dir.
#   1. starts SLAM (robot base + lidar + slam_toolbox) and Xbox teleop (B = E-STOP).
#      If Hiwonder's ROS 1 stack holds the lidar (it does after a reboot), it uses that stack
#      instead through camera/ros1_relay.py: its sensors in, our drive commands out.
#   2. records a bag, starts the ping, runs speed_test.py
#   3. analyses the bag, draws graphs (graphs/*.png) and zips everything with the code/YAML used
# Every command is printed (+ ...) before it runs. Ctrl+C stops the robot and everything.
source "$(dirname "$0")/env.sh"
cd "$TOOLKIT/scripts"
cfg() { python3 -c "import yaml,sys;d=yaml.safe_load(open('speed_test.yaml'))
for k in sys.argv[1].split('.'): d=d[k]
print(' '.join(map(str,d)) if isinstance(d,list) else d)" "$1"; }
run() { echo "+ $*"; "$@"; }
pids=()
bg() { local log=$1; shift; echo "+ $* > $log &"; "$@" > "$log" 2>&1 & pids+=($!); }
cleanup() {
    timeout 3 ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
    # Background scripts ignore SIGINT, so use TERM (children first: teleop sends a zero on TERM).
    [ ${#pids[@]} -gt 0 ] && kill_tree "${pids[@]}"
    wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

NAME=speed_test_$(date +%Y%m%d_%H%M)
ROOT=$(realpath -m "$(cfg record.out_dir)")
OUT=$ROOT/$NAME
run mkdir -p "$OUT/logs" "$OUT/analysis" "$OUT/code"

ros1_lidar() { timeout 6 python3 -c "
import xmlrpc.client,socket,sys; socket.setdefaulttimeout(3)
try: _,_,s=xmlrpc.client.ServerProxy('http://127.0.0.1:11311').getSystemState('/check')
except Exception: sys.exit(1)
sys.exit(0 if '/scan' in dict(s[0]) else 1)"; }
if ros1_lidar && ! node_running odom_publisher; then
    echo "Hiwonder's ROS 1 stack holds the lidar and motor board: using it through the relay."
    bg "$OUT/logs/relay.log" python3 camera/ros1_relay.py
    export SDL_GAMECONTROLLERCONFIG=$(grep -oP "SDL_GAMECONTROLLERCONFIG='\K[^']+" teleop_xbox.sh)
    for js in /sys/class/input/js*; do [ -e "$js" ] && ensure_dev /sys/class/input "input/$(basename $js)"; done
    if ls /dev/input/js* >/dev/null 2>&1; then
        bg "$OUT/logs/joy.log" ros2 run joy joy_node --ros-args --params-file xbox_teleop.yaml
        bg "$OUT/logs/teleop.log" python3 xbox_teleop.py --ros-args --params-file xbox_teleop.yaml
    else
        echo "  (no Xbox controller: E-STOP is Ctrl+C here or ./stop_all.sh)"
    fi
    echo "  (waiting 8 s for the relay)"; sleep 8
elif ! node_running slam_toolbox; then
    if node_running odom_publisher; then
        echo "A robot base is running without SLAM (e.g. from teleop_xbox.sh). Run ./stop_all.sh first."
        exit 1
    fi
    bg "$OUT/logs/slam.log" env NO_RVIZ=1 ./slam_start.sh
    echo "  (waiting 15 s for SLAM)"; sleep 15
fi
if ! node_running xbox_teleop && ! [ -e "$OUT/logs/teleop.log" ]; then
    bg "$OUT/logs/teleop.log" ./teleop_xbox.sh
    sleep 5
fi

bg "$OUT/logs/bag.log" ros2 bag record -o "$OUT/bag" $(cfg record.topics)
BAG=${pids[-1]}
sleep 4
bg "$OUT/ping.txt" sudo ping -c "$(cfg ping.count)" -i "$(cfg ping.interval)" "$(cfg ping.target)"
PING=${pids[-1]}

echo "+ python3 speed_test.py | tee $OUT/speed_test_log.txt"
python3 speed_test.py | tee "$OUT/speed_test_log.txt"
sleep 1
run kill -INT $BAG; wait $BAG
echo "  (waiting for the ping to finish: $(cfg ping.count) x $(cfg ping.interval) s)"
wait $PING
tail -3 "$OUT/ping.txt"

run sh -c "ros2 bag info '$OUT/bag' > '$OUT/bag_info.txt' 2>/dev/null"
run python3 speed_analysis.py "$OUT/bag" "$OUT/analysis" "$OUT/ping.txt"
run python3 speed_plot.py "$OUT"
run cp speed_test.yaml speed_test.py speed_analysis.py speed_plot.py run_speed_test.sh \
       xbox_teleop.yaml xbox_teleop.py teleop_xbox.sh map_xbox.sh \
       camera/ros1_relay.py camera/ros1_relay.yaml camera/ros1_sub.py "$OUT/code/"
run sh -c "cd '$ROOT' && zip -qr '$NAME.zip' '$NAME'"
echo "Done: $ROOT/$NAME.zip"
