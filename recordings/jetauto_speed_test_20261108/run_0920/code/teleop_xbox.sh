#!/bin/bash
# Drive with an Xbox controller (USB cable, or Bluetooth after pairing on host).
# Hold LB to move. Left stick = drive/strafe, right stick = rotate, RB = turbo,
# D-pad up/down = speed level, B = E-STOP (latches), Start = release e-stop.
# Settings: xbox_teleop.yaml. Extra args go to xbox_teleop.py (e.g. -p max_angular:=0.3).
source "$(dirname "$0")/env.sh"
for js in /sys/class/input/js*; do [ -e "$js" ] && ensure_dev /sys/class/input "input/$(basename $js)"; done
for ev in /sys/class/input/event*; do ensure_dev /sys/class/input "input/$(basename $ev)" >/dev/null; done
if ! ls /dev/input/js* >/dev/null 2>&1; then
    echo "No joystick found. Plug the Xbox controller in by USB (or pair via Bluetooth on the host) and retry."
    exit 1
fi
echo "Joysticks: $(ls /dev/input/js*)"
# Same fix as drive_pygame.py: SDL 2.0.20 reads LB as BACK on the Xbox Series pad over Bluetooth.
export SDL_GAMECONTROLLERCONFIG='050000005e040000130b000009050000,Xbox Series X Controller,a:b0,b:b1,x:b3,y:b4,leftshoulder:b6,rightshoulder:b7,back:b10,start:b11,guide:b12,leftstick:b13,rightstick:b14,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a5,righttrigger:a4,dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,misc1:b15,platform:Linux,'
PARAMS="$TOOLKIT/scripts/xbox_teleop.yaml"
pids=()
if ! node_running odom_publisher; then
    echo "Starting robot base..."; ros2 launch controller controller.launch.py & pids+=($!); sleep 6
fi
ros2 run joy joy_node --ros-args --params-file "$PARAMS" & pids+=($!)
# Background children ignore SIGINT, so stop them (and their children) with TERM.
stop_children() { kill_tree "${pids[@]}"; wait; }
trap stop_children EXIT
trap 'exit 130' INT TERM
python3 "$TOOLKIT/scripts/xbox_teleop.py" --ros-args --params-file "$PARAMS" "$@"
