#!/usr/bin/env bash
# Free RAM on the Jetson Nano and show what is using it.
#
#   ./scripts/free_memory.sh                 # report + safe cleanup (nothing the robot needs is stopped)
#   ./scripts/free_memory.sh --robot         # also stop all robot software (scripts/stop_all.sh)
#   ./scripts/free_memory.sh --editor        # also kill VS Code server / language servers on the robot
#
# Works on the host or inside the ROS 2 container; steps that need host root are skipped if unavailable.
# Background and one-time host settings: docs/JETSON_MEMORY.md
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

mem() { free -m | awk '/^Mem:/ {printf "RAM used %d / %d MB (available %d MB)", $3, $2, $7} /^Swap:/ {printf ", swap used %d / %d MB\n", $3, $2}'; }

echo "== before: $(mem)"

for a in "$@"; do
  case "$a" in
    --robot)
      echo "-- stopping robot software"
      "$HERE/stop_all.sh" >/dev/null 2>&1 || true ;;
    --editor)
      echo "-- stopping editor processes (VS Code server, Pylance, extensions)"
      pkill -f '\.vscode-server/' 2>/dev/null || true
      pkill -f 'vscode-pylance' 2>/dev/null || true ;;
  esac
done

# ROS 2 CLI daemon: ~60-100 MB, restarts by itself the next time a ros2 command needs it
command -v ros2 >/dev/null 2>&1 && ros2 daemon stop >/dev/null 2>&1 && echo "-- stopped ros2 daemon"

# Leftover ros2 launch / node processes from crashed runs (only when no launch is running)
if ! pgrep -f "ros2 launch" >/dev/null; then
  n=$(pgrep -fc '/opt/ros/humble/lib/|/install/ame505_bringup/lib/' 2>/dev/null || echo 0)
  if [ "${n:-0}" -gt 0 ]; then
    echo "-- $n leftover ROS node processes with no launch running: stopping them"
    pkill -f '/opt/ros/humble/lib/|/install/ame505_bringup/lib/' 2>/dev/null || true
  fi
fi

# Page cache: frees "buff/cache" (needs host root; harmless, the kernel refills it as needed)
if sudo -n true 2>/dev/null; then
  sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && echo "-- dropped page cache"
fi

sleep 1
echo "== after:  $(mem)"
echo
echo "== top 10 processes by memory (RSS MB)"
ps -eo rss,pid,comm,args --sort=-rss | awk 'NR>1 && NR<=11 {printf "%6d MB  %-7s %s\n", $1/1024, $2, substr($0, index($0,$4), 90)}'
echo
if command -v docker >/dev/null 2>&1 && docker ps -q >/dev/null 2>&1; then
  echo "== containers"
  docker stats --no-stream --format '{{.Name}}: {{.MemUsage}}' 2>/dev/null
fi
free -m | awk '/^Mem:/ && $7 < 800 {print "\nWARNING: under 800 MB available. Navigation will lag. See docs/JETSON_MEMORY.md"}'
