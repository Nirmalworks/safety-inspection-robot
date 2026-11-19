#!/usr/bin/env bash
# Start the dashboard against the SIMULATED robot (laptop, no hardware): http://localhost:8505
#   scripts/sim_dashboard.sh                  # lidar world = the sample hallway (sim_hall)
#   scripts/sim_dashboard.sh lab2d_11_18_2    # lidar world = that saved map, so it can be driven like the real room
# scripts/dashboard.sh is for the robot: it loads the hardware environment (scripts/env.sh).
# Needs ROS 2 Humble + Nav2 + slam_toolbox + python3-flask and a built workspace (docs/SIMULATION.md),
# or use the container: docker/Dockerfile.
source "$(dirname "$0")/sim_env.sh"
if [ -n "${1:-}" ]; then
  export AME505_SIM_WORLD="$AME505_DATA/maps_2d/$1.yaml"       # read by robot.launch.py (sim_world default)
  [ -f "$AME505_SIM_WORLD" ] || { echo "No map $AME505_SIM_WORLD (copy it there, or: scripts/backup_map.sh $1 --restore)"; exit 1; }
fi
echo "Simulated dashboard -> http://localhost:${AME505_DASHBOARD_PORT:-8505}   world: ${AME505_SIM_WORLD:-sim_hall}"
exec ros2 run ame505_bringup dashboard
