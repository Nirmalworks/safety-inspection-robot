#!/usr/bin/env bash
# Runs inside the container (docker/Dockerfile): build, then start the dashboard (8505) and the map editor (8506).
# The build goes to /sim_ws, not into the mounted repo, so a build made on the host is left alone.
set -e
source /opt/ros/humble/setup.bash
mkdir -p /sim_ws/src
ln -sfn /ws/src/ame505_bringup /sim_ws/src/ame505_bringup
ln -sfn /ws/src/ame505_description /sim_ws/src/ame505_description
(cd /sim_ws && colcon build --symlink-install)
cd /ws
source scripts/sim_env.sh
source /sim_ws/install/local_setup.bash
echo "Dashboard:  http://localhost:8505      Map editor: http://localhost:8506"
python3 scripts/map_editor.py &
exec ros2 run ame505_bringup dashboard
