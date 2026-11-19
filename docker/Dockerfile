# Simulation + dashboard + map editor in a container, for laptops without ROS 2 Humble (macOS, Windows).
# No robot hardware, no RViz: everything is used through the browser. See docs/SIMULATION.md.
#   docker build -t ame505-sim -f docker/Dockerfile .
#   docker run --rm -it -p 8505:8505 -p 8506:8506 -v "$PWD":/ws -v ame505_sim_data:/root/ame505_sim_data ame505-sim
# ros:humble-ros-base has amd64 and arm64 builds, so it runs natively on Apple Silicon.
FROM ros:humble-ros-base

# Only what the simulated stack needs (no RTAB-Map, no RViz, no Hiwonder drivers).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
      ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
      ros-humble-joy ros-humble-sensor-msgs-py \
      python3-flask python3-opencv python3-scipy python3-yaml python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
EXPOSE 8505 8506
CMD ["/ws/docker/sim_entry.sh"]
