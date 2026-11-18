# Jetson Nano memory: why it lags and how to fix it

The Nano has **4 GB RAM shared between CPU and GPU**. When it runs out, the kernel swaps, every
process slows down, Nav2 misses its deadlines and the robot stutters or aborts goals. A run
recorded on 2026-11-13 (`debug_logs/20261113_195519/system.txt`) had **159 MB free and the 2 GB swap
100 % full**; VS Code with its extensions on the robot used about 1.3 GB of that.

Target: **at least 1 GB "available"** in `free -m` before starting navigation, and more (~1.5 GB) once
the camera and the detector run.

## 1. Every session (1 minute)

```bash
cd $AME505_ROOT && source scripts/env.sh
./scripts/free_memory.sh            # report + safe cleanup
```
It prints RAM / swap before and after, the 10 largest processes and each container's memory, and
warns if less than 800 MB is available. Options: `--editor` also kills VS Code server processes on the
robot, `--robot` also stops all robot software.

Then follow these rules while the robot software runs:

| Do | Don't |
|---|---|
| Edit code on the laptop, `git push`, then `git pull` on the robot | Open VS Code Remote-SSH on the robot (≈ 1–1.5 GB) |
| Use the dashboard from the **laptop browser** (`http://<robot-ip>:8505`) | Open a browser or RViz on the robot (`use_rviz:=false` is the default) |
| Use plain `ssh` terminals | Run a command-line assistant or editor on the robot while navigating; if you use one for setup, close it before **Start navigation** |
| Tick **camera** only when needed (inspection, dataset) | Leave the camera bridge running while only mapping |

## 2. One-time host settings (biggest gains)

Run these on the **Jetson host** (not inside the ROS 2 container): open a terminal on the robot's
screen, or `ssh` to the robot and do not `docker exec` into the container. Each step can be undone.

### 2a. Boot without the desktop: frees ≈ 400–800 MB

The Ubuntu desktop (GNOME / LXDE and its apps) stays in memory even when nobody looks at the screen.

```bash
free -m                                       # note "available"
sudo systemctl set-default multi-user.target  # text console at boot
sudo reboot
# after reboot: free -m  -> compare
```
Undo (desktop back): `sudo systemctl set-default graphical.target && sudo reboot`.
Check first that nothing you need runs only in the desktop session: after the reboot, `docker ps` must
still list the Hiwonder and ROS 2 containers (they start from Docker, not from the desktop). If a
container is missing, start it as before or undo this step.

### 2b. Add an 8 GB swap file on the SSD: stops out-of-memory crashes

Swap does not make the robot faster, but it keeps it from freezing or killing nodes when memory runs
out. The SSD is far faster than the SD card / eMMC. Find where the SSD is mounted on the host:

```bash
df -h | grep sda          # e.g. /dev/sda1  916G ... /mnt/ssd   -> SSD=/mnt/ssd
SSD=<that mount point>
sudo fallocate -l 8G $SSD/swapfile && sudo chmod 600 $SSD/swapfile
sudo mkswap $SSD/swapfile && sudo swapon --priority 5 $SSD/swapfile
echo "$SSD/swapfile none swap sw,pri=5 0 0" | sudo tee -a /etc/fstab
swapon --show             # zram (in RAM, fast) keeps its higher priority; the SSD file is the overflow
```
Prefer RAM over swap until memory is really short:
```bash
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf && sudo sysctl --system
```
Undo: `sudo swapoff $SSD/swapfile`, delete the line from `/etc/fstab`, `sudo rm $SSD/swapfile`.

### 2c. Turn off services the robot does not use

```bash
systemctl list-units --type=service --state=running     # see what runs
sudo systemctl disable --now cups cups-browsed 2>/dev/null          # printing
sudo systemctl disable --now whoopsie apport 2>/dev/null            # crash reporting
sudo systemctl disable --now packagekit unattended-upgrades 2>/dev/null   # background updates
```
Do **not** disable `docker`, `ssh`, `NetworkManager`, `bluetooth` (Xbox controller) or anything
named `nv*` / `jetson*`.

### 2d. Maximum performance power mode

Not memory, but it stops the CPU from throttling under load:
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
sudo nvpmodel -q          # should say MAXN
```

## 3. Check the result

With navigation running (camera off), run on the host:
```bash
free -m
tegrastats --interval 1000     # Ctrl+C after ~30 s; "RAM used/total" and CPU load per core
uptime                         # load average should stay below ~6 on 4 cores
```

| Measurement | Before (2026-11-13) | After |
|---|---|---|
| RAM available with navigation running | 697 MB (swap full) | ______ MB |
| Swap used | 1,977 / 1,977 MB | ______ MB |
| Load average | 18–19 | ______ |

## 4. If it still lags

1. `./scripts/free_memory.sh` and read the top-10 list: anything not robot-related goes.
2. `docker stats --no-stream`: the Hiwonder ROS 1 container is only needed for the **camera**.
3. `./scripts/save_debug_bundle.sh 6 "lag"` and push it, so the logs can be analysed on a laptop.
