#!/usr/bin/env bash
# Copy a map and everything placed on it (checkpoints, home, landmarks) between the SSD data
# folder and the repo's maps/ folder, so it can be committed and pushed.
#   scripts/backup_map.sh lab_new_maps            # SSD -> repo maps/lab_new_maps/   (then git add/commit/push)
#   scripts/backup_map.sh lab_new_maps --restore  # repo -> SSD (fresh robot / container, or after a pull)
set -euo pipefail
NAME="${1:?usage: backup_map.sh <map name> [--restore]}"
DATA="${AME505_DATA:-/home/ubuntu/share/tmp/ame505_data}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/maps/$NAME"
# SSD sub-folder : file names
PARTS=("maps_2d:$NAME.pgm $NAME.yaml" "checkpoints:$NAME.yaml" "home:$NAME.yaml" "landmarks:$NAME.yaml")

for part in "${PARTS[@]}"; do
  sub="${part%%:*}"
  for f in ${part#*:}; do
    if [ "${2:-}" = --restore ]; then src="$REPO/$sub/$f"; dst="$DATA/$sub/$f"
    else src="$DATA/$sub/$f"; dst="$REPO/$sub/$f"; fi
    [ -f "$src" ] || continue
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst" && echo "  $sub/$f"
  done
done
[ "${2:-}" = --restore ] && echo "Restored to $DATA" || echo "Copied to $REPO. Commit: git add maps/$NAME && git commit -m 'Map $NAME'"
