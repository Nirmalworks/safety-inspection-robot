#!/usr/bin/env bash
# Copy rosbags recorded while mapping (SSD $AME505_DATA/bags/) into the repo's recordings/mapping/,
# so they can be committed and pushed.
#   scripts/backup_bag.sh lab2d_11_25_6       # every bag of that map (lab2d_11_25_6_<date>_<time>)
#   scripts/backup_bag.sh lab2d_11_25_6_20261126_101500   # one bag
#   scripts/backup_bag.sh --today             # every bag recorded today
#   scripts/backup_bag.sh --restore <name>    # repo -> SSD
# Bags still recording (no metadata.yaml yet) are skipped. GitHub refuses files over 100 MB, so a
# bag with a bigger file is rewritten zstd-compressed in pieces of at most 90 MB.
set -euo pipefail
DATA="${AME505_DATA:-/home/ubuntu/share/tmp/ame505_data}/bags"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/recordings/mapping"
LIMIT=$((95 * 1024 * 1024))

restore=0
[ "${1:-}" = --restore ] && { restore=1; shift; }
WHAT="${1:?usage: backup_bag.sh <map name | bag name | --today> (or --restore <name>)}"
if [ $restore = 1 ]; then SRC="$REPO"; DST="$DATA"; else SRC="$DATA"; DST="$REPO"; fi

if [ "$WHAT" = --today ]; then
  pattern="*_$(date +%Y%m%d)_*"
elif [ -d "$SRC/$WHAT" ]; then
  pattern="$WHAT"
else
  pattern="${WHAT}_[0-9]*_[0-9]*"
fi

copied=()
shopt -s nullglob
for bag in "$SRC"/$pattern/; do
  bag="${bag%/}"; name="$(basename "$bag")"
  if [ ! -f "$bag/metadata.yaml" ]; then
    echo "  skip $name: no metadata.yaml (still recording, or crashed: ros2 bag reindex $bag)"; continue
  fi
  if [ -e "$DST/$name" ]; then echo "  skip $name: already in $DST"; continue; fi
  mkdir -p "$DST"
  big=$(find "$bag" -type f -size +"$LIMIT"c | head -1)
  if [ $restore = 0 ] && [ -n "$big" ]; then
    echo "  $name: has files over 95 MB, writing a compressed copy..."
    set +u; source /opt/ros/humble/setup.bash; set -u   # ROS setup reads unset variables
    cfg="$(mktemp --suffix=.yaml)"
    cat > "$cfg" <<EOF
output_bags:
- uri: $DST/$name
  storage_id: sqlite3
  max_bag_size: $((90 * 1024 * 1024))
  compression_mode: file
  compression_format: zstd
  all: true
EOF
    ros2 bag convert -i "$bag" -o "$cfg" >/dev/null
    rm -f "$cfg"
  else
    cp -rp "$bag" "$DST/$name"
  fi
  echo "  $name ($(du -sh "$DST/$name" | cut -f1))"
  copied+=("$name")
done

if [ ${#copied[@]} = 0 ]; then echo "No bags matching '$WHAT' to copy from $SRC"; exit 0; fi
if [ $restore = 1 ]; then echo "Restored to $DATA"
else echo "Copied to $REPO. Commit: git add recordings/mapping && git commit -m 'Mapping bags ${WHAT#--}'"; fi
