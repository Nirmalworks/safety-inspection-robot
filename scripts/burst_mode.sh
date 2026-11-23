#!/usr/bin/env bash
# Burst mode through the running dashboard (same as its "Burst mode (dataset)" card).
#
# Burst HERE (you put the robot on the checkpoint yourself): the robot stays there, takes PHOTOS
# photos (default 120) and nudges itself between groups of them (turn +-8/16 deg, slide/back 5 cm).
#   scripts/burst_mode.sh here cp_03                                   # normal scene, session run1
#   SESSION=run2_obstructed SCENE=hazard HAZARD=obstructed_walkway OBJECTS=box,chair \
#     scripts/burst_mode.sh here cp_03
#   PHOTOS=200 scripts/burst_mode.sh here cp_05
# Automatic route (Nav2 drives to each saved checkpoint, needs navigation + localization):
#   scripts/burst_mode.sh route all          or      scripts/burst_mode.sh route cp_01,cp_03
#   scripts/burst_mode.sh stop
# Needs: dashboard running and the robot started (Drive or Navigation); the camera starts itself.
# Photos: $AME505_DATA/dataset/<session>/<checkpoint>/ + labels in $AME505_DATA/dataset/index.csv;
# run log: $AME505_DATA/bursts/.
set -euo pipefail
URL="http://localhost:${AME505_DASHBOARD_PORT:-8505}"
post() { curl -s -m 10 -X POST "$URL/api/$1" -H 'Content-Type: application/json' -d "${2:-{\}}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["msg"]); sys.exit(0 if d["ok"] else 1)'; }
MODE="${1:-}"
case "$MODE" in
  stop) post burst_stop; exit ;;
  here|route) ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
TARGET="${2:-}"
[ "$MODE" = here ] && [ -z "$TARGET" ] && { echo "Which checkpoint is the robot on? e.g. $0 here cp_03"; exit 2; }
BODY=$(python3 -c 'import json,sys; a=sys.argv; print(json.dumps({"mode": a[1], "spot": a[2], "checkpoints": a[2] or "all",
  "photos": int(a[3]), "positions": int(a[4]), "session": a[5], "scene": a[6], "hazard": a[7], "objects": a[8],
  "nudge_cm": float(a[9]), "nudge_deg": float(a[10])}))' "$MODE" "$TARGET" "${PHOTOS:-120}" "${POSITIONS:-12}" \
  "${SESSION:-run1}" "${SCENE:-normal}" "${HAZARD:-none}" "${OBJECTS:-}" "${NUDGE_CM:-5}" "${NUDGE_DEG:-8}")
post burst_start "$BODY"
trap 'echo; echo "Stopping burst mode..."; post burst_stop || true; exit 130' INT
last=""
while true; do
  line=$(curl -s -m 5 "$URL/api/status" | python3 -c '
import json, sys
b = json.load(sys.stdin)["burst"]
print("%s|%s %s/%s photos: %s" % (b["state"], b.get("cp", ""), b.get("frames", 0), b.get("target", 0), b.get("msg", "")))')
  [ "$line" != "$last" ] && echo "  ${line#*|}" && last="$line"
  case "${line%%|*}" in done) exit 0 ;; failed|stopped) exit 1 ;; esac
  sleep 1
done
