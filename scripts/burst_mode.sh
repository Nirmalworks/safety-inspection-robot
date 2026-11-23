#!/usr/bin/env bash
# Burst mode through the running dashboard (same as its "Burst mode" card): visit the checkpoints
# and take several bursts of photos at each, nudging the robot between bursts. Photos go to the SSD
# ($AME505_DATA/dataset/<session>/<checkpoint>/), the run log to $AME505_DATA/bursts/.
#   scripts/burst_mode.sh                              # all checkpoints, 4 bursts x 6 photos, session "burst"
#   scripts/burst_mode.sh cp_01,cp_03                  # only these checkpoints
#   BURSTS=6 PHOTOS=8 SESSION=s2 scripts/burst_mode.sh
#   SCENE=hazard HAZARD=blocked_exit OBJECTS=box scripts/burst_mode.sh cp_02
#   scripts/burst_mode.sh stop                         # stop a running burst mode
# Needs: dashboard running, navigation started with "camera" ticked, localization good.
set -euo pipefail
URL="http://localhost:${AME505_DASHBOARD_PORT:-8505}"
post() { curl -s -m 5 -X POST "$URL/api/$1" -H 'Content-Type: application/json' -d "${2:-{\}}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["msg"]); sys.exit(0 if d["ok"] else 1)'; }
if [ "${1:-}" = stop ]; then post burst_stop; exit; fi
BODY=$(python3 -c 'import json,sys; a=sys.argv; print(json.dumps({"checkpoints": a[1], "bursts": int(a[2]),
  "photos": int(a[3]), "session": a[4], "scene": a[5], "hazard": a[6], "objects": a[7],
  "nudge_cm": float(a[8]), "nudge_deg": float(a[9])}))' "${1:-all}" "${BURSTS:-4}" "${PHOTOS:-6}" \
  "${SESSION:-burst}" "${SCENE:-normal}" "${HAZARD:-none}" "${OBJECTS:-}" "${NUDGE_CM:-5}" "${NUDGE_DEG:-8}")
post burst_start "$BODY"
trap 'echo; echo "Stopping burst mode..."; post burst_stop || true; exit 130' INT
last=""
while true; do
  line=$(curl -s -m 5 "$URL/api/status" | python3 -c 'import json,sys; b=json.load(sys.stdin)["burst"]
print(f"{b[\"state\"]}|{b.get(\"index\",0)}/{b.get(\"total\",0)} {b.get(\"cp\",\"\")} {b.get(\"frames\",0)} photos: {b.get(\"msg\",\"\")}")')
  [ "$line" != "$last" ] && echo "  ${line#*|}" && last="$line"
  case "${line%%|*}" in done) exit 0 ;; failed|stopped) exit 1 ;; esac
  sleep 1
done
