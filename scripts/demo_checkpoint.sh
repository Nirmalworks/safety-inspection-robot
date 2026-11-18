#!/usr/bin/env bash
# Demo at a checkpoint, through the running dashboard (same as its "Demo at checkpoint" card):
# drive to the checkpoint, YOLO on, hold still, capture photos, save a snapshot.
#   scripts/demo_checkpoint.sh                         # cp_01, normal scene, 6 photos, session "demo"
#   scripts/demo_checkpoint.sh cp_01 hazard blocked_exit box
#   N=10 SESSION=s1 scripts/demo_checkpoint.sh cp_02 normal
# Needs: dashboard running, navigation started with "camera" ticked, robot localized.
set -euo pipefail
CP="${1:-cp_01}"; SCENE="${2:-normal}"; HAZARD="${3:-none}"; OBJECTS="${4:-}"
URL="http://localhost:${AME505_DASHBOARD_PORT:-8505}"
BODY=$(python3 -c 'import json,sys; print(json.dumps({"id": sys.argv[1], "scene": sys.argv[2], "hazard": sys.argv[3],
  "objects": sys.argv[4], "n": int(sys.argv[5]), "session": sys.argv[6], "note": sys.argv[7]}))' \
  "$CP" "$SCENE" "$HAZARD" "$OBJECTS" "${N:-6}" "${SESSION:-demo}" "${NOTE:-}")
curl -s -m 5 -X POST "$URL/api/demo_checkpoint" -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["msg"]); sys.exit(0 if d["ok"] else 1)'
last=""
while true; do
  line=$(curl -s -m 5 "$URL/api/status" | python3 -c 'import json,sys; d=json.load(sys.stdin)["demo"]; print(d["state"] + "|" + d.get("msg", ""))')
  [ "$line" != "$last" ] && echo "  ${line/|/: }" && last="$line"
  case "${line%%|*}" in done) exit 0 ;; failed) exit 1 ;; esac
  sleep 1
done
