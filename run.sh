#!/usr/bin/env bash
set -euo pipefail
URL="${1:-https://webtv.feratel.com/webtv/?cam=15111}"
OUT="${2:-./out}"
DUR="${3:-85}"
FPS="${4:-2.5}"
python feratel_pano_grabber.py "$URL" "$OUT" "$DUR" "$FPS"
