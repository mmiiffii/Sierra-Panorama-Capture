#!/usr/bin/env python3
"""
Sierra Nevada OCR Capture — day-rollover safe

Fixes:
- All time comparisons are *local naive* (no aware/naive mixups).
- If the latest saved filename timestamp is > FUTURE_BUF_MIN minutes ahead of *now*
  and we are before MORNING_FIX_CUTOFF (local hour), auto-relabel it to yesterday.
- Even if we don't rename, a "future" last file will not block morning saves.
- Visual de-dup (aHash) still avoids duplicate frames.

Env (tune from workflow or terminal):
  OCR_LANG=eng
  CAM_TZ=Europe/Madrid
  FUTURE_BUF_MIN=90       # minutes ahead → consider "future" (default 60)
  MORNING_FIX_CUTOFF=11   # before this hour, future-last will be auto-renamed to yesterday
  DUP_HASH_TOL=6          # aHash Hamming distance for dedup (0..64)
  DEBUG_ROI=0             # 1 to dump debug/roi_*.jpg once
"""

import os, re, glob, time, json, requests
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import cv2

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# --- Cameras (unchanged URLs) ---
CAMS: Dict[str, str] = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}

BASE_DIR = "images"
TIMEOUT  = 20
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (SierraNevadaOCR/1.4)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# ROI defaults (percent)
DEFAULT_ROI = dict(top=0.90, left=0.85, bottom=0.985, right=0.995)

# --- Tunables via env ---
OCR_LANG = os.environ.get("OCR_LANG", "eng")
CAM_T
