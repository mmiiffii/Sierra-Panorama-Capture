#!/usr/bin/env python3
"""
Capture Feratel JPG webcams once.

For each camera:
  - Fetch bytes (cache-busted).
  - Crop a *fixed bottom-right* region (from config).
  - OCR timestamp from that region (strict; required).
  - If that timestamp is newer than the latest saved for that camera,
    write images/<cam>/<cam>_YYMMDD_HHMMSS.jpg

Config:
  Put per-camera ROIs in config/roi.yml (percentages in [0..1]):
    borreguiles: {top: 0.86, left: 0.78, bottom: 0.98, right: 0.98}
    stadium:     {top: 0.86, left: 0.78, bottom: 0.98, right: 0.98}
    satelite:    {top: 0.86, left: 0.78, bottom: 0.98, right: 0.98}
    veleta:      {top: 0.86, left: 0.78, bottom: 0.98, right: 0.98}
  If a camera is missing in the file, DEFAULT_ROI is used.

Env (optional):
  OCR_LANG=eng          # tess language(s), e.g. 'eng' or 'eng+spa'
  DEBUG_ROI=0|1         # save cropped ROI previews to ./debug
"""

import os, re, glob, time, json, hashlib, requests
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import cv2

# -------- Cameras --------
CAMS: Dict[str, str] = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}

BASE_DIR = "images"
TIMEOUT  = 20
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (FeratelOCR/1.0)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Default ROI if not specified in config (bottom-right box)
DEFAULT_ROI = dict(top=0.86, left=0.78, bottom=0.98, right=0.98)

OCR_LANG = os.environ.get("OCR_LANG", "eng")
DEBUG_ROI = os.environ.get("DEBUG_ROI", "0") in ("1","true","yes")


# ---------- small utils ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def fetch(url: str) -> bytes | None:
    sep = "&" if "?" in url else "?"
    bust = f"{sep}_ts={int(time.time())}"
    try:
        r = requests.get(url + bust, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def latest_saved_ts(cam: str) -> datetime | None:
    """Grabs the newest timestamp from filenames in images/<cam>/*.jpg"""
    pattern = os.path.join(BASE_DIR, cam, f"{cam}_*.jpg")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    # filenames end with _YYMMDD_HHMMSS.jpg (maybe _N suffix)
    name = os.path.basename(files[-1])
    m = re.search(rf"^{re.escape(cam)}_(\d{{6}})_(\d{{6}})", name)
    if not m:
        return None
    d6, t6 = m.groups()
    try:
        return datetime.strptime(d6 + t6, "%y%m%d%H%M%S")
    except ValueError:
        return None

# ---- config loader (YAML/JSON) ----
def load_roi_config() -> Dict[str, Dict[str, float]]:
    cfg_paths = ["config/roi.yml", "config/roi.yaml", "config/roi.json"]
    for p in cfg_paths:
        if os.path.isfile(p):
            try:
                if p.endswith((".yml", ".yaml")):
                    import yaml  # type: ignore
                    data = yaml.safe_load(open(p, "r", encoding="utf-8"))
                else:
                    data = json.load(open(p, "r", encoding="utf-8"))
                return data or {}
            except Exception:
                return {}
    return {}

def get_roi_for(cam: str, cfg: Dict[str, Dict[str, float]]) -> Tuple[float,float,float,float]:
    r = (cfg.get(cam) or {})
    top    = float(r.get("top",    DEFAULT_ROI["top"]))
    left   = float(r.get("left",   DEFAULT_ROI["left"]))
    bottom = float(r.get("bottom", DEFAULT_ROI["bottom"]))
    right  = float(r.get("right",  DEFAULT_ROI["right"]))
    # clamp to [0,1]
    top = max(0.0, min(1.0, top)); left = max(0.0, min(1.0, left))
    bottom = max(0.0, min(1.0, bottom)); right = max(0.0, min(1.0, right))
    return top, left, bottom, right

# ------------- OCR -------------
def ocr_from_roi(img_bgr: np.ndarray, roi_box: Tuple[float,float,float,float]) -> str:
    h, w = img_bgr.shape[:2]
    top, left, bottom, right = roi_box
    y0, y1 = int(h*top), int(h*bottom)
    x0, x1 = int(w*left), int(w*right)
    if y1 <= y0 or x1 <= x0:
        return ""

    roi = img_bgr[y0:y1, x0:x1]

    if DEBUG_ROI:
        ensure_dir("debug")
        cv2.imwrite(f"debug/roi_{int(time.time()*1000)}.jpg", roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.9, beta=24)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr  = cv2.medianBlur(thr, 3)
    thr  = cv2.dilate(thr, np.ones((2,2), np.uint8), iterations=1)

    import pytesseract  # imported here so script runs even if not installed locally
    cfg = f"--psm 7 -l {OCR_LANG} -c tessedit_char_whitelist=0123456789./:- "
    try:
        text = pytesseract.image_to_string(thr, config=cfg)
    except Exception:
        text = ""
    return " ".join(text.split())

def parse_overlay_ts(text: str) -> datetime | None:
    # Typical formats on these feeds:
    patterns = [
        r"(?P<d>\d{2})[.\-/](?P<m>\d{2})[.\-/](?P<Y>\d{4})\s+(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?",
        r"(?P<Y>\d{4})[.\-/](?P<m>\d{2})[.\-/](?P<d>\d{2})\s+(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?",
        r"(?P<d>\d{2})[.\-/](?P<m>\d{2})[.\-/](?P<y>\d{2})\s+(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        gd = m.groupdict()
        Y = int(gd.get("Y") or (2000 + int(gd["y"])))
        M = int(gd["m"]); D = int(gd["d"])
        H = int(gd.get("H") or 0); Mi = int(gd.get("M") or 0); S = int(gd.get("S") or 0)
        try:
            return datetime(Y, M, D, H, Mi, S)
        except ValueError:
            continue
    return None

def fmt(dt: datetime) -> Tuple[str,str]:
    return dt.strftime("%y%m%d"), dt.strftime("%H%M%S")

def save_if_new(cam: str, img_bytes: bytes, ts: datetime) -> str | None:
    day, hhmmss = fmt(ts)
    cam_dir = os.path.join(BASE_DIR, cam)
    ensure_dir(cam_dir)
    out = os.path.join(cam_dir, f"{cam}_{day}_{hhmmss}.jpg")
    if os.path.exists(out):
        # already saved this timestamp
        return None
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(img_bytes)
    os.replace(tmp, out)
    return out


# ------------- main -------------
def main() -> int:
    # load per-camera ROI config
    roi_cfg = load_roi_config()

    ensure_dir(BASE_DIR)
    for cam in CAMS:
        ensure_dir(os.path.join(BASE_DIR, cam))

    total_saved = 0

    for cam, url in CAMS.items():
        print(f"[{cam}] fetch…", flush=True)
        b = fetch(url)
        if not b:
            print(f"[{cam}] fetch failed", flush=True)
            continue

        # decode once, OCR ROI
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{cam}] decode failed", flush=True)
            continue

        roi = get_roi_for(cam, roi_cfg)
        text = ocr_from_roi(img, roi)
        ts   = parse_overlay_ts(text)
        if not ts:
            print(f"[{cam}] OCR failed (text='{text}') — skipping", flush=True)
            continue

        # only save if timestamp is new compared to last saved
        last = latest_saved_ts(cam)
        if last and ts <= last:
            print(f"[{cam}] timestamp not newer ({ts} <= {last}) — skip", flush=True)
            continue

        out = save_if_new(cam, b, ts)
        if out:
            print(f"[{cam}] saved {os.path.basename(out)}", flush=True)
            total_saved += 1
        else:
            print(f"[{cam}] already have this timestamp — skip", flush=True)

    print(f"saved_count={total_saved}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
