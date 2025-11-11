#!/usr/bin/env python3
"""
Capture all Feratel JPG webcams once:
- For each camera, fetch image bytes (cache-busted).
- If bytes differ from the last saved image for that camera, save a new file
  to images/<cam>/<cam>_YYMMDD_HHMMSS.jpg
- Filename time comes from the overlay (OCR) taken from the BOTTOM-RIGHT ROI ONLY.
  If OCR fails and REQUIRE_OCR=0, fall back to current UTC.

Env:
  REQUIRE_OCR=0|1   (default 0 -> allow UTC fallback if OCR fails)
  OCR_LANG=eng      (e.g. 'eng+spa')
  # ROI as percentages [0..1] (BOTTOM-RIGHT rectangle):
  ROI_TOP=0.82
  ROI_LEFT=0.55
  ROI_BOTTOM=0.98
  ROI_RIGHT=0.98
"""

import os, glob, re, hashlib, requests, time
from datetime import datetime, timezone
import numpy as np
import cv2

CAMS = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}
BASE_DIR = "images"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (FeratelCapture/1.1)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
TIMEOUT = 20

REQUIRE_OCR = os.environ.get("REQUIRE_OCR", "0").lower() in ("1","true","yes")
OCR_LANG = os.environ.get("OCR_LANG", "eng")

ROI_TOP = float(os.environ.get("ROI_TOP", "0.82"))
ROI_LEFT = float(os.environ.get("ROI_LEFT", "0.55"))
ROI_BOTTOM = float(os.environ.get("ROI_BOTTOM", "0.98"))
ROI_RIGHT = float(os.environ.get("ROI_RIGHT", "0.98"))

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


def ensure_dir(p): os.makedirs(p, exist_ok=True)
def log(*a): print(*a, flush=True)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def fetch(url: str) -> bytes | None:
    # add a cache-busting query param so CDNs don't serve stale bytes
    sep = "&" if "?" in url else "?"
    bust = f"{sep}_ts={int(time.time())}"
    try:
        r = requests.get(url + bust, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None

def latest_saved_path(cam: str) -> str | None:
    files = sorted(glob.glob(os.path.join(BASE_DIR, cam, f"{cam}_*.jpg")))
    return files[-1] if files else None

def latest_saved_md5(cam: str) -> str | None:
    p = latest_saved_path(cam)
    if not p: return None
    try:
        with open(p, "rb") as f: return md5_bytes(f.read())
    except Exception: return None


# ---------- OCR (bottom-right only) ----------
def ocr_bottom_right(b: bytes) -> str:
    """Read only the bottom-right overlay area (faster & more reliable)."""
    if not OCR_AVAILABLE:
        return ""
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    y0, y1 = int(h*ROI_TOP), int(h*ROI_BOTTOM)
    x0, x1 = int(w*ROI_LEFT), int(w*ROI_RIGHT)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return ""

    # preprocess for crisp digits
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.medianBlur(thr, 3)
    # small dilation can connect thin digits
    thr = cv2.dilate(thr, np.ones((2,2), np.uint8), iterations=1)

    # try a single-line assumption first, then generic
    cfg1 = f"--psm 7 -l {OCR_LANG} -c tessedit_char_whitelist=0123456789./:- "
    cfg2 = f"--psm 6 -l {OCR_LANG} -c tessedit_char_whitelist=0123456789./:- "
    try:
        txt = pytesseract.image_to_string(thr, config=cfg1)
        if not txt.strip():
            txt = pytesseract.image_to_string(thr, config=cfg2)
    except Exception:
        txt = ""
    return " ".join(txt.split())

def parse_timestamp(text: str):
    # Common European overlays
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

def timestamp_from_image_or_now(b: bytes):
    txt = ocr_bottom_right(b)
    dt = parse_timestamp(txt) if txt else None
    if dt:
        return dt
    return None if REQUIRE_OCR else datetime.now(timezone.utc).replace(tzinfo=None)
# --------------------------------------------


def fmt(dt: datetime): return dt.strftime("%y%m%d"), dt.strftime("%H%M%S")

def save_image(cam: str, b: bytes, dt: datetime) -> str:
    d, t = fmt(dt)
    cam_dir = os.path.join(BASE_DIR, cam); ensure_dir(cam_dir)
    path = os.path.join(cam_dir, f"{cam}_{d}_{t}.jpg")
    base = path[:-4]; idx = 1
    while os.path.exists(path):
        path = f"{base}_{idx}.jpg"; idx += 1
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: f.write(b)
    os.replace(tmp, path)
    return path


def main():
    ensure_dir(BASE_DIR)
    total_saved = 0
    for cam, url in CAMS.items():
        ensure_dir(os.path.join(BASE_DIR, cam))
        b = fetch(url)
        if not b:
            log(f"[{cam}] fetch failed"); continue

        cur = md5_bytes(b)
        prev = latest_saved_md5(cam)
        if prev is not None and prev == cur:
            log(f"[{cam}] duplicate, skip"); continue

        dt = timestamp_from_image_or_now(b)
        if not dt:
            log(f"[{cam}] OCR failed and REQUIRE_OCR=1 -> skip"); continue

        p = save_image(cam, b, dt)
        log(f"[{cam}] saved {os.path.basename(p)}")
        total_saved += 1

    log(f"saved_count={total_saved}")

if __name__ == "__main__":
    main()
