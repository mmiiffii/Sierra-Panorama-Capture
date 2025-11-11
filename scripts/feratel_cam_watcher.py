#!/usr/bin/env python3
"""
One-pass webcam grabber for Feratel JPG endpoints.

Modes
-----
--bootstrap   Save the current image for every camera immediately (first run).
(no flag)     Save only if that specific camera's bytes changed vs its last saved file.

Output
------
images/<cam>/<cam>_YYMMDD_HHMMSS.jpg
Filename time is parsed from the overlay by OCR; if OCR fails and REQUIRE_OCR=0, we fall back to UTC.

Env
---
REQUIRE_OCR=0|1   (default 0 -> allow UTC fallback)
OCR_LANG=eng      (you can set 'eng+spa' etc.)
"""

import os, glob, re, hashlib, requests, argparse, time
from datetime import datetime, timezone
import numpy as np
import cv2

# --------- CONFIG ---------
CAMS = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (FeratelWatcher/1.1)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
TIMEOUT = 20
BASE_DIR = "images"

REQUIRE_OCR = os.environ.get("REQUIRE_OCR", "0").lower() in ("1","true","yes")
OCR_LANG = os.environ.get("OCR_LANG", "eng")

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
# --------------------------

def log(msg): print(msg, flush=True)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def fetch(url: str) -> bytes | None:
    # add a cache-busting query param
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

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=18)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.medianBlur(thr, 3)
    return thr

def try_ocr_regions(b: bytes) -> str:
    if not OCR_AVAILABLE: return ""
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return ""
    h, w = img.shape[:2]
    rois = [
        img[int(h*0.82):int(h*0.98), int(w*0.04):int(w*0.96)],  # bottom strip
        img[int(h*0.02):int(h*0.18), int(w*0.55):int(w*0.96)],  # top-right
        img[int(h*0.02):int(h*0.18), int(w*0.04):int(w*0.45)],  # top-left
    ]
    cfg = f"--psm 6 -l {OCR_LANG} -c tessedit_char_whitelist=0123456789./:- "
    pieces = []
    for roi in rois:
        thr = preprocess_for_ocr(roi)
        try: pieces.append(pytesseract.image_to_string(thr, config=cfg))
        except Exception: pass
    # whole-frame fallback (downscaled)
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    thr = preprocess_for_ocr(small)
    try: pieces.append(pytesseract.image_to_string(thr, config=cfg))
    except Exception: pass
    return " ".join(" ".join(pieces).split())

def parse_timestamp(text: str):
    pats = [
        r"(?P<d>\d{2})[.\-/](?P<m>\d{2})[.\-/](?P<Y>\d{4})\s+(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?",
        r"(?P<Y>\d{4})[.\-/](?P<m>\d{2})[.\-/](?P<d>\d{2})\s+(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?",
        r"(?P<d>\d{2})[.\-/](?P<m>\d{2})[.\-/](?P<y>\d{2})\s+(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if not m: continue
        gd = m.groupdict()
        Y = int(gd.get("Y") or (2000 + int(gd["y"])))
        M = int(gd["m"]); D = int(gd["d"])
        H = int(gd.get("H") or 0); Mi = int(gd.get("M") or 0); S = int(gd.get("S") or 0)
        try: return datetime(Y, M, D, H, Mi, S)
        except ValueError: continue
    return None

def timestamp_from_image_or_now(b: bytes):
    txt = try_ocr_regions(b)
    dt = parse_timestamp(txt) if txt else None
    if dt: return dt
    return None if REQUIRE_OCR else datetime.now(timezone.utc).replace(tzinfo=None)

def fmt(dt: datetime): return dt.strftime("%y%m%d"), dt.strftime("%H%M%S")

def save_image(cam: str, b: bytes, dt: datetime) -> str:
    d, t = fmt(dt)
    cam_dir = os.path.join(BASE_DIR, cam)
    ensure_dir(cam_dir)
    path = os.path.join(cam_dir, f"{cam}_{d}_{t}.jpg")
    # avoid collision on reruns with same timestamp
    base = path[:-4]; idx = 1
    while os.path.exists(path):
        path = f"{base}_{idx}.jpg"; idx += 1
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: f.write(b)
    os.replace(tmp, path)
    return path

def run_bootstrap() -> list[str]:
    """Save current image for every cam (ignores change checks)."""
    saved = []
    for cam, url in CAMS.items():
        log(f"[{cam}] bootstrap fetch…")
        b = fetch(url)
        if not b:
            log(f"[{cam}] bootstrap fetch failed"); continue
        dt = timestamp_from_image_or_now(b)
        if not dt:
            log(f"[{cam}] OCR failed and REQUIRE_OCR=1 -> skipping"); continue
        p = save_image(cam, b, dt)
        log(f"[{cam}] bootstrap saved {os.path.basename(p)}")
        saved.append(p)
    return saved

def run_once() -> list[str]:
    """Save only when a cam changed vs last saved file."""
    saved = []
    for cam, url in CAMS.items():
        log(f"[{cam}] check…")
        b = fetch(url)
        if not b:
            log(f"[{cam}] fetch failed"); continue
        prev_md5 = latest_saved_md5(cam)
        cur_md5  = md5_bytes(b)
        if prev_md5 is not None and prev_md5 == cur_md5:
            log(f"[{cam}] no change"); continue
        dt = timestamp_from_image_or_now(b)
        if not dt:
            log(f"[{cam}] OCR failed and REQUIRE_OCR=1 -> skipping"); continue
        p = save_image(cam, b, dt)
        log(f"[{cam}] saved {os.path.basename(p)}")
        saved.append(p)
    return saved

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true", help="Save current frame for every camera now.")
    args = ap.parse_args()

    ensure_dir(BASE_DIR)
    for cam in CAMS: ensure_dir(os.path.join(BASE_DIR, cam))

    if args.bootstrap:
        run_bootstrap()
    else:
        run_once()
