#!/usr/bin/env python3
"""
Watches multiple Feratel JPG webcams individually. When a camera's image bytes change,
extract the timestamp from the overlay via OCR and save to:

  images/<cam>/<cam>_YYMMDD_HHMMSS.jpg

Requires: requests, numpy, opencv-python-headless, pytesseract, and system tesseract-ocr.
"""

import os, time, glob, re, hashlib, requests
from datetime import datetime
import numpy as np
import cv2

# ---------- CONFIG ----------
CAMS = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (FeratelWatcher/1.0)"}
TIMEOUT = 20
BASE_DIR = "images"
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "60"))  # seconds between checks

# OCR must be available; if not, we skip saves (you said you want the timestamp from the image).
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
# ---------------------------


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def fetch(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None


def last_saved_md5(cam: str) -> str | None:
    """Compute MD5 of the newest saved file for this camera (raw bytes)."""
    cam_dir = os.path.join(BASE_DIR, cam)
    files = sorted(glob.glob(os.path.join(cam_dir, f"{cam}_*.jpg")))
    if not files:
        return None
    latest = files[-1]
    try:
        with open(latest, "rb") as f:
            return md5_bytes(f.read())
    except Exception:
        return None


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Basic enhance + binarize suitable for timestamp overlays."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=18)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.medianBlur(thr, 3)
    return thr


def try_ocr_regions(b: bytes) -> str:
    """Try several likely overlay regions; return concatenated OCR text."""
    txt_all = []
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""

    h, w = img.shape[:2]
    rois = [
        img[int(h*0.82):int(h*0.98), int(w*0.04):int(w*0.96)],  # bottom strip
        img[int(h*0.02):int(h*0.18), int(w*0.55):int(w*0.96)],  # top-right
        img[int(h*0.02):int(h*0.18), int(w*0.04):int(w*0.45)],  # top-left
        img[int(h*0.80):int(h*0.98), int(w*0.55):int(w*0.96)],  # bottom-right
    ]
    cfg = "--psm 6 -c tessedit_char_whitelist=0123456789./:- "
    for roi in rois:
        thr = preprocess_for_ocr(roi)
        try:
            txt = pytesseract.image_to_string(thr, config=cfg)
        except Exception:
            txt = ""
        txt_all.append(txt)

    # Fallback: small downscaled whole-frame OCR (in case overlay position differs)
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    thr = preprocess_for_ocr(small)
    try:
        txt_all.append(pytesseract.image_to_string(thr, config=cfg))
    except Exception:
        pass

    # Normalize
    return " ".join(" ".join(txt_all).split())


def parse_timestamp_from_text(text: str) -> datetime | None:
    """
    Try multiple numeric date/time patterns.
    Examples:
      11.11.2025 12:45:33
      2025-11-11 12:45
      11/11/25 12:45:33
    """
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
        M = int(gd["m"])
        D = int(gd["d"])
        H = int(gd.get("H") or 0)
        Mi = int(gd.get("M") or 0)
        S = int(gd.get("S") or 0)
        try:
            return datetime(Y, M, D, H, Mi, S)
        except ValueError:
            continue
    return None


def extract_timestamp(b: bytes) -> datetime | None:
    if not OCR_AVAILABLE:
        return None
    text = try_ocr_regions(b)
    if not text:
        return None
    return parse_timestamp_from_text(text)


def fmt(dt: datetime) -> tuple[str, str]:
    return dt.strftime("%y%m%d"), dt.strftime("%H%M%S")


def save_image(cam: str, b: bytes, dt: datetime) -> str:
    day, t = fmt(dt)
    cam_dir = os.path.join(BASE_DIR, cam)
    ensure_dir(cam_dir)
    path = os.path.join(cam_dir, f"{cam}_{day}_{t}.jpg")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, path)
    return path


def check_once_for_cam(cam: str, url: str) -> str | None:
    """Return saved filepath if saved, else None."""
    b = fetch(url)
    if not b:
        return None

    prev_md5 = last_saved_md5(cam)
    if prev_md5 == md5_bytes(b):
        # No change for this camera
        return None

    # Must have timestamp from image (per user instruction)
    dt = extract_timestamp(b)
    if not dt:
        # Skip saving if we can't read the timestamp from the image
        print(f"[{cam}] OCR failed; skipping this frame.")
        return None

    saved = save_image(cam, b, dt)
    return saved


def main_loop():
    # Ensure base and cam folders
    ensure_dir(BASE_DIR)
    for cam in CAMS:
        ensure_dir(os.path.join(BASE_DIR, cam))

    while True:
        any_saved = []
        for cam, url in CAMS.items():
            try:
                saved = check_once_for_cam(cam, url)
                if saved:
                    any_saved.append(saved)
                    print(f"[{cam}] saved {os.path.basename(saved)}")
            except Exception as e:
                print(f"[{cam}] error: {e}")

        # Print a compact heartbeat
        if not any_saved:
            print("No updates.")
        time.sleep(max(5, CHECK_INTERVAL))


if __name__ == "__main__":
    main_loop()
