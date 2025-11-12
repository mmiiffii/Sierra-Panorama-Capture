#!/usr/bin/env python3
"""
Capture Sierra Nevada JPG webcams once, using OCR from a fixed bottom-right ROI.

- Reads HH:MM[:SS] from overlay (strict OCR).
- Date comes from Europe/Madrid on runner time (image often shows time-only).
- Writes only if timestamp is newer than last saved file per camera.
- Output: images/<cam>/<cam>_YYMMDD_HHMMSS.jpg

Config (percentages in [0..1], per camera). If missing, DEFAULT_ROI is used:
  config/roi.yml or config/roi.json:
    borreguiles: {top: 0.90, left: 0.85, bottom: 0.985, right: 0.995}
    stadium:     {top: 0.90, left: 0.85, bottom: 0.985, right: 0.995}
    satelite:    {top: 0.90, left: 0.85, bottom: 0.985, right: 0.995}
    veleta:      {top: 0.90, left: 0.85, bottom: 0.985, right: 0.995}

Env:
  OCR_LANG=eng            # e.g. 'eng' or 'eng+spa'
  CAM_TZ=Europe/Madrid    # timezone to source the date from
  DEBUG_ROI=0|1           # save ROI & threshold previews under ./debug
"""

import os, re, glob, time, json, hashlib, requests
from datetime import datetime, date
from typing import Dict, Tuple, Optional

import numpy as np
import cv2

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # fallback later

# ---------- Cameras ----------
CAMS: Dict[str, str] = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}

BASE_DIR = "images"
TIMEOUT  = 20
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (SierraNevadaOCR/1.1)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DEFAULT_ROI = dict(top=0.90, left=0.85, bottom=0.985, right=0.995)  # wide & safe for your sample

OCR_LANG = os.environ.get("OCR_LANG", "eng")
CAM_TZ   = os.environ.get("CAM_TZ", "Europe/Madrid")
DEBUG_ROI = os.environ.get("DEBUG_ROI", "0").lower() in ("1","true","yes")

# ---------- Small utils ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def fetch(url: str) -> Optional[bytes]:
    sep = "&" if "?" in url else "?"
    bust = f"{sep}_ts={int(time.time())}"
    try:
        r = requests.get(url + bust, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def latest_saved_ts(cam: str) -> Optional[datetime]:
    """Newest timestamp from filenames images/<cam>/<cam>_YYMMDD_HHMMSS.jpg"""
    files = sorted(glob.glob(os.path.join(BASE_DIR, cam, f"{cam}_*.jpg")))
    if not files: return None
    name = os.path.basename(files[-1])
    m = re.search(rf"^{re.escape(cam)}_(\d{{6}})_(\d{{6}})", name)
    if not m: return None
    d6, t6 = m.groups()
    try:
        return datetime.strptime(d6+t6, "%y%m%d%H%M%S")
    except ValueError:
        return None

def local_today(tz_name: str) -> date:
    if ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
            return datetime.now(tz).date()
        except Exception:
            pass
    return datetime.utcnow().date()

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
    # clamp
    top = max(0.0, min(1.0, top)); left = max(0.0, min(1.0, left))
    bottom = max(0.0, min(1.0, bottom)); right = max(0.0, min(1.0, right))
    return top, left, bottom, right

# ---------- OCR helpers ----------
def preprocess_variants(roi_bgr: np.ndarray) -> list:
    """Return multiple binarized variants to try with Tesseract."""
    # upscale to help Tesseract
    h, w = roi_bgr.shape[:2]
    scale = 2 if max(h, w) < 200 else 1
    if scale != 1:
        roi_bgr = cv2.resize(roi_bgr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g1 = clahe.apply(gray)
    g2 = cv2.convertScaleAbs(gray, alpha=1.9, beta=24)

    variants = []
    for g in (g1, g2):
        # Otsu
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(th)
        # Inverted
        _, thi = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(thi)
        # Adaptive (mean)
        am = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, 10)
        variants.append(am)
        # Adaptive inverted
        ami = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)
        variants.append(ami)
    # tidy shapes
    variants = [cv2.medianBlur(v, 3) for v in variants]
    return variants

def ocr_text_from_roi(img_bgr: np.ndarray, roi_box) -> str:
    y0,y1,x0,x1 = roi_box  # absolute px coords
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0: return ""
    if DEBUG_ROI:
        ensure_dir("debug")
        cv2.imwrite(f"debug/roi_{int(time.time()*1000)}.jpg", roi)

    import pytesseract
    # Try a few variants + PSMs; stop at first regex match
    for th in preprocess_variants(roi):
        for psm in (7, 13):  # single line, raw line
            cfg = f"--psm {psm} -l {OCR_LANG} -c tessedit_char_whitelist=0123456789:"
            try:
                txt = pytesseract.image_to_string(th, config=cfg)
            except Exception:
                txt = ""
            txt = " ".join(txt.split())
            if re.search(r"\b\d{2}:\d{2}(:\d{2})?\b", txt):
                return txt
    return txt  # last attempt, may be empty

def parse_time(text: str) -> Optional[Tuple[int,int,int]]:
    m = re.search(r"\b(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?\b", text)
    if not m: return None
    H = int(m.group("H")); Mi = int(m.group("M")); S = int(m.group("S") or 0)
    if not (0 <= H < 24 and 0 <= Mi < 60 and 0 <= S < 60): return None
    return H, Mi, S

def format_dt(dt: datetime) -> Tuple[str,str]:
    return dt.strftime("%y%m%d"), dt.strftime("%H%M%S")

def save_if_new(cam: str, img_bytes: bytes, ts: datetime) -> Optional[str]:
    day, hms = format_dt(ts)
    cam_dir = os.path.join(BASE_DIR, cam); ensure_dir(cam_dir)
    out = os.path.join(cam_dir, f"{cam}_{day}_{hms}.jpg")
    if os.path.exists(out):  # already saved this timestamp
        return None
    tmp = out + ".tmp"
    with open(tmp, "wb") as f: f.write(img_bytes)
    os.replace(tmp, out)
    return out

# ---------- Main ----------
def main() -> int:
    roi_cfg = load_roi_config()

    ensure_dir(BASE_DIR)
    for cam in CAMS: ensure_dir(os.path.join(BASE_DIR, cam))

    # timezone for date part
    today_local = local_today(CAM_TZ)

    total_saved = 0
    for cam, url in CAMS.items():
        print(f"[{cam}] fetch…", flush=True)
        b = fetch(url)
        if not b:
            print(f"[{cam}] fetch failed", flush=True); continue

        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{cam}] decode failed", flush=True); continue

        h,w = img.shape[:2]
        top,left,bottom,right = get_roi_for(cam, roi_cfg)
        y0,y1 = int(h*top), int(h*bottom)
        x0,x1 = int(w*left), int(w*right)

        text = ocr_text_from_roi(img, (y0,y1,x0,x1))
        t_hms = parse_time(text)
        if not t_hms:
            print(f"[{cam}] OCR failed (txt='{text}') — skip", flush=True)
            continue

        H,Mi,S = t_hms
        ts = datetime(today_local.year, today_local.month, today_local.day, H, Mi, S)

        last = latest_saved_ts(cam)
        if last and ts <= last:
            print(f"[{cam}] not newer ({ts} <= {last}) — skip", flush=True)
            continue

        out = save_if_new(cam, b, ts)
        if out:
            print(f"[{cam}] saved {os.path.basename(out)}", flush=True)
            total_saved += 1
        else:
            print(f"[{cam}] already have {ts} — skip", flush=True)

    print(f"saved_count={total_saved}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
