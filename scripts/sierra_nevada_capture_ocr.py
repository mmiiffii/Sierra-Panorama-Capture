#!/usr/bin/env python3
"""
Sierra Nevada capture — byte-hash first, OCR second.

What changes:
- Decide to save based on the JPG's SHA-1 (server actually changed = save).
- OCR is only for naming; if OCR fails, we save using local time anyway.
- Overnight "frozen frame" is naturally ignored (bytes identical).
- Optional auto-repair renames a forward-dated last file to yesterday.

Env knobs:
  CAM_TZ=Europe/Madrid
  OCR_LANG=eng
  FUTURE_BUF_MIN=90        # minutes ahead -> consider last file 'future' (default 60)
  MORNING_CUTOFF_H=11      # before this hour, auto-repair allowed
  AUTO_REPAIR=0            # set 1 to auto-rename future-dated last file to yesterday
  DEBUG_ROI=0              # 1 to dump ROI crops (only for debugging OCR)
"""

import os, re, glob, time, json, hashlib, requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---- Cameras (unchanged endpoints) ----
CAMS: Dict[str, str] = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}

BASE_DIR  = "images"
STATE_DIR = "state"   # stores last SHA-1 per camera

TIMEOUT = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (SierraNevadaCapture/2.0)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# OCR / ROI (used only for naming; saving doesn't depend on it)
DEFAULT_ROI = dict(top=0.90, left=0.85, bottom=0.985, right=0.995)

CAM_TZ_NAME     = os.environ.get("CAM_TZ", "Europe/Madrid")
OCR_LANG        = os.environ.get("OCR_LANG", "eng")
DEBUG_ROI       = os.environ.get("DEBUG_ROI", "0").lower() in ("1","true","yes")
FUTURE_BUF_MIN  = int(os.environ.get("FUTURE_BUF_MIN", "60"))
MORNING_CUTOFF  = int(os.environ.get("MORNING_CUTOFF_H", "11"))
AUTO_REPAIR     = os.environ.get("AUTO_REPAIR", "0").lower() in ("1","true","yes")

def now_local() -> datetime:
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(CAM_TZ_NAME)).replace(tzinfo=None)
        except Exception:
            pass
    return datetime.utcnow()

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def fetch_bytes(url: str) -> Optional[bytes]:
    bust = ("&" if "?" in url else "?") + f"_ts={int(time.time())}"
    try:
        r = requests.get(url + bust, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def sha1_hex(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def state_path(cam: str) -> str:
    return os.path.join(STATE_DIR, f"{cam}.sha1")

def read_last_sha(cam: str) -> Optional[str]:
    p = state_path(cam)
    if os.path.isfile(p):
        return open(p, "r", encoding="utf-8").read().strip() or None
    # fallback: hash latest saved file if state missing
    last = latest_saved_path(cam)
    if last:
        try:
            return sha1_hex(open(last, "rb").read())
        except Exception:
            return None
    return None

def write_last_sha(cam: str, hexval: str):
    ensure_dir(STATE_DIR)
    tmp = state_path(cam) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(hexval + "\n")
    os.replace(tmp, state_path(cam))

def latest_saved_path(cam: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(BASE_DIR, cam, f"{cam}_*.jpg")))
    return files[-1] if files else None

def parse_fname_ts(cam: str, name: str) -> Optional[datetime]:
    m = re.match(rf"^{re.escape(cam)}_(\d{{6}})_(\d{{6}})\.jpg$", name, re.I)
    if not m: return None
    d6, t6 = m.groups()
    try:
        return datetime.strptime(d6 + t6, "%y%m%d%H%M%S")
    except ValueError:
        return None

# ---- OCR just for naming ----
def load_roi_config() -> Dict[str, Dict[str, float]]:
    for p in ("config/roi.yml","config/roi.yaml","config/roi.json"):
        if os.path.isfile(p):
            try:
                if p.endswith((".yml",".yaml")):
                    import yaml  # type: ignore
                    return (yaml.safe_load(open(p,"r",encoding="utf-8")) or {})
                else:
                    import json as _json
                    return (_json.load(open(p,"r",encoding="utf-8")) or {})
            except Exception:
                return {}
    return {}

def roi_for(cam: str, cfg: Dict[str, Dict[str, float]]):
    r = (cfg.get(cam) or {})
    clamp = lambda x: min(max(float(x), 0.0), 1.0)
    return (
        clamp(r.get("top",    DEFAULT_ROI["top"])),
        clamp(r.get("left",   DEFAULT_ROI["left"])),
        clamp(r.get("bottom", DEFAULT_ROI["bottom"])),
        clamp(r.get("right",  DEFAULT_ROI["right"])),
    )

def ocr_hms(img_bgr: np.ndarray, box) -> Optional[Tuple[int,int,int]]:
    y0,y1,x0,x1 = box
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0: return None
    if DEBUG_ROI:
        ensure_dir("debug")
        cv2.imwrite(f"debug/roi_{int(time.time()*1000)}.jpg", roi)

    import pytesseract
    def variants(roi):
        h,w = roi.shape[:2]
        if max(h,w) < 200:
            roi = cv2.resize(roi, (w*2,h*2), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0,(8,8))
        g1 = clahe.apply(g)
        g2 = cv2.convertScaleAbs(g, alpha=1.9, beta=24)
        outs = []
        for x in (g1,g2):
            outs.append(cv2.threshold(x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
            outs.append(cv2.threshold(x,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1])
            outs.append(cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10))
            outs.append(cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10))
        return [cv2.medianBlur(v,3) for v in outs]

    for th in variants(roi):
        for psm in (7,13):
            cfg = f"--psm {psm} -l {OCR_LANG} -c tessedit_char_whitelist=0123456789:"
            try:
                txt = pytesseract.image_to_string(th, config=cfg)
            except Exception:
                txt = ""
            txt = " ".join(txt.split())
            m = re.search(r"\b(\d{2}):(\d{2})(?::(\d{2}))?\b", txt)
            if m:
                H,M,S = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
                if 0<=H<24 and 0<=M<60 and 0<=S<60:
                    return H,M,S
    return None

def map_overlay_to_datetime(now_dt: datetime, H: int, M: int, S: int) -> datetime:
    cand = now_dt.replace(hour=H, minute=M, second=S, microsecond=0)
    if (cand - now_dt) > timedelta(minutes=FUTURE_BUF_MIN):
        cand -= timedelta(days=1)
    return cand

# ---- saving & optional repair ----
def save_with_ts(cam: str, img_bytes: bytes, ts: datetime) -> str:
    outdir = os.path.join(BASE_DIR, cam); ensure_dir(outdir)
    out = os.path.join(outdir, f"{cam}_{ts.strftime('%y%m%d_%H%M%S')}.jpg")
    if os.path.exists(out):
        # rare collision; add suffix
        base, ext = os.path.splitext(out); k = 1
        while os.path.exists(f"{base}_{k}{ext}"): k += 1
        out = f"{base}_{k}{ext}"
    tmp = out + ".tmp"
    with open(tmp, "wb") as f: f.write(img_bytes)
    os.replace(tmp, out)
    return out

def rename_last_to_yesterday(path: str) -> Optional[str]:
    b = os.path.basename(path)
    m = re.match(r"^(.*)_(\d{6})_(\d{6})\.jpg$", b, re.I)
    if not m: return None
    stem, d6, t6 = m.groups()
    ts = datetime.strptime(d6+t6, "%y%m%d%H%M%S") - timedelta(days=1)
    new = os.path.join(os.path.dirname(path), f"{stem}_{ts.strftime('%y%m%d_%H%M%S')}.jpg")
    if os.path.exists(new):
        base, ext = os.path.splitext(new); k = 1
        while os.path.exists(f"{base}_{k}{ext}"): k += 1
        new = f"{base}_{k}{ext}"
    os.replace(path, new)
    return new

# ---- main loop ----
def main() -> int:
    ensure_dir(BASE_DIR); ensure_dir(STATE_DIR)
    roi_cfg = load_roi_config()
    now_dt = now_local()

    for cam, url in CAMS.items():
        print(f"[{cam}] fetch…", flush=True)
        b = fetch_bytes(url)
        if not b:
            print(f"[{cam}] fetch failed", flush=True); continue

        cur_sha = sha1_hex(b)
        last_sha = read_last_sha(cam)
        if last_sha == cur_sha:
            print(f"[{cam}] no server update (SHA-1 match) — skip", flush=True)
            continue

        # Attempt OCR (optional naming); if OCR fails, fallback to now_dt
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{cam}] decode failed — skip", flush=True)
            continue

        h,w = img.shape[:2]
        top,left,bottom,right = roi_for(cam, roi_cfg)
        box = (int(h*top), int(h*bottom), int(w*left), int(w*right))

        Hms = ocr_hms(img, box)
        if Hms:
            ts = map_overlay_to_datetime(now_dt, *Hms)
        else:
            # fallback: use local time so the viewer still gets a new frame
            ts = now_dt

        # Optional morning repair: if the last saved name is "in the future", relabel to yesterday
        last_path = latest_saved_path(cam)
        if AUTO_REPAIR and last_path:
            last_ts = parse_fname_ts(cam, os.path.basename(last_path))
            if last_ts and (last_ts - now_dt) > timedelta(minutes=FUTURE_BUF_MIN) and now_dt.hour < MORNING_CUTOFF:
                newp = rename_last_to_yesterday(last_path)
                if newp:
                    print(f"[{cam}] repaired forward-dated last → {os.path.basename(newp)}", flush=True)

        out = save_with_ts(cam, b, ts)
        print(f"[{cam}] saved {os.path.basename(out)}", flush=True)

        # update state
        write_last_sha(cam, cur_sha)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
