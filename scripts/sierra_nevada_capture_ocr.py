#!/usr/bin/env python3
"""
Sierra Nevada OCR Capture — ignore overnight zombie frames & unblock mornings.

Key ideas:
- Mask out the timestamp ROI and hash the rest. If content hasn't changed, skip saving.
- If the latest saved filename time is "in the future" vs now (frozen 18:xx in the morning),
  treat it as yesterday for ordering so morning/daytime frames aren't blocked.
- Optional AUTO_REPAIR=1: if it's early and last file is 'future' AND current content equals last,
  rename that last file to yesterday to clean the timeline.

Env (tune in workflow or terminal):
  OCR_LANG=eng
  CAM_TZ=Europe/Madrid
  FUTURE_BUF_MIN=90        # minutes ahead → consider last file "future" (default 60)
  MORNING_CUTOFF_H=11      # before this hour, AUTO_REPAIR is allowed
  DUP_HASH_TOL=6           # aHash Hamming threshold (0..64) after ROI masking
  DEBUG_ROI=0              # 1 => write debug/roi_*.jpg for tuning
  AUTO_REPAIR=0            # 1 => rename forward-dated last file to yesterday when safe
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

# --- Cameras ---
CAMS: Dict[str, str] = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}

BASE_DIR = "images"
TIMEOUT  = 20
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (SierraNevadaOCR/1.5)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DEFAULT_ROI = dict(top=0.90, left=0.85, bottom=0.985, right=0.995)

# --- Tunables via env ---
OCR_LANG        = os.environ.get("OCR_LANG", "eng")
CAM_TZ_NAME     = os.environ.get("CAM_TZ", "Europe/Madrid")
DEBUG_ROI       = os.environ.get("DEBUG_ROI", "0").lower() in ("1","true","yes")
FUTURE_BUF_MIN  = int(os.environ.get("FUTURE_BUF_MIN", "60"))
MORNING_CUTOFF  = int(os.environ.get("MORNING_CUTOFF_H", "11"))
DUP_HASH_TOL    = int(os.environ.get("DUP_HASH_TOL", "6"))
AUTO_REPAIR     = os.environ.get("AUTO_REPAIR", "0").lower() in ("1","true","yes")

# ---------- time helpers ----------
def now_local() -> datetime:
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(CAM_TZ_NAME)).replace(tzinfo=None)
        except Exception:
            pass
    # fallback UTC naive
    return datetime.utcnow()

def map_overlay_to_datetime(now_dt: datetime, H: int, M: int, S: int) -> datetime:
    """If overlay time is far in the future vs now, assume it's yesterday."""
    candidate = now_dt.replace(hour=H, minute=M, second=S, microsecond=0)
    if (candidate - now_dt) > timedelta(minutes=FUTURE_BUF_MIN):
        candidate -= timedelta(days=1)
    return candidate

def effective_last_ts(last_ts: Optional[datetime], now_dt: datetime) -> Optional[datetime]:
    """Shift 'future' last_ts back by one day so it doesn't block the morning."""
    if not last_ts:
        return None
    if (last_ts - now_dt) > timedelta(minutes=FUTURE_BUF_MIN):
        return last_ts - timedelta(days=1)
    return last_ts

# ---------- file helpers ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def latest_saved_path(cam: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(BASE_DIR, cam, f"{cam}_*.jpg")))
    return files[-1] if files else None

def parse_fname_ts(cam: str, name: str) -> Optional[datetime]:
    m = re.match(rf"^{re.escape(cam)}_(\d{{6}})_(\d{{6}})", name, re.I)
    if not m: return None
    d6, t6 = m.groups()
    try:
        return datetime.strptime(d6 + t6, "%y%m%d%H%M%S")
    except ValueError:
        return None

def save_frame(cam: str, img_bytes: bytes, ts: datetime) -> Optional[str]:
    outdir = os.path.join(BASE_DIR, cam); ensure_dir(outdir)
    name = f"{cam}_{ts.strftime('%y%m%d_%H%M%S')}.jpg"
    out = os.path.join(outdir, name)
    if os.path.exists(out):  # already saved
        return None
    tmp = out + ".tmp"
    with open(tmp, "wb") as f: f.write(img_bytes)
    os.replace(tmp, out)
    return out

def rename_to_yesterday(path: str) -> Optional[str]:
    """Rename an already-saved forward-dated file to yesterday (same clock time)."""
    base = os.path.basename(path)
    m = re.match(r"^(.*)_(\d{6})_(\d{6})\.jpg$", base, re.I)
    if not m: return None
    stem, d6, t6 = m.groups()
    ts = datetime.strptime(d6 + t6, "%y%m%d%H%M%S") - timedelta(days=1)
    newname = f"{stem}_{ts.strftime('%y%m%d_%H%M%S')}.jpg"
    dst = os.path.join(os.path.dirname(path), newname)
    if os.path.exists(dst):
        # add suffix to avoid collision
        k = 1
        root, ext = os.path.splitext(dst)
        while os.path.exists(f"{root}_{k}{ext}"): k += 1
        dst = f"{root}_{k}{ext}"
    os.replace(path, dst)
    return dst

# ---------- hashing (mask ROI first) ----------
def ahash64(img_gray: np.ndarray) -> int:
    small = cv2.resize(img_gray, (8,8), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small > avg).astype(np.uint8).flatten()
    v = 0
    for b in bits: v = (v << 1) | int(b)
    return v

def masked_hash(img_bgr: np.ndarray, roi_pct) -> int:
    h, w = img_bgr.shape[:2]
    top,left,bottom,right = roi_pct
    y0,y1 = int(h*top), int(h*bottom)
    x0,x1 = int(w*left), int(w*right)
    img = img_bgr.copy()
    # zero out the ROI so only scene content participates in the hash
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,0,0), thickness=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ahash64(gray)

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ---------- OCR bits ----------
def load_roi_config() -> Dict[str, Dict[str, float]]:
    for p in ("config/roi.yml", "config/roi.yaml", "config/roi.json"):
        if os.path.isfile(p):
            try:
                if p.endswith((".yml",".yaml")):
                    import yaml  # type: ignore
                    return (yaml.safe_load(open(p, "r", encoding="utf-8")) or {})
                else:
                    return (json.load(open(p, "r", encoding="utf-8")) or {})
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

def preprocess_variants(roi_bgr: np.ndarray):
    h,w = roi_bgr.shape[:2]
    if max(h,w) < 200:
        roi_bgr = cv2.resize(roi_bgr, (w*2,h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8))
    g1 = clahe.apply(gray)
    g2 = cv2.convertScaleAbs(gray, alpha=1.9, beta=24)
    outs = []
    for g in (g1,g2):
        outs.append(cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
        outs.append(cv2.threshold(g, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1])
        outs.append(cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10))
        outs.append(cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10))
    return [cv2.medianBlur(v,3) for v in outs]

def ocr_time_from_roi(img_bgr: np.ndarray, box) -> Optional[Tuple[int,int,int]]:
    y0,y1,x0,x1 = box
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0: return None
    if DEBUG_ROI:
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite(f"debug/roi_{int(time.time()*1000)}.jpg", roi)
    import pytesseract
    for th in preprocess_variants(roi):
        for psm in (7,13):
            cfg = f"--psm {psm} -l {OCR_LANG} -c tessedit_char_whitelist=0123456789:"
            try:
                txt = pytesseract.image_to_string(th, config=cfg)
            except Exception:
                txt = ""
            txt = " ".join(txt.split())
            m = re.search(r"\b(\d{2}):(\d{2})(?::(\d{2}))?\b", txt)
            if m:
                H, M, S = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
                if 0<=H<24 and 0<=M<60 and 0<=S<60:
                    return H,M,S
    return None

# ---------- main ----------
def main() -> int:
    roi_cfg = load_roi_config()
    for cam in CAMS: ensure_dir(os.path.join(BASE_DIR, cam))
    now_dt = now_local()

    for cam, url in CAMS.items():
        print(f"[{cam}] fetch…")
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                print(f"[{cam}] http {r.status_code}"); continue
            buf = r.content
        except Exception as e:
            print(f"[{cam}] fetch error: {e}"); continue

        arr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{cam}] decode failed"); continue

        # ---- content dedup: masked hash (ignore ROI / clock) ----
        top,left,bottom,right = roi_for(cam, roi_cfg)
        h,w = img.shape[:2]
        box = (int(h*top), int(h*bottom), int(w*left), int(w*right))

        cur_hash = masked_hash(img, (top,left,bottom,right))

        last_path = latest_saved_path(cam)
        if last_path:
            last_img = cv2.imdecode(np.fromfile(last_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            last_hash = masked_hash(last_img, (top,left,bottom,right)) if last_img is not None else None
        else:
            last_hash = None

        if last_hash is not None:
            d = hamming(cur_hash, last_hash)
            if d <= DUP_HASH_TOL:
                print(f"[{cam}] no scene change (d={d}) — skip")
                continue

        # ---- OCR the time (only after we know content changed) ----
        H_M_S = ocr_time_from_roi(img, box)
        if not H_M_S:
            print(f"[{cam}] OCR failed — skip")
            continue
        H, M, S = H_M_S
        ts = map_overlay_to_datetime(now_dt, H, M, S)

        # ---- unblock morning: compare vs effective last_ts ----
        last_ts = parse_fname_ts(cam, os.path.basename(last_path)) if last_path else None
        eff_last = effective_last_ts(last_ts, now_dt)

        if eff_last and ts <= eff_last:
            print(f"[{cam}] ts {ts} <= effective last {eff_last} — skip")
            # Optional morning auto-repair: rename forward-dated last if it's clearly the same frozen frame
            if AUTO_REPAIR and last_ts and (last_ts - now_dt) > timedelta(minutes=FUTURE_BUF_MIN) and now_dt.hour < MORNING_CUTOFF and last_hash is not None:
                d = hamming(cur_hash, last_hash)
                if d <= DUP_HASH_TOL:
                    newp = rename_to_yesterday(last_path)
                    if newp:
                        print(f"[{cam}] repaired forward-dated last → {os.path.basename(newp)}")
            continue

        out = save_frame(cam, buf, ts)
        if out:
            print(f"[{cam}] saved {os.path.basename(out)}")
        else:
            print(f"[{cam}] already exists — skip")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
