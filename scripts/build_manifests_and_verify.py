#!/usr/bin/env python3
"""
Build per-camera manifests from images/* and verify filenames via OCR.

Outputs:
  docs/manifests/<cam>.json  (always rebuilt)
  docs/manifests/report.json (summary of OCR checks)

By default this is a dry-run verifier (no renames). Use --rename-bad to rename.

Usage:
  python scripts/build_manifests_and_verify.py --write
  python scripts/build_manifests_and_verify.py --write --rename-bad
  python scripts/build_manifests_and_verify.py --write --debug-roi

Deps:
  pip install opencv-python-headless pytesseract pyyaml
"""

import os, re, json, time, argparse, shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2

# ---------- Cameras (keep in sync with capture) ----------
CAMS = ["borreguiles", "stadium", "satelite", "veleta"]

# Default ROI (percentages) if config/roi.(yml|yaml|json) is absent
DEFAULT_ROI = dict(top=0.90, left=0.85, bottom=0.985, right=0.995)

# ---------- ROI config loader ----------
def load_roi_config() -> Dict[str, Dict[str, float]]:
    for p in ("config/roi.yml", "config/roi.yaml", "config/roi.json"):
        f = Path(p)
        if not f.exists():
            continue
        try:
            if f.suffix in (".yml", ".yaml"):
                import yaml  # type: ignore
                return yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            else:
                return json.loads(f.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
            return {}
    return {}

def roi_for(cam: str, cfg: Dict[str, Dict[str, float]]):
    r = cfg.get(cam, {})
    t = float(r.get("top", DEFAULT_ROI["top"]))
    l = float(r.get("left", DEFAULT_ROI["left"]))
    b = float(r.get("bottom", DEFAULT_ROI["bottom"]))
    rr = float(r.get("right", DEFAULT_ROI["right"]))
    # clamp to [0,1]
    t = min(max(t, 0.0), 1.0)
    l = min(max(l, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    rr = min(max(rr, 0.0), 1.0)
    return t, l, b, rr

# ---------- OCR helpers (match capture quality) ----------
def preprocess_variants(roi_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = roi_bgr.shape[:2]
    scale = 2 if max(h, w) < 200 else 1
    if scale != 1:
        roi_bgr = cv2.resize(roi_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    g2 = cv2.convertScaleAbs(gray, alpha=1.9, beta=24)

    variants = []
    for g in (g1, g2):
        variants.append(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        variants.append(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        am = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        ami = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        variants.extend([am, ami])
    variants = [cv2.medianBlur(v, 3) for v in variants]
    return variants

def ocr_time_from_roi(img_bgr: np.ndarray, box_px, ocr_lang: str, debug_dir: Optional[Path]) -> Optional[Tuple[int,int,Optional[int],str]]:
    y0, y1, x0, x1 = box_px
    if y1 <= y0 or x1 <= x0:
        return None
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"roi_{int(time.time()*1000)}.jpg"), roi)

    import pytesseract
    last_txt = ""
    for th in preprocess_variants(roi):
        for psm in (7, 13):  # single line / raw line
            cfg = f"--psm {psm} -l {ocr_lang} -c tessedit_char_whitelist=0123456789:"
            try:
                txt = pytesseract.image_to_string(th, config=cfg)
            except Exception:
                txt = ""
            txt = " ".join(txt.split())
            last_txt = txt or last_txt
            m = re.search(r"\b(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?\b", txt)
            if m:
                H = int(m.group("H")); Mi = int(m.group("M")); S = m.group("S")
                S = int(S) if S is not None else None
                return H, Mi, S, txt
    # nothing clean; return last text for diagnostics
    if last_txt:
        return None
    return None

# ---------- Filename parsing ----------
def parse_name_dt(cam: str, name: str) -> Optional[Tuple[int,int,int,int,int,int]]:
    # <cam>_YYMMDD_HHMMSS(.suffix)?.jpg
    m = re.match(rf"^{re.escape(cam)}_(\d{{6}})_(\d{{6}})", name, re.I)
    if not m:
        return None
    d6, t6 = m.group(1), m.group(2)
    try:
        yy = 2000 + int(d6[0:2]); mo = int(d6[2:4]); da = int(d6[4:6])
        hh = int(t6[0:2]); mi = int(t6[2:4]); ss = int(t6[4:6])
        return yy, mo, da, hh, mi, ss
    except ValueError:
        return None

# ---------- Main build/verify ----------
def scan_camera(cam: str, images_dir: Path, roi_cfg, ocr_lang: str, debug: bool, rename_bad: bool):
    cam_dir = images_dir / cam
    files = []
    misnamed = []
    ocr_failed = []

    if not cam_dir.is_dir():
        return files, misnamed, ocr_failed

    t, l, b, r = roi_for(cam, roi_cfg)
    names = sorted([n for n in os.listdir(cam_dir) if n.lower().endswith((".jpg", ".jpeg"))])

    dbg_dir = Path("debug") / f"verify_{cam}" if debug else None

    for name in names:
        p = cam_dir / name
        parsed = parse_name_dt(cam, name)
        if not parsed:
            # skip files that don't follow our naming scheme
            continue

        # decode and OCR
        data = p.read_bytes()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            ocr_failed.append({"name": name, "reason": "decode-failed"})
            continue

        H, W = img.shape[:2]
        y0, y1 = int(H * t), int(H * b)
        x0, x1 = int(W * l), int(W * r)

        res = ocr_time_from_roi(img, (y0, y1, x0, x1), ocr_lang, dbg_dir)
        if not res:
            ocr_failed.append({"name": name, "reason": "ocr-time-not-found"})
            continue

        oH, oM, oS, raw_txt = res
        yy, mo, da, fH, fM, fS = parsed

        # Compare time components:
        # - If OCR lacks seconds, compare only HH:MM.
        # - If OCR has seconds, compare HH:MM:SS exactly.
        mismatch = False
        if oS is None:
            if (oH != fH) or (oM != fM):
                mismatch = True
        else:
            if (oH != fH) or (oM != fM) or (oS != fS):
                mismatch = True

        if mismatch:
            suggested = f"{cam}_{str(yy)[2:]}{mo:02d}{da:02d}_{oH:02d}{oM:02d}{(oS if oS is not None else fS):02d}.jpg"
            misnamed.append({
                "name": name,
                "ocr_time": f"{oH:02d}:{oM:02d}:{(oS if oS is not None else 0):02d}",
                "file_time": f"{fH:02d}:{fM:02d}:{fS:02d}",
                "raw": raw_txt,
                "suggested": suggested
            })
            if rename_bad:
                src = cam_dir / name
                dst = cam_dir / suggested
                if dst.exists():
                    # avoid clobber; add suffix
                    stem, ext = os.path.splitext(suggested)
                    k = 1
                    while (cam_dir / f"{stem}_{k}{ext}").exists():
                        k += 1
                    dst = cam_dir / f"{stem}_{k}{ext}"
                src.rename(dst)
                name = dst.name  # update in manifest

        files.append({"name": name, "path": f"images/{cam}/{name}"})

    # sort files by name (chronological by convention)
    files.sort(key=lambda x: x["name"])
    return files, misnamed, ocr_failed

def write_json_if_changed(path: Path, obj) -> bool:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old != data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="images", help="root images dir")
    ap.add_argument("--docs-dir", default="docs", help="docs root")
    ap.add_argument("--ocr-lang", default="eng", help="tesseract languages, e.g. 'eng' or 'eng+spa'")
    ap.add_argument("--write", action="store_true", help="write manifests to docs/manifests")
    ap.add_argument("--rename-bad", action="store_true", help="rename files whose OCR time mismatches filename time")
    ap.add_argument("--debug-roi", action="store_true", help="save ROI crops under debug/verify_<cam>")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    docs_dir = Path(args.docs_dir)
    man_dir = docs_dir / "manifests"

    roi_cfg = load_roi_config()

    global_report = {"cameras": {}, "renamed": bool(args.rename_bad)}
    any_changed = False

    for cam in CAMS:
        files, misnamed, fails = scan_camera(
            cam, images_dir, roi_cfg, args.ocr_lang, args.debug_roi, args.rename_bad
        )
        global_report["cameras"][cam] = {
            "count": len(files),
            "misnamed": misnamed,
            "ocr_failed": fails,
        }
        if args.write:
            changed = write_json_if_changed(man_dir / f"{cam}.json", files)
            any_changed = any_changed or changed

    if args.write:
        idx_changed = write_json_if_changed(man_dir / "index.json", {c: f"{c}.json" for c in CAMS})
        any_changed = any_changed or idx_changed

    # Write the verification report
    if args.write:
        write_json_if_changed(man_dir / "report.json", global_report)

    # Console summary
    print("\n=== Manifest & OCR Verification ===")
    for cam, data in global_report["cameras"].items():
        print(f"{cam:11s} :: files={data['count']:4d}  misnamed={len(data['misnamed']):3d}  ocr_failed={len(data['ocr_failed']):3d}")
        for m in data["misnamed"][:5]:
            print(f"    - {m['name']}  -> OCR {m['ocr_time']} (file {m['file_time']})  suggest: {m['suggested']}")
        if len(data["misnamed"]) > 5:
            print(f"    … {len(data['misnamed'])-5} more")
        for f in data["ocr_failed"][:3]:
            print(f"    ! OCR fail: {f['name']} [{f['reason']}]")
        if len(data["ocr_failed"]) > 3:
            print(f"    … {len(data['ocr_failed'])-3} more")
    print(f"\nWrote manifests: {bool(args.write)}; Any changed: {any_changed}; Renames applied: {bool(args.rename_bad)}")

if __name__ == "__main__":
    main()
