#!/usr/bin/env python3
"""
Build docs/manifests/<cam>.json from images/<cam>/*.jpg.
Optionally verify filenames vs OCR’d time in the bottom-right ROI.

It never errors if OCR libs aren’t installed; it will just skip verification.

Usage:
  python scripts/make_manifests.py --write
  python scripts/make_manifests.py --write --verify-ocr
  python scripts/make_manifests.py --write --verify-ocr --rename-bad
"""

import os, re, json, argparse
from pathlib import Path

# Cameras you use
CAMS = ["borreguiles", "stadium", "satelite", "veleta"]

# Default ROI (percentages) if no config present
DEFAULT_ROI = dict(top=0.90, left=0.85, bottom=0.985, right=0.995)

def load_roi():
    for p in ("config/roi.yml", "config/roi.yaml", "config/roi.json"):
        f = Path(p)
        if f.exists():
            try:
                if f.suffix in (".yml", ".yaml"):
                    import yaml  # type: ignore
                    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {})
                else:
                    return (json.loads(f.read_text(encoding="utf-8")) or {})
            except Exception:
                return {}
    return {}

def roi_for(cam, cfg):
    r = cfg.get(cam, {})
    top = float(r.get("top", DEFAULT_ROI["top"]))
    left = float(r.get("left", DEFAULT_ROI["left"]))
    bottom = float(r.get("bottom", DEFAULT_ROI["bottom"]))
    right = float(r.get("right", DEFAULT_ROI["right"]))
    clamp = lambda x: min(max(x, 0.0), 1.0)
    return clamp(top), clamp(left), clamp(bottom), clamp(right)

def parse_name(cam, name):
    m = re.match(rf"^{re.escape(cam)}_(\d{{6}})_(\d{{6}})", name, re.I)
    return m.groups() if m else None

def build_list_for_cam(cam, images_root: Path):
    cam_dir = images_root / cam
    if not cam_dir.is_dir():
        return []
    files = [n for n in os.listdir(cam_dir) if n.lower().endswith((".jpg", ".jpeg"))]
    files = [n for n in files if parse_name(cam, n)]
    files.sort()
    return [{"name": n, "path": f"images/{cam}/{n}"} for n in files]

def write_if_changed(p: Path, obj) -> bool:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    old = p.read_text("utf-8") if p.exists() else None
    if old != data:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, "utf-8")
        return True
    return False

# ------------- Optional OCR verification -------------
def try_import_ocr():
    try:
        import cv2, numpy as np, pytesseract  # noqa: F401
        return True
    except Exception:
        return False

def ocr_time(path: Path, roi_pct, ocr_lang="eng"):
    import cv2, numpy as np, pytesseract, re
    b = path.read_bytes()
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, "decode-failed"
    H, W = img.shape[:2]
    top, left, bottom, right = roi_pct
    y0, y1 = int(H*top), int(H*bottom)
    x0, x1 = int(W*left), int(W*right)
    if y1 <= y0 or x1 <= x0: return None, "bad-roi"
    roi = img[y0:y1, x0:x1]

    # simple preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.9, beta=24)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    for psm in (7, 13):
        cfg = f"--psm {psm} -l {ocr_lang} -c tessedit_char_whitelist=0123456789:"
        try:
            txt = pytesseract.image_to_string(thr, config=cfg)
        except Exception:
            txt = ""
        txt = " ".join(txt.split())
        m = re.search(r"\b(\d{2}):(\d{2})(?::(\d{2}))?\b", txt)
        if m:
            Hh, Mm, Ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            return (Hh, Mm, Ss), txt
    return None, "no-time-found"

def verify_cam(cam, files, images_root: Path, roi_cfg, ocr_lang="eng"):
    problems = {"misnamed": [], "ocr_failed": []}
    roi = roi_for(cam, roi_cfg)
    for item in files:
        name = item["name"]
        parsed = parse_name(cam, name)
        if not parsed:
            continue
        d6, t6 = parsed
        fH, fM, fS = int(t6[0:2]), int(t6[2:4]), int(t6[4:6])
        path = images_root / cam / name
        got, raw = ocr_time(path, roi, ocr_lang)
        if got is None:
            problems["ocr_failed"].append({"name": name, "reason": raw})
            continue
        oH, oM, oS = got
        if (oH != fH) or (oM != fM) or (oS != fS):
            problems["misnamed"].append({
                "name": name,
                "file_time": f"{fH:02d}:{fM:02d}:{fS:02d}",
                "ocr_time":  f"{oH:02d}:{oM:02d}:{oS:02d}",
                "raw": raw
            })
    return problems

def maybe_rename(cam, problems, images_root: Path):
    # rename misnamed to match OCR time; avoids clashes with a suffix
    for m in problems["misnamed"]:
        old = images_root / cam / m["name"]
        yy = m["name"][len(cam)+1:len(cam)+3]     # YY from filename (keep same day)
        mo = m["name"][len(cam)+3:len(cam)+5]
        dd = m["name"][len(cam)+5:len(cam)+7]
        oH, oM, oS = m["ocr_time"].split(":")
        new = f"{cam}_{yy}{mo}{dd}_{oH}{oM}{oS}.jpg"
        dst = images_root / cam / new
        if dst.exists():
            stem, ext = os.path.splitext(new); k = 1
            while (images_root / cam / f"{stem}_{k}{ext}").exists(): k += 1
            dst = images_root / cam / f"{stem}_{k}{ext}"
        old.rename(dst)
        m["renamed_to"] = dst.name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--docs",   default="docs")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify-ocr", action="store_true")
    ap.add_argument("--rename-bad", action="store_true")
    ap.add_argument("--ocr-lang", default="eng")
    args = ap.parse_args()

    images_root = Path(args.images)
    manifests_root = Path(args.docs) / "manifests"

    roi_cfg = load_roi()
    all_changed = False
    report = {"cameras": {}, "renamed": False}

    # Build lists
    per_cam = {}
    for cam in CAMS:
        per_cam[cam] = build_list_for_cam(cam, images_root)
        if args.write:
            out = manifests_root / f"{cam}.json"
            changed = write_if_changed(out, per_cam[cam])
            all_changed = all_changed or changed

    # Verify via OCR (optional, graceful if OCR libs missing)
    if args.verify-ocr:
        if not try_import_ocr():
            print("[warn] OCR libs not found (opencv/pytesseract). Skipping verification.")
        else:
            for cam in CAMS:
                problems = verify_cam(cam, per_cam[cam], images_root, roi_cfg, args.ocr_lang)
                report["cameras"][cam] = problems
                if args.rename-bad and problems["misnamed"]:
                    maybe_rename(cam, problems, images_root)
                    report["renamed"] = True
                    # rebuild list after renames
                    per_cam[cam] = build_list_for_cam(cam, images_root)
                    if args.write:
                        out = manifests_root / f"{cam}.json"
                        changed = write_if_changed(out, per_cam[cam])
                        all_changed = all_changed or changed

    # Write index + report
    if args.write:
        all_changed = write_if_changed(manifests_root / "index.json",
                                       {c: f"{c}.json" for c in CAMS}) or all_changed
        write_if_changed(manifests_root / "report.json", report)

    # Console summary
    print("manifests_written:", bool(args.write), "any_changed:", all_changed)
    if report.get("cameras"):
        for cam, probs in report["cameras"].items():
            print(f"{cam:11s} misnamed={len(probs['misnamed']):3d}  ocr_failed={len(probs['ocr_failed']):3d}")

if __name__ == "__main__":
    main()
