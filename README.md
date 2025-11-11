# Sierra Panorama Capture

Capture stills from Feratel webcams, extract the on-image timestamp via OCR, save only new frames, and publish a simple web viewer (GitHub Pages) that browses each camera’s history.

> Built to be Codespaces-friendly: run everything from a terminal, no local installs required.

---

## What’s here

- `scripts/feratel_capture_ocr.py` — fetch each camera JPEG, OCR the bottom-right clock, save `images/<cam>/<cam>_YYMMDD_HHMMSS.jpg` only if **new**.
- `scripts/make_manifests.py` — build `docs/manifests/<cam>.json` from `images/…` (+ optional OCR verify/auto-rename).
- `docs/index.html` — single-file viewer that reads manifests and loads images via `raw.githubusercontent.com`.
- `config/roi.yml` — per-camera ROI (percent of width/height) where the timestamp sits.
- `.github/workflows/*` — optional Actions to run capture, build manifests, and verify.

---

## Quick start (GitHub Codespaces)

Open the repo in Codespaces, then:

```bash
# 1) Python env (once per Codespace)
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Run a capture pass (downloads any new frames)
OCR_LANG=eng CAM_TZ=Europe/Madrid DEBUG_ROI=0 \
python scripts/feratel_capture_ocr.py

# 3) Build (or rebuild) manifests for the viewer
python scripts/make_manifests.py --write

# 4) Commit & push
git add images docs/manifests
git commit -m "capture + manifests"
git push
