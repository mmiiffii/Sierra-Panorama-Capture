<<<<<<< HEAD
# Sierra Panorama Capture

Capture stills from Feratel webcams, extract the on-image timestamp via OCR, save only new frames, and publish a simple web viewer (GitHub Pages) that browses each camera’s history.
=======
# Sierra Nevada Panorama Grabber

One-stop utility to automatically capture a full sweep of Sierra Nevada's **360° panorama** player and stitch the frames into a single large panorama image.
>>>>>>> rename-sierra

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

<<<<<<< HEAD
# 4) Commit & push
git add images docs/manifests
git commit -m "capture + manifests"
git push
=======
## Docker (Reproducible)

```bash
docker build -t Sierra Nevada-pano .
docker run --rm -it -e HEADLESS=1 -v "$PWD/images:/app/images" feratel-pano "https://webtv.feratel.com/webtv/?cam=15111" /app/images 90 2.0
```

## GitHub Actions (images/ output)

This repo includes a workflow `.github/workflows/Sierra Nevada-panorama.yml`.
Trigger it from the "Actions" tab and it will place results in an `images/` folder and upload it as an artifact.

## Arguments

```
python feratel_pano_grabber.py [URL] [OUTPUT_DIR] [DURATION_SEC] [FPS]

Defaults:
URL           = https://webtv.feratel.com/webtv/?cam=15111
OUTPUT_DIR    = ./output
DURATION_SEC  = 75
FPS           = 2.0
```

## How it Works

- Launches Chromium (Playwright), opens the page, clicks **360° panorama**, hides overlays/cookies, finds the largest `<video>`/`<canvas>` and captures **element screenshots** at a high device pixel ratio.
- Picks a subset of frames and stitches them using OpenCV's panorama engine.

## Notes

- This tool performs standard browser screenshots (no DRM bypass).  
- Be mindful of the website's Terms of Use.  
- If stitching fails, try a longer duration (ensure a full rotation), lower FPS (more overlap), or increase the subset size in `pick_subset()`.

## License

MIT
>>>>>>> rename-sierra
