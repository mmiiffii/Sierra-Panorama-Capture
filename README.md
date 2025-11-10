# Feratel Panorama Grabber

One-stop utility to automatically capture a full sweep of Feratel's **360° panorama** player and stitch the frames into a single large panorama image.

## Quick Start (Local)

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m playwright install chromium
# Optional: show the browser locally
HEADLESS=0 python feratel_pano_grabber.py "https://webtv.feratel.com/webtv/?cam=15111" ./images 85 2.5
```

- Frames → `./images/frames_*`  
- Panorama → `./images/panorama_*.jpg`

> Tip: Edit `device_scale_factor` inside the script to `3` for even sharper element screenshots (heavier).

## Docker (Reproducible)

```bash
docker build -t feratel-pano .
docker run --rm -it -e HEADLESS=1 -v "$PWD/images:/app/images" feratel-pano "https://webtv.feratel.com/webtv/?cam=15111" /app/images 90 2.0
```

## GitHub Actions (images/ output)

This repo includes a workflow `.github/workflows/feratel-panorama.yml`.
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
