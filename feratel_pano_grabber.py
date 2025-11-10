#!/usr/bin/env python3
"""
feratel_pano_grabber.py
A one-stop tool that:
  1) opens a Feratel webcam page,
  2) switches to 360° panorama,
  3) captures high-res element screenshots across a full rotation (time-based or drag-based),
  4) stitches them into a single wide panorama image.

Usage:
  python feratel_pano_grabber.py [URL] [OUTPUT_DIR] [DURATION_SEC] [FPS]

Defaults:
  URL           = https://webtv.feratel.com/webtv/?cam=15111
  OUTPUT_DIR    = ./output
  DURATION_SEC  = 75
  FPS           = 2.0

Environment (optional):
  HEADLESS=1|0        # default 1 in CI
  DRAG_MODE=1|0       # default 1 (forces rotation via mouse dragging)
  DRAG_STEPS=72       # number of incremental drags across the viewport (drag mode)
  DRAG_PX=35          # pixels per step (horizontal) (drag mode)
  DRAG_PAUSE_MS=200   # pause between steps (drag mode)
  DEVICE_SCALE=2      # 2..3 for crispness

Requirements:
  pip install -r requirements.txt
  python -m playwright install chromium

Note:
  This script uses normal browser screenshots via Playwright (no DRM bypass).
  Please respect the website's Terms of Use for any media you generate.
"""
import os, sys, time
from datetime import datetime

import numpy as np
import cv2
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://webtv.feratel.com/webtv/?cam=15111"

def log(*a):
    print("[feratel-pano]", *a, flush=True)

def getenv_bool(name, default):
    v = os.environ.get(name)
    if v is None: return default
    return str(v).lower() not in ("0","false","no","off","")

def getenv_int(name, default):
    v = os.environ.get(name)
    try: return int(v) if v is not None else default
    except: return default

def getenv_float(name, default):
    v = os.environ.get(name)
    try: return float(v) if v is not None else default
    except: return default

def safe_click_text(page, text, timeout=3000):
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout)
        log(f"Clicked text: {text}")
        return True
    except Exception:
        return False

def try_selectors(page, selectors):
    for sel in selectors:
        try:
            page.locator(sel).first.click(timeout=1500)
            log(f"Clicked selector: {sel}")
            return True
        except Exception:
            pass
    return False

def accept_cookies(page):
    texts = [
        "Accept all", "Agree", "I agree", "I accept", "Accept", "OK", "Got it",
        "Alles akzeptieren", "Ich stimme zu", "Aceptar todo", "Aceptar",
        "Tout accepter", "Accepter", "Accetta tutto"
    ]
    for t in texts:
        if safe_click_text(page, t):
            return
    try_selectors(page, [
        "#didomi-notice-agree-button",
        "button[aria-label*='Accept']",
        "button[mode='primary']",
        "button.cookie-accept",
        "button#onetrust-accept-btn-handler",
    ])

def enter_panorama(page):
    for key in ["360° panorama", "360°", "Panorama"]:
        if safe_click_text(page, key, timeout=4000):
            time.sleep(2.0)
            return

def hide_ui(page):
    css = """
    *[class*="controls"], *[class*="menu"], *[class*="overlay"],
    .cookie, #cookie, [id*="cookie"], footer, header, nav,
    .share, [class*="poi"], [class*="Hotspot"], [class*="sidebar"],
    [class*="poi-"], [class*="poi_"], [class*="Poi"] {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    body { overflow: hidden !important; cursor: none !important; }
    """
    try:
        page.add_style_tag(content=css)
    except Exception:
        pass

def mark_capture_element(page):
    page.evaluate("""() => {
        const nodes = Array.from(document.querySelectorAll('video, canvas'));
        let best = null, bestArea = 0;
        for (const el of nodes) {
            const r = el.getBoundingClientRect();
            const area = Math.max(0, r.width) * Math.max(0, r.height);
            if (area > bestArea && r.width >= 300 && r.height >= 200) {
                best = el; bestArea = area;
            }
        }
        if (best) { best.setAttribute('data-capture', '1'); return; }
        const all = Array.from(document.querySelectorAll('body *'));
        best = null; bestArea = 0;
        for (const el of all) {
            const r = el.getBoundingClientRect();
            if (r.width < 300 || r.height < 200) continue;
            const hasChild = el.querySelector && el.querySelector('video, canvas');
            if (!hasChild) continue;
            const area = Math.max(0, r.width) * Math.max(0, r.height);
            if (area > bestArea) { best = el; bestArea = area; }
        }
        if (best) best.setAttribute('data-capture', '1');
    }""")

def get_bbox(page, locator):
    try:
        box = locator.bounding_box()
        if box: return box
    except Exception:
        pass
    return page.evaluate("""(el) => {
        const r = el.getBoundingClientRect();
        return {x: r.x, y: r.y, width: r.width, height: r.height};
    }""", locator.element_handle())

def capture_drag_sequence(page, el, steps=72, px=35, pause_ms=200, out_dir="output/frames_drag"):
    os.makedirs(out_dir, exist_ok=True)
    el.scroll_into_view_if_needed(timeout=5000)
    box = get_bbox(page, el)

    cx = box["x"] + box["width"]/2
    cy = box["y"] + box["height"]/2

    page.mouse.move(cx, cy)
    page.mouse.down()
    saved = []
    for i in range(steps):
        nx = cx - (i+1)*px
        page.mouse.move(nx, cy, steps=2)
        page.wait_for_timeout(pause_ms)
        path = os.path.join(out_dir, f"frame_{i:05d}.png")
        el.screenshot(path=path, animations="disabled", timeout=8000, scale="device")
        saved.append(path)
    page.mouse.up()
    log(f"Saved {len(saved)} drag frames to {out_dir}")
    return saved

def capture_time_based(page, el, duration_sec=75, fps=2.0, out_dir="output/frames"):
    os.makedirs(out_dir, exist_ok=True)
    el.scroll_into_view_if_needed(timeout=5000)
    interval = 1.0 / max(0.1, float(fps))
    t_end = time.time() + float(duration_sec)
    idx = 0
    saved = []
    log(f"Capturing frames for {duration_sec:.1f}s at {fps} fps...")
    while time.time() < t_end:
        path = os.path.join(out_dir, f"frame_{idx:05d}.png")
        try:
            el.screenshot(path=path, animations="disabled", timeout=8000, scale="device")
            saved.append(path)
            idx += 1
        except Exception:
            pass
        time.sleep(interval)
    log(f"Saved {len(saved)} frames to {out_dir}")
    return saved

def center_crop(img, top_pct=0.06, bottom_pct=0.94):
    h, w = img.shape[:2]
    top = int(h * top_pct)
    bottom = int(h * bottom_pct)
    return img[top:bottom, :, :]

def stitch_try_stitcher(imgs):
    for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
        stitcher = cv2.Stitcher_create(mode)
        try:
            stitcher.setPanoConfidenceThresh(0.6)
        except Exception:
            pass
        status, pano = stitcher.stitch(imgs)
        if status == cv2.Stitcher_OK:
            return pano, status
    return None, status

def stitch_detail_pipeline(imgs):
    try:
        import cv2 as cv
        from cv2 import detail as d

        # Features
        finder = d.AKAZEFeaturesFinder_create()
        features = [d.computeImageFeatures2(finder, im) for im in imgs]

        # Matching
        matcher = d.BestOf2NearestMatcher_create(try_use_gpu=False, match_conf=0.3)
        matches = matcher.apply2(features)
        d.leaveBiggestComponent(features, matches, 0.4)

        # Estimate camera params
        estimator = d.HomographyBasedEstimator_create()
        ok, cameras = estimator.apply(features, matches, None)
        if not ok: return None
        adjuster = d.BundleAdjusterRay_create()
        adjuster.setConfThresh(1.0)
        ok, cameras = adjuster.apply(features, matches, cameras)
        if not ok: return None

        warper = d.CylindricalWarper_create()
        corners, sizes, images_warped, masks_warped = [], [], [], []
        for idx, im in enumerate(imgs):
            K = cameras[idx].K().astype(np.float32)
            img_warp = warper.warp(im, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)[0]
            images_warped.append(img_warp)
            corners.append((0,0))
            sizes.append((img_warp.shape[1], img_warp.shape[0]))
            mask = 255*np.ones((im.shape[0], im.shape[1]), np.uint8)
            masks_warped.append(warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)[0])

        compensator = d.ExposureCompensator_createDefault(d.ExposureCompensator_CHANNELS)
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        seam_finder = d.SeamFinder_createDefault(d.SeamFinder_DP_COLORGRAD)
        masks_seam = seam_finder.find(images_warped, corners, masks_warped)

        blender = d.MultiBandBlender()
        blender.setNumBands(5)
        dst_roi = d.resultRoi(corners=corners, sizes=sizes)
        blender.prepare(dst_roi)
        for i in range(len(images_warped)):
            blender.feed(images_warped[i], masks_seam[i], corners[i])
        result, _ = blender.blend(None, None)
        return result
    except Exception:
        return None

def stitch_panorama(paths, out_path):
    imgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        im = center_crop(im, 0.06, 0.94)
        imgs.append(im)
    if len(imgs) < 4:
        raise RuntimeError("Not enough valid frames after loading/cropping.")

    log(f"Stitching {len(imgs)} frames...")
    pano, status = stitch_try_stitcher(imgs)
    if pano is None:
        log(f"Simple stitcher failed with status {status}. Trying detail pipeline...")
        pano = stitch_detail_pipeline(imgs)
    if pano is None:
        raise RuntimeError(f"OpenCV stitching failed, status code: {status}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, pano)
    log(f"Wrote panorama: {out_path}  ({pano.shape[1]}x{pano.shape[0]} px)")

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out_root = sys.argv[2] if len(sys.argv) > 2 else "output"
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

    HEADLESS = getenv_bool("HEADLESS", True)
    DRAG_MODE = getenv_bool("DRAG_MODE", True)
    DRAG_STEPS = getenv_int("DRAG_STEPS", 72)
    DRAG_PX = getenv_int("DRAG_PX", 35)
    DRAG_PAUSE_MS = getenv_int("DRAG_PAUSE_MS", 200)
    DEVICE_SCALE = getenv_int("DEVICE_SCALE", 2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(out_root, "frames_" + ts)
    out_path = os.path.join(out_root, f"panorama_{ts}.jpg")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=HEADLESS, args=[
            "--disable-notifications",
            "--no-default-browser-check",
            "--disable-infobars",
            "--use-gl=swiftshader",
            "--disable-gpu-compositing",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
        ])
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=DEVICE_SCALE,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
        page = ctx.new_page()
        log(f"Opening {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(3000)
        accept_cookies(page)
        page.wait_for_timeout(1000)
        enter_panorama(page)
        hide_ui(page)
        mark_capture_element(page)
        loc = page.locator("[data-capture='1']").first
        if loc.count() == 0:
            page.wait_for_timeout(2000)
            mark_capture_element(page)
            loc = page.locator("[data-capture='1']").first
        try:
            loc.scroll_into_view_if_needed(timeout=4000)
        except Exception:
            pass
        page.wait_for_timeout(1500)

        if DRAG_MODE:
            saved = capture_drag_sequence(page, loc, steps=DRAG_STEPS, px=DRAG_PX,
                                          pause_ms=DRAG_PAUSE_MS, out_dir=frames_dir)
        else:
            saved = capture_time_based(page, loc, duration_sec=duration, fps=fps, out_dir=frames_dir)

        ctx.close()
        browser.close()

    if len(saved) < 6:
        raise RuntimeError(f"Too few frames captured ({len(saved)}). Increase DRAG_STEPS or duration.")
    stitch_panorama(saved, out_path)

if __name__ == "__main__":
    main()
