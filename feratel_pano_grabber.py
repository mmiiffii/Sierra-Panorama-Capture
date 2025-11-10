#!/usr/bin/env python3
"""
feratel_pano_grabber.py
A one-stop tool that:
  1) opens a Feratel webcam page,
  2) switches to 360° panorama,
  3) captures high-res element screenshots across a full rotation,
  4) stitches them into a single wide panorama image.

Usage:
  python feratel_pano_grabber.py [URL] [OUTPUT_DIR] [DURATION_SEC] [FPS]

Defaults:
  URL           = https://webtv.feratel.com/webtv/?cam=15111
  OUTPUT_DIR    = ./output
  DURATION_SEC  = 75   (recording time to cover one full sweep)
  FPS           = 2.0  (how many frames to capture per second)

Requirements:
  pip install -r requirements.txt
  playwright install chromium

Note:
  This script uses normal browser screenshots via Playwright (no DRM bypass).
  Please respect the website's Terms of Use for any media you generate.
"""
import os, sys, time
from datetime import datetime

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://webtv.feratel.com/webtv/?cam=15111"

def log(*a):
    print("[feratel-pano]", *a, flush=True)

def safe_click_text(page, text, timeout=3000):
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout)
        log(f"Clicked text: {text}")
        return True
    except Exception as e:
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
    # Try common consent buttons & frameworks
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
    # Click the "360° panorama" button/link if present
    for key in ["360° panorama", "360°", "Panorama"]:
        if safe_click_text(page, key, timeout=4000):
            time.sleep(2.0)
            return

def hide_ui(page):
    # Hide overlays/menus to get a clean capture
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
    # Pick the largest <video> or <canvas> on the page and mark it for capture
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
        if (best) { best.setAttribute('data-capture', '1'); }
    }""")
    # Fallback: choose a container that contains a video/canvas
    if page.locator('[data-capture="1"]').count() == 0:
        page.evaluate("""() => {
            const all = Array.from(document.querySelectorAll('body *'));
            let best = null, bestArea = 0;
            for (const el of all) {
                const r = el.getBoundingClientRect();
                if (r.width < 300 || r.height < 200) continue;
                if (!el.querySelector) continue;
                const hasChild = el.querySelector('video, canvas');
                if (!hasChild) continue;
                const area = Math.max(0, r.width) * Math.max(0, r.height);
                if (area > bestArea) { best = el; bestArea = area; }
            }
            if (best) best.setAttribute('data-capture', '1');
        }""")

def capture_frames(page, duration_sec=75, fps=2.0, out_dir="output/frames"):
    os.makedirs(out_dir, exist_ok=True)
    el = page.locator('[data-capture="1"]')
    if el.count() == 0:
        raise RuntimeError("Could not find a video/canvas element to capture.")
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
        except Exception as e:
            # If a frame fails, carry on
            pass
        time.sleep(interval)
    log(f"Saved {len(saved)} frames to {out_dir}")
    return saved

def pick_subset(paths, target_count=40):
    if len(paths) <= target_count:
        return paths
    step = max(1, len(paths) // target_count)
    subset = paths[::step]
    return subset[:target_count]

def center_crop(img, top_pct=0.05, bottom_pct=0.95):
    h, w = img.shape[:2]
    top = int(h * top_pct)
    bottom = int(h * bottom_pct)
    return img[top:bottom, :, :]

def stitch_panorama(paths, out_path):
    if len(paths) < 4:
        raise RuntimeError("Not enough frames to stitch (need at least 4).")
    imgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None: 
            continue
        # minor crop to remove UI edges if any
        im = center_crop(im, 0.05, 0.95)
        imgs.append(im)
    if len(imgs) < 4:
        raise RuntimeError("Not enough valid frames after loading/cropping.")
    log(f"Stitching {len(imgs)} frames (this can take a minute)...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"OpenCV stitching failed, status code: {status}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, pano)
    log(f"Wrote panorama: {out_path}  ({pano.shape[1]}x{pano.shape[0]} px)")

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out_root = sys.argv[2] if len(sys.argv) > 2 else "output"
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(out_root, "frames_" + ts)
    out_path = os.path.join(out_root, f"panorama_{ts}.jpg")

    HEADLESS = os.environ.get("HEADLESS", "1").lower() not in ("0","false","no")

    from playwright.sync_api import Error as PWError

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=HEADLESS, args=[
            "--disable-notifications",
            "--no-default-browser-check",
            "--disable-infobars",
        ])
        # High device scale factor to capture sharper screenshots of the element
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=2, # increase to 3 for ultra-sharp if your GPU/CPU can handle it
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
        page = ctx.new_page()
        log(f"Opening {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        # small wait to let player init
        page.wait_for_timeout(3000)
        accept_cookies(page)
        page.wait_for_timeout(1000)
        enter_panorama(page)
        hide_ui(page)
        # Mark and find capture candidate
        mark_capture_element(page)
        if page.locator("[data-capture='1']").count() == 0:
            # try one more time after a brief wait
            page.wait_for_timeout(2000)
            mark_capture_element(page)
        # Ensure the element is in view and reasonably large; attempt to scroll
        try:
            page.locator("[data-capture='1']").first.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass

        # Give it a second for the panorama to start rotating
        page.wait_for_timeout(2000)
        saved = capture_frames(page, duration_sec=duration, fps=fps, out_dir=frames_dir)

        ctx.close()
        browser.close()

    # Choose a subset for reliable stitching (too many frames can confuse the matcher)
    subset = pick_subset(saved, target_count=40)
    stitch_panorama(subset, out_path)

if __name__ == "__main__":
    main()
