#!/usr/bin/env python3
"""
feratel_pano_grabber.py
Capture Feratel's 360° panorama and stitch into one wide image.

Usage:
  python feratel_pano_grabber.py [URL] [OUTPUT_DIR] [DURATION_SEC] [FPS]

Defaults:
  URL           = https://webtv.feratel.com/webtv/?cam=15111
  OUTPUT_DIR    = ./images
  DURATION_SEC  = 75   (only used if DRAG_MODE=0 and AUTO_SCAN=0)
  FPS           = 2.0  (only used if DRAG_MODE=0 and AUTO_SCAN=0)

Environment (tuned for CI):
  # Capture strategy
  HEADLESS=1|0           (default 1)
  AUTO_SCAN=1|0          (default 1)   -> measure pan period and capture evenly over one rotation
  DRAG_MODE=1|0          (default 1)   -> fallback to mouse drag steps if auto-scan fails
  TARGET_FRAMES=16       (default 16)  -> evenly spaced frames across one full sweep
  DEVICE_SCALE=2         (2..3)        -> sharpness of element screenshots

  # Drag settings (fallback)
  DRAG_STEPS=96          (used only if DRAG_MODE path runs)
  DRAG_PX=22
  DRAG_PAUSE_MS=150

  # Frame output / size control
  SAVE_FORMAT=jpg        (jpg|png; jpg recommended)
  JPEG_QUALITY=88        (for jpg)
  MAX_HEIGHT=720         (downscale height before saving)
  CROP_TOP_PCT=0.08      (remove overlays; percentages of img height)
  CROP_BOTTOM_PCT=0.92
  CROP_LEFT_PCT=0.03
  CROP_RIGHT_PCT=0.97

  # Stitching limits
  MAX_FRAMES=16          (frames actually used for stitching)
  STITCH_TIMEOUT_SEC=90  (time budget; falls back to mosaic if exceeded)
  NUM_THREADS=1          (OpenCV threads)

  # Split workflow (optional)
  CAPTURE_ONLY=0         (1=only capture)
  STITCH_ONLY=0          (1=only stitch)
  FRAMES_DIR=""          (used when STITCH_ONLY=1)

Notes:
  - Uses normal element screenshots via Playwright (no DRM bypass).
  - Please respect the website's Terms of Use.
"""
import os, sys, time, glob
from datetime import datetime

import numpy as np
import cv2
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://webtv.feratel.com/webtv/?cam=15111"

# ---------------------------- helpers ----------------------------
def log(*a): print("[feratel-pano]", *a, flush=True)

def getenv_bool(n, d):
    v = os.environ.get(n)
    return d if v is None else (str(v).lower() not in ("0","false","no","off",""))

def getenv_int(n, d):
    v = os.environ.get(n)
    try: return int(v) if v is not None else d
    except: return d

def getenv_float(n, d):
    v = os.environ.get(n)
    try: return float(v) if v is not None else d
    except: return d

def pick_subset(seq, target):
    if len(seq) <= target: return seq
    step = max(1, len(seq)//target)
    return seq[::step][:target]

# perceptual hash (aHash) for loop/dup detection
def phash_from_bytes(b, size=32):
    arr = np.frombuffer(b, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    mean = im.mean()
    return (im > mean).astype(np.uint8).reshape(-1)

def hamming(a, b):
    if a is None or b is None: return 1e9
    return int(np.count_nonzero(a != b))

def safe_click_text(page, text, timeout=3000):
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout)
        log(f"Clicked text: {text}"); return True
    except Exception:
        return False

def try_selectors(page, selectors):
    for sel in selectors:
        try:
            page.locator(sel).first.click(timeout=1500)
            log(f"Clicked selector: {sel}"); return True
        except Exception:
            pass
    return False

def accept_cookies(page):
    texts = ["Accept all","Agree","I agree","I accept","Accept","OK","Got it",
             "Alles akzeptieren","Ich stimme zu","Aceptar todo","Aceptar",
             "Tout accepter","Accepter","Accetta tutto"]
    for t in texts:
        if safe_click_text(page,t): return
    try_selectors(page, [
        "#didomi-notice-agree-button","button[aria-label*='Accept']",
        "button[mode='primary']","button.cookie-accept",
        "button#onetrust-accept-btn-handler"
    ])

def enter_panorama(page):
    for key in ["360° panorama","360°","Panorama"]:
        if safe_click_text(page, key, timeout=4000):
            page.wait_for_timeout(1200); return

def hide_ui(page):
    css = """
    *[class*="controls"], *[class*="menu"], *[class*="overlay"],
    .cookie, #cookie, [id*="cookie"], footer, header, nav, .logo,
    .share, [class*="poi"], [class*="Hotspot"], [class*="sidebar"],
    [class*="poi-"], [class*="poi_"], [class*="Poi"], [class*="text"],
    [class*="weather"], [class*="label"], [class*="title"] {
        display:none!important; opacity:0!important; visibility:hidden!important;
    }
    body { overflow:hidden!important; cursor:none!important; }
    """
    try: page.add_style_tag(content=css)
    except Exception: pass

def mark_capture_element(page):
    page.evaluate("""() => {
        const nodes=[...document.querySelectorAll('video,canvas')];
        let best=null, area=0;
        for (const el of nodes){
          const r=el.getBoundingClientRect(); const a=r.width*r.height;
          if (a>area && r.width>=300 && r.height>=200){best=el; area=a;}
        }
        if (!best){
          const all=[...document.querySelectorAll('body *')];
          for (const el of all){
            const r=el.getBoundingClientRect(); if (r.width<300||r.height<200) continue;
            if (!el.querySelector) continue;
            if (!el.querySelector('video,canvas')) continue;
            const a=r.width*r.height; if (a>area){best=el; area=a;}
          }
        }
        if (best) best.setAttribute('data-capture','1');
    }""")

def get_bbox(page, locator):
    try:
        box = locator.bounding_box()
        if box: return box
    except Exception:
        pass
    return page.evaluate("""(el) => {
        const r=el.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height};
    }""", locator.element_handle())

def center_crop_pct(img, t=0.08, b=0.92, l=0.03, r=0.97):
    h, w = img.shape[:2]
    y0, y1 = int(h*t), int(h*b)
    x0, x1 = int(w*l), int(w*r)
    return img[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]

def save_bytes_cropped_jpeg(b, path, max_h=720, t=0.08, btm=0.92, l=0.03, r=0.97, q=88):
    arr = np.frombuffer(b, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None: return False
    im = center_crop_pct(im, t, btm, l, r)
    if im.shape[0] > max_h:
        sc = max_h / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1]*sc), max_h), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if ok:
        with open(path, "wb") as f: f.write(bytes(enc))
    return ok

# --------------- capture modes ---------------
def capture_drag_even(page, el, n, px=22, pause_ms=150, out_dir="images/frames", quality=88, max_h=720,
                      t=0.08, btm=0.92, l=0.03, r=0.97):
    os.makedirs(out_dir, exist_ok=True)
    el.scroll_into_view_if_needed(timeout=5000)
    box = get_bbox(page, el)
    cx, cy = box["x"]+box["width"]/2, box["y"]+box["height"]/2

    saved=[]
    for i in range(n):
        # small bounded drag each time so pointer never leaves the element
        page.mouse.move(cx, cy); page.mouse.down()
        direction = -1  # drag left
        page.mouse.move(cx + direction*px, cy, steps=2)
        page.wait_for_timeout(pause_ms)
        page.mouse.up()

        b = el.screenshot(animations="disabled", timeout=8000, scale="device")
        path = os.path.join(out_dir, f"frame_{i:05d}.jpg")
        save_bytes_cropped_jpeg(b, path, max_h=max_h, t=t, btm=btm, l=l, r=r, q=quality)
        saved.append(path)
    log(f"Saved {len(saved)} drag frames to {out_dir}")
    return saved

def scan_pan_period(page, el, max_sec=120, interval=0.5, t=0.08, btm=0.92, l=0.03, r=0.97):
    """Return (period_seconds, start_sig) or (None, first_sig)"""
    start = time.time()
    b0 = el.screenshot(animations="disabled", timeout=8000, scale="device")
    s0 = phash_from_bytes(b0)
    last_time = start
    while time.time() - start < max_sec:
        page.wait_for_timeout(int(interval*1000))
        b = el.screenshot(animations="disabled", timeout=8000, scale="device")
        s = phash_from_bytes(b)
        # consider return to start when very similar & not too soon
        if time.time() - start > 10 and hamming(s, s0) < 150:
            period = time.time() - start
            return (period, s0)
    return (None, s0)

def capture_timed_cycle(page, el, n, period, out_dir="images/frames", quality=88, max_h=720,
                        t=0.08, btm=0.92, l=0.03, r=0.97):
    """Capture n frames evenly spaced over one rotation period (seconds)."""
    os.makedirs(out_dir, exist_ok=True)
    # Wait for the next "cycle start" by watching similarity to the next ref
    b0 = el.screenshot(animations="disabled", timeout=8000, scale="device")
    s0 = phash_from_bytes(b0)
    while True:
        page.wait_for_timeout(300)
        b = el.screenshot(animations="disabled", timeout=8000, scale="device")
        if hamming(phash_from_bytes(b), s0) < 150:
            break  # near start orientation

    saved=[]
    step_ms = int((period / n) * 1000)
    for i in range(n):
        b = el.screenshot(animations="disabled", timeout=8000, scale="device")
        path = os.path.join(out_dir, f"frame_{i:05d}.jpg")
        save_bytes_cropped_jpeg(b, path, max_h=max_h, t=t, btm=btm, l=l, r=r, q=quality)
        saved.append(path)
        page.wait_for_timeout(step_ms)
    log(f"Saved {len(saved)} timed frames to {out_dir}")
    return saved

# ------------------------ stitching strategies ------------------------
def try_opencv_stitcher(imgs, mode):
    stitcher = cv2.Stitcher_create(mode)
    try:
        if hasattr(cv2, "detail") and hasattr(cv2.detail, "createWarperByName"):
            warper = cv2.detail.createWarperByName('cylindrical')
            if hasattr(stitcher, "setWarper"): stitcher.setWarper(warper)
        if hasattr(stitcher, "setPanoConfidenceThresh"):
            stitcher.setPanoConfidenceThresh(0.6)
    except Exception:
        pass
    return stitcher.stitch(imgs)  # -> (status, pano)

def _create_feature_and_norm():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), cv2.NORM_L2
    if hasattr(cv2, "AKAZE_create"):
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    return cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING

def sequential_feature_stitch(imgs):
    feat, norm = _create_feature_and_norm()
    bf = cv2.BFMatcher(norm, crossCheck=False)
    n=len(imgs); mid=n//2
    Hs=[np.eye(3,dtype=np.float64) for _ in range(n)]

    def match(a,b):
        ka,da=feat.detectAndCompute(a,None)
        kb,db=feat.detectAndCompute(b,None)
        if da is None or db is None or len(da)<8 or len(db)<8: return None
        matches = bf.knnMatch(da,db,k=2)
        good=[m for m,nm in matches if m.distance < 0.7*nm.distance]
        if len(good)<8: return None
        src=np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,_=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        return H

    for i in range(mid, n-1):
        H = match(imgs[i], imgs[i+1])
        if H is None: return None
        Hs[i+1] = Hs[i] @ H
    for i in range(mid, 0, -1):
        H = match(imgs[i], imgs[i-1])
        if H is None: return None
        Hs[i-1] = Hs[i] @ H

    h,w = imgs[0].shape[:2]
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    all_pts=np.vstack([cv2.perspectiveTransform(corners,H).reshape(-1,2) for H in Hs])
    minx,miny = np.floor(all_pts.min(axis=0)).astype(int)
    maxx,maxy = np.ceil(all_pts.max(axis=0)).astype(int)
    Tx, Ty = -minx, -miny
    T = np.array([[1,0,Tx],[0,1,Ty],[0,0,1]], dtype=np.float64)
    out_w, out_h = int(maxx-minx), int(maxy-miny)

    result = np.zeros((out_h,out_w,3), np.uint8)
    weight = np.zeros((out_h,out_w), np.float32)
    for img,H in zip(imgs,Hs):
        HH = T @ H
        warped = cv2.warpPerspective(img, HH, (out_w,out_h))
        mask = (warped.sum(axis=2)>0).astype(np.float32)
        result = (result.astype(np.float32)*weight[...,None] + warped.astype(np.float32)*mask[...,None]) / np.maximum(weight+mask,1e-3)[...,None]
        weight = np.clip(weight+mask,0,10)
    return result.astype(np.uint8)

def strip_mosaic_fallback(imgs, slice_ratio=0.28):
    if not imgs: return None
    slices=[]
    for im in imgs:
        h,w=im.shape[:2]
        sw=int(max(8, w*slice_ratio))
        x0=w//2 - sw//2; x1=x0+sw
        slices.append(im[:,x0:x1])
    return np.hstack(slices)

def stitch_with_budget(paths, out_path, max_h=720, budget_sec=90):
    start = time.time()
    imgs=[]
    for p in paths:
        im=cv2.imread(p)
        if im is None: continue
        if im.shape[0] > max_h:
            sc=max_h/im.shape[0]
            im=cv2.resize(im,(int(im.shape[1]*sc),max_h), interpolation=cv2.INTER_AREA)
        imgs.append(im)
    if len(imgs)<2:
        fb=strip_mosaic_fallback(imgs,0.28)
        if fb is not None: cv2.imwrite(out_path, fb)
        return

    log(f"Stitching {len(imgs)} frames (budget {budget_sec}s)...")
    status, pano = try_opencv_stitcher(imgs, cv2.Stitcher_PANORAMA)
    if status == cv2.Stitcher_OK: cv2.imwrite(out_path,pano); return
    if time.time()-start > budget_sec: fb=strip_mosaic_fallback(imgs,0.28); cv2.imwrite(out_path,fb); return

    status, pano = try_opencv_stitcher(imgs, cv2.Stitcher_SCANS)
    if status == cv2.Stitcher_OK: cv2.imwrite(out_path,pano); return
    if time.time()-start > budget_sec: fb=strip_mosaic_fallback(imgs,0.28); cv2.imwrite(out_path,fb); return

    pano = sequential_feature_stitch(imgs)
    if pano is not None: cv2.imwrite(out_path,pano); return

    fb = strip_mosaic_fallback(imgs,0.28)
    if fb is not None: cv2.imwrite(out_path, fb)

# ------------------------------ main ---------------------------------
def main():
    try:
        cv2.setNumThreads(getenv_int("NUM_THREADS", 1))
        if hasattr(cv2, "ocl"): cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    url = sys.argv[1] if len(sys.argv)>1 else DEFAULT_URL
    out_root = sys.argv[2] if len(sys.argv)>2 else "images"
    duration = float(sys.argv[3]) if len(sys.argv)>3 else 75.0
    fps = float(sys.argv[4]) if len(sys.argv)>4 else 2.0

    HEADLESS      = getenv_bool("HEADLESS", True)
    AUTO_SCAN     = getenv_bool("AUTO_SCAN", True)
    DRAG_MODE     = getenv_bool("DRAG_MODE", True)
    TARGET_FRAMES = getenv_int("TARGET_FRAMES", 16)
    DEVICE_SCALE  = getenv_int("DEVICE_SCALE", 2)

    # frame IO / crop
    SAVE_FORMAT   = (os.environ.get("SAVE_FORMAT") or "jpg").lower()
    JPEG_QUALITY  = getenv_int("JPEG_QUALITY", 88)
    MAX_HEIGHT    = getenv_int("MAX_HEIGHT", 720)
    CROP_TOP_PCT  = getenv_float("CROP_TOP_PCT", 0.08)
    CROP_BOTTOM_PCT = getenv_float("CROP_BOTTOM_PCT", 0.92)
    CROP_LEFT_PCT = getenv_float("CROP_LEFT_PCT", 0.03)
    CROP_RIGHT_PCT= getenv_float("CROP_RIGHT_PCT", 0.97)

    # drag fallback
    DRAG_STEPS    = getenv_int("DRAG_STEPS", 96)
    DRAG_PX       = getenv_int("DRAG_PX", 22)
    DRAG_PAUSE_MS = getenv_int("DRAG_PAUSE_MS", 150)

    # stitch limits
    MAX_FRAMES    = getenv_int("MAX_FRAMES", min(16, TARGET_FRAMES))
    STITCH_TIMEOUT= getenv_int("STITCH_TIMEOUT_SEC", 90)

    # split-flow
    CAPTURE_ONLY  = getenv_bool("CAPTURE_ONLY", False)
    STITCH_ONLY   = getenv_bool("STITCH_ONLY", False)
    FRAMES_DIR    = os.environ.get("FRAMES_DIR") or ""

    os.makedirs(out_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(out_root, "frames_"+ts)
    out_path = os.path.join(out_root, f"panorama_{ts}.jpg")

    if not STITCH_ONLY:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=HEADLESS, args=[
                "--disable-notifications","--no-default-browser-check","--disable-infobars",
                "--use-gl=swiftshader","--disable-gpu-compositing","--enable-webgl","--ignore-gpu-blocklist",
            ])
            ctx = browser.new_context(
                viewport={"width":1920,"height":1080},
                device_scale_factor=DEVICE_SCALE,
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
            )
            page = ctx.new_page()
            log(f"Opening {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(2500)
            accept_cookies(page)
            page.wait_for_timeout(800)
            enter_panorama(page)
            hide_ui(page)
            mark_capture_element(page)
            loc = page.locator("[data-capture='1']").first
            if loc.count()==0:
                page.wait_for_timeout(1500); mark_capture_element(page); loc = page.locator("[data-capture='1']").first
            try: loc.scroll_into_view_if_needed(timeout=4000)
            except Exception: pass
            page.wait_for_timeout(900)

            saved=[]
            if AUTO_SCAN:
                period, _ = scan_pan_period(page, loc, max_sec=120, interval=0.5,
                                            t=CROP_TOP_PCT, btm=CROP_BOTTOM_PCT,
                                            l=CROP_LEFT_PCT, r=CROP_RIGHT_PCT)
                if period:
                    log(f"Detected pan period ≈ {period:.1f}s")
                    saved = capture_timed_cycle(page, loc, TARGET_FRAMES, period,
                                                out_dir=frames_dir, quality=JPEG_QUALITY,
                                                max_h=MAX_HEIGHT, t=CROP_TOP_PCT,
                                                btm=CROP_BOTTOM_PCT, l=CROP_LEFT_PCT, r=CROP_RIGHT_PCT)
                else:
                    log("Auto-scan failed; falling back to drag mode.")
            if not saved and DRAG_MODE:
                saved = capture_drag_even(page, loc, TARGET_FRAMES, px=DRAG_PX, pause_ms=DRAG_PAUSE_MS,
                                          out_dir=frames_dir, quality=JPEG_QUALITY, max_h=MAX_HEIGHT,
                                          t=CROP_TOP_PCT, btm=CROP_BOTTOM_PCT, l=CROP_LEFT_PCT, r=CROP_RIGHT_PCT)

            ctx.close(); browser.close()

        if CAPTURE_ONLY:
            log(f"Captured {len(saved)} frames to {frames_dir}"); return
    else:
        frames_dir = FRAMES_DIR or (sorted(glob.glob(os.path.join(out_root,"frames_*")))[-1] if glob.glob(os.path.join(out_root,"frames_*")) else "")
        if not frames_dir: raise SystemExit("No frames directory found. Set FRAMES_DIR or run capture first.")
        saved = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg"))) or sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        log(f"Using {len(saved)} existing frames from {frames_dir}")

    if len(saved) < 2:
        fb = strip_mosaic_fallback([], 0.28)
        if fb is not None: cv2.imwrite(out_path, fb)
        log("Too few frames; wrote fallback mosaic."); return

    subset = pick_subset(saved, MAX_FRAMES)
    log(f"Stitching {len(subset)} frames (subset of {len(saved)})...")
    stitch_with_budget(subset, out_path, max_h=MAX_HEIGHT, budget_sec=STITCH_TIMEOUT)
    log(f"Done. Output: {out_path}")

if __name__=="__main__":
    main()
