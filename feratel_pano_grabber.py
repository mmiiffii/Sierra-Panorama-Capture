#!/usr/bin/env python3
"""
feratel_pano_grabber.py
Capture Feratel's 360° panorama and stitch into one wide image.

Usage:
  python feratel_pano_grabber.py [URL] [OUTPUT_DIR] [DURATION_SEC] [FPS]

Defaults:
  URL           = https://webtv.feratel.com/webtv/?cam=15111
  OUTPUT_DIR    = ./output
  DURATION_SEC  = 75
  FPS           = 2.0

Environment (optional):
  HEADLESS=1|0        # default 1 (CI)
  DRAG_MODE=1|0       # default 1 (forces rotation via mouse)
  DRAG_STEPS=96       # more steps = more overlap (recommended 96–140)
  DRAG_PX=22          # pixels per step (smaller = more overlap)
  DRAG_PAUSE_MS=150   # ms between steps
  DEVICE_SCALE=3      # 2..3 for crispness

This script does normal element screenshots via Playwright.
Please respect the website's Terms of Use.
"""
import os, sys, time
from datetime import datetime

import numpy as np
import cv2
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://webtv.feratel.com/webtv/?cam=15111"

# ---------------------------- helpers ----------------------------
def log(*a): print("[feratel-pano]", *a, flush=True)
def getenv_bool(n,d): 
    v=os.environ.get(n); 
    return d if v is None else (str(v).lower() not in ("0","false","no","off",""))
def getenv_int(n,d):
    try: return int(os.environ.get(n,d))
    except: return d

def safe_click_text(page, text, timeout=3000):
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout)
        log(f"Clicked text: {text}"); return True
    except Exception: return False

def try_selectors(page, selectors):
    for sel in selectors:
        try:
            page.locator(sel).first.click(timeout=1500)
            log(f"Clicked selector: {sel}"); return True
        except Exception: pass
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
            page.wait_for_timeout(1500); return

def hide_ui(page):
    css = """
    *[class*="controls"], *[class*="menu"], *[class*="overlay"],
    .cookie, #cookie, [id*="cookie"], footer, header, nav,
    .share, [class*="poi"], [class*="Hotspot"], [class*="sidebar"],
    [class*="poi-"], [class*="poi_"], [class*="Poi"] {
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
    except Exception: pass
    return page.evaluate("""(el) => {
        const r=el.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height};
    }""", locator.element_handle())

def capture_drag_sequence(page, el, steps=96, px=22, pause_ms=150, out_dir="output/frames_drag"):
    os.makedirs(out_dir, exist_ok=True)
    el.scroll_into_view_if_needed(timeout=5000)
    box = get_bbox(page, el)
    cx, cy = box["x"]+box["width"]/2, box["y"]+box["height"]/2

    page.mouse.move(cx, cy); page.mouse.down()
    saved=[]
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
    interval = 1.0/max(0.1,float(fps)); t_end = time.time()+float(duration_sec)
    idx=0; saved=[]; log(f"Capturing frames for {duration_sec:.1f}s at {fps} fps...")
    while time.time()<t_end:
        path=os.path.join(out_dir, f"frame_{idx:05d}.png")
        try:
            el.screenshot(path=path, animations="disabled", timeout=8000, scale="device")
            saved.append(path); idx+=1
        except Exception: pass
        time.sleep(interval)
    log(f"Saved {len(saved)} frames to {out_dir}")
    return saved

def center_crop(img, top_pct=0.06, bottom_pct=0.94):
    h,w=img.shape[:2]; top=int(h*top_pct); bot=int(h*bottom_pct)
    return img[top:bot,:,:]

# ------------------------ stitching strategies ------------------------
def try_opencv_stitcher(imgs, mode):
    stitcher = cv2.Stitcher_create(mode)
    try:
        # Prefer cylindrical warping when available
        if hasattr(cv2, "detail") and hasattr(cv2.detail, "createWarperByName"):
            warper = cv2.detail.createWarperByName('cylindrical')
            if hasattr(stitcher, "setWarper"): stitcher.setWarper(warper)
        if hasattr(stitcher, "setPanoConfidenceThresh"):
            stitcher.setPanoConfidenceThresh(0.6)
    except Exception:
        pass
    return stitcher.stitch(imgs)

def sequential_sift_stitch(imgs):
    # Simple sequential stitch: SIFT + BF + RANSAC + accumulate H to middle frame
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    n=len(imgs); mid=n//2
    Hs=[np.eye(3,dtype=np.float64) for _ in range(n)]

    def match(a,b):
        ka,da=sift.detectAndCompute(a,None)
        kb,db=sift.detectAndCompute(b,None)
        if da is None or db is None or len(da)<8 or len(db)<8: return None
        matches = bf.knnMatch(da,db,k=2)
        good=[]
        for m,nm in matches:
            if m.distance < 0.7*nm.distance: good.append(m)
        if len(good)<8: return None
        src=np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        return H

    # forward from mid
    for i in range(mid, n-1):
        H = match(imgs[i], imgs[i+1])
        if H is None: return None
        Hs[i+1] = Hs[i] @ H
    # backward from mid
    for i in range(mid, 0, -1):
        H = match(imgs[i], imgs[i-1])
        if H is None: return None
        Hs[i-1] = Hs[i] @ H

    # compute canvas bounds
    h,w = imgs[0].shape[:2]
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    all_pts=[]
    for H in Hs:
        pts=cv2.perspectiveTransform(corners, H).reshape(-1,2)
        all_pts.append(pts)
    all_pts=np.vstack(all_pts)
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
        # feather blend
        result = (result.astype(np.float32)*weight[...,None] + warped.astype(np.float32)*mask[...,None]) / np.maximum(weight+mask,1e-3)[...,None]
        weight = np.clip(weight+mask,0,10)

    return result.astype(np.uint8)

def strip_mosaic_fallback(imgs, slice_ratio=0.33):
    # Guaranteed output: take central vertical slice from each frame and tile
    slices=[]
    for im in imgs:
        h,w=im.shape[:2]
        sw=int(max(8, w*slice_ratio))
        x0=w//2 - sw//2; x1=x0+sw
        slices.append(im[:,x0:x1])
    return np.hstack(slices)

def stitch_panorama(paths, out_path):
    imgs=[]
    for p in paths:
        im=cv2.imread(p)
        if im is None: continue
        im=center_crop(im,0.06,0.94)
        # limit height for robustness/speed in CI; keep aspect
        max_h=900
        if im.shape[0]>max_h:
            scale=max_h/im.shape[0]
            im=cv2.resize(im,(int(im.shape[1]*scale),max_h))
        imgs.append(im)
    if len(imgs)<4: raise RuntimeError("Not enough valid frames to stitch.")

    log(f"Stitching {len(imgs)} frames (robust pipeline)...")
    # 1) OpenCV PANORAMA
    status, pano = try_opencv_stitcher(imgs, cv2.Stitcher_PANORAMA)
    if status==cv2.Stitcher_OK: 
        cv2.imwrite(out_path,pano); return out_path

    # 2) OpenCV SCANS
    status, pano = try_opencv_stitcher(imgs, cv2.Stitcher_SCANS)
    if status==cv2.Stitcher_OK:
        cv2.imwrite(out_path,pano); return out_path

    # 3) Sequential SIFT
    pano = sequential_sift_stitch(imgs)
    if pano is not None:
        cv2.imwrite(out_path,pano); return out_path

    # 4) Strip mosaic fallback (always succeeds)
    log("All stitchers failed; writing strip-mosaic fallback.")
    fb = strip_mosaic_fallback(imgs, slice_ratio=0.28)
    cv2.imwrite(out_path.replace(".jpg","_mosaic.jpg"), fb)
    raise RuntimeError("Stitching failed; wrote mosaic fallback instead.")

# ------------------------------ main ---------------------------------
def main():
    url = sys.argv[1] if len(sys.argv)>1 else DEFAULT_URL
    out_root = sys.argv[2] if len(sys.argv)>2 else "output"
    duration = float(sys.argv[3]) if len(sys.argv)>3 else 75.0
    fps = float(sys.argv[4]) if len(sys.argv)>4 else 2.0

    HEADLESS   = getenv_bool("HEADLESS", True)
    DRAG_MODE  = getenv_bool("DRAG_MODE", True)
    DRAG_STEPS = getenv_int("DRAG_STEPS", 96)
    DRAG_PX    = getenv_int("DRAG_PX", 22)
    DRAG_PAUSE_MS = getenv_int("DRAG_PAUSE_MS", 150)
    DEVICE_SCALE  = getenv_int("DEVICE_SCALE", 3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(out_root, "frames_"+ts)
    out_path = os.path.join(out_root, f"panorama_{ts}.jpg")

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
        page.wait_for_timeout(3000)
        accept_cookies(page)
        page.wait_for_timeout(1000)
        enter_panorama(page)
        hide_ui(page)
        mark_capture_element(page)
        loc = page.locator("[data-capture='1']").first
        if loc.count()==0:
            page.wait_for_timeout(2000); mark_capture_element(page); loc = page.locator("[data-capture='1']").first
        try: loc.scroll_into_view_if_needed(timeout=4000)
        except Exception: pass
        page.wait_for_timeout(1200)

        if DRAG_MODE:
            saved = capture_drag_sequence(page, loc, steps=DRAG_STEPS, px=DRAG_PX,
                                          pause_ms=DRAG_PAUSE_MS, out_dir=frames_dir)
        else:
            saved = capture_time_based(page, loc, duration_sec=duration, fps=fps, out_dir=frames_dir)

        ctx.close(); browser.close()

    if len(saved)<6: raise RuntimeError(f"Too few frames ({len(saved)}). Increase DRAG_STEPS or duration.")
    stitch_panorama(saved, out_path)

if __name__=="__main__":
    main()
