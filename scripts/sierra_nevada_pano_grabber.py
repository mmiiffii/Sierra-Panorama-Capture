#!/usr/bin/env python3
"""
Sierra Nevada 360° Panorama Grabber
- Opens the Feratel web player, switches to 360° mode, clicks Download,
  and saves the equirectangular panorama image.
- Filenames use Europe/Madrid local time (configurable via CAM_TZ).

Output:
  images/panoramas/<cam>/<cam>_YYMMDD_HHMMSS_pano.jpg

Env:
  CAM_TZ   (default: Europe/Madrid)
  CAMS     Comma-separated cams to run; default is all in CAMS map below.
"""

import os
import pathlib
from datetime import datetime
from typing import Dict, List

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# --- Config ---
CAM_TZ = os.environ.get("CAM_TZ", "Europe/Madrid")

# Known 360°-capable “webtv” pages (the hotspot links usually don’t expose 360)
CAMS: Dict[str, str] = {
    "borreguiles": "https://webtv.feratel.com/webtv/?cam=15111",
    "stadium":     "https://webtv.feratel.com/webtv/?cam=15112",
    # If these don't expose 360° on your end they'll be skipped gracefully:
    "satelite":    "https://webtv.feratel.com/webtv/?cam=15111",
    "veleta":      "https://webtv.feratel.com/webtv/?cam=15112",
}

TARGET_ROOT = pathlib.Path("images/panoramas")

CONSENT_LABELS = [
    "Accept all", "Accept & continue", "Akzeptieren", "Alle akzeptieren",
    "Aceptar", "Aceptar todo", "OK", "Ok", "Got it", "I agree"
]

def now_local(tz_name: str) -> datetime:
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.utcnow()

def ts_path(cam: str, ext: str) -> pathlib.Path:
    ts = now_local(CAM_TZ).strftime("%y%m%d_%H%M%S")
    return TARGET_ROOT / cam / f"{cam}_{ts}_pano{ext}"

def ensure_dirs():
    for cam in CAMS:
        (TARGET_ROOT / cam).mkdir(parents=True, exist_ok=True)

def click_text(page, text: str, timeout=2000) -> bool:
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout)
        return True
    except PWTimeout:
        return False

def click_css(page, selector: str, timeout=2000) -> bool:
    try:
        page.locator(selector).first.click(timeout=timeout)
        return True
    except PWTimeout:
        return False

def accept_cookies(page):
    for label in CONSENT_LABELS:
        if click_text(page, label, timeout=1200):
            return True
    # a11y-style buttons
    for sel in ("button[aria-label*=Accept i]", "button[aria-label*=Akzept i]"):
        if click_css(page, sel, timeout=800):
            return True
    return False

def open_share_menu(page):
    for key in ("Share now", "Teilen", "Condividi", "Compartir", "Share"):
        if click_text(page, key, timeout=1200):
            return True
    return False

def switch_to_pano(page) -> bool:
    # Text varies per locale/skin; try a few
    for key in ("360° panorama", "360°", "Panorama 360°", "Panorama"):
        if click_text(page, key, timeout=2500):
            return True
    # Some skins use an icon/button with title/aria-label
    if click_css(page, "button[aria-label*='360'], [title*='360']", timeout=1500):
        return True
    return False

def save_download(page, cam: str) -> pathlib.Path | None:
    """
    Click a visible Download control; if hidden, open Share first.
    Only keep image files (jpg/png). If the player offers an .mp4, skip it.
    """
    tried_share = False
    for _attempt in range(2):
        try:
            with page.expect_download(timeout=8000) as dlinfo:
                # Prefer a literal 'Download' control
                if not click_text(page, "Download", timeout=3000):
                    if not tried_share:
                        open_share_menu(page)
                        tried_share = True
                        continue
                    # generic download affordances
                    if not click_css(page, "a[download], button[download], a[aria-label*=Download i], button[aria-label*=Download i]", timeout=3000):
                        return None
            dl = dlinfo.value
            suggested = (dl.suggested_filename or "panorama.jpg").lower()
            ext = pathlib.Path(suggested).suffix or ".jpg"
            if ext not in (".jpg", ".jpeg", ".png"):  # ignore videos/others
                return None
            out = ts_path(cam, ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Avoid clobber if two downloads land in the same second
            final = out
            i = 1
            while final.exists():
                final = out.with_name(out.stem + f"_{i}" + out.suffix)
                i += 1
            dl.save_as(str(final))
            return final
        except PWTimeout:
            if not tried_share:
                open_share_menu(page)
                tried_share = True
                continue
            return None

def run_for_cams(cam_list: List[str]) -> int:
    ensure_dirs()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        saved = 0
        for cam in cam_list:
            url = CAMS.get(cam)
            if not url:
                print(f"[skip] unknown cam: {cam}")
                continue

            print(f"[{cam}] open {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            accept_cookies(page)

            if not switch_to_pano(page):
                print(f"[{cam}] 360° control not found — skip")
                continue

            page.wait_for_timeout(1200)  # allow mode switch

            out = save_download(page, cam)
            if out:
                print(f"[{cam}] saved → {out}")
                saved += 1
            else:
                print(f"[{cam}] no image download found (maybe video/timeout)")

        ctx.close()
        browser.close()

    print(f"saved_count={saved}")
    return 0

if __name__ == "__main__":
    # CAMS env like: CAMS=borreguiles,stadium
    cams_env = os.environ.get("CAMS", "")
    to_run = [c.strip() for c in cams_env.split(",") if c.strip()] or list(CAMS.keys())
    raise SystemExit(run_for_cams(to_run))
