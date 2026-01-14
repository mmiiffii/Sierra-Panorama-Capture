#!/usr/bin/env python3
"""
Sierra Nevada 360° Panorama Grabber (single URL)

Opens a Feratel WebTV page (e.g. https://webtv.feratel.com/webtv/?cam=15111),
switches to the 360° panorama mode, clicks Download, and saves the pano image.

Filename uses local time in Europe/Madrid (configurable via CAM_TZ).
No OCR. Robust fallback if the UI's Download isn't found.

Env (or CLI):
  WEBTV_URL  (required) – full URL like https://webtv.feratel.com/webtv/?cam=15111
  NAME       (optional) – folder/name prefix; default inferred from cam=<id> or 'pano'
  CAM_TZ     (optional) – default 'Europe/Madrid'

Usage (Codespaces):
  python scripts/sierra_nevada_pano_grabber.py --url "https://webtv.feratel.com/webtv/?cam=15111" --name borreguiles
"""

import os
import re
import argparse
import pathlib
import hashlib
from urllib.parse import urlparse, parse_qs
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# ---------- config ----------
DEFAULT_TZ = os.environ.get("CAM_TZ", "Europe/Madrid")
TARGET_ROOT = pathlib.Path("images/panoramas")
STATE_DIR = pathlib.Path("state")

CONSENT_LABELS = [
    "Accept all","Accept & continue","Akzeptieren","Alle akzeptieren",
    "Aceptar","Aceptar todo","OK","Ok","Got it","I agree"
]

# ---------- helpers ----------
def now_local(tz_name: str) -> datetime:
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.utcnow()

def infer_name(url: str) -> str:
    # Try to derive from ?cam=15111 -> "cam15111"
    q = parse_qs(urlparse(url).query)
    if "cam" in q and q["cam"]:
        return f"cam{q['cam'][0]}"
    # fallback: last hostname label or 'pano'
    host = urlparse(url).netloc.split(".")
    return host[0] if host else "pano"

def out_path(name: str, tz: str, ext: str = ".jpg") -> pathlib.Path:
    ts = now_local(tz).strftime("%y%m%d_%H%M%S")
    return TARGET_ROOT / name / f"{name}_{ts}_pano{ext}"

def ensure_dirs(name: str):
    (TARGET_ROOT / name).mkdir(parents=True, exist_ok=True)

def sha1_hex(data: bytes) -> str:
    """Compute SHA-1 hash of bytes."""
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()

def state_path(name: str) -> pathlib.Path:
    """Path to the state file storing last image hash."""
    return STATE_DIR / f"{name}_pano.sha1"

def read_last_hash(name: str) -> str | None:
    """Read the last saved image hash from state file."""
    p = state_path(name)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None
    return None

def write_last_hash(name: str, hexval: str):
    """Write the image hash to state file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(name).write_text(hexval + "\n", encoding="utf-8")

def click_text(page, text: str, timeout=2500) -> bool:
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout)
        return True
    except PWTimeout:
        return False

def click_css(page, selector: str, timeout=2500) -> bool:
    try:
        page.locator(selector).first.click(timeout=timeout)
        return True
    except PWTimeout:
        return False

def accept_cookies(page):
    for label in CONSENT_LABELS:
        if click_text(page, label, timeout=1200):
            return True
    for sel in ("button[aria-label*=Accept i]", "button[aria-label*=Akzept i]"):
        if click_css(page, sel, timeout=1000):
            return True
    return False

def switch_to_pano(page) -> bool:
    # Try common labels/aria for the 360 tab/button
    for key in ("360° panorama","360°","Panorama 360°","Panorama"):
        if click_text(page, key, timeout=3000):
            return True
    if click_css(page, "button[aria-label*='360'], [title*='360']", timeout=2000):
        return True
    return False

def open_share(page):
    for key in ("Share now","Teilen","Condividi","Compartir","Share"):
        if click_text(page, key, timeout=1500):
            return True
    return False

def try_ui_download(page, name: str, tz: str) -> tuple[pathlib.Path, bytes] | None:
    """Preferred path: capture Playwright download when clicking 'Download'."""
    tried_share = False
    for _ in range(2):
        try:
            with page.expect_download(timeout=8000) as dlinfo:
                if not click_text(page, "Download", timeout=3000):
                    if not tried_share:
                        open_share(page)
                        tried_share = True
                        continue
                    # generic download affordances
                    if not click_css(page, "a[download], button[download], a[aria-label*=Download i], button[aria-label*=Download i]", timeout=3000):
                        return None
            dl = dlinfo.value
            suggested = (dl.suggested_filename or "panorama.jpg").lower()
            ext = pathlib.Path(suggested).suffix or ".jpg"
            if ext not in (".jpg",".jpeg",".png"):
                return None
            # Save to temp location first to read bytes
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp_path = pathlib.Path(tmp.name)
            dl.save_as(str(tmp_path))
            data = tmp_path.read_bytes()
            tmp_path.unlink()
            
            out = out_path(name, tz, ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            return (out, data)
        except PWTimeout:
            if not tried_share:
                open_share(page)
                tried_share = True
                continue
            return None

def try_link_scrape(page, name: str, tz: str) -> tuple[pathlib.Path, bytes] | None:
    """
    Fallback: look for <a href="*.jpg|*.jpeg|*.png"> in the UI and fetch it via Playwright's context.
    """
    # gather candidate hrefs
    candidates = []
    anchors = page.locator("a[href]").all()
    for a in anchors:
        try:
            href = a.get_attribute("href") or ""
        except Exception:
            href = ""
        if re.search(r"\.(jpg|jpeg|png)(\?.*)?$", href, re.I):
            candidates.append(href)
    if not candidates:
        # sometimes the large pano is in an <img>
        imgs = page.locator("img[src]").all()
        for im in imgs:
            try:
                src = im.get_attribute("src") or ""
            except Exception:
                src = ""
            if re.search(r"\.(jpg|jpeg|png)(\?.*)?$", src, re.I):
                candidates.append(src)

    if not candidates:
        return None

    # resolve relative to page and download with same cookies using context.request
    href = candidates[0]
    ctx = page.context
    resp = ctx.request.get(page.urljoin(href) if hasattr(page, "urljoin") else href, timeout=15000)
    if resp.ok:
        data = resp.body()
        # guess ext from url
        m = re.search(r"\.(jpg|jpeg|png)(\?.*)?$", href, re.I)
        ext = "." + (m.group(1).lower() if m else "jpg")
        out = out_path(name, tz, ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        return (out, data)
    return None

def grab_one(url: str, name: str, tz: str) -> int:
    ensure_dirs(name)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        print(f"[pano] open {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=35000)
        accept_cookies(page)

        if not switch_to_pano(page):
            print("[pano] 360° control not found — abort")
            ctx.close(); browser.close()
            return 2

        # allow UI to switch
        page.wait_for_timeout(1200)

        result = try_ui_download(page, name, tz)
        if not result:
            result = try_link_scrape(page, name, tz)

        ctx.close(); browser.close()

    if result:
        out_path, data = result
        # Check if image is identical to last saved
        current_hash = sha1_hex(data)
        last_hash = read_last_hash(name)
        
        if current_hash == last_hash:
            print(f"[pano] image unchanged (hash={current_hash[:8]}...) — skipping save")
            return 0
        
        # Save the new image
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        write_last_hash(name, current_hash)
        print(f"[pano] saved → {out_path}")
        return 0
    else:
        print("[pano] no image download found (maybe skin changed or video-only)")
        return 0  # Changed from 3 to 0 - not finding an image is not an error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("WEBTV_URL", ""), help="WebTV URL like https://webtv.feratel.com/webtv/?cam=15111")
    ap.add_argument("--name", default=os.environ.get("NAME", ""), help="Name/prefix for folder and file")
    ap.add_argument("--tz",   default=os.environ.get("CAM_TZ", DEFAULT_TZ), help="Timezone for filename (IANA)")
    args = ap.parse_args()

    if not args.url:
        print("ERROR: --url (or WEBTV_URL) is required")
        return 64

    name = args.name.strip() or infer_name(args.url)
    return grab_one(args.url, name, args.tz)

if __name__ == "__main__":
    raise SystemExit(main())
