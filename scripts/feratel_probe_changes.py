#!/usr/bin/env python3
"""
Probe if any camera image changed vs last saved file (per-camera MD5).
Prints only 'true' or 'false' to stdout (so the workflow can read it).
Returns exit 0 always.
"""

import os, glob, hashlib, requests, time

CAMS = {
    "borreguiles": "https://wtvpict.feratel.com/picture/35/15111.jpeg",
    "stadium":     "https://wtvpict.feratel.com/picture/35/15112.jpeg",
    "satelite":    "https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg",
    "veleta":      "https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg",
}
BASE_DIR = "images"
HEADERS = {"User-Agent": "Mozilla/5.0 (FeratelProbe/1.0)", "Cache-Control": "no-cache", "Pragma": "no-cache"}
TIMEOUT = 20

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def latest_saved_md5(cam: str):
    files = sorted(glob.glob(os.path.join(BASE_DIR, cam, f"{cam}_*.jpg")))
    if not files: return None
    try:
        with open(files[-1], "rb") as f: return md5_bytes(f.read())
    except Exception:
        return None

def fetch(url: str):
    sep = "&" if "?" in url else "?"
    bust = f"{sep}_ts={int(time.time())}"
    try:
        r = requests.get(url + bust, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200: return r.content
    except Exception:
        return None
    return None

def main():
    # If any cam has no previous file OR bytes differ -> changed=true
    for cam, url in CAMS.items():
        prev = latest_saved_md5(cam)
        b = fetch(url)
        if not b:
            continue
        cur = md5_bytes(b)
        if prev is None or cur != prev:
            print("true")
            return
    print("false")

if __name__ == "__main__":
    main()
