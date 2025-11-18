#!/usr/bin/env python3
"""
Create a timelapse for a single Sierra Nevada camera.

- Select frames for the last N days (based on date in filename).
- Sort chronologically.
- Skip "glitched" frames:
    * unreadable JPG
    * resolution mismatch
    * exact byte duplicate (same SHA-1 as previous)
- Write an MP4 file (mp4v via OpenCV).

Usage:

    python scripts/create_timelapse.py \
        --camera borreguiles \
        --days 2 \
        --fps 24 \
        --output timelapse_borreguiles.mp4

IMPORTANT: --camera must match the folder under images/, e.g.
    images/borreguiles
    images/satelite
    images/stadium
    images/veleta
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import re

import cv2
import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = ROOT / "images"

# Match ..._YYMMDD_HHMMSS or ..._YYYYMMDD_HHMMSS
TS_RE = re.compile(r"(?P<date>\d{6}|\d{8})_(?P<time>\d{6})")


@dataclass
class FrameInfo:
    dt: datetime
    path: Path
    sha1: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Sierra Nevada timelapse")
    p.add_argument(
        "--camera",
        required=True,
        help="Camera folder name under images/ "
             "(borreguiles, satelite, stadium, veleta)",
    )
    p.add_argument("--days", type=int, default=1, help="How many days back to include (>=1)")
    p.add_argument("--fps", type=float, default=24.0, help="Frames per second in the output video")
    p.add_argument("--output", required=True, help="Output MP4 file path")
    return p.parse_args()


def local_today() -> date:
    if ZoneInfo:
        try:
            tz = ZoneInfo("Europe/Madrid")
            return datetime.now(tz).date()
        except Exception:
            pass
    return datetime.utcnow().date()


def parse_timestamp_from_name(name: str) -> Optional[datetime]:
    """
    Parse datetime from filename pattern:
      *_YYMMDD_HHMMSS.jpg   or   *_YYYYMMDD_HHMMSS.jpg
    """
    m = TS_RE.search(name)
    if not m:
        return None

    d = m.group("date")
    t = m.group("time")

    if len(d) == 8:
        yyyy = int(d[0:4])
        mm = int(d[4:6])
        dd = int(d[6:8])
    else:
        yy = int(d[0:2])
        mm = int(d[2:4])
        dd = int(d[4:6])
        yyyy = 2000 + yy

    hh = int(t[0:2])
    mi = int(t[2:4])
    ss = int(t[4:6])

    try:
        return datetime(yyyy, mm, dd, hh, mi, ss)
    except ValueError:
        return None


def iter_frames_for_camera(cam_folder: str, days: int) -> List[FrameInfo]:
    cam_dir = IMAGES_ROOT / cam_folder
    print(f"[timelapse] IMAGES_ROOT = {IMAGES_ROOT}", file=sys.stderr)
    print(f"[timelapse] cam_dir     = {cam_dir}", file=sys.stderr)

    if not cam_dir.exists():
        print(f"[timelapse] ERROR: camera folder not found: {cam_dir}", file=sys.stderr)
        return []

    today = local_today()
    min_day = today - timedelta(days=max(days, 1) - 1)

    frames: List[FrameInfo] = []

    for p in sorted(cam_dir.rglob("*.jpg")):
        dt = parse_timestamp_from_name(p.name)
        if dt is None:
            continue
        if dt.date() < min_day:
            continue
        if dt.date() > today:
            continue
        frames.append(FrameInfo(dt=dt, path=p))

    frames.sort(key=lambda f: f.dt)
    return frames


def sha1_for_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[timelapse] WARN: failed to hash {path}: {e}", file=sys.stderr)
        return None


def read_image(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[timelapse] WARN: failed to decode {path}", file=sys.stderr)
    return img


def build_timelapse(frames: List[FrameInfo], fps: float, out_path: Path) -> bool:
    if not frames:
        print("[timelapse] No frames to build video.", file=sys.stderr)
        return False

    kept = []
    prev_sha = None
    base_size: Optional[Tuple[int, int]] = None

    for fi in frames:
        sha = sha1_for_file(fi.path)
        if sha is None:
            continue

        # Skip exact duplicates
        if prev_sha is not None and sha == prev_sha:
            print(f"[timelapse] skip duplicate {fi.path.name}")
            continue

        img = read_image(fi.path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if base_size is None:
            base_size = (w, h)
        else:
            if (w, h) != base_size:
                print(
                    f"[timelapse] skip glitched (size change) {fi.path.name} "
                    f"({w}x{h} != {base_size[0]}x{base_size[1]})"
                )
                continue

        kept.append((fi, img))
        prev_sha = sha

    if not kept:
        print("[timelapse] All frames were skipped (duplicates or glitches).", file=sys.stderr)
        return False

    width, height = base_size
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[timelapse] ERROR: could not open VideoWriter", file=sys.stderr)
        return False

    print(f"[timelapse] writing video {out_path} ({len(kept)} frames @ {fps} fps)")
    for fi, img in kept:
        writer.write(img)

    writer.release()
    print("[timelapse] done.")
    return True


def main() -> int:
    args = parse_args()

    cam_folder = args.camera.strip()
    frames = iter_frames_for_camera(cam_folder, args.days)
    print(f"[timelapse] found {len(frames)} candidate frames for folder '{cam_folder}' over last {args.days} days")

    out_path = Path(args.output)
    ok = build_timelapse(frames, args.fps, out_path)
    if not ok:
        return 1

    print(f"[timelapse] OUTPUT_VIDEO={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
