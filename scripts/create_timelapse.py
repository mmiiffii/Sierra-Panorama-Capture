#!/usr/bin/env python3
"""
Create a timelapse for a single Sierra Nevada camera.

Features:
- Select frames for the last N days (based on date in filename).
- Sort chronologically.
- Skip "glitched" frames:
    * unreadable JPG
    * resolution mismatch
    * exact byte duplicate (same SHA-1 as previous).
- Optional downscale to a max width (e.g. 1920).
- Optional crossfade over large time gaps (e.g. night → morning).
- Write an MP4 file (mp4v via OpenCV).

Usage:

    python scripts/create_timelapse.py \
        --camera borreguiles \
        --days 2 \
        --fps 24 \
        --max-width 1920 \
        --fade-gaps \
        --output timelapses/timelapse_borreguiles.mp4

IMPORTANT: --camera must match the folder under images/:
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

# Match ..._YYMMDD_HHMMSS or ..._YYYYMMDD_HHMMSS anywhere in the filename
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
    p.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Optional max video width in pixels (0 = use native size)",
    )
    p.add_argument(
        "--fade-gaps",
        action="store_true",
        help="Crossfade over big time gaps between frames (e.g. night → morning)",
    )
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


def decide_output_size(
    base_size: Tuple[int, int],
    max_width: int,
) -> Tuple[int, int, float]:
    """Return (out_w, out_h, scale_factor) given base size and max_width."""
    base_w, base_h = base_size
    if max_width <= 0 or base_w <= max_width:
        return base_w, base_h, 1.0
    scale = max_width / float(base_w)
    out_w = int(round(base_w * scale))
    out_h = int(round(base_h * scale))
    return out_w, out_h, scale


def build_timelapse(
    frames: List[FrameInfo],
    fps: float,
    out_path: Path,
    max_width: int,
    fade_gaps: bool,
    gap_hours_threshold: float = 8.0,
    gap_fade_frames: int = 10,
) -> bool:
    if not frames:
        print("[timelapse] No frames to build video.", file=sys.stderr)
        return False

    kept = []
    prev_sha = None
    base_size: Optional[Tuple[int, int]] = None

    # First pass: load images, enforce consistent size, de-dup, keep in memory
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

    assert base_size is not None
    out_w, out_h, scale = decide_output_size(base_size, max_width)

    if scale != 1.0:
        print(
            f"[timelapse] downscaling from {base_size[0]}x{base_size[1]} "
            f"to {out_w}x{out_h} (scale={scale:.3f})",
            file=sys.stderr,
        )
    else:
        print(f"[timelapse] using native size {out_w}x{out_h}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("[timelapse] ERROR: could not open VideoWriter", file=sys.stderr)
        return False

    print(f"[timelapse] writing video {out_path} ({len(kept)} base frames @ {fps} fps)")

    prev_frame_written: Optional[np.ndarray] = None
    prev_dt: Optional[datetime] = None

    for idx, (fi, img) in enumerate(kept):
        # Resize if needed
        if scale != 1.0:
            frame = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame = img

        # Gap fading (night → morning, etc.)
        if (
            fade_gaps
            and prev_frame_written is not None
            and prev_dt is not None
        ):
            gap_hours = (fi.dt - prev_dt).total_seconds() / 3600.0
            if gap_hours >= gap_hours_threshold:
                print(
                    f"[timelapse] gap {gap_hours:.1f}h between "
                    f"{prev_dt} and {fi.dt} → inserting {gap_fade_frames} fade frames",
                    file=sys.stderr,
                )
                for k in range(1, gap_fade_frames + 1):
                    alpha = k / float(gap_fade_frames + 1)
                    blend = cv2.addWeighted(prev_frame_written, 1.0 - alpha, frame, alpha, 0.0)
                    writer.write(blend)

        # Write current frame
        writer.write(frame)

        prev_frame_written = frame
        prev_dt = fi.dt

    writer.release()
    print("[timelapse] done.")
    return True


def main() -> int:
    args = parse_args()

    cam_folder = args.camera.strip()
    frames = iter_frames_for_camera(cam_folder, args.days)
    print(
        f"[timelapse] found {len(frames)} candidate frames for folder "
        f"'{cam_folder}' over last {args.days} days"
    )

    out_path = Path(args.output)
    ok = build_timelapse(
        frames,
        args.fps,
        out_path,
        args.max_width,
        args.fade_gaps,
    )
    if not ok:
        return 1

    print(f"[timelapse] OUTPUT_VIDEO={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
