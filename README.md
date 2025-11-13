# Sierra Nevada — Webcam Archive

An ongoing, lightweight archive of Sierra Nevada webcams. Frames are captured throughout the day, named by local time, and published to a simple website for easy scrolling.

**Open the viewer:** https://mmiiffii.github.io/Sierra-Panorama-Capture/

---

## What this repo does

- Captures still images from select Sierra Nevada webcams and stores them in date-stamped folders.
- Skips near-duplicates so the overnight “frozen frame” doesn’t clutter the morning.
- Builds tiny JSON lists so the web viewer can load quickly and let you scrub through time.
- Optionally saves ready-made 360° panoramas from the camera’s panorama view.

---

## Quick camera links

- **Borreguiles — still image**  
  https://wtvpict.feratel.com/picture/35/15111.jpeg  
  **Panorama view:** https://webtv.feratel.com/webtv/?cam=15111&t=9

- **Stadium — still image**  
  https://wtvpict.feratel.com/picture/35/15112.jpeg  
  **Panorama view:** https://webtv.feratel.com/webtv/?cam=15112&t=9

- **Satelite — still image**  
  https://webtvhotspot.feratel.com/hotspot/35/15111/0.jpeg

- **Veleta — still image**  
  https://webtvhotspot.feratel.com/hotspot/35/15112/1.jpeg

---

Times are recorded in **Europe/Madrid**. The website is served from the `docs/` folder and reads the manifests there to display the latest images.
