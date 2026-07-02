"""Extract tracked, masked object crops from footage for the embedding benchmark.

Reuses v1's exact perception path (YOLOE track + padded_crop with mask fill) so
measured separations transfer to the rewrite. Output: crops/<clip>/<tid>_<frame>.jpg
plus manifest.json with {clip, tid, cls, conf, frame} per crop, and a per-clip
contact sheet for manual same-item verification.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from app.detector import ObjectDetector  # noqa: E402
from app.vision import padded_crop  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("crops_out")
CLIPS = ["footage/mission.mp4", "footage/7578546-hd_1920_1080_30fps.mp4",
         "footage/7578549-hd_1920_1080_30fps.mp4"]
FRAME_STEP = 6          # 30fps -> 5fps, matches live ingest
MAX_SAMPLED = 260       # cap per clip (~52s of video)
MIN_CONF = 0.45
MIN_CROPS_PER_TRACK = 5
MAX_CROPS_PER_TRACK = 12
MIN_GAP_FRAMES = 12     # spacing between kept crops (angle variety, not near-dupes)

OUT.mkdir(parents=True, exist_ok=True)
manifest = []

det = ObjectDetector()
det.warm()

for clip_path in CLIPS:
    clip = Path(clip_path).stem.split("-")[0]
    clip_dir = OUT / clip
    clip_dir.mkdir(exist_ok=True)
    det.reset()

    cap = cv2.VideoCapture(clip_path)
    per_track = defaultdict(list)  # tid -> list of (frame_idx, cls, conf, crop PIL)
    last_kept = {}
    fidx = sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok or sampled >= MAX_SAMPLED:
            break
        if fidx % FRAME_STEP:
            fidx += 1
            continue
        sampled += 1
        dets, _ = det.track(frame)
        for d in dets:
            if d.conf < MIN_CONF:
                continue
            if len(per_track[d.track_id]) >= MAX_CROPS_PER_TRACK:
                continue
            if fidx - last_kept.get(d.track_id, -999) < MIN_GAP_FRAMES:
                continue
            crop = padded_crop(frame, d.box, d.mask)
            if crop.width < 48 or crop.height < 48:
                continue
            per_track[d.track_id].append((fidx, d.cls, d.conf, crop))
            last_kept[d.track_id] = fidx
        fidx += 1
    cap.release()

    kept_tracks = {tid: rows for tid, rows in per_track.items()
                   if len(rows) >= MIN_CROPS_PER_TRACK}
    print(f"{clip}: {sampled} frames sampled, {len(per_track)} tracks seen, "
          f"{len(kept_tracks)} kept (>= {MIN_CROPS_PER_TRACK} crops)")

    for tid, rows in kept_tracks.items():
        for fi, cls, conf, crop in rows:
            name = f"{tid}_{fi}.jpg"
            crop.save(clip_dir / name, quality=90)
            manifest.append({"clip": clip, "tid": tid, "cls": cls,
                             "conf": round(conf, 3), "frame": fi, "file": f"{clip}/{name}"})

    # Contact sheet: one row per track (tid + cls + up to 6 thumbs) for manual
    # same-item verification.
    tile, per_row = 112, 6
    rows_sorted = sorted(kept_tracks.items())
    sheet = Image.new("RGB", (140 + tile * per_row, tile * max(len(rows_sorted), 1)), "black")
    drw = ImageDraw.Draw(sheet)
    for r, (tid, rows) in enumerate(rows_sorted):
        drw.text((4, r * tile + tile // 2 - 12), f"t{tid}", fill="yellow")
        drw.text((4, r * tile + tile // 2 + 2), rows[0][1][:16], fill="white")
        step = max(1, len(rows) // per_row)
        for c, (fi, cls, conf, crop) in enumerate(rows[::step][:per_row]):
            th = crop.copy()
            th.thumbnail((tile, tile))
            sheet.paste(th, (140 + c * tile, r * tile))
    sheet.save(OUT / f"sheet_{clip}.jpg", quality=88)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"total crops: {len(manifest)} -> {OUT}")
