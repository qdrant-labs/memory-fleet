"""Detector spike: class-agnostic proposal quality + speed on MPS.

Candidates (labels discarded — hive-mind names come from vector search only):
  - FastSAM-s / FastSAM-x : segment-anything-style, inherently class-agnostic
  - yoloe-11l-seg-pf      : YOLOE prompt-free (built-in 4.5k vocab), labels ignored
  - yoloe-11s-seg-pf      : small prompt-free variant

Measures ms/frame (after warmup), proposals/frame (area-filtered), and saves
annotated frames for visual review.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, FastSAM

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CLIPS = ["footage/mission.mp4", "footage/7578546-hd_1920_1080_30fps.mp4"]
N_FRAMES = 12          # sampled per clip
IMGSZ = 640
MIN_AREA, MAX_AREA = 0.0008, 0.55   # normalized box area filter (v1-like)

frames = []
for cp in CLIPS:
    cap = cv2.VideoCapture(cp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, total - 1, N_FRAMES, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append((Path(cp).stem.split("-")[0], int(i), fr))
    cap.release()
print(f"{len(frames)} frames sampled on {DEV}")

MODELS = [
    ("fastsam-s", lambda: FastSAM("FastSAM-s.pt"), {"conf": 0.4, "iou": 0.9}),
    ("fastsam-x", lambda: FastSAM("FastSAM-x.pt"), {"conf": 0.4, "iou": 0.9}),
    ("yoloe-11l-pf", lambda: YOLO("yoloe-11l-seg-pf.pt"), {"conf": 0.25}),
    ("yoloe-11s-pf", lambda: YOLO("yoloe-11s-seg-pf.pt"), {"conf": 0.25}),
]

for name, loader, kw in MODELS:
    try:
        model = loader()
    except Exception as e:
        print(f"{name:14s} LOAD FAILED: {e}")
        continue
    # warmup
    model.predict(frames[0][2], device=DEV, imgsz=IMGSZ, verbose=False, **kw)

    times, counts = [], []
    for idx, (clip, fi, fr) in enumerate(frames):
        t0 = time.perf_counter()
        res = model.predict(fr, device=DEV, imgsz=IMGSZ, verbose=False,
                            max_det=64, **kw)[0]
        times.append((time.perf_counter() - t0) * 1000)

        h, w = fr.shape[:2]
        keep = []
        if res.boxes is not None:
            for bi, xyxy in enumerate(res.boxes.xyxy):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                area = ((x2 - x1) / w) * ((y2 - y1) / h)
                if MIN_AREA <= area <= MAX_AREA:
                    keep.append(bi)
        counts.append(len(keep))

        if idx % 6 == 0:  # save a few annotated frames for eyeballing
            vis = fr.copy()
            if res.masks is not None:
                for bi in keep:
                    if bi < len(res.masks.xy):
                        poly = np.asarray(res.masks.xy[bi], dtype=np.int32)
                        if len(poly) >= 3:
                            cv2.polylines(vis, [poly], True, (0, 255, 180), 2)
            for bi in keep:
                x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[bi])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 160, 255), 2)
            small = cv2.resize(vis, (1280, 720))
            cv2.imwrite(str(OUT / f"{name}_{clip}_{fi}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])

    print(f"{name:14s} {np.median(times):6.0f} ms/frame (p90 {np.percentile(times, 90):.0f})"
          f" | proposals/frame med {np.median(counts):.0f} (min {min(counts)}, max {max(counts)})")
