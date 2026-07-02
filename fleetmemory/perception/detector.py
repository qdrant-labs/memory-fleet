"""Class-agnostic object proposals + tracking (PLAN.md §9.1: YOLOE-11L prompt-free).

The detector's built-in vocabulary drives detection; its labels are discarded.
Names come from vector search only. Tracking (BoT-SORT via ultralytics) gives
frame-to-frame continuity — never identity (PLAN.md §12.5).
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

logger = logging.getLogger(__name__)

WEIGHTS = "yoloe-11l-seg-pf.pt"  # auto-downloads to repo root (gitignored)
IMGSZ = 640
MAX_DET = 64
DEFAULT_CONF = 0.30  # spike used 0.25; expected to tune up on live webcam scenes
# Normalized box-area band (spike values): drops speck noise and room-spanning blobs.
MIN_AREA, MAX_AREA = 0.0008, 0.55


def area_band_ok(box: tuple[float, float, float, float]) -> bool:
    """box is (x1, y1, x2, y2) normalized to [0, 1]."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    return MIN_AREA <= area <= MAX_AREA


@dataclass(slots=True)
class Proposal:
    """One tracked region. No label — the detector has no say in what things are."""

    tid: int  # tracker id: continuity only, never identity
    conf: float
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    mask: np.ndarray | None  # segmentation polygon, frame pixel coords


class Detector:
    """YOLOE-11L-seg prompt-free wrapper: load once, track frames, labels dropped."""

    def __init__(self, conf: float = DEFAULT_CONF):
        self.model = None
        self.device = None
        self.conf = conf  # live-tunable

    def load(self):
        if self.model is not None:
            return
        import torch
        from ultralytics import YOLO

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        repo_weights = Path(__file__).resolve().parents[2] / WEIGHTS
        logger.info("loading %s on %s", WEIGHTS, self.device)
        self.model = YOLO(str(repo_weights) if repo_weights.exists() else WEIGHTS)

    def warm(self):
        self.load()
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, imgsz=IMGSZ, verbose=False)

    def reset(self):
        """Fresh tracker state for a new session (track ids keep counting up)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def track(self, frame_bgr: np.ndarray) -> tuple[list[Proposal], float]:
        """Detect + track one frame. Returns (proposals, detect_ms)."""
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter_ns()
        result = self.model.track(
            frame_bgr,
            device=self.device,
            conf=self.conf,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            # one physical item must not survive NMS as several class-named boxes
            agnostic_nms=True,
            persist=True,
            verbose=False,
        )[0]
        detect_ms = (time.perf_counter_ns() - t0) / 1e6

        proposals: list[Proposal] = []
        boxes = result.boxes
        polys = result.masks.xy if result.masks is not None else None
        if boxes is not None and boxes.id is not None:
            triples = zip(boxes.id, boxes.conf, boxes.xyxy, strict=False)
            for i, (tid, cf, xyxy) in enumerate(triples):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                if not area_band_ok(box):
                    continue
                mask = polys[i] if polys is not None and i < len(polys) else None
                proposals.append(Proposal(int(tid), float(cf), box, mask))
        return proposals, detect_ms
