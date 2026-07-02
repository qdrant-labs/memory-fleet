"""Masked crops for embedding — v1's exact crop path (PLAN.md §2).

The embedding benchmark (§9.2) measured separations through this code; the
mask fill (background flattened to neutral gray inside the segmentation
polygon) is what makes recognition survive background and hand changes.
Keep in lockstep with docs/spikes — thresholds were derived through it.
"""

import cv2
import numpy as np
from PIL import Image

CROP_PAD = 0.12  # small margin; the mask removes the background anyway
FILL = (124, 124, 124)


def padded_crop(
    frame_bgr: np.ndarray,
    box: tuple[float, float, float, float],
    mask: np.ndarray | None = None,
) -> Image.Image:
    """Crop one item for embedding; box normalized, mask in frame pixel coords."""
    h, w = frame_bgr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int((box[0] - bw * CROP_PAD) * w))
    y1 = max(0, int((box[1] - bh * CROP_PAD) * h))
    x2 = min(w, int((box[2] + bw * CROP_PAD) * w))
    y2 = min(h, int((box[3] + bh * CROP_PAD) * h))
    region = frame_bgr[y1:y2, x1:x2]

    if mask is not None and len(mask) >= 3 and region.size:
        full = np.zeros((h, w), np.uint8)
        cv2.fillPoly(full, [np.asarray(mask, dtype=np.int32)], 255)
        full = cv2.dilate(full, np.ones((7, 7), np.uint8), iterations=1)
        inside = full[y1:y2, x1:x2] > 0
        region = region.copy()
        region[~inside] = FILL

    return Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))


def crop_quality(frame_bgr: np.ndarray, box, conf: float) -> float:
    """Bigger, sharper, more confident crops make better object portraits."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    region = frame_bgr[y1:y2, x1:x2]
    small = cv2.resize(region, (96, 96), interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    area = (box[2] - box[0]) * (box[3] - box[1])
    return conf * (area**0.5) * min(sharp, 1200.0)
