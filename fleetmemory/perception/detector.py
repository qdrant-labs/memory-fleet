"""Class-agnostic object proposals + tracking (YOLOE-11, prompt-free).

The detector's built-in vocabulary drives detection; its labels are discarded.
Names come from vector search only. Tracking (BoT-SORT via ultralytics) gives
frame-to-frame continuity — never identity.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

try:
    # torch-MPS autoreleases Metal objects per inference; a pure-Python loop
    # never drains the pool, leaking ~80 MB/min. Every model call must run
    # inside this pool so the objects are released each iteration.
    from objc import autorelease_pool
except ImportError:  # non-macOS: nothing to drain
    from contextlib import nullcontext as autorelease_pool

logger = logging.getLogger(__name__)

WEIGHTS = "yoloe-11l-seg-pf.pt"  # GPU default (MPS/CUDA); auto-downloads to repo root
CPU_WEIGHTS = "yoloe-11m-seg-pf.pt"  # lighter default when no GPU is present
IMGSZ = 640
MAX_DET = 64
DEFAULT_CONF = 0.30  # tuned for live webcam scenes; live-tunable in the UI
# Normalized box-area band: drops speck noise and oversized phantom regions.
# Live desk scenes produce empty quarter-screen phantom proposals; demo objects
# are hand-held scale, so the area cap stays tight.
MIN_AREA, MAX_AREA = 0.0008, 0.20


def area_band_ok(box: tuple[float, float, float, float], max_area: float = MAX_AREA) -> bool:
    """box is (x1, y1, x2, y2) normalized to [0, 1]."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    return MIN_AREA <= area <= max_area


# People and body parts are suppressed at the proposal level: hands and faces
# must not flood the unknowns queue. This is the ONE use of the detector's class
# names in the app — object names still come from vector search.
PERSON_WORDS = frozenset(
    "person people man men woman women boy girl child kid baby human humans face "
    "faces head hair ear eye eyes nose mouth lip lips chin cheek forehead beard "
    "mustache moustache neck shoulder arm arms elbow wrist hand hands finger "
    "fingers thumb fist chest torso waist hip leg legs knee ankle foot feet toe "
    "toes skin body "
    # hair/face vocabulary the model actually emits for people (its classifier
    # flickers between these and the plain words frame to frame)
    "wig ponytail braid bangs afro dreadlock dreadlocks mane haircut hairstyle "
    "eyebrow eyebrows eyelash eyelashes lash lashes freckle freckles jaw scalp "
    "sideburn sideburns goatee tongue tooth teeth throat nostril manicure "
    "businessman fisherman fireman airman craftsman".split()
)


def is_person_like(class_name: str) -> bool:
    return any(w in PERSON_WORDS for w in class_name.lower().replace("-", " ").split())


@dataclass(slots=True)
class Proposal:
    """One tracked region. The detector's class name rides along as a teach-time
    HINT only — names still enter memory exclusively through humans."""

    tid: int  # tracker id: continuity only, never identity
    conf: float
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    mask: np.ndarray | None  # segmentation polygon, frame pixel coords
    cls: str = ""  # YOLOE's guess, surfaced as a suggestion chip in the teach popover


def pick_device() -> str:
    """cuda > mps > cpu. FM_DEVICE overrides (e.g. force cpu on a shared GPU box)."""
    forced = os.environ.get("FM_DEVICE")
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(override: str | None = None) -> str:
    """FM_MODEL override wins; otherwise the heavier model on GPU, lighter on CPU."""
    if override:
        return override
    return WEIGHTS if pick_device() in ("cuda", "mps") else CPU_WEIGHTS


class Detector:
    """YOLOE-11 seg prompt-free wrapper: load once, track frames, labels dropped."""

    def __init__(self, conf: float = DEFAULT_CONF, model: str | None = None):
        self.model = None
        self.weights = model  # FM_MODEL override; None -> chosen by device in load()
        self.device = None
        self.conf = conf  # live-tunable
        self.max_area = MAX_AREA  # live-tunable: biggest proposal kept, frame fraction
        # Person suppression is sticky per track: the classifier flickers (a hair
        # patch reads "hair" one frame, "wig" or "fur" the next), so a track that
        # has EVER looked person-like stays suppressed for its lifetime.
        self._person_tids: set[int] = set()

    def load(self):
        if self.model is not None:
            return
        from ultralytics import YOLO

        self.device = pick_device()
        self.weights = resolve_weights(self.weights)
        repo_weights = Path(__file__).resolve().parents[2] / self.weights
        logger.info("loading %s on %s", self.weights, self.device)
        self.model = YOLO(str(repo_weights) if repo_weights.exists() else self.weights)

    def warm(self):
        self.load()
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        with autorelease_pool():
            self.model.predict(dummy, device=self.device, imgsz=IMGSZ, verbose=False)

    def reset(self):
        """Fresh tracker state for a new session (track ids keep counting up)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
        self._person_tids.clear()

    def track(self, frame_bgr: np.ndarray) -> tuple[list[Proposal], float]:
        """Detect + track one frame. Returns (proposals, detect_ms)."""
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter_ns()
        with autorelease_pool():
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
            quads = zip(boxes.id, boxes.cls, boxes.conf, boxes.xyxy, strict=False)
            for i, (tid, cls, cf, xyxy) in enumerate(quads):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                # person check runs before the area band so an oversized face box
                # still poisons its track id for later, smaller frames
                tid = int(tid)
                cls_name = result.names.get(int(cls), "")
                if tid in self._person_tids:
                    continue
                if is_person_like(cls_name):
                    self._person_tids.add(tid)
                    continue
                if not area_band_ok(box, self.max_area):
                    continue
                mask = polys[i] if polys is not None and i < len(polys) else None
                proposals.append(Proposal(tid, float(cf), box, mask, cls_name))
        if len(self._person_tids) > 4096:  # ids only grow; keep the recent flags
            self._person_tids = set(sorted(self._person_tids)[-1024:])
        return proposals, detect_ms
