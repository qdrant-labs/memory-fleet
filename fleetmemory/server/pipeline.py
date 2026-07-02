"""Perception -> core glue (PLAN.md §3.2): capture/detect on one worker thread,
embeds on another; results enter the core as messages. The server owns no logic —
this file is scheduling and plumbing only.
"""

import base64
import logging
import queue
import threading
import time

import cv2
import numpy as np

from fleetmemory.memory.core import Core, Ingest, TrackDied
from fleetmemory.perception.cadence import EmbedScheduler
from fleetmemory.perception.crops import crop_quality, padded_crop
from fleetmemory.perception.detector import Detector
from fleetmemory.perception.embedder import Embedder

logger = logging.getLogger(__name__)

BROADCAST_WIDTH = 960
THUMB_SIZE = 128
DRIVE_FPS = 5.0  # drive mode: synthetic clock, deterministic against fed frames


class CameraSource:
    def __init__(self, index: int):
        self.index = index
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index)
        return self.cap.isOpened()

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class DriveSource:
    """Frames arrive over the WebSocket (tests/drive). JPEG bytes queue unbounded
    (small); decode happens on the capture thread so the event loop never blocks."""

    def __init__(self):
        self.frames: queue.Queue = queue.Queue()

    def open(self) -> bool:
        return True

    def push(self, jpg: bytes):
        self.frames.put(jpg)

    def read(self):
        try:
            jpg = self.frames.get(timeout=0.25)
        except queue.Empty:
            return None
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        pass


class Pipeline:
    """One capture thread + one embed thread; everything else is messages."""

    def __init__(self, core: Core, source, broadcast, drive_mode: bool = False):
        self.core = core
        self.source = source
        self.broadcast = broadcast  # thread-safe callable(dict)
        self.drive_mode = drive_mode
        self.detector = Detector()
        self.embedder = Embedder()
        self.scheduler = EmbedScheduler()
        self.burst_tids: set[int] = set()  # updated from core events (GIL-safe set ops)
        self._embed_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._frame_count = 0
        # camera runs only while someone is watching; drive mode is always on
        self._active = threading.Event()
        if drive_mode:
            self._active.set()

    # -- lifecycle --

    def start(self):
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._embed_loop, name="embed", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        self._embed_q.put(None)
        for t in self._threads:
            t.join(timeout=5)
        self.source.close()

    def set_active(self, on: bool):
        """First viewer connects -> camera on; last one leaves -> camera released."""
        if on or self.drive_mode:
            self._active.set()
        else:
            self._active.clear()

    def note_event(self, ev: dict):
        """Track which tids are mid-burst so the scheduler embeds them every tick."""
        if ev.get("type") != "track_update":
            return
        tid = ev["tid"]
        if ev["state"] == "capturing":
            self.burst_tids.add(tid)
        else:
            self.burst_tids.discard(tid)

    # -- capture thread --

    def _capture_loop(self):
        self.detector.warm()
        src_open = False
        while not self._stop.is_set():
            if not self._active.is_set():
                if src_open:
                    self.source.close()  # camera light goes off
                    src_open = False
                self._active.wait(timeout=0.25)
                continue
            if not src_open:
                if not self.source.open():
                    self.broadcast({"type": "error", "message": "video source unavailable"})
                    self._stop.wait(2.0)  # retry while a viewer is connected
                    continue
                src_open = True
            frame = self.source.read()
            if frame is None:
                if self.drive_mode:
                    continue  # waiting for injected frames
                self.broadcast({"type": "error", "message": "camera read failed — retrying"})
                self.source.close()
                src_open = False
                self._stop.wait(1.0)
                continue
            self._tick(frame)
        if src_open:
            self.source.close()

    def _tick(self, frame):
        self._frame_count += 1
        now = self._frame_count / DRIVE_FPS if self.drive_mode else time.time()
        props, detect_ms = self.detector.track(frame)

        to_embed, died = self.scheduler.tick(
            [p.tid for p in props], now, burst_tids=self.burst_tids
        )
        for inc in died:
            self.core.submit(TrackDied(tid=inc.tid, epoch=inc.epoch))

        due = {i.tid: i.epoch for i in to_embed}
        jobs = []
        for p in props:
            if p.tid in due:
                crop = padded_crop(frame, p.box, p.mask)
                q = crop_quality(frame, p.box, p.conf)
                jobs.append((p.tid, due[p.tid], crop, q, now))
        if jobs:
            self._embed_q.put(jobs)

        self.broadcast(
            {
                "type": "frame",
                "jpg": _encode_frame(frame),
                "detect_ms": round(detect_ms, 1),
                "boxes": [
                    {
                        "tid": p.tid,
                        "epoch": self.scheduler.epoch(p.tid),
                        "box": [round(v, 4) for v in p.box],
                        "conf": round(p.conf, 2),
                        "stability": self.scheduler.stability(p.tid),
                    }
                    for p in props
                ],
            }
        )

    # -- embed thread --

    def _embed_loop(self):
        self.embedder.load()
        while not self._stop.is_set():
            jobs = self._embed_q.get()
            if jobs is None:
                return
            try:
                vecs = self.embedder.embed([crop for _, _, crop, _, _ in jobs])
            except Exception:
                logger.exception("embed batch failed")
                continue
            for (tid, epoch, crop, quality, t), vec in zip(jobs, vecs, strict=True):
                self.core.submit(
                    Ingest(
                        tid=tid,
                        epoch=epoch,
                        vec=vec,
                        thumb_jpeg=_thumb_jpeg(crop),
                        quality=quality,
                        t=t,
                    )
                )


def _encode_frame(frame) -> str:
    h, w = frame.shape[:2]
    if w > BROADCAST_WIDTH:
        frame = cv2.resize(frame, (BROADCAST_WIDTH, int(h * BROADCAST_WIDTH / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode() if ok else ""


def _thumb_jpeg(crop) -> bytes:
    img = np.asarray(crop)[:, :, ::-1]  # PIL RGB -> BGR
    h, w = img.shape[:2]
    s = THUMB_SIZE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""
