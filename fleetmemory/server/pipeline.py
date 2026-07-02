"""Perception -> core glue. A grabber thread owns the camera and streams display
frames, a detect thread runs YOLOE at a tunable cadence on the latest frame and
emits boxes-only, and an embed thread runs Unicom on scheduled crops. Results
enter the single-threaded memory core as messages. The server owns no logic —
scheduling and plumbing only.
"""

import base64
import contextlib
import logging
import queue
import sys
import threading
import time
from collections import Counter

import cv2
import numpy as np

from fleetmemory.memory.core import Core, Ingest, TrackDied
from fleetmemory.perception.cadence import EmbedScheduler
from fleetmemory.perception.crops import crop_quality, padded_crop
from fleetmemory.perception.detector import Detector
from fleetmemory.perception.embedder import Embedder

logger = logging.getLogger(__name__)

# The detect thread's Python-side glue (ultralytics pre/post, BYTETrack) holds
# the GIL in chunks; with the default 5 ms switch interval the grabber misses
# camera frames (macOS AVFoundation keeps only the latest) and video halves to
# ~15 fps. Shorter slices keep the stream near camera rate (~25 fps).
sys.setswitchinterval(0.002)

BROADCAST_WIDTH = 960
THUMB_SIZE = 224  # big enough for the lightbox; only ONE rides in a fleet payload
# Detection cadence. Unpaced, the detector runs MPS at 100% duty (~13 fps) and
# cooks the laptop for nothing — the demo's ingest target is ~5-8 fps.
TARGET_FPS = 8.0
# Video is decoupled from detection: 8 fps video looks choppy. The grabber thread
# owns the camera and streams JPEG frames at up to DISPLAY_FPS (CPU-cheap,
# ~1.3 ms/frame), while the detect thread takes the latest frame at TARGET_FPS
# and broadcasts boxes-only messages; the client interpolates boxes between
# ticks. They must be separate threads: a ~90 ms MPS detect call on the read
# loop caps video at ~10 fps.
DISPLAY_FPS = 30.0


class CameraSource:
    def __init__(self, index: int):
        self.index = index
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            # 1080p drags this sensor to ~20 fps and buys nothing (broadcast is
            # 960w and YOLOE resizes to its own input); 720p reads at ~30 fps
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap.isOpened()

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class Pipeline:
    """Three threads: a grabber that owns the camera and streams frames, a detect
    thread that runs YOLOE on the latest frame, and an embed thread. Everything
    else flows to the core as messages."""

    def __init__(self, core: Core, source, broadcast):
        self.core = core
        self.source = source
        self.broadcast = broadcast  # thread-safe callable(dict)
        self.detector = Detector()
        self.embedder = Embedder()
        self.scheduler = EmbedScheduler()
        self.burst_tids: set[int] = set()  # updated from core events (GIL-safe set ops)
        # bounded: if the embed thread dies or stalls, crops must not pile up forever
        self._embed_q: queue.Queue = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # camera runs only while someone is watching AND the user has it enabled
        self._active = threading.Event()
        self._viewers = False
        self.user_enabled = True
        self.target_fps = TARGET_FPS  # live-tunable detection cadence (UI dial)
        # rolling perf counters for the HUD's pipeline ticker
        self.last_embed_ms = 0.0
        self._tick_times: list[float] = []
        self._last_perf = 0.0
        self._cls_seen: dict[int, Counter] = {}  # tid -> detector class guesses
        self._last_tick = 0.0
        self._next_display = 0.0
        # latest camera frame handed from the grabber to the detect thread;
        # the session counter tells the detect thread to reset tracker state
        self._latest_lock = threading.Lock()
        self._latest = None  # (seq, frame)
        self._frame_seq = 0
        self._session = 0

    # -- lifecycle --

    def start(self):
        workers = [("grab", self._grab_loop), ("detect", self._detect_loop)]
        workers.append(("embed", self._embed_loop))
        self._threads = [
            threading.Thread(target=fn, name=name, daemon=True) for name, fn in workers
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._embed_q.put_nowait(None)
        for t in self._threads:
            t.join(timeout=5)
        # only close from here if the source-owning thread is truly gone — it
        # closes the source itself on exit, and yanking cv2 out from under a
        # live read() crashes (slow first-run warm can outlive the join timeout)
        if not any(t.is_alive() for t in self._threads):
            self.source.close()

    def set_active(self, on: bool):
        """First viewer connects -> camera on; last one leaves -> camera released."""
        self._viewers = on
        self._recompute()

    def set_user_enabled(self, on: bool):
        """The UI's camera on/off switch; wins over viewer presence."""
        self.user_enabled = on
        self._recompute()
        self.broadcast({"type": "camera", "on": on})

    def _recompute(self):
        if self._viewers and self.user_enabled:
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

    # -- grabber thread (live only): owns the camera, streams display frames --

    def _grab_loop(self):
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
                self._session += 1  # detect thread resets tracker for the new session
            frame = self.source.read()
            if frame is None:
                self.broadcast({"type": "error", "message": "camera read failed — retrying"})
                self.source.close()
                src_open = False
                self._stop.wait(1.0)
                continue
            self._frame_seq += 1
            with self._latest_lock:
                self._latest = (self._frame_seq, frame)
            now = time.time()
            if now >= self._next_display:
                # slot accumulator, not a stamp: a ~29 fps camera against a
                # stamped gate beats down to every other frame (~15 fps)
                self._next_display = max(
                    self._next_display + 1.0 / DISPLAY_FPS, now - 1.0 / DISPLAY_FPS
                )
                self.broadcast({"type": "frame", "jpg": _encode_frame(frame)})
        if src_open:
            self.source.close()

    # -- detect thread (live only): latest frame at TARGET_FPS, boxes-only --

    def _detect_loop(self):
        try:
            self.detector.warm()
        except Exception:
            logger.exception("detector failed to load")
            self.broadcast({"type": "error", "message": "detector failed to load — see server log"})
            return
        last_session = -1  # force a tracker reset on the first tick
        last_seq = 0
        while not self._stop.is_set():
            if not self._active.is_set():
                self._active.wait(timeout=0.25)
                continue
            wait = self._last_tick + 1.0 / self.target_fps - time.time()
            if wait > 0:
                self._stop.wait(min(wait, 0.05))
                continue
            with self._latest_lock:
                item = self._latest
            if item is None or item[0] == last_seq:
                self._stop.wait(0.01)  # no fresh frame yet
                continue
            last_seq, frame = item
            if self._session != last_session:
                last_session = self._session
                self.detector.reset()  # fresh tracker state for the new camera session
            self._last_tick = time.time()
            self._tick(frame)

    def _tick(self, frame):
        now = time.time()
        props, detect_ms = self.detector.track(frame)
        self._perf(detect_ms, len(props))

        to_embed, died = self.scheduler.tick(
            [p.tid for p in props], now, burst_tids=self.burst_tids
        )
        for inc in died:
            self.core.submit(TrackDied(tid=inc.tid, epoch=inc.epoch))
            self._cls_seen.pop(inc.tid, None)

        # accumulate YOLOE's per-track class guesses (teach-popover hints —
        # the class flickers frame to frame, so the top 3 are genuine options)
        for p in props:
            if p.cls:
                self._cls_seen.setdefault(p.tid, Counter())[p.cls] += 1

        due = {i.tid: i.epoch for i in to_embed}
        jobs = []
        for p in props:
            if p.tid in due:
                crop = padded_crop(frame, p.box, p.mask)
                q = crop_quality(frame, p.box, p.conf)
                jobs.append((p.tid, due[p.tid], crop, q, now))
        if jobs:
            with contextlib.suppress(queue.Full):  # shed embeds rather than balloon
                self._embed_q.put_nowait(jobs)

        # boxes-only: video streams from the grabber thread
        self.broadcast(
            {
                "type": "boxes",
                "detect_ms": round(detect_ms, 1),
                "boxes": [
                    {
                        "tid": p.tid,
                        "epoch": self.scheduler.epoch(p.tid),
                        "box": [round(v, 4) for v in p.box],
                        "conf": round(p.conf, 2),
                        "stability": self.scheduler.stability(p.tid),
                        "hints": [
                            c for c, _ in self._cls_seen.get(p.tid, Counter()).most_common(3)
                        ],
                    }
                    for p in props
                ],
            }
        )

    def _perf(self, detect_ms: float, n_tracks: int):
        t = time.time()
        self._tick_times = [x for x in self._tick_times if t - x < 3.0] + [t]
        if t - self._last_perf >= 1.0:
            self._last_perf = t
            self.broadcast(
                {
                    "type": "perf",
                    "fps": round(len(self._tick_times) / 3.0, 1),
                    "detect_ms": round(detect_ms, 0),
                    "embed_ms": round(self.last_embed_ms, 1),
                    "tracks": n_tracks,
                }
            )

    # -- embed thread --

    def _embed_loop(self):
        try:
            self.embedder.load()
        except Exception:
            logger.exception("embedder failed to load")
            self.broadcast({"type": "error", "message": "embedder failed to load — see server log"})
            while not self._stop.is_set():  # keep draining so the queue can't grow
                self._embed_q.get()
            return
        while not self._stop.is_set():
            jobs = self._embed_q.get()
            if jobs is None:
                return
            try:
                t0 = time.perf_counter_ns()
                vecs = self.embedder.embed([crop for _, _, crop, _, _ in jobs])
                self.last_embed_ms = (time.perf_counter_ns() - t0) / 1e6 / len(jobs)
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
