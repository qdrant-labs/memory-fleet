"""Sync lifecycle (PLAN.md §3.6): pull every ~30 s (or on demand), push only
from curation. Downloads happen on this worker; shard mutations happen as
queued core messages. Sync failures degrade to "fleet offline", never crash.
"""

import logging
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from fleetmemory.memory.core import (
    ApplyPartialSnapshot,
    Core,
    ManifestRequest,
    MarkPushed,
    PreparePush,
    fold_rows,
)
from fleetmemory.memory.matcher import S_SUGGEST
from fleetmemory.sync.client import FleetClient

logger = logging.getLogger(__name__)

PULL_INTERVAL = 30.0
REPLY_TIMEOUT = 10.0


class SyncManager:
    def __init__(
        self, core: Core, client: FleetClient, on_event=None, interval: float = PULL_INTERVAL
    ):
        self.core = core
        self.client = client
        self.interval = interval
        self.online: bool | None = None  # None = never tried yet
        self._on_event = on_event or (lambda e: None)
        self._jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _set_online(self, on: bool):
        """The demo is local-first; the cloud fleet is a bonus. Status events
        fire on TRANSITIONS only — an unreachable fleet must not nag."""
        if self.online != on:
            self.online = on
            self._on_event({"type": "fleet_status", "online": on})

    # ---------- lifecycle ----------

    def start(self):
        self._thread = threading.Thread(target=self._run, name="sync", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._jobs.put(None)
        if self._thread:
            self._thread.join(timeout=10)

    def request_pull(self):
        self._jobs.put(("pull",))

    def request_push(self, object_ids: list):
        self._jobs.put(("push", object_ids))

    # ---------- worker ----------

    def _run(self):
        self._ensured = False
        try:
            self.client.ensure_collection()
            self._ensured = True
            self.pull_once()  # seed / catch-up on boot
            self._set_online(True)
        except Exception as e:
            logger.info("sync: fleet unreachable at boot (%s) — running local-first", e)
            self._set_online(False)
        next_pull = time.time() + self.interval
        while not self._stop.is_set():
            timeout = max(0.2, next_pull - time.time())
            try:
                job = self._jobs.get(timeout=timeout)
            except queue.Empty:
                job = ("pull",)
            if job is None:
                return
            try:
                if not self._ensured:  # fleet came back after an offline boot
                    self.client.ensure_collection()
                    self._ensured = True
                if job[0] == "pull":
                    self.pull_once()
                    next_pull = time.time() + self.interval
                elif job[0] == "push":
                    self.push(job[1])
                self._set_online(True)
            except Exception as e:
                logger.warning("sync: %s failed: %s", job[0], e)
                self._set_online(False)
                if job[0] == "push":  # a user-initiated action deserves a reply
                    self._on_event(
                        {"type": "fleet_error", "message": "push failed — fleet unreachable"}
                    )

    # ---------- pull: manifest (core) -> download (here) -> apply (core) ----------

    def pull_once(self):
        reply: queue.Queue = queue.Queue()
        self.core.submit(ManifestRequest(reply=reply))
        manifest = reply.get(timeout=REPLY_TIMEOUT)
        if manifest is None:
            return  # no immutable mirror (local mode)
        # The core owns the snapshot file from the moment it's submitted — it
        # deletes it after applying. Never tie the file's lifetime to a timeout
        # here: a busy core (first boot) had the temp dir yanked mid-unpack.
        workdir = Path(tempfile.mkdtemp(prefix="fm-pull-"))
        try:
            dest = workdir / "partial.snapshot"
            self.client.download_partial_snapshot(manifest, dest)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        self.core.submit(ApplyPartialSnapshot(path=str(dest), cleanup=True))

    # ---------- push: prepare (core) -> fleet ops (here) -> mark (core) ----------

    def push(self, object_ids: list):
        reply: queue.Queue = queue.Queue()
        self.core.submit(PreparePush(object_ids=object_ids, reply=reply))
        objs = reply.get(timeout=REPLY_TIMEOUT)
        if not objs:  # everything filtered (blocklist/synthetic/missing): still ack
            self.core.submit(MarkPushed(items=[]))
            return
        items = []
        now = time.time()
        for o in objs:
            existing, existing_sim = None, 0.0
            if o["rows"]:
                ours = np.stack([np.asarray(r, dtype=np.float32) for r in o["rows"]])
                for rec in self.client.find_by_label(o["label"]):
                    frs = (rec.vector or {}).get("exemplars") or []
                    if not frs:
                        continue
                    sim = float(np.max(ours @ np.asarray(frs, dtype=np.float32).T))
                    if sim > existing_sim:
                        existing, existing_sim = rec, sim
            if existing is not None and (str(existing.id) == o["id"] or existing_sim >= S_SUGGEST):
                # the SAME physical item already on the fleet (same id, or same
                # name + it actually looks like it): fold views into that point.
                # A same-named but different-looking item stays its own point.
                frows = [
                    np.asarray(r, dtype=np.float32)
                    for r in (existing.vector or {}).get("exemplars", [])
                ]
                fviews = list((existing.payload or {}).get("views") or [])
                rows, views = fold_rows(frows, fviews, o["rows"], o["views"])
                payload = {
                    **(existing.payload or {}),
                    "views": views,
                    "t_sync": now,
                    "label_key": o["label"].strip().lower(),
                }
                self.client.upsert_object(str(existing.id), o["label"], rows, payload)
                items.append(
                    {
                        "old_id": o["id"],
                        "fleet_id": str(existing.id),
                        "label": o["label"],
                        "rows": rows,
                        "views": views,
                        "neg": o["neg"],
                        "thumb": o["thumb"],
                        "t_sync": now,
                        "fingerprint": o.get("fingerprint"),
                    }
                )
            else:
                payload = {
                    "kind": "object",
                    "label": o["label"],
                    "label_key": o["label"].strip().lower(),
                    "device": o["device"],
                    "event": o["event"],
                    "t_created": o["t_created"],
                    "t_sync": now,
                    "views": o["views"],
                    "neg": o["neg"],
                    "thumb": o["thumb"],
                }
                self.client.upsert_object(o["id"], o["label"], o["rows"], payload)
                items.append(
                    {"old_id": o["id"], "t_sync": now, "fingerprint": o.get("fingerprint")}
                )
        self.core.submit(MarkPushed(items=items))
