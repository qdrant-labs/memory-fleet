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
        self._on_event = on_event or (lambda e: None)
        self._jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
        try:
            self.client.ensure_collection()
            self.pull_once()  # seed / catch-up on boot
        except Exception as e:
            self._fleet_error("fleet unreachable at boot", e)
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
                if job[0] == "pull":
                    self.pull_once()
                    next_pull = time.time() + self.interval
                elif job[0] == "push":
                    self.push(job[1])
            except Exception as e:
                self._fleet_error(f"{job[0]} failed", e)

    def _fleet_error(self, msg: str, e: Exception):
        logger.warning("sync: %s: %s", msg, e)
        self._on_event({"type": "fleet_error", "message": msg})

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
            existing = self.client.find_by_label(o["label"])
            if existing is not None and str(existing.id) != o["id"]:
                # name == identity extended to the fleet: fold into the fleet point
                frows = [
                    np.asarray(r, dtype=np.float32)
                    for r in (existing.vector or {}).get("exemplars", [])
                ]
                fviews = list((existing.payload or {}).get("views") or [])
                rows, views = fold_rows(frows, fviews, o["rows"], o["views"])
                payload = {**(existing.payload or {}), "views": views, "t_sync": now}
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
                    }
                )
            else:
                payload = {
                    "kind": "object",
                    "label": o["label"],
                    "device": o["device"],
                    "event": o["event"],
                    "t_created": o["t_created"],
                    "t_sync": now,
                    "views": o["views"],
                    "neg": o["neg"],
                    "thumb": o["thumb"],
                }
                self.client.upsert_object(o["id"], o["label"], o["rows"], payload)
                items.append({"old_id": o["id"], "t_sync": now})
        self.core.submit(MarkPushed(items=items))
