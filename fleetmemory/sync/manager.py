"""Sync lifecycle: every ~30 s tick pulls, then auto-pushes any dirty
confirmed objects in one batch (curation can still push explicitly). Offline
ticks skip both — dirty objects wait and ride the first tick after reconnect.
Downloads happen on this worker; shard mutations happen as queued core
messages. Sync failures degrade to "fleet offline", never crash.
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
    NEG_CAP,
    ApplyPartialSnapshot,
    Core,
    DrainSightings,
    ManifestRequest,
    MarkPushed,
    PreparePush,
    fold_rows,
)
from fleetmemory.memory.matcher import S_FLEET_FOLD
from fleetmemory.sync.client import FleetClient

logger = logging.getLogger(__name__)

PULL_INTERVAL = 30.0
REPLY_TIMEOUT = 10.0


def build_fold(
    existing_payload: dict, o: dict, frows, fviews, now: float, device: str, same_id: bool
):
    """Compose the folded (rows, views, payload) for a push into an existing
    fleet point. same_id folds LOCAL-as-base: the local copy is the successor of
    that fleet point, so its views lead. This preserves an at-cap human-view
    replacement (the fold stops at VIEW_CAP before the fleet's old views get a
    turn). It does NOT preserve a below-cap local prune of a view the fleet still
    holds — that view is diverse by construction, so it refolds through the
    diversity gate; honoring a prune across the fleet would need a view_id
    tombstone (accepted debt). A cross-instance label fold folds fleet-as-base —
    we are adding this local instance's views to a point another unit owns.
    Negatives from both sides are unioned (deduped, newest kept) and capped, so a
    Reject the user made never evaporates on the next fold."""
    if same_id:
        rows, views = fold_rows(o["rows"], o["views"], frows, fviews)
    else:
        rows, views = fold_rows(frows, fviews, o["rows"], o["views"])
    negs: list = []
    for n in list((existing_payload or {}).get("neg") or []) + list(o["neg"] or []):
        if n in negs:  # a hydrated copy re-carries the fleet's negs each push
            negs.remove(n)  # keep it, at its newest position
        negs.append(n)
    payload = {
        **(existing_payload or {}),
        "views": views,
        "neg": negs[-NEG_CAP:],
        "t_sync": now,
        "t_seen": now,  # decay freshness: pushing IS an interaction
        # ...so the room must move too, or recall pairs "just now" with a stale
        # sibling's device and names the wrong room
        "last_seen_device": device,
        # label AND label_key: a renamed local copy carries the new display name
        # into the fold, not just the lookup key
        "label": o["label"],
        "label_key": o["label"].strip().lower(),
    }
    return rows, views, payload


def _fold_survived(written, views) -> bool:
    """False if our folded views are not all present on the fleet point after the
    upsert: a concurrent fold clobbered us, or the point vanished (fleet-sleep
    folded or archived it mid-push, which is documented as racing pushes). Either
    way, don't stamp t_sync — stay dirty and retry next tick."""
    if written is None:
        return False
    have = {v.get("view_id") for v in (written.payload or {}).get("views") or []}
    return {v.get("view_id") for v in views} <= have


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
                    self.touch_sightings()  # decay must spare what units still see
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
        # here: a busy core (first boot) can outlast it and get the temp dir
        # yanked mid-unpack.
        workdir = Path(tempfile.mkdtemp(prefix="fm-pull-"))
        try:
            dest = workdir / "partial.snapshot"
            self.client.download_partial_snapshot(manifest, dest)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        self.core.submit(ApplyPartialSnapshot(path=str(dest), cleanup=True))

    def touch_sightings(self):
        """Stamp t_seen on fleet points recognized since the last tick.
        Recognition never dirties a point, so without this heartbeat the decay
        job would fade memories the fleet still sees every day."""
        reply: queue.Queue = queue.Queue()
        self.core.submit(DrainSightings(reply=reply))
        seen = reply.get(timeout=REPLY_TIMEOUT)
        if seen:
            self.client.touch_seen(seen, time.time(), device=self.core.device_name)

    # ---------- push: prepare (core) -> fleet ops (here) -> mark (core) ----------

    def push(self, object_ids: list | None):
        """object_ids=None is the auto-push sweep: all dirty confirmed objects,
        silent when there is nothing to send (no 'nothing to push' toast)."""
        reply: queue.Queue = queue.Queue()
        self.core.submit(PreparePush(object_ids=object_ids, reply=reply))
        objs = reply.get(timeout=REPLY_TIMEOUT)
        if not objs:  # everything filtered (blocklist/synthetic/missing)
            if object_ids is not None:  # a user-initiated push still gets an ack
                self.core.submit(MarkPushed(items=[]))
            return
        items = []
        now = time.time()
        for o in objs:
            # same-id first — the id IS the instance, regardless of the current
            # display name (a hydrated fleet object renamed locally must still
            # fold into its own fleet point, not overwrite it)
            existing, existing_sim = self.client.get_point(o["id"]), 0.0
            if existing is None and o["rows"]:
                ours = np.stack([np.asarray(r, dtype=np.float32) for r in o["rows"]])
                for rec in self.client.find_by_label(o["label"]):
                    frs = (rec.vector or {}).get("exemplars") or []
                    if not frs:
                        continue
                    sim = float(np.max(ours @ np.asarray(frs, dtype=np.float32).T))
                    if sim > existing_sim:
                        existing, existing_sim = rec, sim
            same_id = existing is not None and str(existing.id) == o["id"]
            if existing is not None and (same_id or existing_sim >= S_FLEET_FOLD):
                # the SAME physical item already on the fleet (same id, or same
                # name + it actually looks like it, at instance confidence): fold
                # views into that point. A same-named but different-looking item
                # stays its own point.
                frows = [
                    np.asarray(r, dtype=np.float32)
                    for r in (existing.vector or {}).get("exemplars", [])
                ]
                fviews = list((existing.payload or {}).get("views") or [])
                rows, views, payload = build_fold(
                    existing.payload or {}, o, frows, fviews, now, self.core.device_name, same_id
                )
                self.client.upsert_object(str(existing.id), o["label"], rows, payload)
                # ponytail: no server-side CAS, so a concurrent fold into this
                # same point could clobber ours. Re-read and confirm our folded
                # views survived BEFORE letting MarkPushed stamp t_sync + delete
                # the local copy; if not, leave it dirty and retry next tick.
                # NARROWS the lost-then-locally-deleted window at 3-device scale
                # (a competitor writing after our re-read still wins); a full fix
                # needs optimistic locking on the fleet point.
                if not _fold_survived(self.client.get_point(str(existing.id)), views):
                    logger.warning(
                        "fold clobbered or point gone mid-push for %s; staying dirty", existing.id
                    )
                    continue
                items.append(
                    {
                        "old_id": o["id"],
                        "fleet_id": str(existing.id),
                        "label": o["label"],
                        "rows": rows,
                        "views": views,
                        "neg": payload["neg"],
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
                    "t_seen": now,  # decay freshness: pushing IS an interaction
                    "last_seen_device": self.core.device_name,  # keep room paired to t_seen
                    "views": o["views"],
                    "neg": o["neg"],
                    "thumb": o["thumb"],
                }
                self.client.upsert_object(o["id"], o["label"], o["rows"], payload)
                items.append(
                    {"old_id": o["id"], "t_sync": now, "fingerprint": o.get("fingerprint")}
                )
        self.core.submit(MarkPushed(items=items))
