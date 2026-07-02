"""The single-threaded memory core (PLAN.md §3.2).

All memory mutations happen here: ingest results, UI verbs, and sync events
enter as queued messages; events come out through a callback. No locks by
construction. Staleness is handled once, at the single place messages are
consumed: every track-scoped message carries the incarnation (tid, epoch) it
saw, and mismatches are dropped.
"""

import base64
import logging
import queue
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .matcher import Decision, Thresholds, decide
from .store import Store, new_id

logger = logging.getLogger(__name__)

VIEW_CAP = 12  # exemplar rows per object, diversity-gated like v1
DIVERSITY_MAX = 0.95  # skip a new view too similar to a stored row
NEG_CAP = 8
BURST_VIEWS = 6  # teach burst: capture until this many views...
BURST_SECONDS = 3.0  # ...or this much time


def fold_rows(krows: list, kviews: list, frows: list, fviews: list) -> tuple[list, list]:
    """Fold f's exemplar rows into k's: human rows first, diversity-gated, capped.
    Used by merge and by the fleet label-fold push (§3.6). Returns (rows, views)."""
    krows, kviews = list(krows), list(kviews)
    order = sorted(range(len(frows)), key=lambda i: not fviews[i].get("human"))
    for i in order:
        if len(krows) >= VIEW_CAP:
            break
        if krows and max(float(np.dot(frows[i], r)) for r in krows) > DIVERSITY_MAX:
            continue
        krows.append(frows[i])
        kviews.append(fviews[i])
    return krows, kviews


# ---------- messages ----------


@dataclass(slots=True)
class Ingest:
    tid: int
    epoch: int
    vec: np.ndarray  # L2-normalized
    thumb_jpeg: bytes | None
    quality: float
    t: float


@dataclass(slots=True)
class TrackDied:
    tid: int
    epoch: int


@dataclass(slots=True)
class Teach:
    tid: int
    epoch: int
    label: str


@dataclass(slots=True)
class Confirm:
    tid: int
    epoch: int
    object_id: str


@dataclass(slots=True)
class Reject:
    tid: int
    epoch: int
    object_id: str


@dataclass(slots=True)
class IgnoreTrack:
    tid: int
    epoch: int


@dataclass(slots=True)
class IgnoreObject:
    object_id: str


@dataclass(slots=True)
class Forget:
    object_id: str


@dataclass(slots=True)
class Merge:
    keep_id: str
    fold_id: str


@dataclass(slots=True)
class Rename:
    object_id: str
    label: str


@dataclass(slots=True)
class Prune:
    object_id: str
    view_id: str


@dataclass(slots=True)
class SetThresholds:
    s_same: float
    s_suggest: float
    s_ignore: float


@dataclass(slots=True)
class ApplyPartialSnapshot:
    path: str
    cleanup: bool = False  # core deletes the snapshot's directory after consuming it


@dataclass(slots=True)
class InventoryRequest:
    pass


@dataclass(slots=True)
class SearchRequest:
    text: str


@dataclass(slots=True)
class MapRequest:
    pass


@dataclass(slots=True)
class ScaleStunt:
    on: bool
    path: str = "edge-data-scale"


@dataclass(slots=True)
class Call:
    """Run fn() on the core thread (the shards' only legal thread) and reply.
    For tools and tests — verbs stay first-class messages."""

    fn: object
    reply: object | None = None


@dataclass(slots=True)
class ManifestRequest:
    """Sync worker asks for the immutable shard's snapshot manifest; the shard
    is only ever touched on the core thread, so this goes through the queue."""

    reply: object  # queue.Queue the worker blocks on


@dataclass(slots=True)
class PreparePush:
    """Sync worker asks for push-ready copies of mutable objects (kind=object,
    non-synthetic). Reply: list of dicts with rows/views/payload fields."""

    object_ids: list
    reply: object


@dataclass(slots=True)
class MarkPushed:
    """Sync worker reports a completed fleet push. Items either stamp t_sync in
    place ({old_id, t_sync}) or — label-fold (§3.6) — rewrite the local point
    under the fleet point's id ({old_id, fleet_id, label, rows, views, neg,
    thumb, t_sync}) so the §3.3 id-present dedup applies verbatim on next pull."""

    items: list


# ---------- track state ----------


@dataclass(slots=True)
class TrackState:
    epoch: int
    state: str = "pending"  # pending|unknown|suggest|recognized|ignored|capturing
    object_id: str | None = None
    label: str = ""
    score: float = 0.0
    last_vec: np.ndarray | None = None
    last_thumb: bytes | None = None
    last_quality: float = 0.0
    last_seen: float = 0.0
    vetoed: set[str] = field(default_factory=set)
    burst_until: float = 0.0
    burst_want: int = 0
    burst_have: int = 0


class Core:
    """Consume commands, mutate the store, emit events."""

    def __init__(self, store: Store, device_name: str = "", event_tag: str = "", on_event=None):
        self.store = store
        self.device_name = device_name
        self.event_tag = event_tag
        self.thresholds = Thresholds()
        self.tracks: dict[int, TrackState] = {}
        # session-scoped negatives for fleet (immutable) objects — not persisted (§3.5)
        self.session_negs: dict[str, list] = {}
        self._on_event = on_event or (lambda e: None)
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.thumbs_dir = store.data_dir / "thumbs"
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    # ---------- plumbing ----------

    def submit(self, msg):
        self._queue.put(msg)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="core", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def drain(self):
        """Synchronously process everything queued — the deterministic gate's clock."""
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                return
            if msg is not None:
                self._process(msg)

    def _run(self):
        while not self._stop.is_set():
            msg = self._queue.get()
            if msg is None:
                continue
            try:
                self._process(msg)
            except Exception:
                logger.exception("core: %s failed", type(msg).__name__)
                self._emit({"type": "error", "message": f"{type(msg).__name__} failed"})

    def _emit(self, event: dict):
        self._on_event(event)

    # ---------- the single consumption point (incarnation guard lives here) ----------

    def _process(self, msg):
        tid = getattr(msg, "tid", None)
        if tid is not None:
            ts = self.tracks.get(tid)
            if isinstance(msg, Ingest):
                if ts is None or msg.epoch > ts.epoch:
                    ts = TrackState(epoch=msg.epoch)
                    self.tracks[tid] = ts
                elif msg.epoch < ts.epoch:
                    return  # stale worker result from a dead incarnation
            else:
                if ts is None or ts.epoch != msg.epoch:
                    return  # verb or death notice aimed at a track that re-bound or died
        handler = getattr(self, f"_on_{type(msg).__name__.lower()}")
        handler(msg)

    # ---------- ingest ----------

    def _on_ingest(self, m: Ingest):
        ts = self.tracks[m.tid]
        ts.last_vec = m.vec
        ts.last_seen = m.t
        if m.thumb_jpeg and m.quality >= ts.last_quality:
            ts.last_thumb, ts.last_quality = m.thumb_jpeg, m.quality

        if ts.state == "capturing":
            self._burst_step(m.tid, ts, m)
            return

        res = self.store.recognize(m.vec)
        self._emit({"type": "query", "tid": m.tid, "ms": res.latency_ms, "searched": res.searched})
        d = decide(m.vec, res.candidates, self.thresholds, ts.vetoed, self.session_negs)
        self._apply_decision(m.tid, ts, d, m)

    def _apply_decision(self, tid: int, ts: TrackState, d: Decision, m: Ingest):
        state = {
            "recognized": "recognized",
            "suggest": "suggest",
            "ignored": "ignored",
            "unknown": "unknown",
        }[d.outcome]
        obj = d.candidate.id if d.candidate and state in ("recognized", "suggest") else None
        label = d.candidate.label if obj else ""
        changed = (state, obj) != (ts.state, ts.object_id)
        ts.state, ts.object_id, ts.label, ts.score = state, obj, label, d.score

        if state == "recognized" and d.candidate.from_mutable:
            self._maybe_accrete(d.candidate.id, m, human=False)
        if changed or state == "recognized":
            self._track_event(tid, ts)

    def _track_event(self, tid: int, ts: TrackState):
        self._emit(
            {
                "type": "track_update",
                "tid": tid,
                "epoch": ts.epoch,
                "state": ts.state,
                "object_id": ts.object_id,
                "label": ts.label,
                "score": round(ts.score, 3),
            }
        )

    # ---------- view accretion ----------

    def _maybe_accrete(self, object_id: str, m: Ingest, human: bool) -> bool:
        got = self.store.get_object(object_id)
        if got is None:
            return False
        payload, rows = got
        if len(rows) >= VIEW_CAP:
            return False
        if rows and max(float(m.vec @ r) for r in rows) > DIVERSITY_MAX:
            return False
        view_id = uuid.uuid4().hex[:12]
        views = list(payload.get("views") or [])
        rows.append(m.vec)
        views.append({"view_id": view_id, "human": human})
        self._save_view_thumb(view_id, m.thumb_jpeg)
        self.store.upsert_object(
            object_id,
            payload.get("label", ""),
            rows,
            views,
            neg=payload.get("neg"),
            device=payload.get("device", ""),
            event=payload.get("event", ""),
            t_created=payload.get("t_created"),
            t_sync=payload.get("t_sync"),
            thumb=payload.get("thumb", ""),
        )
        self._emit({"type": "object_updated", "object_id": object_id, "views": len(views)})
        return True

    def _save_view_thumb(self, view_id: str, thumb: bytes | None):
        if thumb:
            (self.thumbs_dir / f"{view_id}.jpg").write_bytes(thumb)

    # ---------- teach + burst ----------

    def _on_teach(self, m: Teach):
        ts = self.tracks[m.tid]
        label = m.label.strip()
        if not label or ts.last_vec is None:
            self._emit({"type": "error", "message": "teach needs a label and a seen track"})
            return

        existing = self.store.find_label(label)
        now = ts.last_seen
        if existing:
            # name == identity: fold into the object already carrying this label
            object_id = existing
            fake = Ingest(m.tid, m.epoch, ts.last_vec, ts.last_thumb, ts.last_quality, now)
            self._maybe_accrete(object_id, fake, human=True)
            self._emit({"type": "object_updated", "object_id": object_id, "folded": True})
        else:
            object_id = new_id()
            view_id = uuid.uuid4().hex[:12]
            self._save_view_thumb(view_id, ts.last_thumb)
            thumb_b64 = base64.b64encode(ts.last_thumb).decode() if ts.last_thumb else ""
            self.store.upsert_object(
                object_id,
                label,
                [ts.last_vec],
                [{"view_id": view_id, "human": True}],
                device=self.device_name,
                event=self.event_tag,
                thumb=thumb_b64,
            )
            self._emit({"type": "object_created", "object_id": object_id, "label": label})

        ts.state, ts.object_id, ts.label = "capturing", object_id, label
        ts.vetoed.discard(object_id)
        ts.burst_until = now + BURST_SECONDS
        ts.burst_want, ts.burst_have = BURST_VIEWS, 0
        self._track_event(m.tid, ts)
        self._emit_stats()

    def _burst_step(self, tid: int, ts: TrackState, m: Ingest):
        if self._maybe_accrete(ts.object_id, m, human=True):
            ts.burst_have += 1
        self._emit(
            {"type": "burst_progress", "tid": tid, "have": ts.burst_have, "want": ts.burst_want}
        )
        if ts.burst_have >= ts.burst_want or m.t >= ts.burst_until:
            ts.state, ts.score = "recognized", 1.0
            self._track_event(tid, ts)

    # ---------- the other verbs ----------

    def _on_confirm(self, m: Confirm):
        ts = self.tracks[m.tid]
        if ts.last_vec is not None:
            fake = Ingest(m.tid, m.epoch, ts.last_vec, ts.last_thumb, ts.last_quality, ts.last_seen)
            self._maybe_accrete(m.object_id, fake, human=True)  # no-op on fleet objects
        got = self.store.get_object(m.object_id, with_vectors=False)
        ts.state, ts.object_id = "recognized", m.object_id
        ts.label = got[0].get("label", "") if got else ts.label
        ts.score = 1.0
        ts.vetoed.discard(m.object_id)
        self._track_event(m.tid, ts)

    def _on_reject(self, m: Reject):
        ts = self.tracks[m.tid]
        if ts.last_vec is None:
            return
        neg = ts.last_vec.tolist()
        got = self.store.get_object(m.object_id)
        if got is not None:  # mutable object: persist the negative
            payload, rows = got
            negs = (list(payload.get("neg") or []) + [neg])[-NEG_CAP:]
            self.store.upsert_object(
                m.object_id,
                payload.get("label", ""),
                rows,
                list(payload.get("views") or []),
                neg=negs,
                device=payload.get("device", ""),
                event=payload.get("event", ""),
                t_created=payload.get("t_created"),
                t_sync=payload.get("t_sync"),
                thumb=payload.get("thumb", ""),
            )
        else:  # fleet object: session-scoped only (§3.5 — re-teach beats persistence machinery)
            self.session_negs.setdefault(m.object_id, []).append(neg)
            self.session_negs[m.object_id] = self.session_negs[m.object_id][-NEG_CAP:]
        ts.vetoed.add(m.object_id)
        ts.state, ts.object_id, ts.label, ts.score = "unknown", None, "", 0.0
        self._track_event(m.tid, ts)

    def _on_ignoretrack(self, m: IgnoreTrack):
        ts = self.tracks[m.tid]
        if ts.last_vec is None:
            return
        view_id = uuid.uuid4().hex[:12]
        self.store.upsert_object(
            new_id(),
            "",
            [ts.last_vec],
            [{"view_id": view_id, "human": True}],
            kind="ignored",
            device=self.device_name,
        )
        ts.state, ts.object_id, ts.label, ts.score = "ignored", None, "", 1.0
        self._track_event(m.tid, ts)
        self._emit_stats()

    def _on_ignoreobject(self, m: IgnoreObject):
        got = self.store.get_object(m.object_id)
        if got is None:
            return
        payload, rows = got
        self.store.upsert_object(
            new_id(),
            "",
            rows,
            list(payload.get("views") or []),
            kind="ignored",
            device=self.device_name,
        )
        self._forget(m.object_id, payload)

    def _on_forget(self, m: Forget):
        got = self.store.get_object(m.object_id, with_vectors=False)
        if got is not None:
            self._forget(m.object_id, got[0])

    def _forget(self, object_id: str, payload: dict):
        for v in payload.get("views") or []:
            (self.thumbs_dir / f"{v['view_id']}.jpg").unlink(missing_ok=True)
        self.store.delete(object_id)
        for tid, ts in self.tracks.items():
            if ts.object_id == object_id:
                ts.state, ts.object_id, ts.label, ts.score = "unknown", None, "", 0.0
                self._track_event(tid, ts)
        self._emit({"type": "object_deleted", "object_id": object_id})
        self._emit_stats()

    def _on_merge(self, m: Merge):
        keep = self.store.get_object(m.keep_id)
        fold = self.store.get_object(m.fold_id)
        if keep is None or fold is None or m.keep_id == m.fold_id:
            return
        kp, krows = keep
        fp, frows = fold
        views = list(kp.get("views") or [])
        krows, views = fold_rows(krows, views, frows, list(fp.get("views") or []))
        negs = (list(kp.get("neg") or []) + list(fp.get("neg") or []))[-NEG_CAP:]
        self.store.upsert_object(
            m.keep_id,
            kp.get("label", ""),
            krows,
            views,
            neg=negs,
            device=kp.get("device", ""),
            event=kp.get("event", ""),
            t_created=kp.get("t_created"),
            t_sync=kp.get("t_sync"),
            thumb=kp.get("thumb", "") or fp.get("thumb", ""),
        )
        self.store.delete(m.fold_id)
        for tid, ts in self.tracks.items():
            if ts.object_id == m.fold_id:
                ts.object_id, ts.label = m.keep_id, kp.get("label", "")
                self._track_event(tid, ts)
        self._emit({"type": "objects_merged", "kept": m.keep_id, "folded": m.fold_id})
        self._emit_stats()

    def _on_rename(self, m: Rename):
        label = m.label.strip()
        got = self.store.get_object(m.object_id)
        if not label or got is None:
            return
        other = self.store.find_label(label)
        if other and other != m.object_id:
            self._emit(
                {
                    "type": "rename_conflict",
                    "object_id": m.object_id,
                    "label": label,
                    "existing_id": other,
                }
            )
            return
        payload, rows = got
        self.store.upsert_object(
            m.object_id,
            label,
            rows,
            list(payload.get("views") or []),
            neg=payload.get("neg"),
            device=payload.get("device", ""),
            event=payload.get("event", ""),
            t_created=payload.get("t_created"),
            t_sync=payload.get("t_sync"),
            thumb=payload.get("thumb", ""),
        )
        for tid, ts in self.tracks.items():
            if ts.object_id == m.object_id:
                ts.label = label
                self._track_event(tid, ts)
        self._emit({"type": "object_updated", "object_id": m.object_id, "label": label})

    def _on_prune(self, m: Prune):
        got = self.store.get_object(m.object_id)
        if got is None:
            return
        payload, rows = got
        views = list(payload.get("views") or [])
        idx = next((i for i, v in enumerate(views) if v.get("view_id") == m.view_id), None)
        if idx is None:
            return
        rows.pop(idx)
        views.pop(idx)
        (self.thumbs_dir / f"{m.view_id}.jpg").unlink(missing_ok=True)
        if not rows:
            self._forget(m.object_id, {**payload, "views": []})
            return
        self.store.upsert_object(
            m.object_id,
            payload.get("label", ""),
            rows,
            views,
            neg=payload.get("neg"),
            device=payload.get("device", ""),
            event=payload.get("event", ""),
            t_created=payload.get("t_created"),
            t_sync=payload.get("t_sync"),
            thumb=payload.get("thumb", ""),
        )
        self._emit({"type": "object_updated", "object_id": m.object_id, "views": len(views)})

    def _on_trackdied(self, m: TrackDied):
        del self.tracks[m.tid]

    def _on_setthresholds(self, m: SetThresholds):
        self.thresholds = Thresholds(m.s_same, m.s_suggest, m.s_ignore)
        self._emit(
            {
                "type": "thresholds",
                "s_same": m.s_same,
                "s_suggest": m.s_suggest,
                "s_ignore": m.s_ignore,
            }
        )

    def _on_applypartialsnapshot(self, m: ApplyPartialSnapshot):
        try:
            self.store.apply_partial_snapshot(m.path)
            removed = self.store.dedup_after_pull()
        except Exception:
            # the mirror is a disposable replica of the fleet: rebuild empty and
            # let the next pull re-seed it, rather than staying wedged
            logger.exception("pull apply failed; rebuilding the fleet mirror")
            self.store.reset_immutable()
            self._emit(
                {"type": "fleet_error", "message": "pull failed — mirror rebuilt, re-syncing"}
            )
            return
        finally:
            if m.cleanup:
                shutil.rmtree(Path(m.path).parent, ignore_errors=True)
        self._emit({"type": "pull_applied", "deduped": len(removed)})
        self._emit_stats()

    # ---------- sync worker handshakes (worker blocks on reply queues) ----------

    def _on_searchrequest(self, m: SearchRequest):
        hits = self.store.search_text(m.text, limit=10) if m.text.strip() else []
        self._emit(
            {
                "type": "search_results",
                "text": m.text,
                "hits": [
                    {"object_id": c.id, "label": c.label, "score": round(c.score, 3)}
                    for c in hits
                    if c.kind == "object" and not c.payload.get("synthetic")
                ],
            }
        )

    def _on_maprequest(self, m: MapRequest):
        objs = self.store.object_mean_vectors()
        points = []
        if len(objs) >= 2:
            vecs = np.stack([v for _, _, _, v in objs])
            vecs = vecs - vecs.mean(axis=0)
            # PCA via SVD — decorative 2D layout, never load-bearing
            _, _, vt = np.linalg.svd(vecs, full_matrices=False)
            xy = vecs @ vt[:2].T
            span = np.abs(xy).max(axis=0)
            span[span == 0] = 1.0
            xy = xy / span * 0.5 + 0.5
            points = [
                {
                    "object_id": oid,
                    "label": label,
                    "local": from_mut,
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                }
                for (oid, label, from_mut, _), (x, y) in zip(objs, xy, strict=True)
            ]
        elif objs:
            oid, label, from_mut, _ = objs[0]
            points = [{"object_id": oid, "label": label, "local": from_mut, "x": 0.5, "y": 0.5}]
        self._emit({"type": "map", "points": points})

    def _on_scalestunt(self, m: ScaleStunt):
        if m.on:
            ok = self.store.attach_scale_shard(m.path)
            if not ok:
                self._emit(
                    {
                        "type": "error",
                        "message": "no prebuilt scale shard — run `make demo-scale` first",
                    }
                )
                return
        else:
            self.store.detach_scale_shard()
        self._emit({"type": "scale", "on": m.on})
        self._emit_stats()

    def _on_call(self, m: Call):
        result = m.fn()
        if m.reply is not None:
            m.reply.put(result)

    def _on_manifestrequest(self, m: ManifestRequest):
        manifest = None
        if self.store.immutable is not None:
            manifest = self.store.immutable.snapshot_manifest()
        m.reply.put(manifest)

    def _on_preparepush(self, m: PreparePush):
        out = []
        for oid in m.object_ids:
            got = self.store.get_object(oid)
            if got is None:
                continue
            payload, rows = got
            if payload.get("kind") != "object" or payload.get("synthetic"):
                continue  # blocklist entries and stunt synthetics never reach the fleet
            out.append(
                {
                    "id": oid,
                    "label": payload.get("label", ""),
                    "rows": rows,
                    "views": list(payload.get("views") or []),
                    "neg": list(payload.get("neg") or []),
                    "thumb": payload.get("thumb", ""),
                    "device": payload.get("device", ""),
                    "event": payload.get("event", ""),
                    "t_created": payload.get("t_created"),
                }
            )
        m.reply.put(out)

    def _on_markpushed(self, m: MarkPushed):
        for it in m.items:
            old_id = it["old_id"]
            if "fleet_id" in it:  # label-fold: local point rewritten under the fleet id
                self.store.delete(old_id)
                self.store.upsert_object(
                    it["fleet_id"],
                    it["label"],
                    it["rows"],
                    it["views"],
                    neg=it.get("neg"),
                    device=self.device_name,
                    event=self.event_tag,
                    thumb=it.get("thumb", ""),
                    t_sync=it["t_sync"],
                )
                for ts in self.tracks.values():
                    if ts.object_id == old_id:
                        ts.object_id = it["fleet_id"]
            else:
                self.store.set_payload(old_id, {"t_sync": it["t_sync"]})
        self._emit({"type": "push_done", "count": len(m.items)})
        self._emit_stats()

    def _on_inventoryrequest(self, m: InventoryRequest):
        items = []
        for pid, pl, from_mut in self.store.scroll_objects():
            items.append(
                {
                    "object_id": pid,
                    "label": pl.get("label", ""),
                    "views": pl.get("views") or [],
                    "thumb": pl.get("thumb", ""),
                    "device": pl.get("device", ""),
                    "local": from_mut,
                    "pushed": bool(pl.get("t_sync")),
                }
            )
        self._emit({"type": "inventory", "items": items})

    def _emit_stats(self):
        self._emit(
            {
                "type": "stats",
                "memories": self.store.count(),
                "fleet": self.store.immutable is not None,
            }
        )
