"""The single-threaded memory core.

All memory mutations happen here: ingest results, UI verbs, and sync events
enter as queued messages; events come out through a callback. No locks by
construction. Staleness is handled once, at the single place messages are
consumed: every track-scoped message carries the incarnation (tid, epoch) it
saw, and mismatches are dropped.
"""

import base64
import contextlib
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

# Exemplar rows per object, diversity-gated. 24 views gives one instance enough
# room to cover varied angles and lighting; MAX_SIM cost stays negligible at
# this scale. When full, a new HUMAN view replaces the most redundant auto view,
# so confirms keep teaching.
VIEW_CAP = 24
DIVERSITY_MAX = 0.95  # skip a new view too similar to a stored row
NEG_CAP = 8
BURST_VIEWS = 6  # teach burst: capture until this many views...
BURST_SECONDS = 3.0  # ...or this much time
ARCHIVE_CAP = 12  # departed unknowns kept teachable (wrist goes down, watch stays)
IGNORE_FOLD = 0.55  # ignoring something this close to a blocklist entry grows that entry


def _push_fingerprint(payload: dict) -> tuple:
    """Identity of a point's pushable content: label + exact view rows + negatives."""
    return (
        payload.get("label", ""),
        tuple(v.get("view_id", "") for v in payload.get("views") or []),
        len(payload.get("neg") or []),
    )


def fold_rows(krows: list, kviews: list, frows: list, fviews: list) -> tuple[list, list]:
    """Fold f's exemplar rows into k's: human rows first, diversity-gated, capped.
    Used by merge and by the fleet label-fold push. Returns (rows, views)."""
    krows, kviews = list(krows), list(kviews)
    n = min(len(frows), len(fviews))  # defensive: never index past shorter metadata
    order = sorted(range(n), key=lambda i: not fviews[i].get("human"))
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
    label: str = ""  # set when the click came from a labeled blocklist pill


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
class DismissUnknown:
    """Drop an archived (departed) unknown from the recent list."""

    tid: int
    epoch: int


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
class ManifestRequest:
    """Sync worker asks for the immutable shard's snapshot manifest; the shard
    is only ever touched on the core thread, so this goes through the queue."""

    reply: object  # queue.Queue the worker blocks on


@dataclass(slots=True)
class PreparePush:
    """Sync worker asks for push-ready copies of mutable objects (kind=object,
    non-synthetic). object_ids=None means every dirty confirmed object —
    the auto-push sweep. Reply: list of dicts with rows/views/payload fields."""

    object_ids: list | None
    reply: object


@dataclass(slots=True)
class DrainSightings:
    """Sync worker asks which objects were recognized since the last drain, to
    stamp t_seen on their fleet points — decay must not fade memories the
    fleet still sees, and recognition alone never dirties a point."""

    reply: object


@dataclass(slots=True)
class MarkPushed:
    """Sync worker reports a completed fleet push. Items either stamp t_sync in
    place ({old_id, t_sync}) or — label-fold — rewrite the local point under the
    fleet point's id ({old_id, fleet_id, label, rows, views, neg, thumb, t_sync})
    so the id-present dedup applies verbatim on next pull."""

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
    guesses: list = field(default_factory=list)  # nearest memories while unknown
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
        # departed-but-unnamed tracks, still teachable from the unknowns drawer
        self.recent_unknowns: dict[tuple[int, int], TrackState] = {}
        # session-scoped negatives for fleet (immutable) objects — not persisted
        self.session_negs: dict[str, list] = {}
        # sighting stats: object_id -> (count, last_seen). Persisted on mutable
        # objects at bind time; session-only for fleet-mirror objects.
        self.sightings: dict[str, tuple[int, float]] = {}
        # objects recognized since the last DrainSightings — the sync worker
        # stamps t_seen on their fleet points so decay spares them
        self._fleet_seen: set[str] = set()
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
                    # a verb aimed at a departed track may still hit its archived
                    # incarnation — the item left frame, but stays teachable
                    key = (tid, msg.epoch)
                    if isinstance(msg, Teach) and key in self.recent_unknowns:
                        self._teach_archived(msg, self.recent_unknowns.pop(key))
                        return
                    if isinstance(msg, IgnoreTrack) and key in self.recent_unknowns:
                        self._ignore_archived(msg, self.recent_unknowns.pop(key))
                        return
                    if isinstance(msg, DismissUnknown) and key in self.recent_unknowns:
                        del self.recent_unknowns[key]
                        self._emit({"type": "unknown_removed", "tid": tid, "epoch": msg.epoch})
                        return
                    return  # otherwise: stale, drop
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
        # nearest memories as picture pills: at suggest tier you pick WHICH
        # instance it is; at unknown tier they're one-click teach options.
        # Ignored entries ride along (flagged) — "it's that thing I ignored"
        # is a one-click answer too.
        ts.guesses = []
        for c in res.candidates:
            if c.payload.get("synthetic") or c.score < 0.3:
                continue
            ignored = c.kind == "ignored"
            if not ignored and not c.label:
                continue
            ts.guesses.append(
                {
                    "object_id": c.id,
                    "label": c.label,  # may be empty for unnamed blocklist entries
                    "score": round(c.score, 2),
                    "thumb": c.payload.get("thumb", ""),
                    "ignored": ignored,
                }
            )
            if len(ts.guesses) == 3:
                break
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

        if state == "recognized" and changed:
            count, _ = self.sightings.get(obj, (0, 0.0))
            self.sightings[obj] = (count + 1, m.t)
            self._fleet_seen.add(obj)
            if d.candidate.from_mutable:
                self.store.set_payload(obj, {"sightings": count + 1, "t_seen": m.t})
        if state == "recognized" and d.candidate.from_mutable:
            self._maybe_accrete(d.candidate.id, m, human=False)
        if changed or state in ("recognized", "suggest"):
            self._track_event(tid, ts)  # suggest re-emits so its pills stay fresh

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
                # suppressed tracks keep their pills too: a faint box is one
                # click from "it's that ignored thing" or a rescue teach
                "guesses": ts.guesses if ts.state in ("unknown", "suggest", "ignored") else [],
            }
        )

    # ---------- view accretion ----------

    def _maybe_accrete(self, object_id: str, m: Ingest, human: bool) -> bool:
        got = self.store.get_object(object_id)
        if got is None and human:
            # fleet object: copy-on-write. Accrete onto the mirror's content;
            # the upsert below writes the result into the MUTABLE shard under
            # the same id, dirty (t_sync=None) — it shadows the mirror
            # (mutable wins ties), survives pull dedup, and same-id-folds back
            # into the fleet point on the next push. The mirror is never
            # written. human=False stays a no-op: auto-captured views must not
            # dirty every fleet object a unit merely sees.
            got = self.store.get_object_from_mirror(object_id)
        if got is None:
            return False
        payload, rows = got
        if rows and max(float(m.vec @ r) for r in rows) > DIVERSITY_MAX:
            return False
        view_id = uuid.uuid4().hex[:12]
        views = list(payload.get("views") or [])
        if len(rows) >= VIEW_CAP:
            # full: a human view may replace the most redundant auto view —
            # otherwise "yes, same" would stop improving coverage forever
            auto = [i for i, v in enumerate(views) if not v.get("human")]
            if not human or not auto:
                return False
            victim = max(
                auto,
                key=lambda i: max(float(rows[i] @ rows[j]) for j in range(len(rows)) if j != i),
            )
            (self.thumbs_dir / f"{views[victim]['view_id']}.jpg").unlink(missing_ok=True)
            rows[victim] = m.vec
            views[victim] = {"view_id": view_id, "human": True}
        else:
            rows.append(m.vec)
            views.append({"view_id": view_id, "human": human})
        self._save_view_thumb(view_id, m.thumb_jpeg)
        self.store.upsert_object(
            object_id,
            payload.get("label", ""),
            rows,
            views,
            kind=payload.get("kind", "object"),  # blocklist entries accrete too
            neg=payload.get("neg"),
            device=payload.get("device", ""),
            event=payload.get("event", ""),
            t_created=payload.get("t_created"),
            t_sync=None,  # local edit: content no longer matches the fleet copy
            thumb=payload.get("thumb", ""),
            base_payload=payload,
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

        existing = self._fold_target(label, ts.last_vec) or self._rescue_ignored(label, ts.last_vec)
        now = ts.last_seen
        if existing:
            # same item re-taught (same name + looks like it), or a rescued
            # blocklist entry that now carries this label
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

    def _fold_target(self, label: str, vec: np.ndarray) -> str | None:
        """Instance identity: an object is one physical thing; the label is its
        display name and may repeat. Teaching folds
        into a same-name object ONLY when the view plausibly IS that object
        (>= s_suggest to its views) — otherwise it's a new instance. Five
        different watches = five clean points, all called "watch". Fleet
        instances are fold targets too — accretion hydrates them copy-on-write
        instead of spawning a local sibling of the same physical thing."""
        best, best_sim = None, 0.0
        for pid, _pl in self.store.find_label_points(label, mutable_only=False):
            rows = self.store.get_rows(pid)
            if not rows:
                continue
            sim = max(float(vec @ r) for r in rows)
            if sim > best_sim:
                best, best_sim = pid, sim
        return best if best_sim >= self.thresholds.s_suggest else None

    def _rescue_ignored(self, label: str, vec: np.ndarray) -> str | None:
        """Naming a view that matches a blocklist entry converts that entry into
        a labeled object, so one physical thing stays one point. Without this,
        the teach would spawn a sibling the blocklist keeps suppressing."""
        res = self.store.recognize(vec)
        best = next((c for c in res.candidates if c.kind == "ignored" and c.from_mutable), None)
        if best is None or best.score < IGNORE_FOLD:
            return None
        got = self.store.get_object(best.id)
        if got is None:
            return None
        payload, rows = got
        self.store.upsert_object(
            best.id,
            label,
            rows,
            list(payload.get("views") or []),
            kind="object",
            neg=payload.get("neg"),
            device=payload.get("device", "") or self.device_name,
            event=payload.get("event", "") or self.event_tag,
            t_created=payload.get("t_created"),
            t_sync=None,
            thumb=payload.get("thumb", ""),
            base_payload=payload,
        )
        self._emit({"type": "object_created", "object_id": best.id, "label": label})
        return best.id

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
            self._maybe_accrete(m.object_id, fake, human=True)  # hydrates fleet objects
        got = self.store.get_object(m.object_id, with_vectors=False)
        ts.state, ts.object_id = "recognized", m.object_id
        ts.label = got[0].get("label", "") if got else ts.label
        ts.score = 1.0
        ts.vetoed.discard(m.object_id)
        self._fleet_seen.add(m.object_id)
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
                t_sync=None,  # local edit: content no longer matches the fleet copy
                thumb=payload.get("thumb", ""),
                base_payload=payload,
            )
        else:  # fleet object: session-scoped only (re-teach beats persistence machinery)
            self.session_negs.setdefault(m.object_id, []).append(neg)
            self.session_negs[m.object_id] = self.session_negs[m.object_id][-NEG_CAP:]
        ts.vetoed.add(m.object_id)
        ts.state, ts.object_id, ts.label, ts.score = "unknown", None, "", 0.0
        self._track_event(m.tid, ts)

    def _on_ignoretrack(self, m: IgnoreTrack):
        ts = self.tracks[m.tid]
        if ts.last_vec is None:
            return
        self._blocklist_view(m.tid, m.epoch, ts, m.label)
        ts.state, ts.object_id, ts.label, ts.score = "ignored", None, "", 1.0
        self._track_event(m.tid, ts)
        self._emit_stats()

    def _blocklist_view(self, tid: int, epoch: int, ts: TrackState, label: str = ""):
        # fold into the nearest existing blocklist entry when it's plausibly the
        # same thing: hair and other shape-shifters need MANY views before the
        # strict suppress threshold covers them — each ignore strengthens ONE
        # entry instead of littering the blocklist with near-duplicates
        res = self.store.recognize(ts.last_vec)
        nearest = next((c for c in res.candidates if c.kind == "ignored"), None)
        if nearest is not None and nearest.from_mutable and nearest.score >= IGNORE_FOLD:
            fake = Ingest(tid, epoch, ts.last_vec, ts.last_thumb, ts.last_quality, ts.last_seen)
            self._maybe_accrete(nearest.id, fake, human=True)
        else:
            # a new look of an ignored thing below IGNORE_FOLD is a new
            # blocklist INSTANCE — it keeps the label the human pointed at
            view_id = uuid.uuid4().hex[:12]
            self._save_view_thumb(view_id, ts.last_thumb)
            self.store.upsert_object(
                new_id(),
                label,
                [ts.last_vec],
                [{"view_id": view_id, "human": True}],
                kind="ignored",
                device=self.device_name,
                thumb=base64.b64encode(ts.last_thumb).decode() if ts.last_thumb else "",
            )

    def _on_ignoreobject(self, m: IgnoreObject):
        got = self.store.get_object(m.object_id)
        if got is None:
            return
        payload, rows = got
        self.store.upsert_object(
            new_id(),
            payload.get("label", ""),  # keep the name — "door", not "(unnamed)"
            rows,
            list(payload.get("views") or []),
            kind="ignored",
            device=self.device_name,
            thumb=payload.get("thumb", ""),
        )
        # delete the object point but keep the view thumbs — they moved to the
        # blocklist entry (same view_ids), where curation can still inspect them
        self.store.delete(m.object_id)
        for tid, ts in self.tracks.items():
            if ts.object_id == m.object_id:
                ts.state, ts.object_id, ts.label, ts.score = "unknown", None, "", 0.0
                self._track_event(tid, ts)
        self._emit({"type": "object_deleted", "object_id": m.object_id})
        self._emit_stats()

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
        if kp.get("kind", "object") != fp.get("kind", "object"):
            self._emit({"type": "error", "message": "can't merge an object with an ignored look"})
            return
        views = list(kp.get("views") or [])
        krows, views = fold_rows(krows, views, frows, list(fp.get("views") or []))
        negs = (list(kp.get("neg") or []) + list(fp.get("neg") or []))[-NEG_CAP:]
        self.store.upsert_object(
            m.keep_id,
            kp.get("label", ""),
            krows,
            views,
            kind=kp.get("kind", "object"),  # merging ignored entries keeps them ignored
            neg=negs,
            device=kp.get("device", ""),
            event=kp.get("event", ""),
            t_created=kp.get("t_created"),
            t_sync=None,  # merged content no longer matches the fleet copy
            thumb=kp.get("thumb", "") or fp.get("thumb", ""),
            base_payload=kp,
        )
        self.store.delete(m.fold_id)
        for tid, ts in self.tracks.items():
            if ts.object_id == m.fold_id:
                ts.object_id, ts.label = m.keep_id, kp.get("label", "")
                self._track_event(tid, ts)
        self._emit({"type": "objects_merged", "kept": m.keep_id, "folded": m.fold_id})
        self._emit_stats()

    def _on_rename(self, m: Rename):
        # labels are display names and may repeat (instance model) — no conflict
        label = m.label.strip()
        got = self.store.get_object(m.object_id)
        if not label or got is None:
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
            t_sync=None,  # local edit: content no longer matches the fleet copy
            thumb=payload.get("thumb", ""),
            base_payload=payload,
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
            t_sync=None,  # local edit: content no longer matches the fleet copy
            thumb=payload.get("thumb", ""),
            base_payload=payload,
        )
        self._emit({"type": "object_updated", "object_id": m.object_id, "views": len(views)})

    def _on_trackdied(self, m: TrackDied):
        ts = self.tracks.pop(m.tid)
        if ts.state in ("unknown", "suggest") and ts.last_vec is not None:
            self.recent_unknowns[(m.tid, ts.epoch)] = ts
            while len(self.recent_unknowns) > ARCHIVE_CAP:
                old_key = next(iter(self.recent_unknowns))
                del self.recent_unknowns[old_key]
                self._emit({"type": "unknown_removed", "tid": old_key[0], "epoch": old_key[1]})
            self._emit(
                {
                    "type": "unknown_archived",
                    "tid": m.tid,
                    "epoch": ts.epoch,
                    "thumb": base64.b64encode(ts.last_thumb).decode() if ts.last_thumb else "",
                    "t": ts.last_seen,
                    "guesses": ts.guesses,
                }
            )

    def _on_dismissunknown(self, m: DismissUnknown):
        pass  # live-track dismiss is a no-op; archived dismiss is handled in _process

    def _teach_archived(self, m: Teach, ts: TrackState):
        """Teach a departed track from its remembered view. No capture burst —
        the item isn't in frame; re-showing it accretes views the normal way."""
        label = m.label.strip()
        if not label:
            return
        existing = self._fold_target(label, ts.last_vec) or self._rescue_ignored(label, ts.last_vec)
        if existing:
            fake = Ingest(m.tid, m.epoch, ts.last_vec, ts.last_thumb, ts.last_quality, ts.last_seen)
            self._maybe_accrete(existing, fake, human=True)
            self._emit({"type": "object_updated", "object_id": existing, "folded": True})
        else:
            object_id = new_id()
            view_id = uuid.uuid4().hex[:12]
            self._save_view_thumb(view_id, ts.last_thumb)
            self.store.upsert_object(
                object_id,
                label,
                [ts.last_vec],
                [{"view_id": view_id, "human": True}],
                device=self.device_name,
                event=self.event_tag,
                thumb=base64.b64encode(ts.last_thumb).decode() if ts.last_thumb else "",
            )
            self._emit({"type": "object_created", "object_id": object_id, "label": label})
        self._emit({"type": "unknown_removed", "tid": m.tid, "epoch": m.epoch})
        self._emit_stats()

    def _ignore_archived(self, m: IgnoreTrack, ts: TrackState):
        """Ignore a departed track from its remembered view — the red pill on an
        archived card means the same thing it means on a live box."""
        self._blocklist_view(m.tid, m.epoch, ts, m.label)
        self._emit({"type": "unknown_removed", "tid": m.tid, "epoch": m.epoch})
        self._emit_stats()

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
        results, ms = self.store.search_text(m.text)
        hits = []
        for c, similar in results:
            if c.kind != "object" or c.payload.get("synthetic"):
                continue
            hits.append(
                {
                    "object_id": c.id,
                    "score": round(c.score, 3),
                    "similar": similar,
                    **self._object_meta(c.id, c.payload, c.from_mutable),
                }
            )
        self._emit({"type": "search_results", "text": m.text, "ms": round(ms, 2), "hits": hits})

    def _object_meta(self, object_id: str, pl: dict, local: bool) -> dict:
        count, last = self.sightings.get(object_id, (0, 0.0))
        return {
            "object_id": object_id,
            "label": pl.get("label", ""),
            "views": len(pl.get("views") or []),
            "thumb": pl.get("thumb", ""),
            "device": pl.get("device", ""),
            "local": local,
            "sightings": count or pl.get("sightings", 0),
            "last_seen": last or pl.get("t_seen", 0.0),
            "created": pl.get("t_created", 0.0),
        }

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

    def _on_manifestrequest(self, m: ManifestRequest):
        manifest = None
        try:
            if self.store.immutable is not None:
                manifest = self.store.immutable.snapshot_manifest()
        except Exception:
            # damaged mirror: rebuild it (disposable replica) rather than
            # leaving the sync worker timing out on every pull forever
            logger.exception("manifest failed; rebuilding the fleet mirror")
            self.store.reset_immutable()
            with contextlib.suppress(Exception):
                manifest = self.store.immutable.snapshot_manifest()
        finally:
            m.reply.put(manifest)

    def _on_drainsightings(self, m: DrainSightings):
        seen, self._fleet_seen = self._fleet_seen, set()
        m.reply.put(list(seen))

    def _on_preparepush(self, m: PreparePush):
        ids = m.object_ids
        if ids is None:  # auto-push sweep: every confirmed object not yet on the fleet
            ids = [
                pid
                for pid, pl, _ in self.store.scroll_objects(mutable_only=True)
                if not pl.get("t_sync") and not pl.get("synthetic")
            ]
        out = []
        for oid in ids:
            got = self.store.get_object(oid)
            if got is None:
                continue
            payload, rows = got
            if payload.get("kind") != "object" or payload.get("synthetic"):
                continue  # blocklist entries and stunt synthetics never reach the fleet
            views = list(payload.get("views") or [])
            out.append(
                {
                    "id": oid,
                    "label": payload.get("label", ""),
                    "rows": rows,
                    "views": views,
                    "neg": list(payload.get("neg") or []),
                    "thumb": payload.get("thumb", ""),
                    "device": payload.get("device", ""),
                    "event": payload.get("event", ""),
                    "t_created": payload.get("t_created"),
                    # what exactly went over the wire — MarkPushed must not stamp
                    # a point the user edited while the network push was in flight
                    "fingerprint": _push_fingerprint(payload),
                }
            )
        m.reply.put(out)

    def _on_markpushed(self, m: MarkPushed):
        for it in m.items:
            old_id = it["old_id"]
            got = self.store.get_object(old_id, with_vectors=False)
            if got is not None and it.get("fingerprint") is not None:
                if _push_fingerprint(got[0]) != it["fingerprint"]:
                    continue  # edited mid-push: stays dirty, next push carries the edit
            if "fleet_id" in it and it["fleet_id"] != old_id:
                # label-fold: rewrite under the fleet id — upsert FIRST, delete
                # after, so a failure never loses the local teaching
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
                self.store.delete(old_id)
                for ts in self.tracks.values():
                    if ts.object_id == old_id:
                        ts.object_id = it["fleet_id"]
            elif "fleet_id" in it:  # same-id fold: merged content, same point
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
            else:
                self.store.set_payload(old_id, {"t_sync": it["t_sync"]})
        self._emit({"type": "push_done", "count": len(m.items)})
        self._emit_stats()

    def _on_inventoryrequest(self, m: InventoryRequest):
        items = []
        for kind in ("object", "ignored"):
            for pid, pl, from_mut in self.store.scroll_objects(
                kind=kind, mutable_only=kind == "ignored"
            ):
                items.append(
                    {
                        **self._object_meta(pid, pl, from_mut),
                        "views": pl.get("views") or [],  # full row-aligned meta for curation
                        "pushed": bool(pl.get("t_sync")),
                        "ignored": kind == "ignored",
                    }
                )
        self._emit({"type": "inventory", "items": items})

    def _emit_stats(self):
        self._emit(
            {
                "type": "stats",
                "memories": self.store.vector_count(),
                "fleet": self.store.immutable is not None,
            }
        )
