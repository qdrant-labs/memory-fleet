"""Two-shard Edge storage (PLAN.md §3.3): mutable (local teachings) + immutable
(fleet mirror, snapshot-fed). Recognition fans out to both, merged, deduped by
point id — mutable wins ties. The shard is the source of truth for vectors;
nothing here caches them (PLAN.md §12.3).
"""

import contextlib
import shutil
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_edge import (
    Bm25,
    Bm25Config,
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeSparseVectorParams,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    Point,
    Query,
    QueryRequest,
    ScrollRequest,
    UpdateOperation,
)

DIM = 512  # Unicom-ViT-B-32; tests use smaller dims via the dim argument
LABEL_DENSE_DIM = 384  # bge-small-en-v1.5 (labels.py); field exists even in BM25 fallback
RRF_K = 2  # Qdrant's reciprocal-rank-fusion default
DENSE_LABEL_FLOOR = 0.6  # bge cosine below this is noise, not a semantic neighbor


def shard_config(dim: int) -> EdgeConfig:
    return EdgeConfig(
        vectors={
            "exemplars": EdgeVectorParams(
                size=dim,
                distance=Distance.Cosine,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MaxSim),
            ),
            "label_dense": EdgeVectorParams(size=LABEL_DENSE_DIM, distance=Distance.Cosine),
        },
        # IDF modifier serves both miniCOIL (requires it) and the BM25 fallback
        sparse_vectors={"label": EdgeSparseVectorParams(modifier=Modifier.Idf)},
    )


@dataclass(slots=True)
class Candidate:
    """One recognition hit, with everything the matcher needs to rerank it."""

    id: str
    score: float
    kind: str
    label: str
    payload: dict
    from_mutable: bool


@dataclass(slots=True)
class RecognitionResult:
    candidates: list[Candidate]
    latency_ms: float
    searched: int  # total exemplar vectors fanned out over


class Store:
    """Owns the Edge shards. All calls happen on the core thread."""

    def __init__(
        self,
        data_dir: str | Path,
        dim: int = DIM,
        with_immutable: bool = False,
        label_embedder=None,
    ):
        self.dim = dim
        self.data_dir = Path(data_dir)
        self._bm25 = Bm25(Bm25Config())
        # miniCOIL + dense (labels.LabelEmbedder) when models are available;
        # None -> BM25-only fallback (deterministic gate, CI-style sync tests)
        self.labels = label_embedder

        mut_dir = self.data_dir / "mutable"
        fresh = not mut_dir.exists()
        mut_dir.mkdir(parents=True, exist_ok=True)
        self.mutable = (
            EdgeShard.create(str(mut_dir), shard_config(dim))
            if fresh
            else EdgeShard.load(str(mut_dir))
        )
        if fresh:
            self.mutable.update(
                UpdateOperation.create_field_index("kind", PayloadSchemaType.Keyword)
            )
        else:
            self._migrate_label_dense(self.mutable)

        self.scale = None  # optional stunt shard, attached on demand
        self._vectors_dirty = True
        self._vector_cache = 0
        self.immutable = None
        immut_dir = self.data_dir / "immutable"
        if immut_dir.exists():
            self.immutable = EdgeShard.load(str(immut_dir))
            self._migrate_label_dense(self.immutable)
        elif with_immutable:
            immut_dir.mkdir(parents=True)
            self.immutable = EdgeShard.create(str(immut_dir), shard_config(dim))

    @staticmethod
    def _migrate_label_dense(shard: EdgeShard):
        """Shards created before the hybrid-search schema gain label_dense in
        place (verified on Edge 0.7.2; no-op error when it already exists)."""
        with contextlib.suppress(Exception):
            shard.update(
                UpdateOperation.create_dense_vector("label_dense", LABEL_DENSE_DIM, Distance.Cosine)
            )

    def close(self):
        self.mutable.close()
        if self.immutable is not None:
            self.immutable.close()
        self.detach_scale_shard()

    # ---------- recognition ----------

    def recognize(self, vec: np.ndarray, limit: int = 10) -> RecognitionResult:
        """MAX_SIM fanout over both shards; this latency is the HUD number."""
        req = QueryRequest(
            query=Query.Nearest([vec.tolist()], using="exemplars"),
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        t0 = time.perf_counter_ns()
        hits = [(p, True) for p in self.mutable.query(req)]
        if self.immutable is not None:
            hits += [(p, False) for p in self.immutable.query(req)]
        if self.scale is not None:
            hits += [(p, False) for p in self.scale.query(req)]
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        best: dict[str, Candidate] = {}
        for p, from_mut in hits:
            pid = str(p.id)
            prev = best.get(pid)
            # same id in both shards: the mutable copy is the local truth and
            # must win outright — an older mirror copy may legitimately score higher
            if prev is not None and prev.from_mutable and not from_mut:
                continue
            if prev is None or (from_mut and not prev.from_mutable) or p.score > prev.score:
                pl = p.payload or {}
                best[pid] = Candidate(
                    id=pid,
                    score=float(p.score),
                    kind=pl.get("kind", "object"),
                    label=pl.get("label", ""),
                    payload=pl,
                    from_mutable=from_mut,
                )
        cands = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return RecognitionResult(cands, latency_ms, self.vector_count())

    def count(self) -> int:
        n = self.mutable.count(CountRequest())
        if self.immutable is not None:
            n += self.immutable.count(CountRequest())
        if self.scale is not None:
            n += self.scale.count(CountRequest())
        return n

    def vector_count(self) -> int:
        """Total exemplar VECTORS in memory — the honest HUD number: an object
        with 12 views is 12 memories, not one. Recomputed lazily after
        mutations (payload scroll, cheap at demo scale); the stunt shard
        contributes 3 rows per point by construction (§9.4), no scroll."""
        if self._vectors_dirty:
            n = 0
            for shard in [self.mutable] + ([self.immutable] if self.immutable else []):
                offset = None
                while True:
                    recs, offset = shard.scroll(
                        ScrollRequest(
                            limit=256, offset=offset, with_payload=True, with_vector=False
                        )
                    )
                    for r in recs:
                        n += len((r.payload or {}).get("views") or []) or 1
                    if offset is None:
                        break
            self._vector_cache = n
            self._vectors_dirty = False
        scale_rows = 3 * self.scale.count(CountRequest()) if self.scale is not None else 0
        return self._vector_cache + scale_rows

    # ---------- mutable-shard writes (local teachings + blocklist) ----------

    def upsert_object(
        self,
        object_id: str,
        label: str,
        rows: list[np.ndarray],
        views: list[dict],
        *,
        kind: str = "object",
        neg: list | None = None,
        device: str = "",
        event: str = "",
        t_created: float | None = None,
        t_sync: float | None = None,
        thumb: str = "",
        base_payload: dict | None = None,
    ):
        """base_payload: when re-upserting an existing point, pass its current
        payload so auxiliary keys (sightings, t_seen, ...) survive the rewrite.
        t_sync=None means "this content is NOT on the fleet": any local edit to
        a pushed object must clear the old stamp, or the next pull's dedup
        would delete the edit (id present in mirror + stale t_sync)."""
        assert len(rows) == len(views), "exemplar rows and view metadata must stay row-aligned"
        payload = {
            **(base_payload or {}),
            "kind": kind,
            "label": label,
            "label_key": label.strip().lower(),  # case-insensitive identity, incl. fleet
            "device": device,
            "event": event,
            "t_created": t_created if t_created is not None else time.time(),
            "views": views,
            "neg": neg or [],
            "thumb": thumb,
        }
        if t_sync is not None:
            payload["t_sync"] = t_sync
        else:
            payload.pop("t_sync", None)
        vector = {"exemplars": [r.tolist() for r in rows]}
        if self.labels is not None:
            sparse, dense = self.labels.embed_doc(label)
            vector["label"] = sparse
            vector["label_dense"] = dense
            payload["label_v"] = 2  # hybrid-era label vectors (miniCOIL + dense)
        else:
            vector["label"] = self._bm25.embed_document(label or " ")
        self.mutable.update(
            UpdateOperation.upsert_points([Point(id=object_id, vector=vector, payload=payload)])
        )
        self._vectors_dirty = True

    def reembed_labels(self) -> int:
        """One-shot migration: points taught before hybrid search carry
        BM25-space label vectors that miniCOIL/dense queries can't see.
        Re-embed them in place. Idempotent (label_v marker); mutable only —
        the mirror re-fills from fleet pushes. Call on the core thread."""
        if self.labels is None:
            return 0
        stale = []
        for kind in ("object", "ignored"):
            for pid, pl, _ in self.scroll_objects(kind=kind, mutable_only=True):
                if pl.get("label_v") != 2:
                    stale.append(pid)
        for pid in stale:
            payload, rows = self.get_object(pid)
            self.upsert_object(
                pid,
                payload.get("label", ""),
                rows,
                list(payload.get("views") or []),
                kind=payload.get("kind", "object"),
                neg=payload.get("neg"),
                device=payload.get("device", ""),
                event=payload.get("event", ""),
                t_created=payload.get("t_created"),
                t_sync=payload.get("t_sync"),
                thumb=payload.get("thumb", ""),
                base_payload=payload,
            )
        return len(stale)

    def get_object(self, object_id: str, with_vectors: bool = True):
        """Mutable-shard point -> (payload, rows) or None."""
        recs = self.mutable.retrieve([object_id], with_payload=True, with_vector=with_vectors)
        if not recs:
            return None
        rec = recs[0]
        rows = []
        if with_vectors:
            rows = [np.asarray(r, dtype=np.float32) for r in rec.vector["exemplars"]]
        return rec.payload or {}, rows

    def delete(self, object_id: str):
        self.mutable.update(UpdateOperation.delete_points([object_id]))
        self._vectors_dirty = True

    def set_payload(self, object_id: str, payload: dict):
        self.mutable.update(UpdateOperation.set_payload(payload=payload, point_ids=[object_id]))

    def scroll_objects(self, kind: str = "object", mutable_only: bool = False) -> list:
        """Payload-only records, both shards (mutable first), synthetic excluded."""
        flt = Filter(must=[FieldCondition(key="kind", match=MatchValue(value=kind))])
        out = []
        shards = [self.mutable] + (
            [] if mutable_only or self.immutable is None else [self.immutable]
        )
        seen = set()
        for shard in shards:
            offset = None
            while True:
                recs, offset = shard.scroll(
                    ScrollRequest(
                        limit=256, offset=offset, filter=flt, with_payload=True, with_vector=False
                    )
                )
                for r in recs:
                    pid = str(r.id)
                    pl = r.payload or {}
                    if pid in seen or pl.get("synthetic"):
                        continue
                    seen.add(pid)
                    out.append((pid, pl, shard is self.mutable))
                if offset is None:
                    break
        return out

    def find_label(self, label: str) -> str | None:
        """Identity == label: A mutable object currently carrying this label
        (there may be several sibling buckets; this returns the first)."""
        pids = self.find_label_points(label)
        return pids[0][0] if pids else None

    def find_label_points(self, label: str) -> list[tuple[str, dict]]:
        """All mutable points carrying this label — identity is the LABEL;
        points are ≤VIEW_CAP-view buckets of it (hive scale: a name grows by
        adding sibling points, never by unbounded multivectors)."""
        needle = label.strip().lower()
        return [
            (pid, pl)
            for pid, pl, _ in self.scroll_objects(mutable_only=True)
            if pl.get("label", "").strip().lower() == needle
        ]

    # ---------- map support (decorative, off the hot path) ----------

    def object_mean_vectors(self):
        """(id, label, from_mutable, mean exemplar vector) per real object —
        synthetic excluded by construction (scale shard never scrolled)."""
        out = []
        for shard, from_mut in [(self.mutable, True), (self.immutable, False)]:
            if shard is None:
                continue
            offset = None
            while True:
                recs, offset = shard.scroll(
                    ScrollRequest(limit=128, offset=offset, with_payload=True, with_vector=True)
                )
                for r in recs:
                    pl = r.payload or {}
                    if pl.get("kind") != "object" or pl.get("synthetic"):
                        continue
                    rows = np.asarray(r.vector["exemplars"], dtype=np.float32)
                    out.append((str(r.id), pl.get("label", ""), from_mut, rows.mean(axis=0)))
                if offset is None:
                    break
        return out

    # ---------- hybrid search (BM25 + substring, dense visual expansion) ----------

    def get_rows(self, object_id: str) -> list[np.ndarray]:
        """Exemplar rows for a point in either shard (mutable first)."""
        for shard in [self.mutable] + ([self.immutable] if self.immutable else []):
            recs = shard.retrieve([object_id], with_payload=False, with_vector=True)
            if recs:
                return [np.asarray(r, dtype=np.float32) for r in recs[0].vector["exemplars"]]
        return []

    def _query_leg(self, query, limit: int) -> tuple[list[Candidate], int]:
        """One prefetch leg: run a query across both shards, dedup by id
        (mutable wins), rank by score. Returns (candidates, engine_ns)."""
        req = QueryRequest(query=query, limit=limit, with_payload=True, with_vector=False)
        best: dict[str, Candidate] = {}
        t0 = time.perf_counter_ns()
        for shard, from_mut in [(self.mutable, True)] + (
            [(self.immutable, False)] if self.immutable is not None else []
        ):
            for p in shard.query(req):
                pid = str(p.id)
                pl = p.payload or {}
                if pl.get("kind", "object") != "object" or pl.get("synthetic"):
                    continue  # blocklist entries and stunt synthetics never match labels
                prev = best.get(pid)
                if prev is not None and prev.from_mutable and not from_mut:
                    continue  # mutable copy is the local truth for a shared id
                if prev is None or (from_mut and not prev.from_mutable) or p.score > prev.score:
                    best[pid] = Candidate(
                        pid, float(p.score), "object", pl.get("label", ""), pl, from_mut
                    )
        engine_ns = time.perf_counter_ns() - t0
        return sorted(best.values(), key=lambda c: c.score, reverse=True), engine_ns

    def search_text(self, text: str, limit: int = 8) -> tuple[list[tuple[Candidate, bool]], float]:
        """Hybrid label search, the Qdrant pattern: two prefetch legs — miniCOIL
        sparse + dense text — fused with reciprocal-rank fusion (Edge 0.7.2
        exports Prefetch/Fusion but doesn't consume them yet, so the RRF step
        runs app-side; same math as the server's FusionQuery). A substring pass
        catches half-typed words, and the best hit's exemplars fan out over
        MAX_SIM to add visually similar memories (similar=True). Returns
        (results, engine_ms) — engine_ms is Edge query time only."""
        text = text.strip()
        if not text:
            return [], 0.0
        engine_ns = 0

        legs: list[list[Candidate]] = []
        if self.labels is not None:
            sparse, dense = self.labels.embed_query(text)
            leg, ns = self._query_leg(Query.Nearest(sparse, using="label"), 20)
            legs.append(leg)
            engine_ns += ns
            leg, ns = self._query_leg(Query.Nearest(dense, using="label_dense"), 20)
            # dense text similarity scores everything; keep semantic neighbors only
            legs.append([c for c in leg if c.score >= DENSE_LABEL_FLOOR])
            engine_ns += ns
        else:  # model-free fallback: single BM25 leg
            leg, ns = self._query_leg(
                Query.Nearest(self._bm25.embed_query(text), using="label"), 20
            )
            legs.append(leg)
            engine_ns += ns

        # reciprocal-rank fusion across the legs
        fused: dict[str, float] = {}
        by_id: dict[str, Candidate] = {}
        for leg in legs:
            for rank, cand in enumerate(leg):
                fused[cand.id] = fused.get(cand.id, 0.0) + 1.0 / (RRF_K + rank + 1)
                if cand.id not in by_id or cand.from_mutable:
                    by_id[cand.id] = cand
        top = max(fused.values(), default=1.0)
        ranked = []
        for pid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            c = by_id[pid]
            ranked.append(Candidate(c.id, score / top, c.kind, c.label, c.payload, c.from_mutable))

        # substring pass so half-typed words ("watc") still land
        needle = text.lower()
        for pid, pl, from_mut in self.scroll_objects():
            if pid not in fused and needle in pl.get("label", "").lower():
                ranked.append(Candidate(pid, 0.3, "object", pl.get("label", ""), pl, from_mut))
                fused[pid] = 0.3

        results: list[tuple[Candidate, bool]] = [(c, False) for c in ranked]

        if ranked:  # visual expansion from the strongest fused hit
            rows = self.get_rows(ranked[0].id)
            if rows:
                probe = np.mean(rows, axis=0)
                probe /= np.linalg.norm(probe) or 1.0
                t0 = time.perf_counter_ns()
                rec = self.recognize(probe, limit=5)
                engine_ns += time.perf_counter_ns() - t0
                extra = 0
                for cand in rec.candidates:
                    if cand.id in fused or cand.kind != "object" or cand.score < 0.45:
                        continue
                    results.append((cand, True))
                    extra += 1
                    if extra >= 3:
                        break
        return results[:limit], engine_ns / 1e6

    # ---------- scale stunt (PLAN.md §4 step 4) ----------

    def attach_scale_shard(self, path: str | Path) -> bool:
        """Fan recognition out over a prebuilt synthetic shard ("a year of robot
        memories"). Inventory/map never see it: its points are synthetic:true
        and scroll_objects only walks mutable+immutable."""
        path = Path(path)
        if self.scale is not None or not path.exists():
            return False
        self.scale = EdgeShard.load(str(path))
        return True

    def detach_scale_shard(self):
        if self.scale is not None:
            self.scale.close()
            self.scale = None

    # ---------- sync support (PLAN.md §3.3 dedup rule) ----------

    def reset_immutable(self):
        """The mirror is a disposable replica of the fleet — on damage, rebuild
        it empty and let the next pull re-seed it."""
        if self.immutable is None:
            return
        self.immutable.close()
        immut_dir = self.data_dir / "immutable"
        shutil.rmtree(immut_dir, ignore_errors=True)
        immut_dir.mkdir(parents=True)
        self.immutable = EdgeShard.create(str(immut_dir), shard_config(self.dim))
        self._vectors_dirty = True

    def apply_partial_snapshot(self, snapshot_path: str | Path) -> bool:
        """Apply a partial snapshot to the mirror. Returns False for an empty
        delta — an up-to-date mirror gets a zero-byte body (or a tar with no
        segments/), and Edge 0.7.2's update_from_snapshot chokes on both
        instead of no-opping."""
        assert self.immutable is not None, "partial snapshot without an immutable mirror"
        snapshot_path = Path(snapshot_path)
        if snapshot_path.stat().st_size == 0:
            return False
        with tarfile.open(snapshot_path) as tar:
            if not any(m.name.startswith("segments") for m in tar):
                return False
        self.immutable.update_from_snapshot(str(snapshot_path))
        self._vectors_dirty = True
        return True

    def dedup_after_pull(self) -> list[str]:
        """Delete a mutable point ONLY when it was pushed (t_sync stamped) AND its
        id now exists in the fresh immutable mirror. Never by bare timestamp —
        unpushed local objects must survive a pull."""
        if self.immutable is None:
            return []
        pushed = []
        offset = None
        while True:  # mutable shard only: hundreds of points, payload filter in-process
            recs, offset = self.mutable.scroll(
                ScrollRequest(limit=256, offset=offset, with_payload=True, with_vector=False)
            )
            pushed += [str(r.id) for r in recs if (r.payload or {}).get("t_sync")]
            if offset is None:
                break
        if not pushed:
            return []
        mirrored = {
            str(r.id)
            for r in self.immutable.retrieve(pushed, with_payload=False, with_vector=False)
        }
        doomed = [pid for pid in pushed if pid in mirrored]
        if doomed:
            self.mutable.update(UpdateOperation.delete_points(doomed))
            self._vectors_dirty = True
        return doomed


def new_id() -> str:
    return str(uuid.uuid4())
