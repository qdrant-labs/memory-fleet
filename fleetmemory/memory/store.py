"""Two-shard Edge storage (PLAN.md §3.3): mutable (local teachings) + immutable
(fleet mirror, snapshot-fed). Recognition fans out to both, merged, deduped by
point id — mutable wins ties. The shard is the source of truth for vectors;
nothing here caches them (PLAN.md §12.3).
"""

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


def shard_config(dim: int) -> EdgeConfig:
    return EdgeConfig(
        vectors={
            "exemplars": EdgeVectorParams(
                size=dim,
                distance=Distance.Cosine,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MaxSim),
            )
        },
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
    searched: int  # total points fanned out over


class Store:
    """Owns the Edge shards. All calls happen on the core thread."""

    def __init__(self, data_dir: str | Path, dim: int = DIM, with_immutable: bool = False):
        self.dim = dim
        self.data_dir = Path(data_dir)
        self._bm25 = Bm25(Bm25Config())

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

        self.scale = None  # optional stunt shard, attached on demand
        self.immutable = None
        immut_dir = self.data_dir / "immutable"
        if immut_dir.exists():
            self.immutable = EdgeShard.load(str(immut_dir))
        elif with_immutable:
            immut_dir.mkdir(parents=True)
            self.immutable = EdgeShard.create(str(immut_dir), shard_config(dim))

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
            if prev is None or p.score > prev.score or (from_mut and not prev.from_mutable):
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
        return RecognitionResult(cands, latency_ms, self.count())

    def count(self) -> int:
        n = self.mutable.count(CountRequest())
        if self.immutable is not None:
            n += self.immutable.count(CountRequest())
        if self.scale is not None:
            n += self.scale.count(CountRequest())
        return n

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
        payload so auxiliary keys (sightings, t_seen, ...) survive the rewrite."""
        assert len(rows) == len(views), "exemplar rows and view metadata must stay row-aligned"
        payload = {
            **(base_payload or {}),
            "kind": kind,
            "label": label,
            "device": device,
            "event": event,
            "t_created": t_created if t_created is not None else time.time(),
            "views": views,
            "neg": neg or [],
            "thumb": thumb,
        }
        if t_sync is not None:
            payload["t_sync"] = t_sync
        self.mutable.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        id=object_id,
                        vector={
                            "exemplars": [r.tolist() for r in rows],
                            "label": self._bm25.embed_document(label or " "),
                        },
                        payload=payload,
                    )
                ]
            )
        )

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
        """name == identity: the mutable object currently carrying this label."""
        for pid, pl, _ in self.scroll_objects(mutable_only=True):
            if pl.get("label", "").strip().lower() == label.strip().lower():
                return pid
        return None

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

    def search_text(self, text: str, limit: int = 8) -> tuple[list[tuple[Candidate, bool]], float]:
        """Hybrid label search. Lexical: BM25 sparse + substring (partial words).
        Then dense expansion: the best hit's exemplars fan out over MAX_SIM to
        surface visually similar memories (flagged similar=True). Returns
        (results, engine_ms) — engine_ms is Edge query time only, the HUD number."""
        text = text.strip()
        if not text:
            return [], 0.0
        shards = [(self.mutable, True)] + (
            [(self.immutable, False)] if self.immutable is not None else []
        )
        engine_ns = 0

        req = QueryRequest(
            query=Query.Nearest(self._bm25.embed_query(text), using="label"),
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        best: dict[str, Candidate] = {}
        t0 = time.perf_counter_ns()
        for shard, from_mut in shards:
            for p in shard.query(req):
                pid = str(p.id)
                if pid not in best or p.score > best[pid].score:
                    pl = p.payload or {}
                    best[pid] = Candidate(
                        pid,
                        float(p.score),
                        pl.get("kind", "object"),
                        pl.get("label", ""),
                        pl,
                        from_mut,
                    )
        engine_ns += time.perf_counter_ns() - t0

        # substring pass so half-typed words ("watc") still land — payload scan,
        # fine at demo scale, never runs against the synthetic stunt shard
        needle = text.lower()
        for pid, pl, from_mut in self.scroll_objects():
            if pid not in best and needle in pl.get("label", "").lower():
                best[pid] = Candidate(pid, 0.5, "object", pl.get("label", ""), pl, from_mut)

        lexical = sorted(best.values(), key=lambda c: c.score, reverse=True)
        results: list[tuple[Candidate, bool]] = [(c, False) for c in lexical]

        if lexical:  # dense expansion from the strongest lexical hit
            rows = self.get_rows(lexical[0].id)
            if rows:
                probe = np.mean(rows, axis=0)
                probe /= np.linalg.norm(probe) or 1.0
                t0 = time.perf_counter_ns()
                rec = self.recognize(probe, limit=5)
                engine_ns += time.perf_counter_ns() - t0
                extra = 0
                for cand in rec.candidates:
                    if cand.id in best or cand.kind != "object" or cand.score < 0.45:
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
        return doomed


def new_id() -> str:
    return str(uuid.uuid4())
