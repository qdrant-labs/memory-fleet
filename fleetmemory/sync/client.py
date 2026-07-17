"""Fleet client: qdrant-client for curated push, raw snapshot endpoints for
pull. Runs against a Qdrant Cloud cluster (the shared fleet).
"""

import logging
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models
from qdrant_edge import Bm25, Bm25Config

from fleetmemory.memory.store import LABEL_DENSE_DIM

logger = logging.getLogger(__name__)

COLLECTION = "fleet"


class FleetClient:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection: str = COLLECTION,
        dim: int = 512,
        label_embedder=None,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key or None
        self.collection = collection
        self.dim = dim
        self.labels = label_embedder  # same hybrid embeddings as on-device
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        self._bm25 = Bm25(Bm25Config())
        self._headers = {"api-key": api_key} if api_key else {}

    # ---------- schema ----------

    def _vectors_config(self):
        return {
            "exemplars": models.VectorParams(
                size=self.dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            "label_dense": models.VectorParams(
                size=LABEL_DENSE_DIM, distance=models.Distance.COSINE
            ),
        }

    def ensure_collection(self):
        """Create the fleet collection with the current schema. An EMPTY
        collection on an older schema (no label_dense) is recreated in place —
        adding named vectors server-side needs Qdrant >= 1.18, and Cloud runs
        1.17. A non-empty old-schema collection is left alone with a warning."""
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vecs = info.config.params.vectors or {}
            if isinstance(vecs, dict) and "label_dense" in vecs:
                self._ensure_indexes()
                return
            if info.points_count == 0:
                self.client.delete_collection(self.collection)
                logger.info("fleet collection %r: empty, old schema — recreating", self.collection)
            else:
                logger.warning(
                    "fleet collection %r predates hybrid search and has data; "
                    "label_dense pushes disabled for it",
                    self.collection,
                )
                self.labels = None
                return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self._vectors_config(),
            sparse_vectors_config={
                "label": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        self._ensure_indexes()
        logger.info("fleet collection %r created", self.collection)

    def _ensure_indexes(self):
        # Cloud rejects filtered scrolls on unindexed payload keys; the
        # label-fold push looks objects up by label_key (case-insensitive)
        self.client.create_payload_index(
            self.collection, "label_key", models.PayloadSchemaType.KEYWORD
        )
        # the decay formula (fleet sleep) reads t_seen server-side — formula
        # variables need an index of a numeric type
        self.client.create_payload_index(self.collection, "t_seen", models.PayloadSchemaType.FLOAT)

    # ---------- pull (native snapshots) ----------

    def download_partial_snapshot(self, manifest: dict, dest: Path) -> Path:
        r = requests.post(
            f"{self.url}/collections/{self.collection}/shards/0/snapshot/partial/create",
            headers=self._headers,
            json=manifest,
            stream=True,
            timeout=120,
        )
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):  # chunked; r.raw corrupts the tar
                f.write(chunk)
        return dest

    # ---------- push (curated upsert) ----------

    def get_point(self, point_id: str):
        """The fleet point with this exact id, or None. Same-id must always
        fold at push time — a renamed local copy would miss the label lookup
        and a plain upsert would clobber views other units folded in."""
        recs = self.client.retrieve(
            self.collection, ids=[point_id], with_payload=True, with_vectors=True
        )
        return recs[0] if recs else None

    def find_by_label(self, label: str, limit: int = 8) -> list:
        """Fleet points carrying this label (instance model: several distinct
        items may share a display name). Case-insensitive via label_key."""
        recs, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="label_key",
                        match=models.MatchValue(value=label.strip().lower()),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        return recs

    def touch_seen(self, ids: list, t: float, device: str = ""):
        """Freshness heartbeat: recognized objects get t_seen stamped so the
        decay job spares them, and last_seen_device records which unit saw them
        (recall's 'which room'). Ids not on the fleet (still-dirty locals) are
        silently skipped by Qdrant."""
        payload = {"t_seen": t}
        if device:
            payload["last_seen_device"] = device
        self.client.set_payload(self.collection, payload=payload, points=ids)

    # ---------- operator console (hidden /fleet-ops view) ----------

    def scroll_all(self, with_vectors: bool = False) -> list:
        """Every fleet point (paged). Payload-only by default — the ops view
        renders labels and thumbs, not vectors."""
        out, offset = [], None
        while True:
            recs, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            out.extend(recs)
            if offset is None:
                return out

    def delete(self, point_id: str):
        self.client.delete(self.collection, points_selector=[point_id])

    def merge(self, keep_id: str, fold_id: str) -> bool:
        """Fold one fleet point into another (dedup duplicate instances). Same
        diversity-gated, view-capped fold as local merge and push; keep's label
        and payload win. Object+object or ignored+ignored only."""
        from fleetmemory.memory.core import fold_rows

        if keep_id == fold_id:
            return False
        keep, fold = self.get_point(keep_id), self.get_point(fold_id)
        if keep is None or fold is None:
            return False
        kp, fp = keep.payload or {}, fold.payload or {}
        if kp.get("kind", "object") != fp.get("kind", "object"):
            return False
        krows = (keep.vector or {}).get("exemplars") or []
        frows = (fold.vector or {}).get("exemplars") or []
        kviews, fviews = list(kp.get("views") or []), list(fp.get("views") or [])
        rows, views = fold_rows(krows, kviews, frows, fviews)
        self.upsert_object(keep_id, kp.get("label", ""), rows, {**kp, "views": views})
        self.delete(fold_id)
        return True

    def prune_view(self, point_id: str, view_id: str) -> bool:
        """Drop one exemplar from a fleet point (a wrong crop that snuck in).
        Rows and views are index-aligned; remove both at that index. The last
        vector gone means the whole point goes (nothing left to recognize)."""
        rec = self.get_point(point_id)
        if rec is None:
            return False
        rows = list((rec.vector or {}).get("exemplars") or [])
        views = list((rec.payload or {}).get("views") or [])
        idx = next((i for i, v in enumerate(views) if v.get("view_id") == view_id), None)
        if idx is None or idx >= len(rows):
            return False
        rows.pop(idx)
        views.pop(idx)
        if not rows:
            self.delete(point_id)
            return True
        pl = rec.payload or {}
        self.upsert_object(point_id, pl.get("label", ""), rows, {**pl, "views": views})
        return True

    def relabel(self, point_id: str, label: str) -> bool:
        """Rename a fleet point in place. The label vectors encode the old
        name, so re-upsert (which re-embeds them) rather than just setting
        the payload; keeps label_key in step for case-insensitive folds."""
        rec = self.get_point(point_id)
        if rec is None:
            return False
        rows = (rec.vector or {}).get("exemplars") or []
        payload = {**(rec.payload or {}), "label": label, "label_key": label.strip().lower()}
        self.upsert_object(point_id, label, rows, payload)
        return True

    def upsert_object(self, point_id: str, label: str, rows: list, payload: dict):
        vector = {"exemplars": [list(map(float, r)) for r in rows]}
        if self.labels is not None:  # miniCOIL sparse + dense, same models as on-device
            sparse, dense = self.labels.embed_doc(label)
            vector["label"] = models.SparseVector(
                indices=list(sparse.indices), values=list(sparse.values)
            )
            vector["label_dense"] = dense
        else:  # model-free fallback (sync tests): on-device BM25
            sv = self._bm25.embed_document(label or " ")
            vector["label"] = models.SparseVector(indices=list(sv.indices), values=list(sv.values))
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
