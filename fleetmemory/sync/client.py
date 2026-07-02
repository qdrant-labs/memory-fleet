"""Fleet client (PLAN.md §3.6): qdrant-client for curated push, raw snapshot
endpoints for pull. Works against Qdrant Cloud (primary) or the bundled Docker
fleet — same code path, verified in the §9.3/§3.6 spikes.
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
