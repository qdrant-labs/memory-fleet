"""Fleet client (PLAN.md §3.6): qdrant-client for curated push, raw snapshot
endpoints for pull. Works against Qdrant Cloud (primary) or the bundled Docker
fleet — same code path, verified in the §9.3/§3.6 spikes.
"""

import logging
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models
from qdrant_edge import Bm25, Bm25Config

logger = logging.getLogger(__name__)

COLLECTION = "fleet"


class FleetClient:
    def __init__(self, url: str, api_key: str | None, collection: str = COLLECTION, dim: int = 512):
        self.url = url.rstrip("/")
        self.api_key = api_key or None
        self.collection = collection
        self.dim = dim
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        self._bm25 = Bm25(Bm25Config())
        self._headers = {"api-key": api_key} if api_key else {}

    # ---------- schema ----------

    def ensure_collection(self):
        """Create the fleet collection with the §3.3 schema if it doesn't exist."""
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "exemplars": models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            },
            sparse_vectors_config={
                "label": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        logger.info("fleet collection %r created", self.collection)

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

    def find_by_label(self, label: str):
        """Exact-label fleet point (name == identity extended to the fleet) or None."""
        recs, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="label", match=models.MatchValue(value=label))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        return recs[0] if recs else None

    def _sparse_label(self, label: str) -> models.SparseVector:
        sv = self._bm25.embed_document(label or " ")  # same BM25 as on-device (qdrant_edge)
        return models.SparseVector(indices=list(sv.indices), values=list(sv.values))

    def upsert_object(self, point_id: str, label: str, rows: list, payload: dict):
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "exemplars": [list(map(float, r)) for r in rows],
                        "label": self._sparse_label(label),
                    },
                    payload=payload,
                )
            ],
        )
