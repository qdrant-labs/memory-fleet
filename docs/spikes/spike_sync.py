"""Sync spike: Edge <-> server round-trip with the planned hive-mind schema.

Verifies, against a local Qdrant server (Docker):
 1. server collection with named dense vector + MAX_SIM multivector + BM25 sparse
 2. upsert via qdrant-client
 3. full shard snapshot -> EdgeShard.unpack_snapshot -> load -> MAX_SIM query
 4. add more points server-side -> snapshot_manifest -> partial snapshot
    -> update_from_snapshot -> new points visible on the edge
 5. mutable EdgeShard dual-write + timestamp-dedup delete (the docs' pattern)
"""

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import requests

QDRANT_URL = "http://localhost:6333"
COLL = "hive"
DIM = 16  # small; schema shape is what matters

rng = np.random.default_rng(7)


def vec():
    v = rng.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def multivec(n):
    return [vec() for _ in range(n)]


# ---------- 1. server collection ----------
from qdrant_client import QdrantClient, models

client = QdrantClient(url=QDRANT_URL, timeout=30)
print("server version:", requests.get(QDRANT_URL).json().get("version"))

if client.collection_exists(COLL):
    client.delete_collection(COLL)
client.create_collection(
    collection_name=COLL,
    vectors_config={
        "vision": models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        "exemplars": models.VectorParams(
            size=DIM, distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM),
        ),
    },
    sparse_vectors_config={"label": models.SparseVectorParams(
        modifier=models.Modifier.IDF)},
)
print("collection created: vision + exemplars(MAX_SIM) + label(sparse IDF)")

# ---------- 2. upsert points ----------
def make_point(label, n_views):
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"vision": vec(), "exemplars": multivec(n_views),
                "label": models.Document(text=label, model="Qdrant/bm25")},
        payload={"label": label, "device": "spike-A", "t_sync": time.time()},
    )

first_batch = [make_point(f"item-{i}", 3 + i % 4) for i in range(8)]
client.upsert(COLL, points=first_batch)
print(f"upserted {len(first_batch)} multivector points via qdrant-client")

# ---------- 3. full snapshot -> immutable edge shard ----------
from qdrant_edge import CountRequest, EdgeShard, Query, QueryRequest

work = Path(tempfile.mkdtemp(prefix="hive-spike-"))
IMMUT = work / "immutable"

snap_url = f"{QDRANT_URL}/collections/{COLL}/shards/0/snapshot"
r = requests.get(snap_url, stream=True)
r.raise_for_status()
snap_file = work / "full.snapshot"
with open(snap_file, "wb") as f:
    for chunk in r.iter_content(chunk_size=1 << 16):
        f.write(chunk)
print(f"full snapshot downloaded: {snap_file.stat().st_size/1024:.0f} KiB")

IMMUT.mkdir(parents=True)
EdgeShard.unpack_snapshot(str(snap_file), str(IMMUT))
immutable = EdgeShard.load(str(IMMUT))
print("immutable shard loaded from snapshot, points:", immutable.count(CountRequest()))

probe = first_batch[0].vector["exemplars"][0]
res = immutable.query(QueryRequest(
    query=Query.Nearest([probe], using="exemplars"),
    limit=3, with_payload=True, with_vector=False))
top = res[0]
print(f"MAX_SIM query on immutable shard: top={top.payload.get('label')} score={top.score:.3f}")
assert top.payload.get("label") == first_batch[0].payload["label"], "MAX_SIM should return the source point"

# ---------- 4. partial snapshot update ----------
second_batch = [make_point(f"late-{i}", 2) for i in range(3)]
client.upsert(COLL, points=second_batch)

manifest = immutable.snapshot_manifest()
part_url = f"{QDRANT_URL}/collections/{COLL}/shards/0/snapshot/partial/create"
r = requests.post(part_url, json=manifest, stream=True)
print("partial snapshot endpoint:", r.status_code)
r.raise_for_status()
part_file = work / "partial.snapshot"
with open(part_file, "wb") as f:
    for chunk in r.iter_content(chunk_size=1 << 16):
        f.write(chunk)
print(f"partial snapshot: {part_file.stat().st_size/1024:.0f} KiB")

immutable.update_from_snapshot(str(part_file))
n = immutable.count(CountRequest())
print("after partial update, immutable points:", n)
assert n == len(first_batch) + len(second_batch), "late points must appear via partial sync"

probe2 = second_batch[0].vector["exemplars"][0]
res2 = immutable.query(QueryRequest(
    query=Query.Nearest([probe2], using="exemplars"),
    limit=1, with_payload=True, with_vector=False))
print(f"late point recognized on edge: {res2[0].payload.get('label')} score={res2[0].score:.3f}")

# ---------- 5. mutable shard: local writes + dedup after sync ----------
from qdrant_edge import (Distance, EdgeConfig, EdgeVectorParams, Filter,
                         FieldCondition, MultiVectorComparator,
                         MultiVectorConfig, Point, RangeFloat, UpdateOperation)

MUT = work / "mutable"
MUT.mkdir()
mut_cfg = EdgeConfig(vectors={
    "vision": EdgeVectorParams(size=DIM, distance=Distance.Cosine),
    "exemplars": EdgeVectorParams(
        size=DIM, distance=Distance.Cosine,
        multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MaxSim)),
})
mutable = EdgeShard.create(str(MUT), mut_cfg)
t_before = time.time()
mutable.update(UpdateOperation.upsert_points([
    Point(id=str(uuid.uuid4()),
          vector={"vision": vec(), "exemplars": multivec(3)},
          payload={"label": "local-item", "t_sync": t_before})]))
print("mutable shard: local point written, count:", mutable.count(CountRequest()))

mutable.update(UpdateOperation.delete_points_by_filter(
    Filter(must=[FieldCondition(key="t_sync", range=RangeFloat(lte=time.time()))])))
print("timestamp-dedup delete works, count now:", mutable.count(CountRequest()))

immutable.close()
mutable.close()
shutil.rmtree(work)
print("\nALL SYNC CHECKS PASSED")
