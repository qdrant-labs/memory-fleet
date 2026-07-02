"""Verify Qdrant CLOUD exposes the shard-snapshot endpoints Edge sync needs.

Creates a tiny throwaway collection, runs the full flow from spike_sync.py
(full snapshot -> EdgeShard -> MAX_SIM query -> partial snapshot -> update),
then deletes the collection. Reads QDRANT_URL / QDRANT_API_KEY from the
hive-mind .env; never prints the key.
"""

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import requests

env = {}
for line in Path("/Users/dylanc/Documents/GitHub/hive-mind/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")

URL = env["QDRANT_URL"].rstrip("/")
KEY = env["QDRANT_API_KEY"]
H = {"api-key": KEY}
COLL = "fleet-endpoint-check"
DIM = 16

print("cluster:", URL.split("//")[1].split(".")[0][:8] + "…")
r = requests.get(URL, headers=H, timeout=15)
print("server version:", r.json().get("version"))

from qdrant_client import QdrantClient, models

client = QdrantClient(url=URL, api_key=KEY, timeout=30)
if client.collection_exists(COLL):
    client.delete_collection(COLL)
client.create_collection(
    collection_name=COLL,
    vectors_config={
        "exemplars": models.VectorParams(
            size=DIM, distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM)),
    },
)

rng = np.random.default_rng(5)


def multivec(n):
    v = rng.normal(size=(n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.tolist()


pts = [models.PointStruct(id=str(uuid.uuid4()),
                          vector={"exemplars": multivec(3)},
                          payload={"label": f"item-{i}", "t_sync": time.time()})
       for i in range(3)]
client.upsert(COLL, points=pts)
print("upserted 3 multivector points")

from qdrant_edge import CountRequest, EdgeShard, Query, QueryRequest

work = Path(tempfile.mkdtemp(prefix="fleet-cloud-check-"))
try:
    # 1. full shard snapshot
    r = requests.get(f"{URL}/collections/{COLL}/shards/0/snapshot",
                     headers=H, stream=True, timeout=60)
    print("GET shards/0/snapshot ->", r.status_code)
    r.raise_for_status()
    snap = work / "full.snapshot"
    with open(snap, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print(f"  downloaded {snap.stat().st_size/1024:.0f} KiB")

    shard_dir = work / "immutable"
    shard_dir.mkdir()
    EdgeShard.unpack_snapshot(str(snap), str(shard_dir))
    shard = EdgeShard.load(str(shard_dir))
    print("  EdgeShard loaded from cloud snapshot, points:", shard.count(CountRequest()))

    probe = pts[0].vector["exemplars"][0]
    res = shard.query(QueryRequest(query=Query.Nearest([probe], using="exemplars"),
                                   limit=1, with_payload=True, with_vector=False))
    assert res[0].payload["label"] == "item-0"
    print("  MAX_SIM query on cloud-born shard: OK")

    # 2. partial snapshot
    client.upsert(COLL, points=[models.PointStruct(
        id=str(uuid.uuid4()), vector={"exemplars": multivec(2)},
        payload={"label": "late", "t_sync": time.time()})])

    manifest = shard.snapshot_manifest()
    r = requests.post(f"{URL}/collections/{COLL}/shards/0/snapshot/partial/create",
                      headers=H, json=manifest, stream=True, timeout=60)
    print("POST partial/create ->", r.status_code)
    r.raise_for_status()
    part = work / "partial.snapshot"
    with open(part, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    shard.update_from_snapshot(str(part))
    n = shard.count(CountRequest())
    print("  after partial update, points:", n)
    assert n == 4
    shard.close()
    print("\nCLOUD SYNC ENDPOINTS: ALL VERIFIED")
finally:
    client.delete_collection(COLL)
    shutil.rmtree(work, ignore_errors=True)
    print("cleaned up (collection deleted)")
