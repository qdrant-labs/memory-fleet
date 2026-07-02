"""Scale spike: can Qdrant Edge hold a 'warehouse memory' and stay fast?

Builds an Edge shard with N objects (512-d vision vector + 3-row MAX_SIM
multivector each), runs optimize(), and measures query latency at increasing
point counts. Validates the '100k memories, still instant' demo stunt.
"""

import shutil
import sys
import tempfile
import time
import uuid

import numpy as np

from qdrant_edge import (CountRequest, Distance, EdgeConfig, EdgeShard,
                         EdgeVectorParams, HnswIndexConfig,
                         MultiVectorComparator, MultiVectorConfig, Point,
                         Query, QueryRequest, UpdateOperation)

DIM = 512
ROWS = 3
STAGES = [1_000, 10_000, 50_000, 100_000]
BATCH = 1_000

rng = np.random.default_rng(3)


def unit(n):
    v = rng.normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


work = tempfile.mkdtemp(prefix="hive-scale-")
config = EdgeConfig(vectors={
    "vision": EdgeVectorParams(size=DIM, distance=Distance.Cosine),
    "exemplars": EdgeVectorParams(
        size=DIM, distance=Distance.Cosine,
        multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MaxSim)),
})
shard = EdgeShard.create(work, config)

probe = unit(1)[0].tolist()


def bench(label):
    for using in ("vision", "exemplars"):
        q = QueryRequest(query=Query.Nearest([probe] if using == "exemplars" else probe,
                                             using=using),
                         limit=10, with_payload=True, with_vector=False)
        times = []
        for _ in range(30):
            t0 = time.perf_counter_ns()
            shard.query(q)
            times.append((time.perf_counter_ns() - t0) / 1e6)
        t = np.array(times)
        print(f"  {label:>8} | {using:9s} med {np.median(t):7.2f} ms  p95 {np.percentile(t, 95):7.2f} ms")


total = 0
t_start = time.time()
for stage in STAGES:
    while total < stage:
        n = min(BATCH, stage - total)
        vis = unit(n)
        pts = [Point(id=str(uuid.uuid4()),
                     vector={"vision": vis[i].tolist(),
                             "exemplars": unit(ROWS).tolist()},
                     payload={"label": f"obj-{total + i}"})
               for i in range(n)]
        shard.update(UpdateOperation.upsert_points(pts))
        total += n
    print(f"{total} points ({time.time() - t_start:.0f}s elapsed), pre-optimize:")
    bench(f"{total//1000}k raw")
    t0 = time.time()
    shard.optimize()
    print(f"  optimize() took {time.time() - t0:.1f}s")
    bench(f"{total//1000}k opt")

print("count:", shard.count(CountRequest()))
shard.close()
size_mb = sum(f.stat().st_size for f in __import__('pathlib').Path(work).rglob('*') if f.is_file()) / 1e6
print(f"shard on disk: {size_mb:.0f} MB")
shutil.rmtree(work)
